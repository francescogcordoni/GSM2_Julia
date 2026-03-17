using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using CUDA
using Statistics: mean

nthreads()

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

#& Stopping power
sp = load_stopping_power()

#~ ============================================================
#~ GSM2 parameters
#~ ============================================================

#& G1 -> 12h
a_G1 = 0.012872261720543399;  b_G1 = 0.04029756109753225;  r_G1 = 2.780479661191086

#& S -> 8h
a_S  = 0.00589118894714544;   b_S  = 0.05794352736120672;  r_S  = 5.84009601901114

#& G2 - M -> 3h + 1h
a_G2 = 0.024306291709970018;  b_G2 = 5.704688326522623e-5; r_G2 = 1.7720064637774506

#& mixed
a  = 0.01481379648786136;     b  = 0.012663276476522422;   r  = 2.5656972960759896
rd = 0.8;                     Rn = 7.2

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)   #! G1
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)   #! S
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)   #! G2 - M
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)    #! mixed

setup_GSM2!(r, a, b, rd, Rn)

#~ ============================================================
#~ Shared simulation settings
#~ ============================================================
dose         = 1.5
tumor_radius = 300.0
times_split  = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10.,
                12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0,
                25., 26., 27., 30., 48.0]
nsim         = 10
phase_keys   = ("G0", "G1", "S", "G2", "M")

mkpath("results")

#~ ============================================================
#~ Helpers
#~ ============================================================
function reset_cell!(dst::DataFrame, src::DataFrame)
    for col in names(src)
        S = src[!, col]; D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S); copyto!(D[i], S[i]); end
        else
            copyto!(D, S)
        end
    end
end

function apply_spheroid_mask!(df::DataFrame; radius::Float64 = 300.0, spacing::Int = 30)
    df.is_cell = ifelse.(
        (df.x .^ 2 .+ df.y .^ 2 .+ df.z .^ 2 .<= radius^2) .&
        ((df.x .÷ spacing .+ df.y .÷ spacing .+ df.z .÷ spacing) .% 2 .== 0),
        1, 0)
    for i in 1:nrow(df)
        df.number_nei[i] = length(df.nei[i]) - sum(df.is_cell[df.nei[i]])
    end
end

#~ ============================================================
#~ CORE: run one split-dose condition
#~   Returns (phase_times_pre, surv_prob) and saves CSV to results/
#~ ============================================================
function run_split_condition(
        label::String,
        E::Float64,
        particle::String,
        dose::Float64,
        tumor_radius::Float64,
        gsm2_cycle::Vector{GSM2},
        times_split::Vector{Float64};
        nsim::Int       = 10,
        phase_keys      = ("G0", "G1", "S", "G2", "M"),
        outdir::String  = "results")

    println("\n", "="^60)
    println("  CONDITION: $label  ($particle  E=$(E) MeV  dose=$(dose) Gy)")
    println("="^60)

    # ── 1. First irradiation ─────────────────────────────────────────────────
    setup(E, particle, dose, tumor_radius)

    cell_df_copy = deepcopy(cell_df)
    cell_df.O   .= 21.0
    apply_spheroid_mask!(cell_df_copy; radius = tumor_radius)
    cell_df_copy.cell_cycle .= "G1"
    cell_df_copy.can_divide .= 0

    @time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy,
                        df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    Ntot = count(cell_df_copy.is_cell .== 1)
    println("  Ntot = $Ntot")

    # ── 2. Second irradiation (instantaneous) ────────────────────────────────
    # Rebuild geometry for second fraction (tumor_radius = 500 in original)
    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr = "false", track_seg = track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox,
                           N_CellsSide, gsm2_cycle[1])
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
    R_beam_2, _, _ = calculate_beam_properties(calc_type, target_geom,
                                                X_box, X_voxel, 500.0)
    cell_df.O .= 21.0

    cell_df_second = deepcopy(cell_df)
    @time MC_dose_fast!(ion, Npar, R_beam_2, irrad_cond, cell_df_second,
                        df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

    # ── 3. ABM split-time loop ───────────────────────────────────────────────
    cell_df_ref = deepcopy(cell_df_copy)
    cell_work   = deepcopy(cell_df_ref)

    phase_times_pre = DataFrame(time   = Float64[], Nalive = Float64[],
                                G0     = Float64[], G1     = Float64[],
                                S      = Float64[], G2     = Float64[],
                                M      = Float64[])
    surv_prob = Vector{Float64}()

    for t in times_split
        println("  ABM t = $t h")

        Nalive_pre_acc = 0.0
        phase_pre_acc  = zeros(5)
        surv_acc       = 0.0
        n_valid        = 0

        for sim in 1:nsim
            try
                reset_cell!(cell_work, cell_df_ref)

                compute_times_domain!(cell_work, gsm2_cycle; terminal_time = t)
                run_simulation_abm!(cell_work; terminal_time = t, verbose = false)

                counts_pre = count_phase_alive(cell_work; phase_col = :cell_cycle)
                alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

                for i in findall(cell_work.is_cell .== 1)
                    cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                    cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
                end

                compute_times_domain!(cell_work, gsm2_cycle)
                alive_post = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

                n_valid        += 1
                Nalive_pre_acc += count(alive_pre)
                surv_acc       += count(alive_post) / Ntot
                for (i, k) in enumerate(phase_keys)
                    phase_pre_acc[i] += counts_pre[k]
                end

            catch e
                @warn "sim=$sim t=$t failed" exception = (e, catch_backtrace())
            end
        end

        inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
        pre   = phase_pre_acc .* inv_n
        push!(phase_times_pre, (t, Nalive_pre_acc * inv_n,
                                pre[1], pre[2], pre[3], pre[4], pre[5]))
        push!(surv_prob, surv_acc * inv_n)

        GC.gc()
    end

    # ── 4. Prepend t = 0 ────────────────────────────────────────────────────
    times_with_zero = vcat(0.0, times_split)
    phase_times_pre[!, :surv_prob] = surv_prob
    phase_times_pre[!, :type]      .= label

    # ── 5. Save ──────────────────────────────────────────────────────────────
    path = joinpath(outdir, "phase_times_$(label).csv")
    CSV.write(path, phase_times_pre)
    println("  Saved → $path")

    return phase_times_pre, surv_prob
end

#~ ============================================================
#~ RUN: 6 conditions
#~ ============================================================
CONDITIONS = [
    ("1H_2",   2.0,   "1H",  dose, tumor_radius),
    ("1H_10",  10.0,  "1H",  dose, tumor_radius),
    ("1H_100", 100.0, "1H",  dose, tumor_radius),
    ("12C_15", 15.0,  "12C", dose, tumor_radius),
    ("12C_20", 20.0,  "12C", dose, tumor_radius),
    ("12C_80", 80.0,  "12C", dose, tumor_radius),
]

results = Dict{String, NamedTuple}()

for (label, E, particle, d, r) in CONDITIONS
    pt_pre, sp = run_split_condition(
        label, E, particle, d, r,
        gsm2_cycle, times_split;
        nsim = nsim, phase_keys = phase_keys)
    results[label] = (phase_times_pre = pt_pre, surv_prob = sp)
end

#~ ============================================================
#~ Combine and save master CSV
#~ ============================================================
all_dfs = [results[lbl].phase_times_pre for (lbl, _, _, _, _) in CONDITIONS]
phase_times_pre_all = vcat(all_dfs...; cols = :union)
CSV.write("results/phase_times_all.csv", phase_times_pre_all)
println("\nSaved: results/phase_times_all.csv")

#~ ============================================================
#~ Assertions
#~ ============================================================
for (label, _, _, _, _) in CONDITIONS
    pt = results[label].phase_times_pre
    sp = results[label].surv_prob
    @assert nrow(pt) == length(sp) "Row/surv_prob mismatch for $label"
end
println("All assertions passed.")

#~ ============================================================
#~ FINAL PRINT
#~ ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO results/")
println("="^60)
for f in sort(readdir("results"))
    println("  results/$f")
end
