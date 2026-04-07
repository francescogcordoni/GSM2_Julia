using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using StatsPlots
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using CUDA
using Statistics: mean

nthreads()

# ============================================================
# Load functions
# ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
include(joinpath(@__DIR__, "..", "src", "migration_ABM.jl"))   # ← random-walk migration

# ============================================================
# Stopping power
# ============================================================
sp = load_stopping_power()

# ============================================================
# GSM2 parameters
# ============================================================
a_G1 = 0.012872261720543399;  b_G1 = 0.04029756109753225;  r_G1 = 2.780479661191086
a_S  = 0.00589118894714544;   b_S  = 0.05794352736120672;  r_S  = 5.84009601901114
a_G2 = 0.024306291709970018;  b_G2 = 5.704688326522623e-5; r_G2 = 1.7720064637774506
a    = 0.01481379648786136;   b    = 0.012663276476522422; r    = 2.5656972960759896
rd   = 0.8;  Rn = 7.2

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)

setup_GSM2!(r, a, b, rd, Rn)

# ============================================================
# Migration parameters
#   D_mig  : random-motility coefficient [µm²/h]
#             set to 0.0 to disable migration and recover
#             the original run_simulation_abm! behaviour exactly
#   h_mig  : lattice spacing = 2·R_cell [µm]
# ============================================================
const D_mig = 200.0          # µm²/h  — tune per cell line
const h_mig = 2.0 * 15.0    # µm     — one cell diameter

# ============================================================
# Helper: reset DataFrame in-place between Monte-Carlo runs
# ============================================================
function reset_cell!(dst::DataFrame, src::DataFrame)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

# ============================================================
# Helper: run one ABM window (with or without migration)
#   Wraps both the DataFrame and CellPopulation paths so the
#   rest of the script never needs to know which runner is used.
# ============================================================
function run_abm!(cell_work::DataFrame;
                  terminal_time::Float64,
                  verbose::Bool = false)
    if D_mig > 0.0
        run_simulation_abm_migration!(cell_work, D_mig, h_mig;
            nat_apo           = 1e-10,
            terminal_time     = terminal_time,
            snapshot_times    = Int[],
            print_interval    = terminal_time + 1.0,
            verbose           = verbose,
            return_dataframes = false,
            update_input      = true)
    else
        run_simulation_abm!(cell_work;
            nat_apo       = 1e-10,
            terminal_time = terminal_time,
            verbose       = verbose)
    end
end

# ============================================================
# Per-scenario simulation loop
#   Runs the full split-dose experiment for one particle/energy
#   combination and returns all outputs needed for plotting.
# ============================================================
function run_scenario(
        label::String,
        E::Float64, particle::String, dose::Float64, tumor_radius::Float64,
        X_box::Float64, X_voxel::Float64, R_cell::Float64,
        target_geom::String, calc_type::String, type_AT::String, track_seg::Bool,
        times_split::Vector{Float64},
        gsm2_cycle::Vector{GSM2};
        nsim::Int = 10)

    println("\n", "="^60)
    println("  SCENARIO: $label")
    println("="^60)

    # ── lattice & population ──────────────────────────────────────────────────
    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)
    R_beam, _, _ = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2_cycle[4])
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
    # at, df_center_x, df_center_y are DataFrames set as globals by setup_irrad_conditions!

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])
    cell_df.O .= 21.0

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    println("  Npar=$(Npar)  R_beam=$(round(R_beam,digits=2))  O2=$(round(O2_mean,digits=3))")

    # ── first irradiation → cell_df_copy ─────────────────────────────────────
    cell_df_copy = deepcopy(cell_df)
    cell_df_copy.is_cell = ifelse.(
        (cell_df_copy.x .^ 2 .+ cell_df_copy.y .^ 2 .+ cell_df_copy.z .^ 2 .<= tumor_radius^2) .&
        ((cell_df_copy.x .÷ Int(2R_cell) .+ cell_df_copy.y .÷ Int(2R_cell) .+
          cell_df_copy.z .÷ Int(2R_cell)) .% 2 .== 0),
        1, 0)
    for i in 1:nrow(cell_df_copy)
        cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                     sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
    end

    @time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy,
                        df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    cell_df_copy.cell_cycle .= "G1"
    cell_df_copy.can_divide .= 0

    Ntot = count(cell_df_copy.is_cell .== 1)
    println("  Ntot=$Ntot")

    # ── second irradiation object ─────────────────────────────────────────────
    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2_cycle[4])
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
    cell_df.O .= 21.0

    cell_df_second = deepcopy(cell_df)

    @time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second,
                        df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

    # ── instantaneous reference survival ─────────────────────────────────────
    cell_df_istant = deepcopy(cell_df_copy)
    cell_irrad     = deepcopy(cell_df_istant)
    MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_irrad,
                  df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_irrad, irrad_cond, gsm2_cycle)
    cell_df_istant.dam_X_dom .+= cell_irrad.dam_X_dom
    cell_df_istant.dam_Y_dom .+= cell_irrad.dam_Y_dom
    compute_times_domain!(cell_df_istant, gsm2_cycle)
    cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
    sp0 = count(.!isfinite.(cell_df_istant_.death_time)) / Ntot

    # ── ABM loop over split times ─────────────────────────────────────────────
    phase_keys  = ("G0", "G1", "S", "G2", "M")
    cell_df_ref = deepcopy(cell_df_copy)
    cell_work   = deepcopy(cell_df_ref)

    phase_times     = DataFrame(time=Float64[], Nalive=Float64[],
                                G0=Float64[], G1=Float64[], S=Float64[],
                                G2=Float64[], M=Float64[])
    phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[],
                                G0=Float64[], G1=Float64[], S=Float64[],
                                G2=Float64[], M=Float64[])
    surv_prob = Vector{Float64}()

    for t in times_split
        println("  [$label] t=$t  migration=$(D_mig>0 ? "ON (D=$(D_mig))" : "OFF")")

        Nalive_pre  = 0.0;  Nalive_post = 0.0
        phase_pre   = zeros(5);  phase_post  = zeros(5)
        surv_acc    = 0.0;  n_valid = 0

        for sim in 1:nsim
            try
                reset_cell!(cell_work, cell_df_ref)

                compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
                run_abm!(cell_work; terminal_time=t)           # ← migration-aware

                counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
                alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

                for i in findall(cell_work.is_cell .== 1)
                    cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                    cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
                end
                compute_times_domain!(cell_work, gsm2_cycle)

                counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
                alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

                n_valid     += 1
                Nalive_pre  += count(alive_pre)
                Nalive_post += count(alive_post)
                surv_acc    += count(alive_post) / Ntot
                for (ki, k) in enumerate(phase_keys)
                    phase_pre[ki]  += counts_pre[k]
                    phase_post[ki] += counts_post[k]
                end

            catch e
                @warn "[$label] sim=$sim t=$t failed" exception=(e, catch_backtrace())
            end
        end

        inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
        pre   = phase_pre  .* inv_n
        post  = phase_post .* inv_n
        push!(phase_times_pre, (t, Nalive_pre*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
        push!(phase_times,     (t, Nalive_post*inv_n, post[1], post[2], post[3], post[4], post[5]))
        push!(surv_prob, surv_acc * inv_n)

        GC.gc()
    end

    return (label           = label,
            times_split     = times_split,
            surv_prob       = surv_prob,
            sp0             = sp0,
            phase_times     = phase_times,
            phase_times_pre = phase_times_pre)
end

# ============================================================
# Shared settings
# ============================================================
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0,
               14., 16., 18., 19., 20., 21., 22., 23.,
               24.0, 25., 26., 27., 30., 48.0]
nsim = 10

# ============================================================
# Run scenarios
# ============================================================

res_1H_50 = run_scenario(
    "1H 40MeV/u (12C)",
    40.0, "12C", 1.5, 300.0,
    300.0, 800.0, 15.0,
    "circle", "full", "KC", true,
    times_split, gsm2_cycle; nsim=nsim)

res_1H_100 = run_scenario(
    "1H 100MeV",
    100.0, "1H", 1.5, 300.0,
    600.0, 300.0, 15.0,
    "circle", "full", "KC", true,
    times_split, gsm2_cycle; nsim=nsim)

res_12C_80 = run_scenario(
    "12C 80MeV/u",
    80.0, "12C", 1.5, 300.0,
    600.0, 300.0, 15.0,
    "circle", "full", "KC", true,
    times_split, gsm2_cycle; nsim=nsim)

# ============================================================
# Plots
# ============================================================

# helper: build (time, surv_prob) vectors with t=0 prepended
function with_t0(res)
    t_full  = vcat([0.0], res.times_split)
    sp_full = vcat([res.sp0], res.surv_prob)
    return t_full, sp_full
end

# ── 1H 50 ────────────────────────────────────────────────────────────────────
t1, sp1 = with_t0(res_1H_50)
p1_1H_50 = Plots.plot(t1, sp1;
    xlabel="Time [h]", ylabel="Survival fraction",
    label=res_1H_50.label, title=res_1H_50.label)

p2_1H_50 = Plots.plot(res_1H_50.phase_times_pre.time, res_1H_50.phase_times_pre.Nalive; label="Alive")
Plots.plot!(p2_1H_50, res_1H_50.phase_times_pre.time, res_1H_50.phase_times_pre.G1; label="G1")
Plots.plot!(p2_1H_50, res_1H_50.phase_times_pre.time, res_1H_50.phase_times_pre.S;  label="S")
Plots.plot!(p2_1H_50, res_1H_50.phase_times_pre.time, res_1H_50.phase_times_pre.G2; label="G2")
Plots.plot!(p2_1H_50, res_1H_50.phase_times_pre.time, res_1H_50.phase_times_pre.M;  label="M")
Plots.plot!(p2_1H_50, res_1H_50.phase_times_pre.time, res_1H_50.phase_times_pre.G0; label="G0")

display(plot(p1_1H_50, p2_1H_50; layout=(2,1), size=(1000,800)))

# ── 1H 100 ───────────────────────────────────────────────────────────────────
t2, sp2 = with_t0(res_1H_100)
p1_1H_100 = Plots.plot(t2, sp2;
    xlabel="Time [h]", ylabel="Survival fraction",
    label=res_1H_100.label, title=res_1H_100.label)

p2_1H_100 = Plots.plot(res_1H_100.phase_times_pre.time, res_1H_100.phase_times_pre.Nalive; label="Alive")
Plots.plot!(p2_1H_100, res_1H_100.phase_times_pre.time, res_1H_100.phase_times_pre.G1; label="G1")
Plots.plot!(p2_1H_100, res_1H_100.phase_times_pre.time, res_1H_100.phase_times_pre.S;  label="S")
Plots.plot!(p2_1H_100, res_1H_100.phase_times_pre.time, res_1H_100.phase_times_pre.G2; label="G2")
Plots.plot!(p2_1H_100, res_1H_100.phase_times_pre.time, res_1H_100.phase_times_pre.M;  label="M")
Plots.plot!(p2_1H_100, res_1H_100.phase_times_pre.time, res_1H_100.phase_times_pre.G0; label="G0")

display(plot(p1_1H_100, p2_1H_100; layout=(2,1), size=(1000,800)))

# ── 12C 80 ───────────────────────────────────────────────────────────────────
t3, sp3 = with_t0(res_12C_80)
p1_12C_80 = Plots.plot(t3, sp3;
    xlabel="Time [h]", ylabel="Survival fraction",
    label=res_12C_80.label, title=res_12C_80.label)

p2_12C_80 = Plots.plot(res_12C_80.phase_times_pre.time, res_12C_80.phase_times_pre.Nalive; label="Alive")
Plots.plot!(p2_12C_80, res_12C_80.phase_times_pre.time, res_12C_80.phase_times_pre.G1; label="G1")
Plots.plot!(p2_12C_80, res_12C_80.phase_times_pre.time, res_12C_80.phase_times_pre.S;  label="S")
Plots.plot!(p2_12C_80, res_12C_80.phase_times_pre.time, res_12C_80.phase_times_pre.G2; label="G2")
Plots.plot!(p2_12C_80, res_12C_80.phase_times_pre.time, res_12C_80.phase_times_pre.M;  label="M")
Plots.plot!(p2_12C_80, res_12C_80.phase_times_pre.time, res_12C_80.phase_times_pre.G0; label="G0")

display(plot(p1_12C_80, p2_12C_80; layout=(2,1), size=(1000,800)))

# ── Combined survival comparison ──────────────────────────────────────────────
p_surv = Plots.plot(t1, sp1; label=res_1H_50.label,
    xlabel="Time [h]", ylabel="Survival fraction",
    title="Split-dose survival  (migration D=$(D_mig) µm²/h)")
Plots.plot!(p_surv, t2, sp2; label=res_1H_100.label)
Plots.plot!(p_surv, t3, sp3; label=res_12C_80.label)
display(p_surv)
