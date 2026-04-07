using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using ProgressBars
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
using SparseArrays
using Printf

nthreads()

# ============================================================
# Load functions
# ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

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
# Simulation parameters
# ============================================================
E            = 50.0
particle     = "1H"
dose         = 1.5
tumor_radius = 150.0
X_box        = 260.0
au           = 4.0

setup(E, particle, dose, tumor_radius; X_box = X_box)

cell_df_copy = deepcopy(cell_df)
cell_df.O   .= 21.0

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0

# ============================================================
# LUT precomputation
# ============================================================
println("Precomputing dose LUT...")
@time lut = MC_precompute_lut!(
    ion, Npar, R_beam, irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg;
    chunk_size = 50_000)

println("Precomputing damage LUT...")
@time damage_lut = precompute_damage_lut!(
    lut, cell_df_copy, irrad_cond, gsm2_cycle, ion;
    chunk_size = 50_000)

jldsave("lut_1H_1p5Gy_50MeV_150radius.jld2"; damage_lut)

Npar_effect = length(damage_lut)
println("Npar_effect = $Npar_effect")

# ============================================================
# Base active-cell template
# ============================================================
active_cells_base = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
for row in eachrow(cell_df_copy)
    row.is_cell != 1 && continue
    active_cells_base[row.index] = (zeros(Int, length(row.dam_X_dom)),
                                    zeros(Int, length(row.dam_Y_dom)))
end
Ntot = length(active_cells_base)
println("Total cells (Ntot) = $Ntot")

datadir = joinpath(@__DIR__, "..", "data", "continuous_doserate_sphere")
mkpath(datadir)
jldsave(joinpath(datadir, "initial_cell_df.jld2"); cell_df_copy)
jldsave(joinpath(datadir, "damage_lut.jld2");      damage_lut)

# ============================================================
# Helpers
# ============================================================
function ts_to_df(ts::SimulationTimeSeries)
    DataFrame(
        time           = ts.time,
        total_cells    = ts.total_cells,
        g0_cells       = ts.g0_cells,
        g1_cells       = ts.g1_cells,
        s_cells        = ts.s_cells,
        g2_cells       = ts.g2_cells,
        m_cells        = ts.m_cells,
        stem_cells     = ts.stem_cells,
        non_stem_cells = ts.non_stem_cells)
end

dr_label(dr_gys::Float64) = @sprintf("%.0e", dr_gys)

# ============================================================
# CORE FUNCTION: dose-rate simulation
# ============================================================
function run_doserate_scenario(
        label::String,
        dose_to_run::Float64,
        doserates_gys::Vector{Float64},
        damage_lut,
        cell_df_copy::DataFrame,
        gsm2_cycle::Vector{GSM2},
        gsm2::GSM2,
        zF::Float64,
        Ntot::Int,
        active_cells_base::Dict;
        au::Float64                 = 4.0,
        post_terminal::Float64      = 72.0,
        post_snapshots::Vector{Int} = [1, 6, 12, 24, 48, 72],
        outdir::String              = "results")

    mkpath(outdir)
    Npar_effect   = length(damage_lut)
    doserates_gyh = doserates_gys .* 3600.0 .* au
    abm_results   = Dict{Float64, NamedTuple}()

    for (j, dose_rate_gyh) in enumerate(doserates_gyh)
        dr_gys = doserates_gys[j]
        println("\n", "="^70)
        println("$label | Dose: $(dose_to_run) Gy | Rate: $dr_gys Gy/s  ($dose_rate_gyh Gy/h)")
        println("="^70)

        dr         = dose_rate_gyh / zF
        N_dose     = round(Int, dose_to_run * Npar_effect)
        times_full = rand(Exponential(1.0 / dr), N_dose)
        lut_order  = mod1.(randperm(N_dose), Npar_effect)

        times_abs   = Float64[]
        lut_indices = Int[]
        t_acc       = 0.0
        for i in 1:N_dose
            t_acc += times_full[i]
            lut_idx = lut_order[i]
            if !isempty(damage_lut[lut_idx])
                push!(times_abs,   t_acc)
                push!(lut_indices, lut_idx)
                t_acc = 0.0
            end
        end
        N_eff = length(lut_indices)
        println("  Effective particles: $N_eff / $N_dose")

        active_cells = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}(
            idx => (copy(X), copy(Y))
            for (idx, (X, Y)) in active_cells_base)

        cell_df_run   = deepcopy(cell_df_copy)
        ts_segments   = SimulationTimeSeries[]
        abm_wall_time = 0.0

        for i in 1:N_eff
            lut_idx = lut_indices[i]

            # 1. Apply damage
            for (idx, (x, y)) in damage_lut[lut_idx]
                !haskey(active_cells, idx)    && continue
                cell_df_run.is_cell[idx] == 0 && continue
                active_cells[idx][1] .+= x
                active_cells[idx][2] .+= y
                cell_df_run.dam_X_dom[idx] .+= x
                cell_df_run.dam_Y_dom[idx] .+= y
            end

            # 2. Repair window delta
            if i < N_eff
                t_repair_h   = times_abs[i]
                t_repair_sim = t_repair_h
            else
                t_repair_h   = 0.0
                t_repair_sim = Inf
            end

            # 3. SSA repair
            to_delete_ssa = Int[]
            for (cell_idx, (X, Y)) in active_cells
                all(iszero, X) && all(iszero, Y) && continue
                death_time, _, _, X_new, Y_new =
                    compute_repair_domain(X, Y, gsm2;
                                          terminal_time = t_repair_sim,
                                          au            = au)
                if isfinite(death_time)
                    push!(to_delete_ssa, cell_idx)
                else
                    active_cells[cell_idx] = (X_new, Y_new)
                end
            end

            # 4. Sync SSA deaths → cell_df_run
            for idx in to_delete_ssa
                delete!(active_cells, idx)
                if cell_df_run.is_cell[idx] == 1
                    cell_df_run.is_cell[idx]      = 0
                    cell_df_run.death_time[idx]   = 0.0
                    cell_df_run.cycle_time[idx]   = Inf
                    cell_df_run.recover_time[idx] = Inf
                    for nb in cell_df_run.nei[idx]
                        cell_df_run.number_nei[nb] =
                            length(cell_df_run.nei[nb]) -
                            sum(cell_df_run.is_cell[cell_df_run.nei[nb]])
                    end
                end
            end

            # 5. ABM repair window
            if t_repair_h > 0.0
                compute_times_domain!(cell_df_run, gsm2_cycle;
                                      nat_apo       = 1e-10,
                                      terminal_time = t_repair_h,
                                      verbose       = false,
                                      summary       = false)

                ts_seg, _ = run_simulation_abm!(cell_df_run;
                                  nat_apo           = 1e-10,
                                  terminal_time     = t_repair_h,
                                  snapshot_times    = Int[],
                                  print_interval    = t_repair_h + 1.0,
                                  verbose           = false,
                                  return_dataframes = false,
                                  update_input      = true)

                ts_seg.time .+= abm_wall_time
                push!(ts_segments, ts_seg)

                # 6. Sync ABM deaths → active_cells
                for idx in collect(keys(active_cells))
                    cell_df_run.is_cell[idx] == 0 && delete!(active_cells, idx)
                end
            end

            abm_wall_time += t_repair_h

            if i % max(1, N_eff ÷ 10) == 0
                n_ssa = length(active_cells)
                n_abm = count(cell_df_run.is_cell .== 1)
                println("  [$(i)/$(N_eff)] t=$(round(abm_wall_time,digits=3))h | " *
                        "alive_ssa=$n_ssa | alive_abm=$n_abm")
            end
        end

        ts_irrad = SimulationTimeSeries()
        for seg in ts_segments
            append!(ts_irrad.time,           seg.time)
            append!(ts_irrad.total_cells,    seg.total_cells)
            append!(ts_irrad.g0_cells,       seg.g0_cells)
            append!(ts_irrad.g1_cells,       seg.g1_cells)
            append!(ts_irrad.s_cells,        seg.s_cells)
            append!(ts_irrad.g2_cells,       seg.g2_cells)
            append!(ts_irrad.m_cells,        seg.m_cells)
            append!(ts_irrad.stem_cells,     seg.stem_cells)
            append!(ts_irrad.non_stem_cells, seg.non_stem_cells)
        end

        n_abm = count(cell_df_run.is_cell .== 1)
        n_ssa = length(active_cells)
        println("\n  Irradiation complete | t=$(round(abm_wall_time,digits=3))h")
        println("  Alive ABM: $n_abm  |  Alive SSA: $n_ssa")

        survival_fraction = n_abm / Ntot
        println("  Survival fraction: $(round(survival_fraction, digits=4))")

        println("  Post-irradiation ABM ($(post_terminal)h)...")
        compute_times_domain!(cell_df_run, gsm2_cycle;
                              nat_apo       = 1e-10,
                              terminal_time = post_terminal,
                              verbose       = false,
                              summary       = true)

        ts_post, snaps_post = run_simulation_abm!(cell_df_run;
                                  nat_apo           = 1e-10,
                                  terminal_time     = post_terminal,
                                  snapshot_times    = post_snapshots,
                                  print_interval    = 6.0,
                                  verbose           = false,
                                  return_dataframes = false,
                                  update_input      = true)
        ts_post.time .+= abm_wall_time

        abm_results[dose_rate_gyh] = (
            dose_rate_gyh     = dose_rate_gyh,
            dose_rate_gys     = dr_gys,
            survival_fraction = survival_fraction,
            ts_irrad          = ts_irrad,
            ts_post           = ts_post,
            snaps_post        = snaps_post,
            cell_df_final     = cell_df_run,
            abm_wall_time     = abm_wall_time)

        lbl = "$(label)_dr_$(dr_label(dr_gys))Gys"
        outdir2 = joinpath(@__DIR__, "..", "data", "continuous_doserate_sphere")
        mkpath(outdir2)

        CSV.write(joinpath(outdir2, "$(lbl)_ts_irrad.csv"),   ts_to_df(ts_irrad))
        CSV.write(joinpath(outdir2, "$(lbl)_ts_post.csv"),    ts_to_df(ts_post))
        CSV.write(joinpath(outdir2, "$(lbl)_ts_combined.csv"),
            vcat(insertcols(ts_to_df(ts_irrad), :phase => "irradiation"),
            insertcols(ts_to_df(ts_post),  :phase => "post")))
        println("  Saved → $outdir/$(lbl)*")

        GC.gc()
    end

    return abm_results
end

# ============================================================
# RUN: 1.5 Gy  with dose rates 1e-6, 1e-5, 1e-2 Gy/s
# ============================================================
abm_results_1p5Gy = run_doserate_scenario(
    "1p5Gy", 1.5,
    [1e-6, 1e-5, 1e-2],
    damage_lut, cell_df_copy, gsm2_cycle, gsm2_cycle[1], zF, Ntot,
    active_cells_base;
    au = au, outdir = "results")

# ============================================================
# SUMMARY CSV
# ============================================================
function summary_csv(abm_results, dose_label, outdir)
    df = DataFrame(
        dose_rate_gys     = Float64[],
        dose_rate_gyh     = Float64[],
        survival_fraction = Float64[],
        irrad_time_h      = Float64[],
        final_abm_cells   = Int[])
    for (_, res) in sort(collect(abm_results), by = x -> x[1])
        push!(df, (res.dose_rate_gys, res.dose_rate_gyh, res.survival_fraction,
                   res.abm_wall_time, count(res.cell_df_final.is_cell .== 1)))
    end
    path = joinpath(@__DIR__, "..", "data", "continuous_doserate_sphere", "summary_$(dose_label).csv")
    CSV.write(path, df)
    println("  Saved summary → $path")
    println(df)
    return df
end

summary_csv(abm_results_1p5Gy, "1p5Gy", "results")

# ============================================================
# SANITY CHECK
# Expected: lower dose rate → more repair time → higher survival
# Order: 1e-6 > 1e-5 > 1e-2 Gy/s
# ============================================================
println("\n", "="^60)
println("SANITY CHECK — expected: lower dose rate → higher SF")
println("="^60)
for (_, res) in sort(collect(abm_results_1p5Gy), by = x -> x[1])
    @printf("  %8.2e Gy/s → SF = %.4f  (irrad time = %.2f h)\n",
            res.dose_rate_gys, res.survival_fraction, res.abm_wall_time)
end

# ============================================================
# FINAL PRINT
# ============================================================

println("ALL RESULTS SAVED TO $datadir/")
println("="^60)
println("Files written:")
for f in sort(readdir(datadir))
    println("  $datadir/$f")
end