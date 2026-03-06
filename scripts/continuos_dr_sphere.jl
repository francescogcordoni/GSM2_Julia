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
using InlineStrings
using CUDA
using Statistics: mean
using SparseArrays

nthreads()

#*Nd is the dimension fo the space, the nucleus of the cell is assumed to be a cylinder. 
#*The cell a sphere around the center.
#*The geometry can be more complicated but for now it is fine

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

#&Stopping power
sp = load_stopping_power()

#~ ==========================================================================================
#~ ============================== Generate GSM2 =============================================
#~ ==========================================================================================

#& Select cell line for GSM2 parameters (to be externalized in the future)
#& cell_line = "HSG"

#& GSM2 model parameters (typical ranges noted in comments)
a  = 0.01    # lethal rate [1/h] 
b  = 0.30    # pairwise lethal rate [1/h]
r  = 4.30    # repair rate [1/h]
rd = 0.80    # domain radius [µm]
Rn = 7.2     # nucleus radius [µm]

#G1 -> 12h
a_G1 = 0.012872261720543399
b_G1 = 0.04029756109753225
r_G1 = 2.780479661191086

#S -> 8h
a_S = 0.00589118894714544
b_S = 0.05794352736120672
r_S = 5.84009601901114

#G2 - M -> 3h + 1h
a_G2 = 0.024306291709970018
b_G2 = 5.704688326522623e-5
r_G2 = 1.7720064637774506

#mixed
a = 0.01481379648786136
b = 0.012663276476522422
r = 2.5656972960759896
rd = 0.8;
Rn = 7.2;    

gsm2_cycle = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn); #! G1
gsm2_cycle[2] = GSM2(r_S, a_S, b_S, rd, Rn); #! S
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn); #! G2 - M
gsm2_cycle[4] = GSM2(r, a, b, rd, Rn); #! mixed


#& Construct the GSM2 object
setup_GSM2!(r, a, b, rd, Rn)

#~ ============================================================
#~ =================== Simulation Parameters ==================
#~ ============================================================

E            = 50.0
particle     = "1H"
dose         = 1.
tumor_radius = 150.0
X_box = 260.
setup(E, particle, dose, tumor_radius, X_box = X_box)
cell_df_copy = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg, single_particle = true)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)
cell_df.O .= 21.

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0
#cell_df_copy = deepcopy(cell_df)

@time lut = MC_precompute_lut!(
    ion, Npar, R_beam, irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg;
    chunk_size = 50_000
)

@time damage_lut = precompute_damage_lut!(
    lut, cell_df_copy, irrad_cond, gsm2_cycle, ion;
    chunk_size = 50_000
)
jldsave("lut_1H_1Gy_50MeV_150radius.jld2"; damage_lut)
#damage_lut = load("lut_1H_1Gy_50MeV_150radius.jld2", "damage_lut")


# ── Configuration ─────────────────────────────────────────────────────────────
au               = 4.0
dose_to_run      = 1.0
dr_to_plot_gys   = 1e-5   # the one dose rate to compare against instantaneous
doserates_to_run_Gys = [dr_to_plot_gys]
doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

Npar_effect = length(damage_lut)
gsm2        = gsm2_cycle[1]

mkpath("results")

# ── Pre-build initial active cells template ───────────────────────────────────
active_cells_base = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
for row in eachrow(cell_df_copy)
    row.is_cell != 1 && continue
    active_cells_base[row.index] = (zeros(Int, length(row.dam_X_dom)),
                                    zeros(Int, length(row.dam_Y_dom)))
end
Ntot = length(active_cells_base)

# ── Save initial state ────────────────────────────────────────────────────────
println("Saving initial cell_df and damage_lut...")
jldsave("results/initial_cell_df.jld2";    cell_df_copy)
jldsave("results/damage_lut.jld2";         damage_lut)
println("  Done.")

# ── Helper: SimulationTimeSeries → DataFrame ──────────────────────────────────
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
        non_stem_cells = ts.non_stem_cells,
    )
end

# ── Helper: save all outputs for one condition ────────────────────────────────
function save_condition(label, cell_df_final, ts_irrad, ts_post, abm_wall_time;
                        outdir = "results")
    mkpath(outdir)
    jldsave(joinpath(outdir, "$(label)_cell_df_final.jld2"); cell_df_final)
    CSV.write(joinpath(outdir, "$(label)_ts_irrad.csv"),  ts_to_df(ts_irrad))
    CSV.write(joinpath(outdir, "$(label)_ts_post.csv"),   ts_to_df(ts_post))
    # Combined time series with phase label
    df_combined = vcat(
        insertcols(ts_to_df(ts_irrad), :phase => "irradiation"),
        insertcols(ts_to_df(ts_post),  :phase => "post")
    )
    CSV.write(joinpath(outdir, "$(label)_ts_combined.csv"), df_combined)
    println("  Saved $label → $outdir")
end


# ═════════════════════════════════════════════════════════════════════════════
# DOSE-RATE SIMULATION  (1e-5 Gy/s)
# ═════════════════════════════════════════════════════════════════════════════
abm_results = Dict{Float64, NamedTuple}()

for (j, dose_rate_gyh) in enumerate(doserates_to_run_Gyh)
    println("\n", "="^70)
    println("Dose rate: $(doserates_to_run_Gys[j]) Gy/s  |  $(dose_rate_gyh) Gy/h")
    println("="^70)

    dr     = dose_rate_gyh / zF
    N_dose = round(Int, dose_to_run * Npar_effect)

    times_full = rand(Exponential(1/dr), N_dose)
    lut_order  = mod1.(randperm(N_dose), Npar_effect)

    times_filtered   = Float64[]
    lut_indices      = Int[]
    accumulated_time = 0.0
    for i in 1:N_dose
        lut_idx = lut_order[i]
        accumulated_time += times_full[i]
        if !isempty(damage_lut[lut_idx])
            push!(times_filtered, accumulated_time)
            push!(lut_indices, lut_idx)
            accumulated_time = 0.0
        end
    end
    N_eff = length(lut_indices)
    println("  Effective particles: $N_eff / $N_dose")

    active_cells = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}(
        idx => (copy(X), copy(Y)) for (idx, (X, Y)) in active_cells_base)
    cell_df_run  = deepcopy(cell_df_copy)
    ts_segments  = SimulationTimeSeries[]
    abm_wall_time = 0.0

    for i in 1:N_eff
        lut_idx = lut_indices[i]

        # Apply damage
        for (idx, (x, y)) in damage_lut[lut_idx]
            !haskey(active_cells, idx) && continue
            active_cells[idx][1] .+= x
            active_cells[idx][2] .+= y
        end
        for (idx, (x, y)) in damage_lut[lut_idx]
            cell_df_run.dam_X_dom[idx] .+= x
            cell_df_run.dam_Y_dom[idx] .+= y
        end

        t_repair_h   = i < N_eff ? times_filtered[i+1] : 0.0
        t_repair_sim = i < N_eff ? times_filtered[i+1] : Inf

        # Survival-only repair
        to_delete = Int[]
        for (cell_idx, (X, Y)) in active_cells
            all(iszero, X) && all(iszero, Y) && continue
            death_time, _, _, X_new, Y_new =
                compute_repair_domain(X, Y, gsm2; terminal_time = t_repair_sim, au = au)
            isfinite(death_time) ? push!(to_delete, cell_idx) : (active_cells[cell_idx] = (X_new, Y_new))
        end
        for idx in to_delete; delete!(active_cells, idx); end

        # ABM repair window
        if t_repair_h > 0.0
            compute_times_domain!(cell_df_run, gsm2_cycle;
                                  nat_apo = 1e-10, terminal_time = t_repair_h,
                                  verbose = false, summary = false)
            ts_seg, _ = run_simulation_abm!(cell_df_run;
                                  nat_apo = 1e-10, terminal_time = t_repair_h,
                                  snapshot_times = Int[], verbose = false,
                                  print_interval = t_repair_h + 1.0,
                                  return_dataframes = false, update_input = true)
            ts_seg.time .+= abm_wall_time
            push!(ts_segments, ts_seg)
        end

        abm_wall_time += t_repair_h

        i % max(1, N_eff ÷ 10) == 0 &&
            println("  [$(i)/$(N_eff)] t=$(round(abm_wall_time,digits=3))h | survival=$(length(active_cells)) | abm=$(count(cell_df_run.is_cell .== 1))")
    end

    # Merge irradiation segments
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

    println("\n  Irradiation done | t=$(round(abm_wall_time,digits=3))h")
    println("  Alive survival=$(length(active_cells)) | ABM=$(count(cell_df_run.is_cell .== 1))")

    # Post-irradiation
    println("  Post-irradiation 3-day ABM...")
    compute_times_domain!(cell_df_run, gsm2_cycle;
                          nat_apo = 1e-10, terminal_time = 72.0,
                          verbose = false, summary = true)

    ts_post, snaps_post = run_simulation_abm!(cell_df_run;
                              nat_apo = 1e-10, terminal_time = 72.0,
                              snapshot_times = [1,6,12,24,48,72],
                              print_interval = 6.0, verbose = false,
                              return_dataframes = false, update_input = true)
    ts_post.time .+= abm_wall_time

    survival_fraction = length(active_cells) / Ntot
    println("  Survival fraction: $(round(survival_fraction, digits=4))")

    abm_results[dose_rate_gyh] = (
        dose_rate_gyh     = dose_rate_gyh,
        dose_rate_gys     = doserates_to_run_Gys[j],
        survival_fraction = survival_fraction,
        ts_irrad          = ts_irrad,
        ts_post           = ts_post,
        snaps_post        = snaps_post,
        cell_df_final     = cell_df_run,
        abm_wall_time     = abm_wall_time,
    )

    label = "doserate_$(doserates_to_run_Gys[j])Gys"
    save_condition(label, cell_df_run, ts_irrad, ts_post, abm_wall_time)
end


# ═════════════════════════════════════════════════════════════════════════════
# INSTANTANEOUS IRRADIATION
# ═════════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("INSTANTANEOUS IRRADIATION  ($(dose_to_run) Gy)")
println("="^70)

cell_df_instant = deepcopy(cell_df_copy)
for i in 1:nrow(cell_df_instant)
    fill!(cell_df_instant.dam_X_dom[i], 0)
    fill!(cell_df_instant.dam_Y_dom[i], 0)
end

# !! Key fix: same number of particles as dose-rate case
lut_order_instant = mod1.(randperm(round(Int, dose_to_run * Npar_effect)), Npar_effect)
for p in lut_order_instant
    for (cell_idx, (x, y)) in damage_lut[p]
        cell_df_instant.dam_X_dom[cell_idx] .+= x
        cell_df_instant.dam_Y_dom[cell_idx] .+= y
    end
end
cell_df_instant[!, :dam_X_total] = sum.(cell_df_instant.dam_X_dom)
cell_df_instant[!, :dam_Y_total] = sum.(cell_df_instant.dam_Y_dom)

println("  Mean X damage: $(mean(cell_df_instant.dam_X_total[cell_df_instant.is_cell .== 1]))")
println("  Mean Y damage: $(mean(cell_df_instant.dam_Y_total[cell_df_instant.is_cell .== 1]))")

compute_times_domain!(cell_df_instant, gsm2_cycle;
                      nat_apo = 1e-10, terminal_time = 72.0,
                      verbose = false, summary = true)

ts_instant, snaps_instant = run_simulation_abm!(cell_df_instant;
                                nat_apo = 1e-10, terminal_time = 72.0,
                                snapshot_times = [1,6,12,24,48,72],
                                print_interval = 6.0, verbose = false,
                                return_dataframes = false, update_input = true)

survival_instant = count(cell_df_instant.is_cell .== 1) / Ntot
println("  Survival fraction (instantaneous): $(round(survival_instant, digits=4))")

# For save_condition, ts_irrad is empty (no irradiation phase steps)
ts_empty = SimulationTimeSeries()
save_condition("instantaneous", cell_df_instant, ts_empty, ts_instant, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT: 1e-5 Gy/s vs instantaneous
# ═════════════════════════════════════════════════════════════════════════════
res_dr = abm_results[dr_to_plot_gys * 3600.0 * au]

t_dr     = vcat(res_dr.ts_irrad.time,       res_dr.ts_post.time)
cells_dr = vcat(res_dr.ts_irrad.total_cells, res_dr.ts_post.total_cells)

p_compare = plot(
    ts_instant.time, ts_instant.total_cells;
    label     = "Instantaneous  (SF=$(round(survival_instant,digits=3)))",
    lw        = 2,
    color     = :black,
    linestyle = :dash,
    xlabel    = "Time (h)",
    ylabel    = "Total cells",
    title     = "$(dose_to_run) Gy  |  instantaneous vs $(dr_to_plot_gys) Gy/s"
)

plot!(p_compare, t_dr, cells_dr;
      label = "$(dr_to_plot_gys) Gy/s  (SF=$(round(res_dr.survival_fraction,digits=3)))",
      lw    = 2,
      color = :steelblue)

vline!(p_compare, [res_dr.abm_wall_time];
       label     = "End irradiation",
       color     = :steelblue,
       lw        = 1.5,
       linestyle = :dot)

display(p_compare)
savefig(p_compare, "results/survival_instant_vs_$(dr_to_plot_gys)Gys.png")




































































# ── Configuration ─────────────────────────────────────────────────────────────
au               = 4.0
dose_to_run      = 3.0
doserates_to_run_Gys = [1e-5, 1e-2]
doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

Npar_effect = length(damage_lut)
gsm2        = gsm2_cycle[1]

# Pre-build initial active cells template
active_cells_base = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
for row in eachrow(cell_df_copy)
    row.is_cell != 1 && continue
    active_cells_base[row.index] = (zeros(Int, length(row.dam_X_dom)),
                                    zeros(Int, length(row.dam_Y_dom)))
end
Ntot = length(active_cells_base)

# ── Results storage ────────────────────────────────────────────────────────────
# For each dose rate: store (ts_irrad, snaps_irrad, ts_post, snaps_post)
abm_results = Dict{Float64, NamedTuple}()

for (j, dose_rate_gyh) in enumerate(doserates_to_run_Gyh)
    println("\n", "="^70)
    println("Running dose rate: $(dose_rate_gyh) Gy/h  [$(doserates_to_run_Gys[j]) Gy/s]")
    println("="^70)

    dr     = dose_rate_gyh / zF
    dose   = dose_to_run
    N_dose = round(Int, dose_to_run * Npar_effect)

    # ── Build particle times & LUT indices ────────────────────────────────
    times_full = rand(Exponential(1/dr), N_dose)
    lut_order  = mod1.(randperm(N_dose), Npar_effect)

    times_filtered   = Float64[]
    lut_indices      = Int[]
    accumulated_time = 0.0

    for i in 1:N_dose
        lut_idx = lut_order[i]
        accumulated_time += times_full[i]
        if !isempty(damage_lut[lut_idx])
            push!(times_filtered, accumulated_time)
            push!(lut_indices, lut_idx)
            accumulated_time = 0.0
        end
    end

    N_eff = length(lut_indices)
    println("  Effective particles: $N_eff / $N_dose")

    # ── Reset active cells & DataFrame for this dose rate ─────────────────
    active_cells = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}(
        idx => (copy(X), copy(Y))
        for (idx, (X, Y)) in active_cells_base
    )

    # Working copy of cell_df for this dose rate run
    cell_df_run = deepcopy(cell_df_copy)

    # ── Storage for irradiation-phase ABM time series ─────────────────────
    # We collect all SimulationTimeSeries segments and merge at the end
    ts_segments   = SimulationTimeSeries[]   # one per inter-hit interval
    snaps_irrad   = Dict{String, CellPopulation}()  # keyed "hit_i_hour_h"

    # Accumulated real time (hours) at which ABM starts each segment
    abm_wall_time = 0.0   # running clock in hours

    # ── Main irradiation loop ──────────────────────────────────────────────
    for i in 1:N_eff
        println(i)
        lut_idx = lut_indices[i]

        # 1) Apply damage to active_cells dict (survival bookkeeping)
        @inbounds for (idx, (x, y)) in damage_lut[lut_idx]
            !haskey(active_cells, idx) && continue
            active_cells[idx][1] .+= x
            active_cells[idx][2] .+= y
        end

        # 2) Apply damage to cell_df_run for ABM
        @inbounds for (idx, (x, y)) in damage_lut[lut_idx]
            cell_df_run.dam_X_dom[idx] .+= x
            cell_df_run.dam_Y_dom[idx] .+= y
        end

        # 3) Inter-hit repair window (hours); last hit gets Inf → use 0 for ABM
        t_repair_h = i < N_eff ? times_filtered[i+1] : 0.0
        # (times_filtered are in hours already if dose_rate is Gy/h; 
        #  adjust if your times are in seconds)

        # 4) Survival-only repair (as before, in original time units)
        t_repair_sim = i < N_eff ? times_filtered[i+1] : Inf

        to_delete = Int[]
        for (cell_idx, (X, Y)) in active_cells
            all(iszero, X) && all(iszero, Y) && continue
            death_time, _, _, X_new, Y_new =
                compute_repair_domain(X, Y, gsm2; terminal_time = t_repair_sim, au = au)
            if isfinite(death_time)
                push!(to_delete, cell_idx)
            else
                active_cells[cell_idx] = (X_new, Y_new)
            end
        end
        for idx in to_delete
            delete!(active_cells, idx)
        end

        # 5) Run ABM for the inter-hit window (skip if window is 0)
        if t_repair_h > 0.0
            # Sync damage state into cell_df_run before compute_times_domain!
            compute_times_domain!(cell_df_run, gsm2_cycle;
                                    nat_apo        = 1e-10,
                                    terminal_time  = t_repair_h,
                                    verbose        = false,
                                    summary        = false)

            ts_seg, snaps_seg = run_simulation_abm!(cell_df_run;
                                    nat_apo          = 1e-10,
                                    terminal_time    = t_repair_h,
                                    snapshot_times   = Int[],   # no fixed snapshots mid-irrad
                                    print_interval   = t_repair_h + 1.0,  # suppress per-step prints
                                    verbose          = false,
                                    return_dataframes = false,
                                    update_input     = true)

            # Offset timestamps so they sit on the global clock
            ts_seg.time .+= abm_wall_time
            push!(ts_segments, ts_seg)

            # Store snapshot keyed by hit index and approximate wall hour
            hit_hour = round(Int, floor(abm_wall_time + t_repair_h))
            snaps_irrad["hit_$(i)_h$(hit_hour)"] = create_snapshot(CellPopulation(cell_df_run))
        end

        abm_wall_time += t_repair_h

        if i % max(1, N_eff ÷ 10) == 0
            println("  [$(i)/$(N_eff)] wall_time=$(round(abm_wall_time,digits=3))h | alive_survival=$(length(active_cells)) | alive_abm=$(count(cell_df_run.is_cell .== 1))")
        end
    end

    # ── Merge irradiation-phase time series ───────────────────────────────
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

    println("\n  Irradiation phase complete.")
    println("  Total irradiation time: $(round(abm_wall_time, digits=3))h")
    println("  Alive (survival model): $(length(active_cells))")
    println("  Alive (ABM):            $(count(cell_df_run.is_cell .== 1))")

    # ── Post-irradiation: final compute_times_domain! + 3-day ABM ─────────
    println("\n  Running post-irradiation repair + 3-day ABM...")

    compute_times_domain!(cell_df_run, gsm2_cycle;
                            nat_apo       = 1e-10,
                            terminal_time = 72.0,    # 3 days
                            verbose       = false,
                            summary       = true)

    post_snapshot_times = [1, 6, 12, 24, 48, 72]

    ts_post, snaps_post = run_simulation_abm!(cell_df_run;
                                nat_apo           = 1e-10,
                                terminal_time     = 72.0,
                                snapshot_times    = post_snapshot_times,
                                print_interval    = 6.0,
                                verbose           = false,
                                return_dataframes = false,
                                update_input      = true)

    # Offset post-irradiation time series to sit after irradiation
    ts_post.time .+= abm_wall_time

    # ── Store all results for this dose rate ──────────────────────────────
    survival_fraction = length(active_cells) / Ntot

    abm_results[dose_rate_gyh] = (
        dose_rate_gyh    = dose_rate_gyh,
        dose_rate_gys    = doserates_to_run_Gys[j],
        survival_fraction = survival_fraction,
        ts_irrad         = ts_irrad,       # SimulationTimeSeries during irradiation
        snaps_irrad      = snaps_irrad,    # Dict{String, CellPopulation}
        ts_post          = ts_post,        # SimulationTimeSeries for 3-day recovery
        snaps_post       = snaps_post,     # Dict{Int, CellPopulation} at post_snapshot_times
        cell_df_final    = cell_df_run,    # final DataFrame state
        abm_wall_time    = abm_wall_time,  # total irradiation duration in hours
    )

    println("\n  Survival fraction (survival model): $(round(survival_fraction, digits=4))")
end

# ── Quick summary ──────────────────────────────────────────────────────────────
println("\n", "="^70)
println("SUMMARY  (dose = $(dose_to_run) Gy)")
println("="^70)
println("  Dose rate (Gy/s)  |  Survival  |  Irrad time (h)  |  Final cells (ABM)")
println("  " * "-"^66)
for (dr, res) in sort(collect(abm_results), by=x->x[1])
    n_final = count(res.cell_df_final.is_cell .== 1)
    @printf("  %-18.2e  %-12.4f  %-18.3f  %d\n",
            res.dose_rate_gys, res.survival_fraction, res.abm_wall_time, n_final)
end


res = first(values(abm_results))  # or abm_results[dose_rate_gyh] if you want a specific one

# Combine irradiation + post-irradiation time series
t_all     = vcat(res.ts_irrad.time,      res.ts_post.time)
cells_all = vcat(res.ts_irrad.total_cells, res.ts_post.total_cells)

p = plot(t_all, cells_all;
         xlabel    = "Time (h)",
         ylabel    = "Total cells",
         label     = "ABM",
         lw        = 2,
         color     = :steelblue,
         title     = "Dose rate = $(round(res.dose_rate_gys, sigdigits=2)) Gy/s")

vline!(p, [res.abm_wall_time];
       label     = "End of irradiation",
       color     = :red,
       lw        = 2,
       linestyle = :dash)

display(p)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: save relevant DataFrames for a result entry
# ─────────────────────────────────────────────────────────────────────────────
function save_abm_result(res, label::String; outdir::String = "results")
    mkpath(outdir)

    # Final cell state
    CSV.write(joinpath(outdir, "$(label)_cell_df_final.csv"),
              res.cell_df_final)

    # Time series — irradiation phase
    ts_irrad_df = DataFrame(
        time           = res.ts_irrad.time,
        total_cells    = res.ts_irrad.total_cells,
        g0_cells       = res.ts_irrad.g0_cells,
        g1_cells       = res.ts_irrad.g1_cells,
        s_cells        = res.ts_irrad.s_cells,
        g2_cells       = res.ts_irrad.g2_cells,
        m_cells        = res.ts_irrad.m_cells,
        stem_cells     = res.ts_irrad.stem_cells,
        non_stem_cells = res.ts_irrad.non_stem_cells,
    )
    CSV.write(joinpath(outdir, "$(label)_ts_irrad.csv"), ts_irrad_df)

    # Time series — post-irradiation phase
    ts_post_df = DataFrame(
        time           = res.ts_post.time,
        total_cells    = res.ts_post.total_cells,
        g0_cells       = res.ts_post.g0_cells,
        g1_cells       = res.ts_post.g1_cells,
        s_cells        = res.ts_post.s_cells,
        g2_cells       = res.ts_post.g2_cells,
        m_cells        = res.ts_post.m_cells,
        stem_cells     = res.ts_post.stem_cells,
        non_stem_cells = res.ts_post.non_stem_cells,
    )
    CSV.write(joinpath(outdir, "$(label)_ts_post.csv"), ts_post_df)

    println("  Saved: $label → $outdir")
end

# Save all dose-rate results
for (dr, res) in abm_results
    label = "dr_$(round(res.dose_rate_gys, sigdigits=2))"
    save_abm_result(res, label)
end


# ─────────────────────────────────────────────────────────────────────────────
# INSTANTANEOUS IRRADIATION
# Apply full damage from all particles at once, then compute_times + ABM
# ─────────────────────────────────────────────────────────────────────────────
println("\n", "="^70)
println("INSTANTANEOUS IRRADIATION")
println("="^70)

cell_df_instant = deepcopy(cell_df_copy)

# Reset damage
for i in 1:nrow(cell_df_instant)
    fill!(cell_df_instant.dam_X_dom[i], 0)
    fill!(cell_df_instant.dam_Y_dom[i], 0)
end

# Apply full damage from all particles at once
for p in 1:length(damage_lut)
    for (cell_idx, (x, y)) in damage_lut[p]
        cell_df_instant.dam_X_dom[cell_idx] .+= x
        cell_df_instant.dam_Y_dom[cell_idx] .+= y
    end
end
cell_df_instant[!, :dam_X_total] = sum.(cell_df_instant.dam_X_dom)
cell_df_instant[!, :dam_Y_total] = sum.(cell_df_instant.dam_Y_dom)

println("  Total damage applied.")
println("  Mean X damage: $(mean(cell_df_instant.dam_X_total[cell_df_instant.is_cell .== 1]))")
println("  Mean Y damage: $(mean(cell_df_instant.dam_Y_total[cell_df_instant.is_cell .== 1]))")

# Compute death/repair times
compute_times_domain!(cell_df_instant, gsm2_cycle;
                      nat_apo       = 1e-10,
                      terminal_time = 72.0,
                      verbose       = false,
                      summary       = true)

# Run 3-day ABM
post_snapshot_times = [1, 6, 12, 24, 48, 72]

ts_instant, snaps_instant = run_simulation_abm!(cell_df_instant;
                                nat_apo           = 1e-10,
                                terminal_time     = 72.0,
                                snapshot_times    = post_snapshot_times,
                                print_interval    = 6.0,
                                verbose           = true,
                                return_dataframes = false,
                                update_input      = true)

survival_instant = count(cell_df_instant.is_cell .== 1) / Ntot
println("\n  Survival fraction (instantaneous): $(round(survival_instant, digits=4))")

# Save instantaneous results
ts_instant_df = DataFrame(
    time           = ts_instant.time,
    total_cells    = ts_instant.total_cells,
    g0_cells       = ts_instant.g0_cells,
    g1_cells       = ts_instant.g1_cells,
    s_cells        = ts_instant.s_cells,
    g2_cells       = ts_instant.g2_cells,
    m_cells        = ts_instant.m_cells,
    stem_cells     = ts_instant.stem_cells,
    non_stem_cells = ts_instant.non_stem_cells,
)
CSV.write("results/instantaneous_ts.csv",       ts_instant_df)
CSV.write("results/instantaneous_cell_df.csv",  cell_df_instant)
println("  Saved: instantaneous results → results/")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT: instantaneous vs dose-rate curves
# ─────────────────────────────────────────────────────────────────────────────
p_compare = plot(ts_instant.time, ts_instant.total_cells;
                 label     = "Instantaneous",
                 lw        = 2,
                 color     = :black,
                 linestyle = :dash,
                 xlabel    = "Time (h)",
                 ylabel    = "Total cells",
                 title     = "Cell survival — instantaneous vs dose rate")

colors = [:steelblue, :darkorange, :green, :purple]
for (ci, (dr, res)) in enumerate(sort(collect(abm_results), by=x->x[1]))
    t_all     = vcat(res.ts_irrad.time,        res.ts_post.time)
    cells_all = vcat(res.ts_irrad.total_cells,  res.ts_post.total_cells)

    plot!(p_compare, t_all, cells_all;
          label     = "$(round(res.dose_rate_gys, sigdigits=2)) Gy/s",
          lw        = 2,
          color     = colors[ci])

    vline!(p_compare, [res.abm_wall_time];
           label     = "",
           color     = colors[ci],
           lw        = 1,
           linestyle = :dot)
end

display(p_compare)
savefig(p_compare, "results/survival_curves.png")
