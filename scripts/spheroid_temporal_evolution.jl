using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using JLD2
using DelimitedFiles
using StatsPlots
using Statistics
using StatsBase
using InlineStrings
using CUDA
using Statistics: mean
using Printf

nthreads()

# ============================================================
# Load functions
# ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

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
# Shared geometry — fixed once for all three conditions
# ============================================================
dose         = 2.0
tumor_radius = 450.0
X_box        = 600.0
X_voxel      = 800.0
R_cell       = 15.0
target_geom  = "circle"
calc_type    = "full"
type_AT      = "KC"
track_seg    = true
terminal_time = 72.0   # hours post-irradiation ABM window

N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

snapshot_hours = [0, 12, 24, 48, 72]   # hours at which 3-D plots are made

mkpath("results")

# ============================================================
# Helper: ts → DataFrame
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

# ============================================================
# Helper: reset cell_df in-place between runs
# ============================================================
function reset_cell!(dst::DataFrame, src::DataFrame)
    for col in names(src)
        S = src[!, col]; D_ = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S); copyto!(D_[i], S[i]); end
        else
            copyto!(D_, S)
        end
    end
end

# ============================================================
# Helper: 3-D half-sphere scatter coloured by cell cycle
#
# Uses scatter() with marker_z mapped to a numeric phase code so
# the colorbar matches a discrete categorical colormap, exactly
# following the style of the oxygen plot reference code.
#
# Phase → integer mapping (for marker_z):
#   G0=0  G1=1  S=2  G2=3  M=4
#
# Accepts a DataFrame with :is_cell, :cell_cycle, :x, :y, :z.
# z must be present — use snapshot_to_plot_df() to add it from
# cell_df_pristine when working with CellPopulation snapshots.
# ============================================================

# Discrete colormap: gray, steelblue, green, orange, red  (G0→M)
const PHASE_CMAP = cgrad([:gray, :steelblue, :green, :orange, :red],
                          5; categorical=true)
const PHASE_CODE = Dict("G0" => 0.0, "G1" => 1.0,
                        "S"  => 2.0, "G2" => 3.0, "M" => 4.0)

function plot_spheroid_halfcut(df::DataFrame, title_str::String)
    # alive cells on x≥0 half only
    alive = df[(df.is_cell .== 1) .& (df.x .>= 0), :]
    phase_num = [get(PHASE_CODE, p, 0.0) for p in alive.cell_cycle]

    p = scatter(
        alive.x, alive.y, alive.z;
        markersize        = 4,
        markerstrokewidth = 0.1,
        marker_z          = phase_num,
        colorbar          = true,
        colorbar_title    = "Phase",
        clims             = (-0.5, 4.5),
        seriescolor       = PHASE_CMAP,
        xlabel = "x (µm)", ylabel = "y (µm)", zlabel = "z (µm)",
        title  = title_str,
        legend = false,
        size   = (900, 700),
        camera = (320, 30)
    )

    # Overlay invisible dummy series for a readable legend
    for (phase, code) in sort(collect(PHASE_CODE), by=x->x[2])
        sub = alive[alive.cell_cycle .== phase, :]
        nrow(sub) == 0 && continue
        scatter!(p, [sub.x[1]], [sub.y[1]], [sub.z[1]];
                 markersize=4, markerstrokewidth=0.1,
                 marker_z=[code], seriescolor=PHASE_CMAP,
                 clims=(-0.5,4.5), label=phase, colorbar=false)
    end
    return p
end

"""
    snapshot_to_plot_df(pop_snap, cell_df_pristine) -> DataFrame

Convert a CellPopulation snapshot to a plottable DataFrame with x, y, z.
CellPopulation does not carry z coordinates in snapshots — they are
looked up from cell_df_pristine by cell index.
"""
function snapshot_to_plot_df(pop_snap::CellPopulation,
                               cell_df_pristine::DataFrame)::DataFrame
    snap_df = to_dataframe(pop_snap; alive_only=false)
    z_lookup = Dict{Int32, Int32}(zip(cell_df_pristine.index,
                                       cell_df_pristine.z))
    snap_df[!, :z] = Int32[get(z_lookup, idx, Int32(0))
                           for idx in snap_df.index]
    return snap_df
end

# ============================================================
# Helper: phase-proportion plot
# ============================================================
function plot_phases(ts_df::DataFrame, label::String)
    t = ts_df.time
    p = plot(t, ts_df.total_cells; label="Alive", lw=2, color=:black,
             xlabel="Time (h)", ylabel="Cell count", title=label)
    plot!(p, t, ts_df.g1_cells; label="G1", lw=1.5, color=:steelblue)
    plot!(p, t, ts_df.s_cells;  label="S",  lw=1.5, color=:green)
    plot!(p, t, ts_df.g2_cells; label="G2", lw=1.5, color=:orange)
    plot!(p, t, ts_df.m_cells;  label="M",  lw=1.5, color=:red)
    plot!(p, t, ts_df.g0_cells; label="G0", lw=1.5, color=:gray,
          linestyle=:dash)
    return p
end

# ============================================================
# CORE: run one irradiation condition
# ============================================================
function run_condition(
        label::String,
        E::Float64,
        particle::String,
        cell_df_irrad::DataFrame,   # already-irradiated, fixed initial state
        gsm2_cycle::Vector{GSM2};
        terminal_time::Float64      = 72.0,
        snapshot_hours::Vector{Int} = [0, 12, 24, 48, 72],
        outdir::String              = "results")

    println("\n", "="^60)
    println("  CONDITION: $label")
    println("="^60)

    Ntot = count(cell_df_irrad.is_cell .== 1)
    println("  Ntot = $Ntot")

    # ── compute stochastic repair/death times ────────────────────────────────
    compute_times_domain!(cell_df_irrad, gsm2_cycle;
                          nat_apo = 1e-10, terminal_time = terminal_time,
                          verbose = false, summary = true)

    # ── run ABM ──────────────────────────────────────────────────────────────
    ts, snaps = run_simulation_abm!(cell_df_irrad;
                    nat_apo           = 1e-10,
                    terminal_time     = terminal_time,
                    snapshot_times    = snapshot_hours,
                    print_interval    = 6.0,
                    verbose           = true,
                    return_dataframes = false,
                    update_input      = true)

    sf = count(cell_df_irrad.is_cell .== 1) / Ntot
    println("  Final survival fraction: $(round(sf, digits=4))")

    # ── save time series ─────────────────────────────────────────────────────
    ts_df = ts_to_df(ts)
    CSV.write(joinpath(outdir, "$(label)_ts.csv"), ts_df)

    # ── save snapshots as CSV ────────────────────────────────────────────────
    for (hr, pop_snap) in snaps
        snap_df = to_dataframe(pop_snap; alive_only=true)
        CSV.write(joinpath(outdir, "$(label)_snap_t$(hr)h.csv"), snap_df)
    end

    # ── save final cell_df ───────────────────────────────────────────────────
    CSV.write(joinpath(outdir, "$(label)_cell_df_final.csv"), cell_df_irrad)

    println("  Saved → $outdir/$(label)_*")

    return (label     = label,
            ts        = ts,
            ts_df     = ts_df,
            snaps     = snaps,
            cell_df   = cell_df_irrad,
            sf        = sf,
            Ntot      = Ntot)
end

# ============================================================
# BUILD THE THREE IRRADIATED INITIAL STATES
# All three share the SAME cell lattice geometry; only the
# particle/energy changes.
# ============================================================

# ── 1. Build geometry once ───────────────────────────────────────────────────
# Use proton 80 MeV just to set up the lattice & population.
# The geometry (positions, neighbours) is particle-independent.
# Cell-cycle phases are assigned randomly by setup_cell_population! —
# no seed is fixed so the distribution is genuinely stochastic.
# The resulting assignment is saved and reused identically for all
# three irradiation conditions so comparisons are fair.
E_ref = 80.0;  particle_ref = "1H"
setup_IonIrrad!(dose, E_ref, particle_ref)
R_beam, _, _ = calculate_beam_properties(calc_type, target_geom,
                                          X_box, X_voxel, tumor_radius)
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                    ParIrr="false", track_seg=track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox,
                       N_CellsSide, gsm2_cycle[4])
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

# Spheroid mask: keep only cells inside tumor_radius in a checkerboard pattern
cell_df.is_cell = ifelse.(
    (cell_df.x .^ 2 .+ cell_df.y .^ 2 .+ cell_df.z .^ 2 .<= tumor_radius^2) .&
    ((cell_df.x .÷ Int(2R_cell) .+ cell_df.y .÷ Int(2R_cell) .+
      cell_df.z .÷ Int(2R_cell)) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df)
    cell_df.number_nei[i] = length(cell_df.nei[i]) -
                             sum(cell_df.is_cell[cell_df.nei[i]])
end

set_oxygen!(cell_df; plot_oxygen=false)
cell_df.O .= 21.0   # uniform normoxia

# Save the pristine (pre-irradiation) geometry as the common reference.
# The random cell-cycle assignment produced by setup_cell_population! is
# preserved here and will be deepcopy'd into each irradiation branch.
cell_df_pristine = deepcopy(cell_df)
Ntot_ref = count(cell_df_pristine.is_cell .== 1)
println("Spheroid cells: $Ntot_ref")
println("Initial phase distribution:")
for phase in ["G0","G1","S","G2","M"]
    n = count(cell_df_pristine.cell_cycle[cell_df_pristine.is_cell .== 1] .== phase)
    println("  $phase : $n  ($(round(100n/Ntot_ref, digits=1))%)")
end
CSV.write("results/cell_df_pristine.csv", cell_df_pristine)

# ── Helper: irradiate a copy of the pristine population ──────────────────────
function irradiate_copy(E::Float64, particle::String,
                         cell_df_pristine::DataFrame,
                         gsm2_cycle::Vector{GSM2};
                         dose::Float64        = 2.0,
                         tumor_radius::Float64 = 300.0,
                         X_box::Float64       = 300.0,
                         X_voxel::Float64     = 800.0,
                         R_cell::Float64      = 15.0,
                         target_geom::String  = "circle",
                         calc_type::String    = "full",
                         type_AT::String      = "KC",
                         track_seg::Bool      = true)

    N_sideVox_   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide_ = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)
    R_beam_, _, _ = calculate_beam_properties(calc_type, target_geom,
                                               X_box, X_voxel, tumor_radius)
    setup_cell_lattice!(target_geom, X_box, R_cell,
                        N_sideVox_, N_CellsSide_;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell,
                           N_sideVox_, N_CellsSide_, gsm2_cycle[4])
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    # Start from the pristine geometry
    cell_copy = deepcopy(cell_df_pristine)

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam_^2 * 1e-8)
    println("  [$particle $(E) MeV] Npar=$Npar  R_beam=$(round(R_beam_,digits=2))")

    @time MC_dose_fast!(ion, Npar, R_beam_, irrad_cond, cell_copy,
                        df_center_x, df_center_y, at,
                        gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_copy, irrad_cond, gsm2_cycle)

    # Keep the random cell-cycle phase assigned at population setup —
    # do NOT override with G1. The phase distribution from cell_df_pristine
    # is identical for all three conditions (same deepcopy source).
    # can_divide is kept as-is (set by setup_cell_population!).

    println("  [$particle $(E) MeV] Mean X damage: " *
            "$(round(mean(sum.(cell_copy.dam_X_dom[cell_copy.is_cell .== 1])), digits=2))")

    return cell_copy
end

# ============================================================
# IRRADIATE: 3 conditions
# ============================================================
println("\n--- Irradiating: 1H 80 MeV ---")
cell_1H_80  = irradiate_copy(150.0,  "1H",  cell_df_pristine, gsm2_cycle)

println("\n--- Irradiating: 1H 30 MeV ---")
cell_1H_30  = irradiate_copy(10.0,  "1H",  cell_df_pristine, gsm2_cycle)

println("\n--- Irradiating: 12C 80 MeV/u ---")
cell_12C_80 = irradiate_copy(80.0,  "12C", cell_df_pristine, gsm2_cycle)

# Save initial irradiated states
CSV.write("results/cell_irrad_1H_80.csv",  cell_1H_80)
CSV.write("results/cell_irrad_1H_30.csv",  cell_1H_30)
CSV.write("results/cell_irrad_12C_80.csv", cell_12C_80)

# ============================================================
# RUN ABM FOR EACH CONDITION
# ============================================================
res_1H_80  = run_condition("1H_80MeV",   150.0, "1H",
                            cell_1H_80,  gsm2_cycle;
                            terminal_time=terminal_time,
                            snapshot_hours=snapshot_hours)

res_1H_30  = run_condition("1H_30MeV",   10.0, "1H",
                            cell_1H_30,  gsm2_cycle;
                            terminal_time=terminal_time,
                            snapshot_hours=snapshot_hours)

res_12C_80 = run_condition("12C_80MeVu", 80.0, "12C",
                            cell_12C_80, gsm2_cycle;
                            terminal_time=terminal_time,
                            snapshot_hours=snapshot_hours)

# ============================================================
# SUMMARY CSV
# ============================================================
summary_df = DataFrame(
    condition         = String[],
    particle          = String[],
    energy_MeV        = Float64[],
    Ntot              = Int[],
    survival_fraction = Float64[],
    final_alive       = Int[])

for (res, part, E_val) in [
        (res_1H_80,  "1H",  150.0),
        (res_1H_30,  "1H",  10.0),
        (res_12C_80, "12C", 80.0)]
    push!(summary_df, (res.label, part, E_val, res.Ntot, res.sf,
                       count(res.cell_df.is_cell .== 1)))
end
CSV.write("results/summary.csv", summary_df)
println("\nSummary:")
println(summary_df)

# ============================================================
# PLOT 1: Temporal evolution of total cell count
# ============================================================
p_total = plot(;
    xlabel="Time (h)", ylabel="Alive cells",
    title="Spheroid response — 2 Gy",
    legend=:topright, size=(900, 500), dpi=150)

for (res, col, ls) in [
        (res_1H_80,  :steelblue, :solid),
        (res_1H_30,  :darkorange, :solid),
        (res_12C_80, :red,       :solid)]
    plot!(p_total, res.ts_df.time, res.ts_df.total_cells;
          label="$(res.label)  (SF=$(round(res.sf,digits=3)))",
          lw=2, color=col, linestyle=ls)
end
display(p_total)
savefig(p_total, "results/total_cells.png")

# ============================================================
# PLOT 2: Phase breakdown — one panel per condition
# ============================================================
p_ph_1H_80  = plot_phases(res_1H_80.ts_df,  "1H 80 MeV — 2 Gy")
p_ph_1H_30  = plot_phases(res_1H_30.ts_df,  "1H 30 MeV — 2 Gy")
p_ph_12C_80 = plot_phases(res_12C_80.ts_df, "12C 80 MeV/u — 2 Gy")

display(plot(p_ph_1H_80, p_ph_1H_30, p_ph_12C_80;
             layout=(1,3), size=(1400,450), dpi=150))
savefig(plot(p_ph_1H_80, p_ph_1H_30, p_ph_12C_80;
             layout=(1,3), size=(1400,450), dpi=150),
        "results/phase_breakdown.png")

# ============================================================
# PLOT 3: Survival fraction over time (normalised)
# ============================================================
p_sf = plot(;
    xlabel="Time (h)", ylabel="Relative cell number (N/N₀)",
    title="Normalised survival — 2 Gy",
    legend=:topright, size=(900,500), dpi=150)

for (res, col) in [
        (res_1H_80,  :steelblue),
        (res_1H_30,  :darkorange),
        (res_12C_80, :red)]
    norm = res.ts_df.total_cells ./ res.Ntot
    plot!(p_sf, res.ts_df.time, norm;
          label=res.label, lw=2, color=col)
end
hline!(p_sf, [1.0]; color=:black, ls=:dash, lw=1, label="N₀")
display(p_sf)
savefig(p_sf, "results/normalised_survival.png")

# ============================================================
# PLOT 4: 3-D half-sphere snapshots
#   4 time points × 3 conditions
#   z coords are joined from cell_df_pristine because
#   CellPopulation snapshots do not carry the z field.
# ============================================================
for (res, part_label) in [
        (res_1H_80,  "1H_80"),
        (res_1H_30,  "1H_30"),
        (res_12C_80, "12C_80")]

    panels = []
    for hr in snapshot_hours
        if haskey(res.snaps, hr)
            # Convert snapshot → DataFrame, join z from pristine geometry
            plot_df = snapshot_to_plot_df(res.snaps[hr], cell_df_pristine)
            push!(panels,
                  plot_spheroid_halfcut(plot_df, "$(part_label) t=$(hr)h"))
        else
            println("  [warn] no snapshot at t=$(hr)h for $(res.label), skipping panel")
        end
    end

    isempty(panels) && continue
    n_panels = length(panels)
    p_3d = plot(panels...;
                layout=(1, n_panels),
                size=(700 * n_panels, 650),
                dpi=100)
    display(p_3d)
    savefig(p_3d, "results/spheroid_3d_$(res.label).png")
    println("Saved: results/spheroid_3d_$(res.label).png")
end

# ============================================================
# PLOT 5: Phase proportions over time (stacked area)
# ============================================================
function plot_phase_fractions(ts_df::DataFrame, label::String)
    t = ts_df.time
    tot = max.(ts_df.total_cells, 1)
    p = plot(;
        xlabel="Time (h)", ylabel="Phase fraction",
        title=label, legend=:topright, ylims=(0,1))
    plot!(p, t, ts_df.g1_cells ./ tot; label="G1", lw=2, color=:steelblue, fill=(0,:steelblue,0.3))
    plot!(p, t, ts_df.s_cells  ./ tot; label="S",  lw=2, color=:green,     fill=(0,:green,    0.3))
    plot!(p, t, ts_df.g2_cells ./ tot; label="G2", lw=2, color=:orange,    fill=(0,:orange,   0.3))
    plot!(p, t, ts_df.m_cells  ./ tot; label="M",  lw=2, color=:red,       fill=(0,:red,      0.3))
    plot!(p, t, ts_df.g0_cells ./ tot; label="G0", lw=2, color=:gray,      fill=(0,:gray,     0.3))
    return p
end

p_frac = plot(
    plot_phase_fractions(res_1H_80.ts_df,  "1H 80 MeV"),
    plot_phase_fractions(res_1H_30.ts_df,  "1H 30 MeV"),
    plot_phase_fractions(res_12C_80.ts_df, "12C 80 MeV/u");
    layout=(1,3), size=(1400,450), dpi=150)
display(p_frac)
savefig(p_frac, "results/phase_fractions.png")

# ============================================================
# FINAL PRINT
# ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO results/")
println("="^60)
println("Files written:")
for f in sort(readdir("results"))
    println("  results/$f")
end
