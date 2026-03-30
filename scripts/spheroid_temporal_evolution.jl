using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using InlineStrings
using CUDA
using Statistics: mean
using Printf
using ProgressMeter

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
tumor_radius = 500.0
X_box        = 500.0
X_voxel      = 800.0
R_cell       = 15.0
target_geom  = "circle"
calc_type    = "full"
type_AT      = "KC"
track_seg    = true
terminal_time = 72.0   # hours post-irradiation ABM window

N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

snapshot_hours = [0, 4, 12, 20, 24, 48, 72]   # hours at which snapshots are saved

datadir = joinpath(@__DIR__, "..", "data", "spheroid_temporal_evolution")
mkpath(datadir)

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

set_oxygen!(cell_df; plot_oxygen=false)

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
CSV.write(joinpath(datadir, "cell_df_pristine.csv"), cell_df_pristine)

# ── Helper: irradiate a copy of the pristine population ──────────────────────
# CRITICAL: does NOT call setup_cell_lattice! or setup_cell_population!
# Those calls overwrite the global cell_df with a new random population,
# destroying the shared pristine geometry and making all conditions identical.
# Only setup_IonIrrad! (updates ion/irrad/LET) and setup_irrad_conditions!
# (rebuilds irrad_cond/df_center_x/df_center_y/at for the new particle) are
# called here. Cell positions and cycle phases come from deepcopy(cell_df_pristine).
function irradiate_copy(E::Float64, particle::String,
                         cell_df_pristine::DataFrame,
                         gsm2_cycle::Vector{GSM2};
                         dose::Float64         = 2.0,
                         tumor_radius::Float64  = 300.0,
                         X_box::Float64        = 300.0,
                         X_voxel::Float64      = 800.0,
                         R_cell::Float64       = 15.0,
                         target_geom::String   = "circle",
                         calc_type::String     = "full",
                         type_AT::String       = "KC",
                         track_seg::Bool       = true)

    # Update ion/irrad/LET for this particle — does NOT touch cell_df
    setup_IonIrrad!(dose, E, particle)
    R_beam_, _, _ = calculate_beam_properties(calc_type, target_geom,
                                               X_box, X_voxel, tumor_radius)

    # Rebuild irrad_cond, df_center_x, df_center_y, at for the new particle.
    # cell_df (global) was built during the initial geometry setup and must
    # remain unchanged — it is the geometry reference for df_center_x/at.
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    # Start from the pristine geometry
    cell_copy = deepcopy(cell_df_pristine)

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam_^2 * 1e-8)
    println("  [$particle $(E) MeV] Npar=$Npar  R_beam=$(round(R_beam_,digits=2))")

    @time MC_dose_CPU!(ion, Npar, R_beam_, irrad_cond, cell_copy,
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
cell_1H_80  = irradiate_copy(80.0,  "1H",  cell_df_pristine, gsm2_cycle)

println("\n--- Irradiating: 1H 30 MeV ---")
cell_1H_30  = irradiate_copy(30.0,  "1H",  cell_df_pristine, gsm2_cycle)

println("\n--- Irradiating: 12C 80 MeV/u ---")
cell_12C_80 = irradiate_copy(80.0,  "12C", cell_df_pristine, gsm2_cycle)

# Save initial irradiated states
CSV.write(joinpath(datadir, "cell_irrad_1H_80.csv"),  cell_1H_80)
CSV.write(joinpath(datadir, "cell_irrad_1H_30.csv"),  cell_1H_30)
CSV.write(joinpath(datadir, "cell_irrad_12C_80.csv"), cell_12C_80)

# ============================================================
# RUN ABM FOR EACH CONDITION
# ============================================================
res_1H_80  = run_condition("1H_80MeV",   80.0, "1H",
                            cell_1H_80,  gsm2_cycle;
                            terminal_time=terminal_time,
                            snapshot_hours=snapshot_hours,
                            outdir=datadir)

res_1H_30  = run_condition("1H_30MeV",   30.0, "1H",
                            cell_1H_30,  gsm2_cycle;
                            terminal_time=terminal_time,
                            snapshot_hours=snapshot_hours,
                            outdir=datadir)

res_12C_80 = run_condition("12C_80MeVu", 80.0, "12C",
                            cell_12C_80, gsm2_cycle;
                            terminal_time=terminal_time,
                            snapshot_hours=snapshot_hours,
                            outdir=datadir)

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
        (res_1H_80,  "1H",  80.0),
        (res_1H_30,  "1H",  30.0),
        (res_12C_80, "12C", 80.0)]
    push!(summary_df, (res.label, part, E_val, res.Ntot, res.sf,
                       count(res.cell_df.is_cell .== 1)))
end
CSV.write(joinpath(datadir, "summary.csv"), summary_df)
println("\nSummary:")
println(summary_df)

# ============================================================
# FINAL PRINT
# ============================================================
println("ALL RESULTS SAVED TO $datadir/")
println("="^60)
println("Files written:")
for f in sort(readdir(datadir))
    println("  $datadir/$f")
end
