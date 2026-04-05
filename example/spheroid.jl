# ============================================================================
# example/spheroid.jl
#
# KEY EXAMPLE — Irradiation of a 3-D spheroid followed by agent-based model
# (ABM) simulation of post-irradiation cell dynamics.
#
# PIPELINE
# --------
#   1. PARAMETERS       — dose, particle, energy, geometry (edit here)
#   2. SETUP            — GSM2, ion/irrad, cell lattice, population, O₂
#   3. DOSE & DAMAGE    — Monte Carlo dose deposition + DNA damage sampling
#   4. SURVIVAL         — GSM2 survival probability per cell
#   5. TIMERS           — stochastic repair / death time assignment
#   6. ABM              — post-irradiation cell dynamics simulation
#   7. OUTPUT           — plots + CSVs saved to example/output/spheroid/
#
# HOW TO RUN
# ----------
#   julia --threads auto example/spheroid.jl
#
# ============================================================================

using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using StatsPlots            # required for density() used in plot_times, plot_damage, etc.
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
using SparseArrays
using Printf

println("Threads available: ", nthreads())

# ── Load all source utilities ─────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS — edit these to change the irradiation condition
# ============================================================================

# Irradiation
PARTICLE      = "12C"      # ion species: "1H", "4He", "12C", "16O"
ENERGY_MEV_U  = 80.0     # kinetic energy per nucleon (MeV/u)
DOSE_GY       = 0.5       # prescribed dose (Gy)

# Spheroid and cell geometry
TUMOR_RADIUS  = 350.0     # spheroid radius (µm)
R_CELL        = 15.0      # cell radius (µm)
X_BOX         = 350.0     # simulation box half-size (µm); match TUMOR_RADIUS
X_VOXEL       = 700.0     # voxel side length for beam-geometry calculation (µm)

# GSM2 domain geometry
RD            = 0.8       # domain radius (µm)
RN            = 7.2       # nucleus radius (µm)

# Phase-specific GSM2 parameters [G1/G0, S, G2/M, average]
# These are fitted to HSG cell line data — replace with your own fits.
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

# Simulation options
const TYPE_AT        = "KC"         # track structure: "KC" (Kiefer-Chatterjee) or "LEM"
const TRACK_SEG      = true         # true = fixed LET across depth (no Bragg-peak buildup)
const TARGET_GEOM    = "circle"     # spheroid cross-section for beam geometry
const CALC_TYPE      = "full"       # beam radius mode: "full" (whole spheroid) or "fast"
const TERMINAL_TIME  = 72.0         # post-irradiation ABM window (h)
const SNAPSHOT_HOURS = [0, 6, 12, 24, 48, 72]   # save population snapshots at these times (h)
const NAT_APO        = 1e-10        # natural apoptosis rate (h⁻¹, background only)

# Output directory
const OUTDIR = joinpath(@__DIR__, "output", "spheroid")
mkpath(OUTDIR)

println("\n", "="^65)
println("  Spheroid Irradiation + ABM Example")
println("  Particle  : $PARTICLE  |  E = $ENERGY_MEV_U MeV/u  |  D = $DOSE_GY Gy")
println("  Spheroid  : radius = $TUMOR_RADIUS µm")
println("  ABM window: $TERMINAL_TIME h")
println("  Output    : $OUTDIR")
println("="^65, "\n")

# ============================================================================
# 2. SETUP — GSM2, ion/irrad, cell lattice, population, oxygenation
#
# NOTE: all of the individual setup steps below (setup_GSM2!, setup_IonIrrad!,
# setup_cell_lattice!, setup_cell_population!, setup_irrad_conditions!, and
# set_oxygen!) can be replaced by a single call to the high-level wrapper:
#
#   out = setup(ENERGY_MEV_U, PARTICLE, DOSE_GY, TUMOR_RADIUS;
#               X_box=X_BOX, X_voxel=X_VOXEL, R_cell=R_CELL,
#               target_geom=TARGET_GEOM, calc_type=CALC_TYPE,
#               type_AT=TYPE_AT, track_seg=TRACK_SEG)
#
#   # out.ion, out.irrad, out.cell_df, out.R_beam, out.Npar, out.zF, ...
#
# The explicit step-by-step form is used here so each stage is visible and
# individually inspectable, which is more instructive for a reference example.
# ============================================================================

# Build phase-dependent GSM2 repair objects
gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(R_G1, A_G1, B_G1, RD, RN)    # G1 / G0
gsm2_cycle[2] = GSM2(R_S,  A_S,  B_S,  RD, RN)    # S
gsm2_cycle[3] = GSM2(R_G2, A_G2, B_G2, RD, RN)    # G2 / M
gsm2_cycle[4] = GSM2(R_AVG, A_AVG, B_AVG, RD, RN) # average (fallback)

# Inject GSM2 + domain centers into Main
setup_GSM2!(R_AVG, A_AVG, B_AVG, RD, RN)

# Inject ion descriptor, Irrad object, and track radii into Main
setup_IonIrrad!(DOSE_GY, ENERGY_MEV_U, PARTICLE; type_AT=TYPE_AT)

# Derived lattice dimensions from box and cell size
N_sideVox   = Int(floor(2 * X_BOX / X_VOXEL))
N_CellsSide = 2 * convert(Int64, floor(X_BOX / (2 * R_CELL)))

# Place cells on a spherical lattice; inject cell_df, nodes_positions into Main
setup_cell_lattice!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide;
                    ParIrr="false", track_seg=TRACK_SEG, full_cycle=true)

# setup_cell_population! reads tumor_radius and X_voxel directly from Main scope
# (normally injected by the setup() wrapper). Inject them explicitly here.
@eval Main begin
    tumor_radius = $TUMOR_RADIUS
    X_voxel      = $X_VOXEL
    X_box        = $X_BOX
end

# Assign biological attributes (cell cycle, damage vectors, neighbour lists, ...)
# Requires gsm2 (from setup_GSM2!) and cell_df (from setup_cell_lattice!) in Main.
setup_cell_population!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide,
                        gsm2_cycle[4]; ParIrr="false")

# Build per-layer AT irradiation conditions (energy step → AT object)
setup_irrad_conditions!(ion, irrad, TYPE_AT, cell_df, TRACK_SEG)

# Assign oxygenation (pO₂) profile across the spheroid
set_oxygen!(cell_df; plot_oxygen=true)

# Print and plot initial (pre-irradiation) state
N_init = count(cell_df.is_cell .== 1)
println("Spheroid: $N_init cells")
print_phase_distribution(cell_df; label="pre-irradiation")

# Beam geometry: radius, center
R_beam, x_beam, y_beam = calculate_beam_properties(
    CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, TUMOR_RADIUS)

# Fluence and particle count estimate
F    = irrad.dose / (1.602e-9 * LET)        # particles / cm²
Npar = round(Int, F * π * R_beam^2 * 1e-8)  # particles through spheroid cross-section
Npar == 0 && error("Npar = 0: check beam radius and dose settings")
zF   = irrad.dose / Npar                     # dose per particle (Gy)
println("Beam: R_beam = $(round(R_beam, digits=1)) µm  |  " *
        "Npar = $Npar  |  zF = $(round(zF, digits=5)) Gy")

# ── Plot 1 — initial cell-cycle distribution (bar + 3-D scatter) ─────────────
p_cycle_init = plot_cell_cycle_distribution(cell_df; phase_plot=false, half_sphere=true)

# ── Plot 1b — phase proportion bar chart (% of alive cells per phase) ─────────
p_phase_init = plot_phase_proportions_alive(cell_df;
                    title_text="Phase distribution – pre-irradiation")

# ── Plot 2 — initial death / recovery time distributions ─────────────────────
p_init_dist = plot_initial_distributions(cell_df)

# Freeze pristine (pre-irradiation) state for before/after comparisons
cell_df_pristine = deepcopy(cell_df)

# ============================================================================
# 3. DOSE & DAMAGE
# ============================================================================

println("\n--- Monte Carlo dose deposition ---")
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df,
                    df_center_x, df_center_y, at,
                    gsm2_cycle, TYPE_AT, TRACK_SEG)

println("\n--- Poisson DNA damage sampling ---")
@time MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)

# ── Plot 4 — dose distribution (density + 3-D spatial) ───────────────────────
p_dose = plot_scalar_cell(cell_df, :dose_cell)

# ── Plot 5 — dose grouped by energy step (depth layers) ──────────────────────
p_dose_layer = plot_scalar_cell(cell_df, :dose_cell; layer_plot=true)

# ── Plot 6 — X-damage density (all active cells, mean line) ──────────────────
# plot_damage computes sum(dam_X_dom[i]) per cell from the per-domain vectors.
p_damage = plot_damage(cell_df)

# ── Plot 7 — X-damage density grouped by energy step (depth layers) ──────────
p_damage_layer = plot_damage(cell_df; layer_plot=true)

# ── Plot 7b — total X-damage per cell: density + 3-D spatial coloured map ─────
# dam_X_total is the scalar sum of all per-domain X lesions for each cell.
# The 3-D panel shows which cells in the spheroid received more DSBs.
p_dam_X = plot_scalar_cell(cell_df, :dam_X_total)

# ── Plot 7c — total Y-damage per cell: density + 3-D spatial coloured map ─────
# dam_Y_total counts lethal (complex/non-repairable) lesions per cell.
# Any cell with dam_Y_total > 0 has zero survival probability in GSM2.
p_dam_Y = plot_scalar_cell(cell_df, :dam_Y_total)

# ── Plot 7d — X-damage vs dose scatter: one point per alive cell ──────────────
let df_a = cell_df[cell_df.is_cell .== 1, :]
    p_dam_vs_dose = scatter(df_a.dose_cell, df_a.dam_X_total;
                            xlabel="Cell dose (Gy)", ylabel="Total X damage (DSBs)",
                            title="X damage vs dose per cell",
                            markersize=3, markerstrokewidth=0, alpha=0.5,
                            label="", color=:steelblue)
#    savefig(p_dam_vs_dose, joinpath(OUTDIR, "07d_damage_vs_dose.png"))
end

# ============================================================================
# 4. SURVIVAL (GSM2)
# ============================================================================

println("\n--- GSM2 survival probability ---")
@time compute_cell_survival_GSM2!(cell_df, gsm2_cycle)

alive_mask = cell_df.is_cell .== 1
SF_mean    = mean(cell_df.sp[alive_mask])
p_hat, ci_lo, ci_hi = survival_ci(cell_df.sp[alive_mask])
println("Mean SP : $(round(SF_mean, digits=4))  " *
        "(Wilson 95% CI: $(round(ci_lo, digits=4)) – $(round(ci_hi, digits=4)))")

# ── Plot 8 — survival probability distribution + 3-D spatial (log10 x-axis) ──
p_sp = plot_scalar_cell(cell_df, :sp; xscale=:log10)

# Snapshot after irradiation, before ABM (for before/after comparison)
cell_df_irradiated = deepcopy(cell_df)

# ── Plot 8b — phase proportions: pre-irradiation vs post-irradiation ──────────
# (Phase assignments do not change during irradiation — the shift visible here
#  reflects cells that received lethal dose and will be removed by the ABM.)
p_phase_irrad   = plot_phase_proportions_alive(cell_df_irradiated;
                      title_text="Phase distribution – post-irradiation")
p_phase_compare = plot(p_phase_init, p_phase_irrad; layout=(1, 2), size=(1000, 400))

# ============================================================================
# 5. REPAIR / DEATH TIMERS
# ============================================================================

println("\n--- Stochastic repair and death timers ---")
@time compute_times_domain!(cell_df, gsm2_cycle;
                             nat_apo       = NAT_APO,
                             terminal_time = TERMINAL_TIME,
                             verbose       = false,
                             summary       = true)

# ── Plot 9 — timer densities (death / recovery / cycle / X-damage) ───────────
p_times = plot_times(cell_df; show_means=true, summary=true)

# ============================================================================
# 6. ABM SIMULATION
# ============================================================================

println("\n--- Agent-based model ($TERMINAL_TIME h) ---")
ts, snaps = @time run_simulation_abm!(cell_df;
                nat_apo           = NAT_APO,
                terminal_time     = TERMINAL_TIME,
                snapshot_times    = SNAPSHOT_HOURS,
                print_interval    = 6.0,
                verbose           = true,
                return_dataframes = false,
                update_input      = true)

N_final = count(cell_df.is_cell .== 1)
SF_abm  = N_final / N_init
println("\nFinal alive: $N_final / $N_init  (ABM SF = $(round(SF_abm, digits=4)))")

print_simulation_summary(ts)

# Convert snapshots to DataFrames for plotting
snap_df = Dict(t => to_dataframe(snaps[t]; alive_only=true)
               for t in SNAPSHOT_HOURS if haskey(snaps, t))

# ============================================================================
# 7. OUTPUT — time-series plots
# ============================================================================

# ── Plot 10 — total cell count vs time ───────────────────────────────────────
p_dyn = plot_cell_dynamics(ts)

# ── Plot 11 — per-phase counts vs time ───────────────────────────────────────
p_phases = plot_phase_dynamics(ts)

# ── Plot 12 — phase percentages vs time ──────────────────────────────────────
p_prop = plot_phase_proportions(ts)

# ── Plot 13 — stacked area chart of phase counts ─────────────────────────────
p_stacked = plot_phase_stacked(ts)

# ── Plot 14 — cycling (G1/S/G2/M) vs quiescent (G0) ─────────────────────────
p_cyq = plot_cycling_vs_quiescent(ts)

# ── Plot 15 — sliding-window population growth rate ───────────────────────────
p_growth = plot_growth_rate(ts; window_size=10)

# ── Plot 16 — stem vs non-stem cell dynamics ─────────────────────────────────
p_stem = plot_stem_dynamics(ts)

# ── Plot 17 — 3-panel simulation summary (dynamics + phases + stem) ───────────
p_summary = plot_simulation_results(ts)

# ── Plot 18 — 5-panel analysis dashboard ─────────────────────────────────────
p_dash = plot_analysis_dashboard(ts)

# ── Plot 20 — phase proportions timeseries (standalone with title) ────────────
p_prop_ts = plot_phase_proportions_timeseries(ts;
                title_text="$PARTICLE $(ENERGY_MEV_U) MeV/u, $(DOSE_GY) Gy — Phase Dynamics")

# ============================================================================
# 8. OUTPUT — snapshot and spatial plots
# ============================================================================

# ── Plot 21 — phase bar chart at each snapshot time ──────────────────────────
p_snap_cmp = plot_snapshot_comparison(snap_df; metric=:cell_cycle)

# ── Plot 22 — division capacity at each snapshot time ────────────────────────
p_snap_div = plot_snapshot_comparison(snap_df; metric=:can_divide)

# ── Plot 23 — cell-cycle distribution: pre vs post irradiation ───────────────
p_before_after = plot_phase_comparison_before_after(cell_df_pristine, cell_df_irradiated)

# ── Plot 24 — cell-cycle distribution: pre-irradiation vs end of ABM ─────────
p_before_final = plot_phase_comparison_before_after(cell_df_pristine, cell_df)

# ── Plot 25 — spatial distribution at t=0 and final snapshot ─────────────────
t_last     = maximum(keys(snap_df))
p_sp_init  = plot_spatial_distribution(cell_df_pristine;
                 color_by=:cell_cycle, title_text="Spatial – pre-irradiation")
p_sp_final = plot_spatial_distribution(snap_df[t_last];
                 color_by=:cell_cycle, title_text="Spatial – t=$(t_last)h")
p_spatial  = plot(p_sp_init, p_sp_final; layout=(1, 2), size=(1200, 500))

# ── Plot 27 — 3-D phase animation over all snapshots ─────────────────────────
anim_file = joinpath(OUTDIR, "27_spatial_animation.gif")
create_spatial_animation(snap_df; output_file=anim_file, fps=2, color_by=:cell_cycle)

# ── Plot 28 — average phase duration proxy (bar chart) ───────────────────────
p_dur = plot_phase_duration_distribution(ts)

# ============================================================================
# 9. DATA EXPORT
# ============================================================================

# Time-series CSV
#export_timeseries_csv(ts, joinpath(OUTDIR, "timeseries.csv"))
#
## Final cell population
#CSV.write(joinpath(OUTDIR, "cell_df_final.csv"), cell_df)
#
## Per-snapshot CSVs
#for (t, df_snap) in snap_df
#    CSV.write(joinpath(OUTDIR, "snapshot_t$(t)h.csv"), df_snap)
#end

# Summary row
summary_df = DataFrame(
    particle          = [PARTICLE],
    energy_MeV_u      = [ENERGY_MEV_U],
    dose_Gy           = [DOSE_GY],
    tumor_radius_um   = [TUMOR_RADIUS],
    N_cells_init      = [N_init],
    N_cells_final     = [N_final],
    Npar              = [Npar],
    SF_GSM2_mean      = [round(SF_mean,  digits=5)],
    SF_GSM2_CI_lo     = [round(ci_lo,    digits=5)],
    SF_GSM2_CI_hi     = [round(ci_hi,    digits=5)],
    SF_ABM            = [round(SF_abm,   digits=5)],
    ABM_hours         = [TERMINAL_TIME],
)
#CSV.write(joinpath(OUTDIR, "summary.csv"), summary_df)

# ============================================================================
# DONE
# ============================================================================

println("\n", "="^65)
println("  DONE")
println("  Particle   : $PARTICLE  |  E = $ENERGY_MEV_U MeV/u  |  D = $DOSE_GY Gy")
println("  Spheroid   : radius = $TUMOR_RADIUS µm  |  N cells = $N_init")
println("  Npar       : $Npar")
println("  SF (GSM2)  : $(round(SF_mean, digits=4))  " *
        "(95% CI $(round(ci_lo,digits=4)) – $(round(ci_hi,digits=4)))")
println("  SF (ABM)   : $(round(SF_abm, digits=4))")
println("  Output     : $OUTDIR/")
println("  Plots      : 28 figures (01_…28_) + 2 CSVs + snapshots")
println("="^65)
