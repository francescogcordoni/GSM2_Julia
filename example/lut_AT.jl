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
using StatsPlots
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
ENERGY_MEV_U  = 250.0      # kinetic energy per nucleon (MeV/u)
DOSE_GY       = 5.        # prescribed dose (Gy)

# Spheroid and cell geometry
TUMOR_RADIUS  = 100.0     # spheroid radius (µm)
R_CELL        = 15.0      # cell radius (µm)
X_BOX         = 550.0     # simulation box half-size (µm); match TUMOR_RADIUS
X_VOXEL       = 700.0     # voxel side length for beam-geometry calculation (µm)

# GSM2 domain geometry
RD            = 0.8      # domain radius (µm)
RN            = 7.2       # nucleus radius (µm)

# Phase-specific GSM2 parameters [G1/G0, S, G2/M, average]
# These are fitted to HSG cell line data — replace with your own fits.
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

# Simulation options
const TYPE_AT        = "KC"         # track structure: "KC" (Kiefer-Chatterjee) or "LEM"
const TRACK_SEG      = false         # true = fixed LET across depth (no Bragg-peak buildup)
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
#set_oxygen!(cell_df; plot_oxygen = true)
# Assign oxygenation (pO₂) profile across the spheroid (analytic diffusion)
set_oxygen_diffusion!(cell_df; density=:mean, plot_oxygen=true, verbose=true)

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

gsm2 = gsm2_cycle[1]

cell_df_is = filter(row -> row.is_cell == 1, cell_df)
nrow(cell_df_is) == 0 && (@warn "No cells with is_cell = 1 → skipping."; return)

grouped_df      = combine(groupby(cell_df_is, [:x, :y]),
                            :index => first => :representative_index)
rep_indices_set = Set(grouped_df.representative_index)

cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
at_single        = filter(row -> row.index in rep_indices_set, at)

mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

Rp = irrad_cond[1].Rp; Rc = irrad_cond[1].Rc
Kp = irrad_cond[1].Kp; Rk = Rp

r_lower            = 0.1 * Rc
core_radius_sq     = r_lower^2
mid_radius_sq      = (gsm2.rd + 150Rc)^2
penumbra_radius_sq = Rp^2

    # Radial dose lookup table
sim_ = 1000
impact_p = 10 .^ range(log10(r_lower), stop=log10(Rk), length=sim_)
dose_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

Threads.@threads for i in 1:sim_
    tid = Threads.threadid()
    track = Track(impact_p[i], 0.0, Rk)
    _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
    dose_lookup_threads[tid][i] = Gyr
end
dose_vec  = sum(dose_lookup_threads)
impact_vec = impact_p
core_dose  = dose_vec[1]

Plots.plot(impact_vec, dose_vec, xscale=:log10, yscale=:log10)

using Trapz  # or implement trapz manually, see below
function yF(impact_vec, lineal_vec)
    num = trapz(impact_vec, lineal_vec .* 2π .* impact_vec)
    den = trapz(impact_vec, 2π .* impact_vec)
    return num / den
end

function yD(impact_vec, lineal_vec)
    num = trapz(impact_vec, lineal_vec .^ 2 .* 2π .* impact_vec)
    den = trapz(impact_vec, lineal_vec .* 2π .* impact_vec)
    return num / den
end

function ystar(impact_vec, lineal_vec, y0)
    num = y0^2 .* trapz(impact_vec, (1 .- exp.(-(lineal_vec ./ y0) .^ 2)) .* 2π .* impact_vec)
    den = trapz(impact_vec, lineal_vec .* 2π .* impact_vec)
    return num / den
end

yF(impact_vec, dose_vec)
yD(impact_vec, dose_vec)
ystar(impact_vec, dose_vec, 150.)

ρ = 1.

energy_vec = 10 .^ range(log10(0.1), stop = log10(250.), length = 1000)
yF_vec     = Array{Float64}(undef, 0)
yD_vec     = Array{Float64}(undef, 0)
ystar_vec  = Array{Float64}(undef, 0)
LET_vec  = Array{Float64}(undef, 0)

#for ENERGY_MEV_U in energy_vec
    println(ENERGY_MEV_U)
    setup(ENERGY_MEV_U, PARTICLE, DOSE_GY, TUMOR_RADIUS)

    Rp = irrad_cond[1].Rp 
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp
    if Rp < Rc
        Rp = Rc
    end

    r_lower            = 0.1 * Rc
    core_radius_sq     = r_lower^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    sim_ = 1000
    impact_p = 10 .^ range(log10(r_lower), stop = log10(gsm2.rd + Rp), length=sim_)
    dose_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        dose_lookup_threads[tid][i] = Gyr
    end
    dose_vec  = sum(dose_lookup_threads)
    impact_vec = impact_p
    core_dose  = dose_vec[1]

    Plots.plot(impact_vec[dose_vec .> 0] * 10^-6, dose_vec[dose_vec .> 0], xscale=:log10, yscale=:log10)

    push!(yF_vec, yF(impact_vec, dose_vec) * (ρ * π * gsm2.rd^2 / 0.1602))
    push!(yD_vec, yD(impact_vec, dose_vec) * (ρ * π * gsm2.rd^2 / 0.1602))
    push!(ystar_vec, ystar(impact_vec, dose_vec, 150.) * (ρ * π * gsm2.rd^2 / 0.1602))
    push!(LET_vec, ion.LET)

Plots.plot(energy_vec, yF_vec, xscale=:log10, label = "yF", xlab = "Energy [MeV]", ylab = "Lineal energy [keV/μm]")
Plots.plot!(energy_vec, yD_vec, xscale=:log10, label = "yD")
Plots.plot!(energy_vec, ystar_vec, xscale=:log10, label = "y*")
Plots.plot!(energy_vec, LET_vec, xscale=:log10, label = "LET")

Plots.plot(energy_vec, 1.5 * yF_vec, xscale=:log10, label = "yF", xlab = "Energy [MeV]", ylab = "Lineal energy [keV/μm]")
Plots.plot!(energy_vec, 1.5 * yD_vec, xscale=:log10, label = "yD")
Plots.plot!(energy_vec, 1.5 * ystar_vec, xscale=:log10, label = "y*")
Plots.plot!(energy_vec, LET_vec, xscale=:log10, label = "LET")



