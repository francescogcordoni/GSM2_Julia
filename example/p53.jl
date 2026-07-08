# ============================================================================
# example/p53.jl
#
# KEY EXAMPLE — Irradiation of a 3-D spheroid followed by GSM2 model with p53 dynamics
# (ABM) simulation of post-irradiation cell dynamics.
#
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
using DifferentialEquations

println("Threads available: ", nthreads())

# ── Load all source utilities ─────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS — edit these to change the irradiation condition
# ============================================================================

# Irradiation
PARTICLE      = "1H"      # ion species: "1H", "4He", "12C", "16O"
ENERGY_MEV_U  = 250.0      # kinetic energy per nucleon (MeV/u)
DOSE_GY       = 5.        # prescribed dose (Gy)

# Spheroid and cell geometry
TUMOR_RADIUS  = 50.0     # spheroid radius (µm)
R_CELL        = 15.0      # cell radius (µm)
X_BOX         = 550.0     # simulation box half-size (µm); match TUMOR_RADIUS
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
set_oxygen!(cell_df; plot_oxygen = true)

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

cell_df.O .= 7.

println("\n--- Monte Carlo dose deposition ---")
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df,
                    df_center_x, df_center_y, at,
                    gsm2_cycle, TYPE_AT, TRACK_SEG)

println("\n--- Poisson DNA damage sampling ---")
@time MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)

active_cells = @view cell_df.index[cell_df.is_cell .== 1]
n_active     = length(active_cells)

# Pre-allocate per-cell storage (index k, thread-safe)
death_times    = Vector{Float64}(undef, n_active)
recover_times  = Vector{Float64}(undef, n_active)
death_types    = Vector{Int}(undef, n_active)
X_finals       = Vector{Vector{Int64}}(undef, n_active)
Y_finals       = Vector{Vector{Int64}}(undef, n_active)
markov_results = Vector{DataFrame}(undef, n_active)
ode_results    = Vector{DataFrame}(undef, n_active)

#@Threads.threads 
for k in eachindex(active_cells)
    i = active_cells[k]

    phase = cell_df.cell_cycle[i]
    gsm2  = if phase == "G1" || phase == "G0"
        gsm2_cycle[1]
    elseif phase == "S"
        gsm2_cycle[2]
    elseif phase == "G2" || phase == "M"
        gsm2_cycle[3]
    else
        gsm2_cycle[4]
    end

    # 1) GSM2 survival probability
    cell_df.sp[i] = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

    # 2) Stochastic repair + p53 dynamics
    death_time, recover_time, death_type, X_f, Y_f, df_markov, df_mol =
        compute_repair_domain_p53_history(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

    # Tag DataFrames with cell index and store at slot k
    df_markov[!, :cell_id] = fill(i, nrow(df_markov))
    if nrow(df_mol) > 0
        df_mol[!, :cell_id] = fill(i, nrow(df_mol))
    end

    if df_markov.sum_Y[size(df_markov, 1)] == 0
        recover_time = df_markov.time[df_markov.sum_X .== 0][1]
        death_time = Inf
    else
        recover_time = Inf
        death_time = df_markov.time[df_markov.sum_Y .> 0][1]
    end

    death_times[k]    = death_time
    recover_times[k]  = recover_time
    death_types[k]    = death_type
    X_finals[k]       = X_f
    Y_finals[k]       = Y_f
    markov_results[k] = df_markov
    ode_results[k]    = df_mol
end

# Build master DataFrames from per-cell results
all_markov = vcat(markov_results...; cols=:union)
all_ode    = vcat(filter(df -> nrow(df) > 0, ode_results)...; cols=:union)

results_df = DataFrame(
    cell_id      = collect(active_cells),
    death_time   = death_times,
    recover_time = recover_times,
    death_type   = death_types,
)

# ── Select a cell to inspect ─────────────────────────────────────────────────
cell_to_plot = active_cells[1]          # change index to inspect a different cell
k_plot       = findfirst(==(cell_to_plot), collect(active_cells))

mk  = filter(:cell_id => ==(cell_to_plot), all_markov)
od  = filter(:cell_id => ==(cell_to_plot), all_ode)
dt  = death_times[k_plot]
rt  = recover_times[k_plot]
dtp = death_types[k_plot]

# ── Plot 1: Markov chain — sum_X and sum_Y vs time ───────────────────────────
p1 = plot(mk.time, mk.sum_X;
            seriestype=:steppost, label="sum_X", lw=2,
            xlabel="Time (h)", ylabel="Lesion count",
            title="Cell $cell_to_plot | type=$dtp")
plot!(p1, mk.time, mk.sum_Y;
        seriestype=:steppost, label="sum_Y", lw=2)
isfinite(dt) && vline!(p1, [dt]; color=:red, lw=1.5, ls=:dash, label="death")
isfinite(rt) && vline!(p1, [rt]; color=:green, lw=1.5, ls=:dash, label="recover")

# ── Plot 2: p53 network ODE ───────────────────────────────────────────────────
p2 = plot(od.time, od.p53s;  label="p53s",  lw=2,
            xlabel="Time (h)", ylabel="Concentration",
            title="p53 network")
plot!(p2, od.time, od.Casp3; label="Casp3", lw=2)
plot!(p2, od.time, od.p21;   label="p21",   lw=2)
plot!(p2, od.time, od.ATMs;  label="ATMs",  lw=2)
plot!(p2, od.time, od.Sen;  label="Sen",  lw=2)
plot!(p2, od.time, od.PTEN;  label="PTEN",  lw=2)
isfinite(dt) && vline!(p2, [dt]; color=:red, lw=1.5, ls=:dash, label="death")
isfinite(rt) && vline!(p2, [rt]; color=:green, lw=1.5, ls=:dash, label="recover")

fig = plot(p1, p2; layout=(1,2), size=(1100, 420),
            left_margin=10Plots.mm, bottom_margin=8Plots.mm, right_margin=5Plots.mm)
display(fig)
savefig(fig, "p53_repair.png")

# ── All-cells p53 dynamics ────────────────────────────────────────────────────
cell_ids = unique(all_ode.cell_id)
palette  = distinguishable_colors(length(cell_ids))

pa = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="p53s — all cells", legend=false)
pb = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="Casp3 — all cells", legend=false)
pc = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="p21 — all cells",  legend=false)
ps = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="Sen — all cells",  legend=false)
pten = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="PTEN — all cells",  legend=false)

for (idx, cid) in enumerate(cell_ids)
    col = palette[idx]
    od_c = filter(:cell_id => ==(cid), all_ode)
    k_c  = findfirst(==(cid), collect(active_cells))
    dt_c = isnothing(k_c) ? Inf : death_times[k_c]

    plot!(pa, od_c.time, od_c.p53s;  color=col, lw=1.2, alpha=0.7)
    plot!(pb, od_c.time, od_c.Casp3; color=col, lw=1.2, alpha=0.7)
    plot!(pc, od_c.time, od_c.p21;   color=col, lw=1.2, alpha=0.7)
    plot!(ps, od_c.time, od_c.Sen;   color=col, lw=1.2, alpha=0.7)
    plot!(pten, od_c.time, od_c.PTEN;   color=col, lw=1.2, alpha=0.7)

    if isfinite(dt_c)
        vline!(pa, [dt_c]; color=col, lw=1, ls=:dash, alpha=0.7)
        vline!(pb, [dt_c]; color=col, lw=1, ls=:dash, alpha=0.7)
        vline!(pc, [dt_c]; color=col, lw=1, ls=:dash, alpha=0.7)
        vline!(ps, [dt_c]; color=col, lw=1, ls=:dash, alpha=0.7)
        vline!(pten, [dt_c]; color=col, lw=1, ls=:dash, alpha=0.7)
    end
end

ps

display(plot(pa, pb, pc, ps; layout=(2,2), size=(1400, 400)))
#savefig(pa, "p53s_all.png")
#savefig(pb, "Casp3_all.png")
#savefig(pc, "p21_all.png")

S_star = 150.
pS = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="S", legend=false)
pG = plot(; xlabel="Time (h)", ylabel="Concentration",
            title="G1 or G2", legend=false)
plot!(pS, od.time, (od.Sen)./(od.Sen .+ S_star);   color=:black, lw=1.2, alpha=0.7)
plot!(pG, od.time, 0.5*(od.Sen)./(od.Sen .+ S_star);   color=:black, lw=1.2, alpha=0.7)
display(plot(pG, pS; layout=(1,2), size=(1400, 400)))


