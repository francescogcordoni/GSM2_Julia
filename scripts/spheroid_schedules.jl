# ============================================================================
# scripts/spheroid_schedules.jl
#
# Two studies, for 1H at 100 MeV and 12C at 80 MeV/u (choose with STUDY, run once each):
#   STUDY = :dose           single acute doses of 0, 2, 4 and 6 Gy (spheroid radius 250 µm)
#   STUDY = :fractionation  the same total dose of 6 Gy delivered as
#                           1 × 6 Gy,  3 × 2 Gy every 24 h,  2 × 3 Gy 48 h apart
#                           on a larger spheroid (radius 350 µm, hypoxic core); the oxygen
#                           profile is recomputed on the current spheroid before every fraction
# Same pipeline as example/spheroid.jl and scripts/spheroid_fractionation.jl
# (dose computed on the current spheroid at every fraction, delayed death).
#
# Each realisation grows the spheroid ONCE for T_PRE hours; every condition then
# starts from a copy of that spheroid, so differences reflect the irradiation only.
#
# Outputs (scripts/output/spheroid_schedules/<STUDY>/):
#   hourly.csv      cell numbers every hour (and right after each fraction): total, clonogenic, doomed;
#                   phases of all cells (G0_all … M_all) and of the clonogenic cells (G0 … M)
#   fractions.csv   per-fraction dose, GSM2 survival probability and surviving fraction
#
# HOW TO RUN
#   julia --threads auto scripts/spheroid_schedules.jl      (with STUDY = :dose, then :fractionation)
# then: julia results/plot_spheroid_schedules.jl
# ============================================================================

using Base.Threads, Distributed, CSV, DataFrames, Distributions, Random, Plots, StatsPlots
using ProgressBars, GLM, JLD2, DelimitedFiles, Statistics, StatsBase, Optim, LsqFit
using ProgressMeter, InlineStrings, CUDA, SparseArrays, Printf
using Statistics: mean

println("Threads available: ", nthreads())
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS
# ============================================================================
const T_PRE  = 24.0          # h of growth before the first fraction
const T_END  = 168.0         # h of follow-up after the FIRST fraction (same for all schedules)
const N_REP  = 3             # stochastic realisations
const IONS = [(tag = "1H_100MeV",  particle = "1H",  E = 100.0),
              (tag = "12C_80MeVu", particle = "12C", E = 80.0)]
# schedules: fraction times (h after the first fraction) and doses (Gy)
const STUDY = :dose          # :dose or :fractionation

const SCHEDULES = STUDY === :dose ? [
    (tag = "control",   label = "0 Gy",                times = Float64[],        doses = Float64[]),
    (tag = "1x2Gy",     label = "2 Gy",                times = [0.0],            doses = [2.0]),
    (tag = "1x4Gy",     label = "4 Gy",                times = [0.0],            doses = [4.0]),
    (tag = "1x6Gy",     label = "6 Gy",                times = [0.0],            doses = [6.0]),
] : [
    (tag = "control",   label = "control",             times = Float64[],        doses = Float64[]),
    (tag = "1x6Gy",     label = "1 × 6 Gy",            times = [0.0],            doses = [6.0]),
    (tag = "3x2Gy_24h", label = "3 × 2 Gy, every 24 h", times = [0.0, 24.0, 48.0], doses = [2.0, 2.0, 2.0]),
    (tag = "2x3Gy_48h", label = "2 × 3 Gy, 48 h apart", times = [0.0, 48.0],     doses = [3.0, 3.0]),
]

# Geometry (box with room for 192 h of growth of the control: the lattice is periodic).
# Fractionation: radius 350 µm > necrosis-onset radius (~206 µm), so the core is hypoxic.
TUMOR_RADIUS = STUDY === :dose ? 250.0 : 350.0
R_CELL       = 15.0
X_BOX        = STUDY === :dose ? 550.0 : 700.0
X_VOXEL      = 2 * X_BOX

RD = 0.8; RN = 7.2
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

const TYPE_AT     = "KC"
const TRACK_SEG   = true
const TARGET_GEOM = "circle"
const CALC_TYPE   = "full"
const NAT_APO     = 1e-10
const FATE        = :incremental
const OXYGEN      = :diffusion        # profile of the current spheroid, recomputed before every fraction
const O2_DENSITY  = :radial           # local cell density per shell: space freed by dead cells reoxygenates
const O2_UNIFORM  = 21.0

const OUTDIR = joinpath(@__DIR__, "output", "spheroid_schedules", String(STUDY))
mkpath(OUTDIR)

# ============================================================================
# 2. SETUP — identical to example/spheroid.jl (top level)
# ============================================================================
# ============================================================================

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(R_G1, A_G1, B_G1, RD, RN)
gsm2_cycle[2] = GSM2(R_S,  A_S,  B_S,  RD, RN)
gsm2_cycle[3] = GSM2(R_G2, A_G2, B_G2, RD, RN)
gsm2_cycle[4] = GSM2(R_AVG, A_AVG, B_AVG, RD, RN)
setup_GSM2!(R_AVG, A_AVG, B_AVG, RD, RN)

setup_IonIrrad!(2.0, 100.0, "1H"; type_AT = TYPE_AT)
N_sideVox   = Int(floor(2 * X_BOX / X_VOXEL))
N_CellsSide = 2 * convert(Int64, floor(X_BOX / (2 * R_CELL)))
setup_cell_lattice!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide;
                    ParIrr = "false", track_seg = TRACK_SEG, full_cycle = true)
@eval Main begin
    tumor_radius = $TUMOR_RADIUS
    X_voxel      = $X_VOXEL
    X_box        = $X_BOX
end
setup_cell_population!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide,
                       gsm2_cycle[4]; ParIrr = "false")
setup_irrad_conditions!(ion, irrad, TYPE_AT, cell_df, TRACK_SEG)
OXYGEN === :diffusion ? set_oxygen_diffusion!(cell_df; density = O2_DENSITY, plot_oxygen = false) :
                        (cell_df.O .= O2_UNIFORM)

# Fate model: set_fate_model! exists only in the latest utilities_abm.jl
if isdefined(Main, :set_fate_model!)
    set_fate_model!(FATE)
else
    @warn "utilities_abm.jl has no set_fate_model!: compute_times_domain! is used as it is. " *
          "For the incremental model with delayed death use the latest utilities_abm.jl."
end
hasproperty(cell_df, :death_type)  || (cell_df.death_type  = fill(-1, nrow(cell_df)))
hasproperty(cell_df, :dam_X_total) || (cell_df.dam_X_total = zeros(Int, nrow(cell_df)))
hasproperty(cell_df, :dam_Y_total) || (cell_df.dam_Y_total = zeros(Int, nrow(cell_df)))

const N_INIT = count(cell_df.is_cell .== 1)
println("Spheroid: $N_INIT cells")
print_phase_distribution(cell_df; label = "pre-irradiation")
cell_df_pristine = deepcopy(cell_df)

# domain tables of the whole lattice (built by setup_cell_population!, cover every site)
const DFX = df_center_x
const DFY = df_center_y
const AT0 = at

# ============================================================================

# ============================================================================
# 3. HELPERS
# ============================================================================
G(name::Symbol) = Base.invokelatest(getfield, Main, name)

# Advance running timers by an idle interval (fallback if utilities_abm.jl lacks it)
if !isdefined(Main, :_advance_idle!)
    function _advance_idle!(cell_df::DataFrame, δ::Float64)
        δ <= 0 && return nothing
        for col in (cell_df.death_time, cell_df.cycle_time, cell_df.recover_time)
            @inbounds for i in eachindex(col)
                cell_df.is_cell[i] == 1 && isfinite(col[i]) && (col[i] = max(0.0, col[i] - δ))
            end
        end
        return nothing
    end
end

"""Counts on the cell DataFrame (alive cells only): doomed = finite death_time."""
function counts(df::DataFrame)
    alive  = df.is_cell .== 1
    doomed = alive .& isfinite.(df.death_time)
    clono  = alive .& .!isfinite.(df.death_time)
    ph = df.cell_cycle[clono]; pa = df.cell_cycle[alive]
    return (total = count(alive), clonogenic = count(clono), doomed = count(doomed),
            G0 = count(==("G0"), ph), G1 = count(==("G1"), ph), S = count(==("S"), ph),
            G2 = count(==("G2"), ph), M = count(==("M"), ph),
            G0_all = count(==("G0"), pa), G1_all = count(==("G1"), pa), S_all = count(==("S"), pa),
            G2_all = count(==("G2"), pa), M_all = count(==("M"), pa))
end

clonogenic(df) = count((df.is_cell .== 1) .& .!isfinite.(df.death_time))

"""Spheroid radius now (µm), from the alive cells."""
current_radius(df) = (a = df.is_cell .== 1;
    maximum(sqrt.(df.x[a] .^ 2 .+ df.y[a] .^ 2 .+ df.z[a] .^ 2)) + R_CELL)

"""
Box check. The lattice is a CUBE with periodic neighbours, so what matters is
whether cells sit on the outermost layer of sites (|x|, |y| or |z| at the face),
not their distance from the centre (corners are far from the faces).
Returns (cells on the outermost layer, max radius, 95th-percentile radius).
"""
function box_check(df)
    a = findall(==(1), df.is_cell)
    face = X_BOX - R_CELL - 1e-6                      # centre of the outermost site layer
    edge = count(i -> max(abs(df.x[i]), abs(df.y[i]), abs(df.z[i])) >= face, a)
    r = sqrt.(df.x[a] .^ 2 .+ df.y[a] .^ 2 .+ df.z[a] .^ 2)
    return edge, maximum(r), quantile(r, 0.95)
end

"""One fraction on the current spheroid, same steps as example/spheroid.jl."""
function irradiate_fraction!(df::DataFrame, c, seed::UInt64)
    setup_IonIrrad!(c.dose, c.E, c.particle; type_AT = TYPE_AT)
    Rnow = current_radius(df)
    R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, Rnow)
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
    OXYGEN === :diffusion && set_oxygen_diffusion!(df; density = O2_DENSITY, plot_oxygen = false)  # before the dose

    F    = G(:irrad).dose / (1.602e-9 * G(:LET))
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    at_  = deepcopy(AT0)
    MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, DFX, DFY, at_,
                 gsm2_cycle, TYPE_AT, TRACK_SEG; seed = seed)
    Random.seed!(seed)
    MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)

    alive = findall(==(1), df.is_cell)
    if FATE === :legacy
        compute_cell_survival_GSM2!(df, gsm2_cycle)     # resets timers: fine in legacy mode
    else
        for i in alive                                   # GSM2 survival without resetting timers
            ph = df.cell_cycle[i]
            g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
            df.sp[i] = domain_GSM2(df.dam_X_dom[i], df.dam_Y_dom[i], g)
        end
    end
    notdoomed = [i for i in alive if !isfinite(df.death_time[i])]
    mean_dose = mean(df.dose_cell[alive])
    mean_sp   = isempty(notdoomed) ? NaN : mean(df.sp[notdoomed])
    compute_times_domain!(df, gsm2_cycle; nat_apo = NAT_APO, terminal_time = Inf,
                          verbose = false, summary = false)
    return (Npar = Npar, R_beam = R_beam, R_spheroid = Rnow, mean_dose = mean_dose,
            mean_sp = mean_sp)
end


"""
ABM for `Δ` hours (may be fractional) starting at time t0 (h after the first
fraction), one hour at a time; counts are recorded at every whole hour.
"""
function run_for!(df, Δ::Float64, t0::Float64, rows::DataFrame, meta)
    t = t0; t_end = t0 + Δ
    while t < t_end - 1e-9
        δ = min(1.0, t_end - t, floor(t + 1 + 1e-9) - t)     # stop at whole hours and at t_end
        δ <= 1e-9 && (δ = min(1.0, t_end - t))
        ts, _ = run_simulation_abm!(df; nat_apo = NAT_APO, terminal_time = δ,
                                    snapshot_times = Int[], verbose = false,
                                    return_dataframes = false, update_input = true)
        _advance_idle!(df, δ - (isempty(ts.time) ? 0.0 : ts.time[end]))
        t += δ
        abs(t - round(t)) < 1e-9 && push!(rows, merge(meta, (t = round(t),), counts(df)))
    end
    return t_end
end

# ============================================================================
# 4. RUN
# ============================================================================
hourly = DataFrame()
fx_rows = DataFrame(ion = String[], schedule = String[], rep = Int[], fraction = Int[], t = Float64[],
                    dose = Float64[], clonogenic_before = Int[], clonogenic_after = Int[], SF = Float64[],
                    mean_sp_GSM2 = Float64[], mean_dose = Float64[], Npar = Int[])

for rep in 1:N_REP
    # common pre-growth for all conditions of this realisation
    grown = deepcopy(cell_df_pristine)
    Random.seed!(hash((:grow, rep)))
    pre = DataFrame()
    run_for!(grown, T_PRE, -T_PRE, pre, (ion = "", schedule = "", rep = rep))
    N0 = count(==(1), grown.is_cell)
    @printf("\n== realisation %d: %d cells at the first fraction ==\n", rep, N0)

    for ion_ in IONS, sc in SCHEDULES
        sc.tag == "control" && ion_ !== IONS[1] && continue          # control is ion-independent
        ion_tag = sc.tag == "control" ? "none" : ion_.tag
        df   = deepcopy(grown)
        rows = DataFrame()
        meta = (ion = ion_tag, schedule = sc.tag, rep = rep)
        push!(rows, merge(meta, (t = 0.0,), counts(df)))
        t = 0.0
        for (f, (tf, D)) in enumerate(zip(sc.times, sc.doses))
            Random.seed!(hash((ion_tag, sc.tag, rep, f, :abm)))
            t = run_for!(df, tf - t, t, rows, meta)
            before = clonogenic(df)
            info = irradiate_fraction!(df, (dose = D, E = ion_.E, particle = ion_.particle),
                                       hash((ion_tag, sc.tag, rep, f)))
            after = clonogenic(df)
            push!(rows, merge(meta, (t = t + 1e-3,), counts(df)))
            push!(fx_rows, (ion_tag, sc.tag, rep, f, t, D, before, after, after / max(before, 1),
                            info.mean_sp, info.mean_dose, info.Npar))
            @printf("  %-11s %-15s fraction %d at %6.2f h: %.2f Gy, SF %.3f (GSM2 %.3f)\n",
                    ion_tag, sc.tag, f, t, info.mean_dose, after / max(before, 1), info.mean_sp)
        end
        Random.seed!(hash((ion_tag, sc.tag, rep, :follow)))
        run_for!(df, T_END - t, t, rows, meta)
        edge, rmax, _ = box_check(df)
        @printf("  %-11s %-15s end: %d cells (clonogenic %d), max radius %.0f µm%s\n",
                ion_tag, sc.tag, count(==(1), df.is_cell), clonogenic(df), rmax,
                edge > 0 ? "  WARNING: cells on the box face" : "")
        rows[!, :N0] .= N0
        append!(hourly, rows)
        CSV.write(joinpath(OUTDIR, "hourly.csv"), hourly)        # saved as it goes
        CSV.write(joinpath(OUTDIR, "fractions.csv"), fx_rows)
    end
end
println("\nSaved in $OUTDIR (study $STUDY; run again with the other STUDY for the second set)")
