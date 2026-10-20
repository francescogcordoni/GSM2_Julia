# ============================================================================
# example/staab_calibration.jl
#
# Calibrate the GSM2 kinetic parameters on V79-379A spheroid survival
# (Staab et al., Radiat. Res. 161, 219–227, 2004) and predict the conditions not
# used in the fit.
#
# PIPELINE
#   1. SETTINGS   conditions (radiation quality × pO2), train/test split, parameters
#   2. TARGETS    survival curves: digitised points from data/staab2004/fig1.csv if
#                 present, otherwise derived from the published LQ fit (X rays,
#                 aerobic spheroids: α = 0.092, β = 0.0301) and the OER/RBE of
#                 Table 1 (dose-modifying approximation)
#   3. SETUP      one spheroid, as in example/spheroid.jl (top level)
#   4. PHYSICS    for every (condition, dose): MC_dose_CPU! once, MC_loop_damage!
#                 N_DAMAGE times; the lesions are stored compactly (histogram of X
#                 lesions per domain + "any Y" flag per cell). Done ONCE.
#   5. FIT        survival = mean over cells of exp(−ρ·mX)·Π_d f(X_d; a, b, r):
#                 exact GSM2 completion probability of the sublethal lesions times
#                 the probability of no directly lethal lesion (ρ = λ/κ). Free: the
#                 kinetic factors (a, b; r if FIT_R) and, if FIT_LETHAL, ρ for the two
#                 carbon qualities. Default: fit on the three AEROBIC curves, predict
#                 the three hypoxic ones (OER and hypoxic RBE are predictions).
#   6. RESULTS    predictions, LQ fits, OER and RBE vs Table 1, PDF figures
#
# Parameters: the phase-specific GSM2 sets are kept and multiplied by global
# factors s_a, s_b, s_r (MODE = :phase_scaled), or one set (a, b, r) is used for
# all cells (MODE = :single). Survival after acute irradiation depends only on
# a/r and b/r, so r is not identifiable from these data: FIT_R = false keeps it
# fixed (recommended); the repair time scale is set elsewhere (τ = au).
#
# HOW TO RUN
#   julia --threads auto example/staab_calibration.jl
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
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
using CUDA
using Statistics: mean
using SparseArrays
using Printf
using Dates

println("Threads available: ", nthreads())
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. SETTINGS
# ============================================================================
const MODE      = :phase_scaled     # :phase_scaled (factors on G1/S/G2 sets) or :single
const FIT_R     = false             # r is not identifiable from acute survival
const TRAIN     = ["X_145"]         # calibration on aerobic X rays only; everything else is predicted
const FIT_LETHAL = false            # true: also fit ρ = λ/κ per carbon quality (then that curve is fitted)
const USE_BRAGG = false             # extended Bragg peak: mixed field, LET spectrum not reported → off
const RHO_X     = 1e-3              # ρ for X rays (code default λ = 10⁻³ κ), kept fixed
const N_DAMAGE  = 3                 # lesion realisations per (condition, dose)
const N_STARTS  = 4
const MAX_ITER  = 300
const S_MIN     = 1e-4              # survival levels below this are not used
const O2_FLOOR  = 0.1               # % O2 used for "0 mmHg" (not strictly anoxic)
const SEED      = 2026

# pO2 (mmHg) → % O2 (units of cell_df.O used by calculate_kappa / OER)
mmHg_to_pct(p) = p == 0 ? O2_FLOOR : 100 * p / 760

# Radiation qualities: LET from the paper; energies found from the stopping-power
# tables so that the track-segment LET matches (Bragg peak: mono-LET approximation)
const QUALITIES = Dict(
    "X"       => (particle = "1H",  LET = 1.7),    # 250 kV X rays: low-LET proxy
    "C_plat"  => (particle = "12C", LET = 18.0),   # 227 MeV/u plateau (measured LET)
    "C_bragg" => (particle = "12C", LET = 150.0),  # extended Bragg peak: effective LET reproducing the OER (table: 60)
)
# Conditions: name => (quality, pO2 in mmHg, doses in Gy)
const CONDITIONS = [
    "X_145"       => ("X",       145.0, [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
    "X_0"         => ("X",         0.0, [2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0]),
    "C_plat_145"  => ("C_plat",  145.0, [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
    "C_plat_0"    => ("C_plat",    0.0, [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]),
    "C_bragg_690" => ("C_bragg", 690.0, [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    "C_bragg_0"   => ("C_bragg",   0.0, [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]),
]
filter!(c -> USE_BRAGG || !startswith(first(c), "C_bragg"), CONDITIONS)

# Published values (Staab et al. 2004)
const ALPHA_X, BETA_X = 0.092, 0.0301                # X rays, spheroids, 145 mmHg
const TABLE1 = (OER = Dict("X" => 2.87, "C_plat" => 2.76, "C_bragg" => 1.40),
                RBE_aer = Dict("C_plat" => 1.39, "C_bragg" => 2.11),
                RBE_hyp = Dict("C_plat" => 1.45, "C_bragg" => 4.31))
const DIGITISED = joinpath(@__DIR__, "..", "data", "staab2004", "fig1.csv")  # condition,dose,S

# Geometry (spheroid as in the other examples; survival is per cell)
TUMOR_RADIUS = 200.0; R_CELL = 15.0; X_BOX = 350.0; X_VOXEL = 700.0
RD = 0.8; RN = 7.2
const TYPE_AT = "KC"; const TRACK_SEG = true; const TARGET_GEOM = "circle"; const CALC_TYPE = "full"

# Phase-specific GSM2 sets (starting point; relative radiosensitivity is kept)
const BASE = Dict("G1" => (r = 2.7805, a = 0.01287, b = 0.04030),
                  "S"  => (r = 5.8401, a = 0.00589, b = 0.05794),
                  "G2" => (r = 1.7720, a = 0.02431, b = 5.705e-5))
const BASE_AVG = (r = 2.5657, a = 0.01481, b = 0.01266)

const OUTDIR = joinpath(@__DIR__, "output", "staab_calibration",
                        "run_" * Dates.format(now(), "yyyymmdd_HHMMSS"))
mkpath(OUTDIR)
println("Results in $OUTDIR")

# ============================================================================
# 2. TARGETS
# ============================================================================
S_X(D) = exp(-ALPHA_X * D - BETA_X * D^2)
"""Target survival: digitised data if available, otherwise Table 1 dose-modifying factors."""
function derived_target(cond, D)
    q, pO2, _ = Dict(CONDITIONS)[cond]
    oer = pO2 == 0 ? TABLE1.OER["X"] : 1.0           # reference: X rays at the same pO2
    q == "X" && return S_X(D / oer)
    rbe = pO2 == 0 ? TABLE1.RBE_hyp[q] : TABLE1.RBE_aer[q]
    return S_X(rbe * D / oer)
end

targets = DataFrame(condition = String[], quality = String[], pO2 = Float64[], dose = Float64[],
                    S = Float64[], source = String[], set = String[])
dig = isfile(DIGITISED) ? CSV.read(DIGITISED, DataFrame) : nothing
for (cond, (q, pO2, doses)) in CONDITIONS
    set = cond in TRAIN ? "train" : "test"
    if dig !== nothing && any(String.(dig.condition) .== cond)
        g = dig[String.(dig.condition) .== cond, :]
        for r in eachrow(g)
            r.S >= S_MIN && push!(targets, (cond, q, pO2, Float64(r.dose), Float64(r.S), "digitised", set))
        end
    else
        for D in doses
            S = derived_target(cond, D)
            S >= S_MIN && push!(targets, (cond, q, pO2, D, S, "Table 1", set))
        end
    end
end
println("\n--- Targets ---"); show(stdout, combine(groupby(targets, [:condition, :set, :source]), nrow => :n_doses)); println()
dig === nothing && println("NOTE: no digitised data ($DIGITISED); targets derived from the published LQ " *
                           "and Table 1 (dose-modifying approximation). Digitise Fig. 1 for a stricter test.")

# ============================================================================
# 3. SETUP (top level, as in example/spheroid.jl)
# ============================================================================
gsm2_cycle = [GSM2(BASE["G1"].r, BASE["G1"].a, BASE["G1"].b, RD, RN),
              GSM2(BASE["S"].r,  BASE["S"].a,  BASE["S"].b,  RD, RN),
              GSM2(BASE["G2"].r, BASE["G2"].a, BASE["G2"].b, RD, RN),
              GSM2(BASE_AVG.r, BASE_AVG.a, BASE_AVG.b, RD, RN)]
setup_GSM2!(BASE_AVG.r, BASE_AVG.a, BASE_AVG.b, RD, RN)
Random.seed!(SEED)
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
setup_cell_population!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide, gsm2_cycle[4]; ParIrr = "false")
setup_irrad_conditions!(ion, irrad, TYPE_AT, cell_df, TRACK_SEG)
const PRISTINE = deepcopy(cell_df)
const ALIVE    = findall(==(1), PRISTINE.is_cell)
const PHASE    = [let p = String(PRISTINE.cell_cycle[i]); p == "G0" ? "G1" : p == "M" ? "G2" : p end for i in ALIVE]
const DFX, DFY, AT0 = df_center_x, df_center_y, at
println("Spheroid: $(length(ALIVE)) cells, phases: ", countmap(PHASE))

G(name::Symbol) = Base.invokelatest(getfield, Main, name)

"""Energy (MeV/u) on the high-energy side of the Bragg peak with the requested LET."""
function energy_for_let(particle, L)
    lo, hi = log(0.5), log(1000.0)
    for _ in 1:80
        m = 0.5 * (lo + hi)
        linear_interpolation(particle, exp(m), sp) > L ? (lo = m) : (hi = m)
    end
    return exp(0.5 * (lo + hi))
end
const ENERGY = Dict(q => energy_for_let(v.particle, v.LET) for (q, v) in QUALITIES)
for (q, v) in QUALITIES
    @printf("  %-8s %s  LET %.1f keV/µm → E = %.1f MeV/u\n", q, v.particle, v.LET, ENERGY[q])
end

# ============================================================================
# 4. PHYSICS — dose and damage ONCE per (condition, dose)
# ============================================================================
"""
Compact lesion record of one realisation: for every cell, the counts of domains
with k = 1, 2, … X lesions (pairs k => n_k) and whether any Y lesion is present.
"""
struct Lesions
    hist::Vector{Vector{Pair{Int,Int}}}   # per cell: (X lesions in a domain => number of such domains)
    anyY::BitVector                       # per cell: a Y lesion was sampled (λ = 10⁻³ κ)
    mX::Vector{Float64}                   # per cell: expected number of X lesions (κ_cell · dose)
end

"""
Compact record of the lesions of every cell. `mX` allows the directly lethal
lesions to be treated analytically, P(no Y) = exp(−ρ·mX), so that the ratio
ρ = λ/κ can be fitted without resampling the damage.
"""
function compact(df::DataFrame, ion_name::String, LET::Float64)
    n = length(ALIVE)
    hist = Vector{Vector{Pair{Int,Int}}}(undef, n); anyY = falses(n); mX = zeros(n)
    for (k, i) in enumerate(ALIVE)
        anyY[k] = any(>(0), df.dam_Y_dom[i])
        cm = countmap(filter(>(0), df.dam_X_dom[i]))
        hist[k] = [x => c for (x, c) in cm]
        mX[k] = 9.0 * calculate_kappa(ion_name, LET, Float64(df.O[i])) * df.dose_cell[i]
    end
    return Lesions(hist, anyY, mX)
end

# The cache key includes EVERYTHING that changes the lesions (radiation qualities and
# energies included), so editing QUALITIES or O2_FLOOR triggers a new computation.
const LIB_KEY  = (CONDITIONS, sort(collect(QUALITIES); by = first), sort(collect(ENERGY); by = first),
                  N_DAMAGE, TUMOR_RADIUS, R_CELL, X_BOX, RD, RN, O2_FLOOR, SEED, TYPE_AT, TRACK_SEG, :lesions_v2)
const LIB_FILE = joinpath(@__DIR__, "output", "staab_calibration",
                          "lesions_" * string(hash(LIB_KEY); base = 16) * ".jld2")
const FORCE_RECOMPUTE = false      # true: ignore any cached lesion library
lib = Dict{Tuple{String,Float64},Vector{Lesions}}()
if isfile(LIB_FILE) && !FORCE_RECOMPUTE
    println("\n--- Loading lesion library $LIB_FILE ---")
    lib = load(LIB_FILE, "lib")
else
    println("\n--- Dose and damage (once per condition and dose) ---")
    for (cond, (q, pO2, _)) in CONDITIONS, D in unique(targets.dose[targets.condition .== cond])
        v = QUALITIES[q]
        df = deepcopy(PRISTINE)
        df.O .= mmHg_to_pct(pO2)
        setup_IonIrrad!(D, ENERGY[q], v.particle; type_AT = TYPE_AT)
        R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, TUMOR_RADIUS)
        setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
        Npar = round(Int, G(:irrad).dose / (1.602e-9 * G(:LET)) * π * R_beam^2 * 1e-8)
        t = @elapsed MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, DFX, DFY, deepcopy(AT0),
                                  gsm2_cycle, TYPE_AT, TRACK_SEG; seed = hash((SEED, cond, D)))
        LETused = G(:irrad_cond)[df.energy_step[ALIVE[1]]].LET
        reals = Lesions[]
        for k in 1:N_DAMAGE
            Random.seed!(hash((SEED, cond, D, k)))
            MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)
            push!(reals, compact(df, G(:ion).ion, LETused))
        end
        lib[(cond, D)] = reals
        @printf("  %-12s D = %5.1f Gy  LET used %.1f keV/µm  O2 %.2f%%  Npar = %9d  mean cell dose %.3f Gy  X/cell %.0f  Y/cell %.2f  (%.1f s)\n",
                cond, D, LETused, df.O[ALIVE[1]], Npar, mean(df.dose_cell[ALIVE]),
                mean(sum.(df.dam_X_dom[ALIVE])), mean(sum.(df.dam_Y_dom[ALIVE])), t)
    end
    jldsave(LIB_FILE; lib)
end

# ============================================================================
# 5. MODEL SURVIVAL AND FIT (a, b, r and, if FIT_LETHAL, ρ = λ/κ of the carbon qualities)
# ============================================================================
const KMAX = 1 + maximum(maximum(first.(h); init = 0) for v in values(lib) for L in v for h in L.hist)

"""log f(k) = log Π_{i=1}^{k} i r / ((r + a) i + b i (i − 1)), k = 0 … KMAX (exact GSM2 completion)."""
function logf_table(r, a, b)
    t = zeros(KMAX + 1)
    for k in 1:KMAX
        t[k + 1] = t[k] + log(k * r) - log((r + a) * k + b * k * (k - 1))
    end
    return t
end

"""GSM2 sets per phase for parameter vector θ (log scale factors or log values)."""
function phase_sets(θ)
    if MODE === :phase_scaled
        sa, sb = exp(θ[1]), exp(θ[2]); sr = FIT_R ? exp(θ[3]) : 1.0
        return Dict(p => (r = BASE[p].r * sr, a = BASE[p].a * sa, b = BASE[p].b * sb) for p in keys(BASE))
    else
        a, b = exp(θ[1]), exp(θ[2]); r = FIT_R ? exp(θ[3]) : BASE_AVG.r
        return Dict(p => (r = r, a = a, b = b) for p in keys(BASE))
    end
end

const NK = FIT_R ? 3 : 2                       # number of kinetic parameters in θ
const QUAL_OF = Dict(c => v[1] for (c, v) in CONDITIONS)
const RHO_IDX = Dict("C_plat" => NK + 1, "C_bragg" => NK + 2)   # ρ_Bragg unused when USE_BRAGG = false

"""Ratio ρ = λ/κ of directly lethal to sublethal lesion yield for a radiation quality."""
rho(θ, q) = (q == "X" || !FIT_LETHAL) ? RHO_X : exp(θ[RHO_IDX[q]])

"""Initial parameter vector: scale factors 1 (or the average set), ρ = 10⁻³."""
theta0() = vcat(MODE === :phase_scaled ? zeros(NK) : log.([BASE_AVG.a, BASE_AVG.b, BASE_AVG.r][1:NK]),
                FIT_LETHAL ? fill(log(RHO_X), 2) : Float64[])

"""
Mean survival of the spheroid cells for (condition, dose) at θ:
S = mean_cells [ exp(−ρ·mX) · Π_d f(X_d; a, b, r) ]  (directly lethal lesions analytic).
"""
function model_S(θ, cond, D)
    sets = phase_sets(θ)
    tabs = Dict(p => logf_table(s.r, s.a, s.b) for (p, s) in sets)
    ρ = rho(θ, QUAL_OF[cond])
    acc = 0.0; n = 0
    for L in lib[(cond, D)]
        for k in eachindex(L.hist)
            n += 1
            t = tabs[PHASE[k]]
            lp = -ρ * L.mX[k]
            for (x, c) in L.hist[k]; lp += c * t[x + 1]; end
            acc += exp(lp)
        end
    end
    return acc / n
end

rows(set) = findall(targets.set .== set)
function loss(θ, idx)
    L = 0.0
    for i in idx
        S = model_S(θ, targets.condition[i], targets.dose[i])
        L += (log(max(S, 1e-12)) - log(targets.S[i]))^2
    end
    return L / length(idx)
end

# self-check of the fast survival against domain_GSM2 on a few cells
let θ0 = theta0(), key0 = first(keys(lib)), L = lib[key0][1]
    sets = phase_sets(θ0); maxdiff = 0.0
    for k in 1:min(20, length(ALIVE))
        L.anyY[k] && continue
        s = sets[PHASE[k]]; g = GSM2(s.r, s.a, s.b, RD, RN)
        xs = Int[]
        for (x, c) in L.hist[k]; append!(xs, fill(x, c)); end
        p_ref = isempty(xs) ? 1.0 : domain_GSM2(xs, zeros(Int, length(xs)), g)
        p_fast = exp(sum(c * logf_table(s.r, s.a, s.b)[x + 1] for (x, c) in L.hist[k]; init = 0.0))
        maxdiff = max(maxdiff, abs(p_ref - p_fast))
    end
    @printf("\nself-check: |fast − domain_GSM2| ≤ %.2e on 20 cells\n", maxdiff)
end

np = length(theta0())
starts = [theta0()]
rng = Xoshiro(SEED)
while length(starts) < N_STARTS; push!(starts, theta0() .+ 0.8 .* randn(rng, np)); end
trace = DataFrame(start = Int[], iter = Int[], loss = Float64[])
best = nothing
println("\n--- Fit on $(TRAIN) ($(length(rows("train"))) points), $(np) parameters, MODE = $MODE ---")
for (s, θ0) in enumerate(starts)
    res = optimize(θ -> loss(θ, rows("train")), θ0, NelderMead(),
                   Optim.Options(iterations = MAX_ITER, store_trace = true, g_tol = 1e-10))
    for tr in Optim.trace(res); push!(trace, (s, tr.iteration, tr.value)); end
    @printf("  start %d: loss %.4e  (%d iterations)\n", s, Optim.minimum(res), Optim.iterations(res))
    (best === nothing || Optim.minimum(res) < Optim.minimum(best)) && (global best = res)
end
θbest = Optim.minimizer(best)
sets_best = phase_sets(θbest)

# ============================================================================
# 6. RESULTS
# ============================================================================
println("\n--- Best GSM2 parameters ---")
pars = DataFrame(phase = String[], r = Float64[], a = Float64[], b = Float64[], a_over_r = Float64[], b_over_r = Float64[])
for p in ("G1", "S", "G2")
    s = sets_best[p]; push!(pars, (p, s.r, s.a, s.b, s.a / s.r, s.b / s.r))
end
show(stdout, pars); println()
MODE === :phase_scaled && @printf("scale factors: s_a = %.3f  s_b = %.3f%s\n", exp(θbest[1]), exp(θbest[2]),
                                  FIT_R ? @sprintf("  s_r = %.3f", exp(θbest[3])) : "")
for q in ("X", "C_plat", "C_bragg")
    @printf("ρ = λ/κ  %-8s %.3e  (×%.1f the default 10⁻³)%s\n", q, rho(θbest, q), rho(θbest, q) / 1e-3,
            q == "X" || !FIT_LETHAL ? "  fixed" : "  fitted")
end

targets.model = [model_S(θbest, targets.condition[i], targets.dose[i]) for i in 1:nrow(targets)]
targets.log_residual = log.(max.(targets.model, 1e-12)) .- log.(targets.S)
loss_tab = combine(groupby(targets, [:condition, :set]), :log_residual => (r -> sqrt(mean(abs2, r))) => :rms_log_residual)
println(); show(stdout, loss_tab); println()

# LQ fits of the model curves → OER and RBE as in the paper (mean over S = 37, 10, 1, 0.1 %)
function lqfit(D, S)
    y = -log.(S); A = hcat(D, D .^ 2); θ = A \ y
    θ[2] < 0 && (θ = [sum(D .* y) / sum(D .^ 2), 0.0])
    return θ[1], θ[2]
end
Dof(α, β, S) = β > 0 ? -α / (2β) + sqrt(0.25 * (α / β)^2 - log(S) / β) : -log(S) / α
const SLEV = [0.37, 0.10, 0.01, 0.001]
lq = Dict{String,Tuple{Float64,Float64}}()
for (cond, _) in CONDITIONS
    g = targets[targets.condition .== cond, :]
    lq[cond] = lqfit(g.dose, g.model)
end
meanratio(c1, c2) = mean(Dof(lq[c1]..., S) / Dof(lq[c2]..., S) for S in SLEV)
cmp = DataFrame(quantity = String[], model = Float64[], Staab2004 = Float64[])
for (name, c1, c2, ref) in (("OER X rays",           "X_0",       "X_145",       TABLE1.OER["X"]),
                            ("OER 12C plateau",      "C_plat_0",  "C_plat_145",  TABLE1.OER["C_plat"]),
                            ("OER 12C Bragg peak",   "C_bragg_0", "C_bragg_690", TABLE1.OER["C_bragg"]),
                            ("RBE plateau, aerobic", "X_145",     "C_plat_145",  TABLE1.RBE_aer["C_plat"]),
                            ("RBE Bragg, aerobic",   "X_145",     "C_bragg_690", TABLE1.RBE_aer["C_bragg"]),
                            ("RBE plateau, hypoxic", "X_0",       "C_plat_0",    TABLE1.RBE_hyp["C_plat"]),
                            ("RBE Bragg, hypoxic",   "X_0",       "C_bragg_0",   TABLE1.RBE_hyp["C_bragg"]))
    haskey(lq, c1) && haskey(lq, c2) && push!(cmp, (name, meanratio(c1, c2), ref))
end
println("\n--- OER and RBE: model vs Staab et al. 2004 (Table 1) ---"); show(stdout, cmp); println()

CSV.write(joinpath(OUTDIR, "best_parameters.csv"), pars)
CSV.write(joinpath(OUTDIR, "predictions.csv"), targets)
CSV.write(joinpath(OUTDIR, "loss_by_condition.csv"), loss_tab)
CSV.write(joinpath(OUTDIR, "oer_rbe.csv"), cmp)
CSV.write(joinpath(OUTDIR, "trace.csv"), trace)
CSV.write(joinpath(OUTDIR, "lq_model.csv"),
          DataFrame(condition = first.(CONDITIONS), alpha = [lq[c][1] for c in first.(CONDITIONS)],
                    beta = [lq[c][2] for c in first.(CONDITIONS)]))
CSV.write(joinpath(OUTDIR, "settings.csv"),
          DataFrame(setting = ["MODE", "FIT_R", "FIT_LETHAL", "rho_X", "rho_C_plat", "rho_C_bragg", "LET_C_bragg",
                               "TRAIN", "N_DAMAGE", "O2_FLOOR", "TUMOR_RADIUS", "N_cells", "targets"],
                    value = string.([MODE, FIT_R, FIT_LETHAL, rho(θbest, "X"), rho(θbest, "C_plat"),
                                     rho(θbest, "C_bragg"), QUALITIES["C_bragg"].LET,
                                     join(TRAIN, ";"), N_DAMAGE, O2_FLOOR, TUMOR_RADIUS, length(ALIVE),
                                     dig === nothing ? "Table 1 derived" : "digitised"])))

# Figures (vector PDF, no titles)
default(framestyle = :box, lw = 1.8, guidefontsize = 12, tickfontsize = 10,
        legendfontsize = 9, legendtitlefontsize = 10, dpi = 300,
        left_margin = 8Plots.mm, bottom_margin = 7Plots.mm, right_margin = 4Plots.mm,
        background_color_legend = RGBA(1, 1, 1, 0.85), foreground_color_legend = nothing)
paper_style!()
const PCOL = Dict(145.0 => :steelblue, 690.0 => :steelblue, 35.0 => :seagreen, 0.0 => :firebrick)
const QLAB = Dict("X" => "250 kV X rays", "C_plat" => "\$^{12}\$C plateau (LET 18 keV/\$\\mu\$m)", "C_bragg" => "\$^{12}\$C Bragg peak")
panels = map(unique(QUAL_OF[c] for (c, _) in CONDITIONS)) do q
    p = plot(yscale = :log10, xlabel = "Dose (Gy)", ylabel = "Surviving fraction",
             legend = :bottomleft, legendtitle = tex(QLAB[q]), ylims = (S_MIN / 2, 1.5))
    for (cond, (qq, pO2, _)) in CONDITIONS
        qq == q || continue
        g = sort(targets[targets.condition .== cond, :], :dose)
        tag = g.set[1] == "train" ? "fit" : "prediction"
        scatter!(p, g.dose, g.S; color = PCOL[pO2], ms = 5, label = tex("$(Int(pO2)) mmHg, data"))
        plot!(p, vcat(0.0, g.dose), vcat(1.0, g.model); color = PCOL[pO2],
              ls = tag == "fit" ? :solid : :dash, label = tex("$(Int(pO2)) mmHg, model ($tag)"))
    end
    p
end
savefig(plot(panels...; layout = (1, length(panels)), size = (600 * length(panels), 480)), joinpath(OUTDIR, "survival_fit_prediction.pdf"))

pc = scatter(cmp.Staab2004, cmp.model; ms = 6, color = :black, label = "",
             xlabel = "Staab et al. 2004", ylabel = "Model", legend = false, size = (520, 480))
lim = (0.8, maximum(vcat(cmp.Staab2004, cmp.model)) * 1.1)
plot!(pc, collect(lim), collect(lim); color = :gray, ls = :dash, xlims = lim, ylims = lim)
for r in eachrow(cmp); annotate!(pc, r.Staab2004, r.model, text("  " * r.quantity, 7, :left)); end
savefig(pc, joinpath(OUTDIR, "oer_rbe_model_vs_data.pdf"))

println("\nSaved in $OUTDIR:")
foreach(f -> println("  ", f), readdir(OUTDIR))
