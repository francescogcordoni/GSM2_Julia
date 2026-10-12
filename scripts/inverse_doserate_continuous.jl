# ============================================================================
# inverse_doserate_continuous.jl
#
# Does the model produce the inverse dose-rate effect (IDRE) under CONTINUOUS
# irradiation of a cycling population? High-energy protons, in vitro setting.
#
# The IDRE (Mitchell, Bedford & Bailey 1979; HeLa between ~1.5 and ~0.4 Gy/h)
# is attributed to cells progressing during exposure and piling up in the
# radiosensitive G2 phase. So the test needs (i) cell-cycle progression during
# irradiation, (ii) phase-dependent GSM2 kinetics, (iii) damage-dependent arrest.
#
# Same tricks as continuous_dr_events.jl
# --------------------------------------
# • High-energy protons (~0.5 keV/µm): a single track adds ≪ 1 lesion per cell,
#   so lesions in each domain form a Poisson process in dose. Each lineage's
#   lesions are drawn once as dose stamps u ∈ [u_birth, D_max]; arrival time is
#   u / Ḋ. Doses are nested (dose D uses stamps ≤ D) and every dose rate sees
#   the SAME lesions: changing Ḋ only rescales time.
# • Common random numbers: the initial cells (phase, residual time, lesion
#   stream, SSA stream) are identical for every dose rate and arrest mode.
# • Nothing is stored except per-checkpoint tallies: memory is trivial.
#
# Clonogenic survival with division during exposure
# --------------------------------------------------
# Cells do not interact (no space limit: exponentially growing culture). Each
# initial cell is followed as a clone; a division creates a daughter lineage
# with its own fresh lesion stream from the current dose onward. At the end of
# irradiation (each dose checkpoint) every living lineage i has a probability
# p_i of completing repair; the clone survives with 1 − Π(1 − p_i).
#   SF_clone  : fraction of irradiated cells leaving ≥ 1 clonogenic descendant
#               (repopulation does not inflate it).
#   SF_plated : clonogenic cells / cells present at the end of exposure
#               (what plating after continuous irradiation measures; doomed
#               cells are assumed still present at plating).
#
# Checkpoint configurations (the mechanism under test)
# ---------------------------------------------------
#   arrest = :all  → clock paused in G1, S, G2 while X > thr   (current ABM: thr = 0)
#   arrest = :g2   → only G2 holds damaged cells (G1/S checkpoint-deficient, e.g. HeLa)
#   arrest = :none → no arrest
#   mthr           → ending M with X > mthr is lethal (current ABM: mthr = 0);
#                    with X ≤ mthr the cell divides and each lesion goes to one
#                    daughter at random.
# Under continuous irradiation X never returns to 0 (lesions keep arriving), so
# thr = mthr = 0 freezes the cycle completely; thr > 0 mimics the known G2/M
# checkpoint threshold (~10–20 DSBs) and lets cells progress.
#
# Run: julia -t auto scripts/inverse_doserate_continuous.jl
# ============================================================================

using Base.Threads
using CSV, DataFrames
using Distributions
using Random
using Statistics
using Printf
using GLM, JLD2, DelimitedFiles, StatsBase, Optim, LsqFit, ProgressMeter
using InlineStrings, SparseArrays, ProgressBars, Distributed
using Plots, StatsPlots

include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# Settings
# ============================================================================
const E_PROTON    = 200.0                 # MeV
const O2_TEST     = 21.0                  # % O2
const AU          = 4.0                   # SSA time rescaling (as in the ABM)
const N_CLONES    = 4000                  # initial cells
const DOSES       = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]       # Gy (nested)
const D_ISO       = 4.0                   # dose for the iso-dose IDRE plot
const RATES_GYH   = [0.1, 0.25, 0.37, 0.5, 0.75, 1.0, 1.54, 3.0, 10.0, 60.0, 1e9]  # Gy/h; 1e9 = acute
const CONFIGS = [
    (name = "pause_all",       arrest = :all,  thr = 0,  mthr = 0),   # current ABM rules
    (name = "g2_only",         arrest = :g2,   thr = 0,  mthr = 0),
    (name = "all_thr15_mit3",  arrest = :all,  thr = 15, mthr = 3),
    (name = "g2_thr15_mit3",   arrest = :g2,   thr = 15, mthr = 3),
    (name = "none",            arrest = :none, thr = 0,  mthr = 0),
]
const SEED        = 2026
const OUTDIR      = joinpath(@__DIR__, "..", "data", "inverse_doserate_continuous")
mkpath(OUTDIR)

# Phase-specific GSM2 rates (M uses G2), as in the paper
const RD, RN = 0.8, 7.2
const GS = (GSM2(2.780479661191086, 0.012872261720543399, 0.04029756109753225, RD, RN),   # G1
            GSM2(5.84009601901114,  0.00589118894714544,  0.05794352736120672, RD, RN),   # S
            GSM2(1.7720064637774506, 0.024306291709970018, 5.704688326522623e-5, RD, RN)) # G2/M
setup_GSM2!(GS[1].r, GS[1].a, GS[1].b, RD, RN)
@inline gsm_of(phase::Int) = phase == 1 ? GS[1] : phase == 2 ? GS[2] : GS[3]

# Lesion yields (same formula as the dose LUT): totals per cell per Gy
const N_REPEAT = floor(Int, RN / RD)
const NCOL     = Base.invokelatest(getfield, Main, :domain)
const NDOM     = N_REPEAT * NCOL
const LET_P    = linear_interpolation("1H", E_PROTON, sp)
const K_TOT    = 9.0 * calculate_kappa("1H", LET_P, O2_TEST)       # X lesions / cell / Gy
const L_TOT    = K_TOT * 1e-3                                      # direct lethal Y / cell / Gy

# Cell cycle: 1=G1, 2=S, 3=G2, 4=M ; durations from PHASE_DURATIONS
const PH_NAMES = ("G1", "S", "G2", "M")
const PH_SHAPE = Tuple(PHASE_DURATIONS[p].shape for p in PH_NAMES)
const PH_SCALE = Tuple(PHASE_DURATIONS[p].scale for p in PH_NAMES)
@inline sample_phase_time(rng, ph::Int) = rand(rng, Gamma(PH_SHAPE[ph], PH_SCALE[ph]))

"""Asynchronous start: phase weighted by mean duration, residual time uniform in a fresh draw."""
function initial_phase(rng)
    w = PH_SHAPE .* PH_SCALE
    u = rand(rng) * sum(w)
    ph = u <= w[1] ? 1 : u <= w[1] + w[2] ? 2 : u <= w[1] + w[2] + w[3] ? 3 : 4
    return ph, rand(rng) * sample_phase_time(rng, ph)
end

# ============================================================================
# Lineage state
# ============================================================================
@inline wdom(x::Int, g::GSM2) = (g.r + g.a) * x + (x > 1 ? g.b * x * (x - 1) : 0.0)

function a0_of(X::Vector{Int}, g::GSM2)
    s = 0.0
    @inbounds for x in X
        x > 0 && (s += wdom(x, g))
    end
    return s
end

mutable struct Lineage
    id::Int
    phase::Int
    τ::Float64            # remaining time in current phase (h)
    X::Vector{Int}
    sumX::Int
    a0::Float64
    u::Vector{Float64}    # future lesion stamps (dose coordinate, Gy), sorted
    d::Vector{Int}        # their domains
    ptr::Int
    yfirst::Float64       # dose stamp of the first direct-lethal lesion (Inf if none)
    t::Float64            # current time (h)
    jnext::Int            # next dose checkpoint
end

"""Fresh lesion stream for a lineage that exists from dose coordinate u0 onward."""
function new_lineage(id, phase, τ, u0, t, jnext, rng)
    rem = max(D_MAX - u0, 0.0)
    n = rand(rng, Poisson(K_TOT * rem))
    u = sort!(u0 .+ rem .* rand(rng, n))
    d = rand(rng, 1:NDOM, n)
    yf = u0 + randexp(rng) / L_TOT
    yf > D_MAX && (yf = Inf)
    return Lineage(id, phase, τ, zeros(Int, NDOM), 0, 0.0, u, d, 1, yf, t, jnext)
end
const D_MAX = maximum(DOSES)

@inline function paused(phase::Int, sumX::Int, cfg)
    (sumX <= cfg.thr || phase == 4) && return false
    cfg.arrest === :all && return true
    cfg.arrest === :g2  && return phase == 3
    return false
end

"""Each lesion goes to one of the two daughters at random; returns the daughter's X."""
function split_damage!(X::Vector{Int}, rng)
    Xd = zeros(Int, length(X))
    @inbounds for i in eachindex(X)
        x = X[i]; x == 0 && continue
        k = rand(rng, Binomial(x, 0.5)); Xd[i] = k; X[i] = x - k
    end
    return Xd
end

"""
One SSA reaction in lineage state (X, sumX, a0) under rates g.
Returns (sumX, a0, alive).
"""
@inline function ssa_react!(X::Vector{Int}, sumX::Int, a0::Float64, g::GSM2, rng)
    u = rand(rng) * a0
    acc = 0.0; j = 0; jlast = 0
    @inbounds for i in eachindex(X)
        x = X[i]; x == 0 && continue
        jlast = i; acc += wdom(x, g)
        if acc >= u; j = i; break; end
    end
    j == 0 && (j = jlast)
    x = X[j]
    if rand(rng) * wdom(x, g) < g.r * x
        a0 -= wdom(x, g) - wdom(x - 1, g)
        X[j] = x - 1; sumX -= 1
        sumX == 0 && (a0 = 0.0)
        return sumX, a0, true
    end
    return sumX, a0, false
end

"""
Probability that a cell with damage X completes repair if irradiation stops now.
Exact (domain_GSM2) when thr = 0 and its phase is frozen until repair ends;
otherwise simulated forward without damage (phases may change, M may kill, a
division with residual damage gives 1 − (1 − p₁)(1 − p₂)).
"""
function completion_prob(X::Vector{Int}, sumX::Int, ph::Int, τ::Float64, a0::Float64,
                         cfg, Yz::Vector{Int}, rng)
    while true
        sumX == 0 && return 1.0
        cfg.thr == 0 && paused(ph, sumX, cfg) && return domain_GSM2(X, Yz, gsm_of(ph))
        run = !paused(ph, sumX, cfg)
        tr  = AU * randexp(rng) / a0
        if run && τ <= tr                            # phase ends first
            if ph == 4
                sumX > cfg.mthr && return 0.0        # mitosis with too much damage
                Xd = split_damage!(X, rng)
                s1 = sum(X); s2 = sum(Xd)
                p1 = completion_prob(X,  s1, 1, sample_phase_time(rng, 1), a0_of(X,  GS[1]), cfg, Yz, rng)
                p2 = completion_prob(Xd, s2, 1, sample_phase_time(rng, 1), a0_of(Xd, GS[1]), cfg, Yz, rng)
                return 1 - (1 - p1) * (1 - p2)
            end
            ph += 1; τ = sample_phase_time(rng, ph); a0 = a0_of(X, gsm_of(ph))
        else
            run && (τ -= tr)
            sumX, a0, ok = ssa_react!(X, sumX, a0, gsm_of(ph), rng)
            ok || return 0.0
        end
    end
end

# Per-clone tallies at each dose checkpoint
struct Tally
    prodq::Vector{Float64}   # Π (1 − p_i) over lineages present at the checkpoint
    num::Vector{Float64}     # Σ p_i
    den::Vector{Int}         # cells present (alive + doomed)
    ph::Matrix{Int}          # 4 × nD alive lineages per phase
end
Tally(nD) = Tally(ones(nD), zeros(nD), zeros(Int, nD), zeros(Int, 4, nD))

function kill!(L::Lineage, T::Tally)
    for j in L.jnext:length(DOSES)
        T.den[j] += 1                       # doomed cell still present, p = 0
    end
    L.jnext = length(DOSES) + 1
end

"""Advance one lineage through all remaining checkpoints; daughters go on `stack`."""
function run_lineage!(L::Lineage, Ḋ::Float64, cfg, T::Tally, stack, next_id::Ref{Int},
                      Yz::Vector{Int}, c::Int, rng)
    nD = length(DOSES)
    while L.jnext <= nD
        g   = gsm_of(L.phase)
        run = !paused(L.phase, L.sumX, cfg)
        t_c = run ? L.t + L.τ : Inf
        t_a = L.ptr <= length(L.u) ? L.u[L.ptr] / Ḋ : Inf
        t_y = L.yfirst / Ḋ
        t_r = L.sumX > 0 ? L.t + AU * randexp(rng) / L.a0 : Inf
        t_j = DOSES[L.jnext] / Ḋ
        tn  = min(t_c, t_a, t_y, t_r, t_j)
        run && (L.τ -= tn - L.t)
        L.t = tn

        if tn == t_j                                        # dose checkpoint
            j = L.jnext
            p = completion_prob(copy(L.X), L.sumX, L.phase, L.τ, L.a0, cfg, Yz,
                                Xoshiro(hash((SEED, c, L.id, j, :cp))))
            T.prodq[j] *= (1 - p); T.num[j] += p; T.den[j] += 1; T.ph[L.phase, j] += 1
            L.jnext += 1
        elseif tn == t_y                                    # direct lethal lesion
            kill!(L, T); return
        elseif tn == t_a                                    # lesion arrives
            dm = L.d[L.ptr]; x = L.X[dm]
            L.a0 += wdom(x + 1, g) - wdom(x, g)
            L.X[dm] = x + 1; L.sumX += 1; L.ptr += 1
        elseif tn == t_r                                    # repair / misrepair
            L.sumX, L.a0, ok = ssa_react!(L.X, L.sumX, L.a0, g, rng)
            ok || (kill!(L, T); return)
        else                                                # phase completes
            if L.phase == 4
                L.sumX > cfg.mthr && (kill!(L, T); return)  # mitotic catastrophe
                next_id[] += 1
                drng = Xoshiro(hash((SEED, c, next_id[], :born)))
                D2 = new_lineage(next_id[], 1, sample_phase_time(drng, 1),
                                 L.t * Ḋ, L.t, L.jnext, drng)
                if L.sumX > 0                                # residual damage split
                    D2.X = split_damage!(L.X, rng)
                    D2.sumX = sum(D2.X); D2.a0 = a0_of(D2.X, GS[1])
                    L.sumX  = sum(L.X)
                end
                push!(stack, D2)
                L.phase = 1; L.τ = sample_phase_time(rng, 1); L.a0 = a0_of(L.X, GS[1])
            else
                L.phase += 1; L.τ = sample_phase_time(rng, L.phase)
                L.a0 = a0_of(L.X, gsm_of(L.phase))
            end
        end
    end
end

"""All clones at one dose rate and mode → per-clone tallies."""
function run_rate(Ḋ::Float64, cfg)
    nD = length(DOSES)
    S   = zeros(nD, N_CLONES); NUM = zeros(nD, N_CLONES); DEN = zeros(Int, nD, N_CLONES)
    PH  = zeros(Int, 4, nD, N_CLONES)
    @threads for c in 1:N_CLONES
        Yz = zeros(Int, NDOM)
        T  = Tally(nD)
        irng = Xoshiro(hash((SEED, c, :init)))              # same for every rate and mode
        ph, τ = initial_phase(irng)
        stack = Lineage[new_lineage(1, ph, τ, 0.0, 0.0, 1, irng)]
        next_id = Ref(1)
        while !isempty(stack)
            L = pop!(stack)
            run_lineage!(L, Ḋ, cfg, T, stack, next_id, Yz, c,
                         Xoshiro(hash((SEED, c, L.id, :ssa))))
        end
        S[:, c] .= 1 .- T.prodq; NUM[:, c] .= T.num; DEN[:, c] .= T.den; PH[:, :, c] .= T.ph
    end
    return S, NUM, DEN, PH
end

# ============================================================================
# Main
# ============================================================================
@printf("threads=%d  E=%.0f MeV  LET=%.2f keV/µm  X lesions/cell/Gy=%.1f  domains=%d  clones=%d\n",
        nthreads(), E_PROTON, LET_P, K_TOT, NDOM, N_CLONES)

rows = DataFrame(mode = String[], dose_rate_Gyh = Float64[], dose_Gy = Float64[],
                 SF_clone = Float64[], SE_clone = Float64[], SF_plated = Float64[],
                 cells_per_clone = Float64[], fG1 = Float64[], fS = Float64[],
                 fG2 = Float64[], fM = Float64[])

for cfg in CONFIGS, Ḋ in RATES_GYH
    mode = cfg.name
    tsim = @elapsed S, NUM, DEN, PH = run_rate(Ḋ, cfg)
    for (j, D) in enumerate(DOSES)
        nph = vec(sum(PH[:, j, :], dims = 2)); tot = max(sum(nph), 1)
        push!(rows, (mode, Ḋ, D, mean(S[j, :]), std(S[j, :]) / sqrt(N_CLONES),
                     sum(NUM[j, :]) / sum(DEN[j, :]), mean(DEN[j, :]),
                     nph[1] / tot, nph[2] / tot, nph[3] / tot, nph[4] / tot))
    end
    jiso = findfirst(==(D_ISO), DOSES)
    @printf("  %-15s Ḋ=%-8.3g Gy/h  SF_clone(%.0f Gy)=%.4f  SF_plated=%.4f  G2+M=%.2f  (%.1f s)\n",
            mode, Ḋ, D_ISO, rows.SF_clone[end - length(DOSES) + jiso],
            rows.SF_plated[end - length(DOSES) + jiso],
            rows.fG2[end - length(DOSES) + jiso] + rows.fM[end - length(DOSES) + jiso], tsim)
    @printf("  %-15s cells per initial cell at end of %.0f Gy: %.2f\n", "", D_ISO,
            rows.cells_per_clone[end - length(DOSES) + jiso])
end
CSV.write(joinpath(OUTDIR, "idre_continuous.csv"), rows)

# ============================================================================
# IDRE detection and plots
# ============================================================================
println("\nInverse dose-rate check at $(D_ISO) Gy (excluding the acute column):")
for mode in (c.name for c in CONFIGS)
    s = sort(rows[(rows.mode .== mode) .& (rows.dose_Gy .== D_ISO) .&
                  (rows.dose_rate_Gyh .< 1e8), :], :dose_rate_Gyh)
    # IDRE = survival DEcreases as dose rate DEcreases somewhere (beyond 2 SE)
    hits = [(s.dose_rate_Gyh[i], s.dose_rate_Gyh[i + 1]) for i in 1:nrow(s) - 1
            if s.SF_clone[i] < s.SF_clone[i + 1] - 2 * hypot(s.SE_clone[i], s.SE_clone[i + 1])]
    println("  ", rpad(mode, 16),
            isempty(hits) ? "no IDRE: survival rises monotonically as the dose rate falls" :
            "IDRE between " * join(["$(a)–$(b) Gy/h" for (a, b) in hits], ", "))
end

let
    default(fontfamily = "Computer Modern", framestyle = :box, grid = true, gridalpha = 0.25,
            minorgrid = false, tickfontsize = 10, guidefontsize = 12, titlefontsize = 13,
            legendfontsize = 9, foreground_color_legend = nothing, background_color_legend = RGBA(1, 1, 1, 0.8),
            linewidth = 2, left_margin = 6Plots.mm, bottom_margin = 5Plots.mm,
            right_margin = 4Plots.mm, top_margin = 3Plots.mm, dpi = 200)
    rl(Ḋ) = Ḋ >= 1e8 ? "acute" : @sprintf("%.3g Gy/h", Ḋ)
    floorSF(x) = max(x, 1e-6)                       # keep zeros on the log axis
    names  = [c.name for c in CONFIGS]
    titles = Dict("pause_all" => "pause all (current ABM)", "g2_only" => "pause G2 only",
                  "all_thr15_mit3" => "pause all, thr 15, mitosis ≤ 3",
                  "g2_thr15_mit3" => "pause G2, thr 15, mitosis ≤ 3", "none" => "no arrest")
    modecol = palette(:tab10)[1:length(names)]
    ratecol = cgrad(:viridis, length(RATES_GYH) - 1, categorical = true)

    # 1. SF_clone vs dose: one panel per configuration, shared axes, one legend panel
    panels = map(names) do mode
        p = plot(yscale = :log10, xlabel = "Dose (Gy)", ylabel = "SF (clonogenic)",
                 title = get(titles, mode, mode), legend = false, ylims = (1e-4, 1.5),
                 yticks = 10.0 .^ (-4:0), xlims = (0, maximum(DOSES) + 0.2))
        for (k, Ḋ) in enumerate(RATES_GYH)
            s = sort(rows[(rows.mode .== mode) .& (rows.dose_rate_Gyh .== Ḋ), :], :dose_Gy)
            acute = Ḋ >= 1e8
            plot!(p, vcat(0.0, s.dose_Gy), floorSF.(vcat(1.0, s.SF_clone));
                  color = acute ? :black : ratecol[k], ls = acute ? :dash : :solid,
                  marker = :circle, ms = 3, markerstrokewidth = 0, label = "")
        end
        p
    end
    leg = plot(framestyle = :none, legend = :left, xlims = (0, 1), ylims = (0, 1),
               legendtitle = "Dose rate", legendtitlefontsize = 10)
    for (k, Ḋ) in enumerate(RATES_GYH)
        acute = Ḋ >= 1e8
        plot!(leg, [NaN], [NaN]; color = acute ? :black : ratecol[k],
              ls = acute ? :dash : :solid, marker = :circle, ms = 3, label = rl(Ḋ))
    end
    savefig(plot(panels..., leg; layout = (2, 3), size = (1500, 850)),
            joinpath(OUTDIR, "sf_vs_dose.png"))

    # 2. Iso-dose survival vs dose rate (IDRE = dip) and G2+M fraction
    p1 = plot(xscale = :log10, yscale = :log10, xlabel = "Dose rate (Gy/h)",
              ylabel = "SF at $(D_ISO) Gy", title = "Iso-dose survival", legend = false)
    p2 = plot(xscale = :log10, xlabel = "Dose rate (Gy/h)", ylims = (0, 1),
              ylabel = "G2+M fraction at end of $(D_ISO) Gy", title = "Cell-cycle redistribution",
              legend = false)
    for (i, mode) in enumerate(names)
        s = sort(rows[(rows.mode .== mode) .& (rows.dose_Gy .== D_ISO) .&
                      (rows.dose_rate_Gyh .< 1e8), :], :dose_rate_Gyh)
        ac = rows[(rows.mode .== mode) .& (rows.dose_Gy .== D_ISO) .&
                  (rows.dose_rate_Gyh .>= 1e8), :]
        sf = floorSF.(s.SF_clone)
        lo = min.(2 .* s.SE_clone, 0.9 .* sf)
        plot!(p1, s.dose_rate_Gyh, sf; yerror = (lo, 2 .* s.SE_clone), color = modecol[i],
              marker = :circle, ms = 4, markerstrokewidth = 0, label = "")
        plot!(p1, s.dose_rate_Gyh, floorSF.(s.SF_plated); color = modecol[i], ls = :dot,
              linewidth = 1.5, label = "")
        isempty(ac) || hline!(p1, [floorSF(ac.SF_clone[1])]; color = modecol[i], ls = :dash,
                              alpha = 0.5, linewidth = 1, label = "")
        plot!(p2, s.dose_rate_Gyh, s.fG2 .+ s.fM; color = modecol[i], marker = :circle,
              ms = 4, markerstrokewidth = 0, label = "")
    end
    leg2 = plot(framestyle = :none, legend = :left, xlims = (0, 1), ylims = (0, 1))
    for (i, mode) in enumerate(names)
        plot!(leg2, [NaN], [NaN]; color = modecol[i], marker = :circle, ms = 4,
              label = get(titles, mode, mode))
    end
    plot!(leg2, [NaN], [NaN]; color = :gray, label = "clonogenic (clone)")
    plot!(leg2, [NaN], [NaN]; color = :gray, ls = :dot, label = "plated")
    plot!(leg2, [NaN], [NaN]; color = :gray, ls = :dash, alpha = 0.5, label = "acute (clone)")
    savefig(plot(p1, p2, leg2; layout = @layout([a b c{0.26w}]), size = (1700, 520)),
            joinpath(OUTDIR, "isodose_vs_doserate.png"))
end
println("\nResults in ", OUTDIR)
