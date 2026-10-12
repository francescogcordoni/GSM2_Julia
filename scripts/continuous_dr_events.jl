# ============================================================================
# continuous_dr_events.jl
#
# Continuous dose-rate survival for 12C 10 MeV/u, 12C 80 MeV/u and 1H 100 MeV,
# static G1 population (no cell cycle), as a verification test against
# Lea–Catcheside.
#
# Design
# ------
# 1. Damage is generated ONCE per replicate for the largest dose D_max, as a
#    sparse list of lesion events (cell, domain, particle index, nX, nY).
#    Doses are never stored. Memory ∝ number of lesions (~60 per cell per Gy),
#    not particles × domains, so no 0.5 Gy LUT reuse is needed.
#       • Near field (|r| < R_NEAR): explicit amorphous-track dose per particle.
#       • Far field  (|r| ≥ R_NEAR): the penumbra of millions of tracks is a
#         uniform background. Its lesions are Poisson with mean κ·D_far per
#         domain and each is stamped with a uniformly random particle index,
#         which gives it the correct arrival time.
# 2. Lower doses use the first N(D) particles (nested), so every dose and
#    every dose rate shares the SAME tracks and the SAME lesions.
# 3. Arrival times t_k = S_k / λ with S_k = cumsum(Exp(1)) fixed once:
#    changing the dose rate only rescales time. Common random numbers also for
#    the SSA (same RNG seed per cell at every dose rate).
# 4. Cells are independent in a static population, so each cell is simulated
#    on its own event list (cost ∝ its lesions, not all cells × all particles).
#    Survival at each dose checkpoint uses the exact GSM2 completion
#    probability (domain_GSM2) of the state at the end of irradiation.
#
# Run:  julia -t auto scripts/continuous_dr_events.jl
# ============================================================================

using Base.Threads
using CSV, DataFrames
using Distributions
using Random
using Statistics
using LinearAlgebra
using Printf
using GLM, JLD2, DelimitedFiles, StatsBase, Optim, LsqFit, ProgressMeter
using InlineStrings, SparseArrays, ProgressBars, Distributed
using Plots, StatsPlots

include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# Settings
# ============================================================================
const QUICK      = true          # true: small spheroids (smoke test)
const AU         = 4.0           # SSA time rescaling, as in compute_repair_domain
const R_NEAR     = 10.0          # µm, radius of explicit track structure
const MAX_REP    = QUICK ? 6 : 30     # max independent damage realisations per ion (lower SF at 6 Gy needs more cells)
const DOSE_RATES = [1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e3]   # Gy/s; 1e3 ≈ acute reference
const O2_TEST    = 21.0          # % O2, in-vitro aerobic test
const OUTDIR     = joinpath(@__DIR__, "..", "data", "continuous_dr_events")
mkpath(OUTDIR)

# G1 parameters (static population → one phase)
const R_G1, A_G1, B_G1 = 2.780479661191086, 0.012872261720543399, 0.04029756109753225
const RD, RN = 0.8, 7.2
const G1 = GSM2(R_G1, A_G1, B_G1, RD, RN)
setup_GSM2!(R_G1, A_G1, B_G1, RD, RN)

# Statistical precision is set by the number of SURVIVING cells at the highest
# dose, not by the number of cells. `min_survivors` = target of
# Σ_cells S_cell(D_max) at the acute rate, pooled over replicates
# (relative SE of SF(D_max) ≈ 1/√min_survivors). Replicates (fresh tracks and
# lesions on the same lattice) are added until the target or MAX_REP is reached.
const CONDITIONS = [
    (tag = "12C_10MeVu", particle = "12C", E = 10.0,
     tumor_radius = QUICK ? 150.0 : 450.0, X_box = QUICK ? 180.0 : 560.0,
     doses = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
     min_survivors = QUICK ? 150 : 600),
    (tag = "12C_80MeVu", particle = "12C", E = 80.0,
     tumor_radius = QUICK ? 150.0 : 250.0, X_box = QUICK ? 180.0 : 300.0,
     doses = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
     min_survivors = QUICK ? 150 : 600),
    (tag = "1H_100MeV",  particle = "1H",  E = 100.0,
     tumor_radius = QUICK ? 150.0 : 250.0, X_box = QUICK ? 180.0 : 300.0,
     doses = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],    # higher range: β needs curvature to be resolved
     min_survivors = QUICK ? 150 : 600),
]

# ============================================================================
# Sparse damage events
# ============================================================================
struct EventBuf
    cpos::Vector{Int32}
    p::Vector{Int32}
    dom::Vector{Int16}
    nx::Vector{Int16}
    ny::Vector{Int8}
end
EventBuf() = EventBuf(Int32[], Int32[], Int16[], Int16[], Int8[])

@inline function push_event!(b::EventBuf, c, p, d, nx, ny)
    push!(b.cpos, c); push!(b.p, p); push!(b.dom, d); push!(b.nx, nx); push!(b.ny, ny)
end

struct DamageEvents
    Npar::Int
    D_max::Float64
    ndom::Int
    ncell::Int
    offs::Vector{Int}      # CSR: events of cell c are offs[c]:offs[c+1]-1, sorted by p
    p::Vector{Int32}
    dom::Vector{Int16}
    nx::Vector{Int16}
    ny::Vector{Int8}
end

"""
    generate_damage_events(a, ion, cell_df, df_center_x, df_center_y, g, type_AT,
                           R_beam, D_max; seed, R_near, O2, x_cb, y_cb) -> DamageEvents

Samples all lesions for dose `D_max` once. See header for the near/far split.
Prints a dose-normalisation check: the dose implied by fluence × ∫kernel should
equal D_max. A large mismatch means the AT kernel or Npar is inconsistent.
"""
function generate_damage_events(a::AT, ion, cell_df::DataFrame,
                                df_center_x::DataFrame, df_center_y::DataFrame,
                                g::GSM2, type_AT::String, R_beam::Float64, D_max::Float64;
                                seed::Integer = 1, R_near::Float64 = R_NEAR,
                                O2::Float64 = O2_TEST,
                                x_cb::Float64 = 0.0, y_cb::Float64 = 0.0,
                                nchunks::Int = 8 * nthreads())
    n_repeat = floor(Int, g.Rn / g.rd)
    live  = cell_df[cell_df.is_cell .== 1, :]
    ncell = nrow(live)
    pos_of = Dict(idx => k for (k, idx) in enumerate(live.index))

    # Domain columns of one representative cell per (x, y)
    reps = representative_indices(live)
    dfx  = select_rows(df_center_x, reps)
    dfy  = select_rows(df_center_y, reps)
    @assert dfx.index == dfy.index
    cols = names(dfx, Not(:index))
    ncol = length(cols)
    nrep = nrow(dfx)
    domx = permutedims(Matrix{Float64}(dfx[:, cols]))     # ncol × nrep
    domy = permutedims(Matrix{Float64}(dfy[:, cols]))
    ndom = ncol * n_repeat
    ndom <= typemax(Int16) || error("too many domains per cell for Int16")

    # All z-layer cells sharing each representative column (track segment)
    xy_pos = Dict{Tuple{Float64,Float64},Vector{Int32}}()
    for (k, row) in enumerate(eachrow(live))
        push!(get!(xy_pos, (Float64(row.x), Float64(row.y)), Int32[]), Int32(k))
    end
    col_cells = Vector{Vector{Int32}}(undef, nrep)
    for j in 1:nrep
        k = pos_of[dfx.index[j]]
        col_cells[j] = xy_pos[(Float64(live.x[k]), Float64(live.y[k]))]
    end

    # Lesion yields per sub-domain per Gy (same formula as precompute_damage_lut!)
    κ = 9.0 * calculate_kappa(ion.ion, a.LET, O2) / (n_repeat * ncol)
    λ = κ * 1e-3

    # Track kernel and its integral (Gy·µm² per particle)
    lut_r, inv_step, b_near2 = build_radial_lut(a, g, type_AT)
    nlut   = length(lut_r)
    b_near = sqrt(b_near2)
    rp2    = a.Rp^2
    R_in   = max(min(R_near, a.Rp), b_near)
    Rin2   = R_in^2
    I_core = π * sum(lut_r) / inv_step
    I_mid  = (min(R_in, a.Rp) > b_near) ? 2π * a.Kp * log(min(R_in, a.Rp) / b_near) : 0.0
    I_far  = (a.Rp > R_in) ? 2π * a.Kp * log(a.Rp / R_in) : 0.0

    Npar     = round(Int, D_max / (1.602e-9 * a.LET) * π * R_beam^2 * 1e-8)
    Npar < typemax(Int32) || error("Npar too large for Int32")
    fl       = Npar / (π * R_beam^2)                  # particles / µm²
    D_kernel = fl * (I_core + I_mid + I_far)
    D_far    = fl * I_far

    @printf("  LET=%.2f keV/µm  Rp=%.1f µm  R_in=%.1f µm  Npar=%d  cells=%d  domains/cell=%d\n",
            a.LET, a.Rp, R_in, Npar, ncell, ndom)
    @printf("  dose check: fluence×∫kernel = %.3f Gy vs D_max = %.3f Gy (ratio %.3f); far-field %.1f%%\n",
            D_kernel, D_max, D_kernel / D_max, 100 * D_far / max(D_kernel, eps()))
    abs(D_kernel / D_max - 1) > 0.05 &&
        @warn "Kernel-implied dose differs from D_max by >5% — check AT normalisation / Npar"

    # Bucket grid over column centres for near-field lookup
    cx = vec(mean(domx, dims = 1)); cy = vec(mean(domy, dims = 1))
    rmax = maximum(hypot.(domx .- cx', domy .- cy'))
    hb = R_in + rmax + 1e-6
    buckets = Dict{Tuple{Int,Int},Vector{Int32}}()
    for j in 1:nrep
        push!(get!(buckets, (floor(Int, cx[j] / hb), floor(Int, cy[j] / hb)), Int32[]), Int32(j))
    end

    # ── Near field: threaded over particle chunks ────────────────────────────
    pranges = chunk_ranges(Npar, nchunks)
    bufs    = [EventBuf() for _ in pranges]
    @sync for (c, pr) in enumerate(pranges)
        Threads.@spawn begin
            rng = Xoshiro(hash((seed, :near, c)))
            buf = bufs[c]
            for p in pr
                x, y = GenerateHit_Circle(rng, x_cb, y_cb, R_beam)
                bx, by = floor(Int, x / hb), floor(Int, y / hb)
                for ix in bx-1:bx+1, iy in by-1:by+1
                    js = get(buckets, (ix, iy), nothing)
                    js === nothing && continue
                    for j in js, col in 1:ncol
                        dx = domx[col, j] - x; dy = domy[col, j] - y
                        d2 = dx * dx + dy * dy
                        d2 >= Rin2 && continue
                        v = d2 < b_near2 ? lut_r[min(trunc(Int, d2 * inv_step) + 1, nlut)] :
                            (d2 < rp2 ? a.Kp / d2 : 0.0)
                        v <= 0.0 && continue
                        mX, mY = κ * v, λ * v
                        for cp in col_cells[j], s in 1:n_repeat
                            nx = rand(rng, Poisson(mX))
                            ny = rand(rng, Poisson(mY))
                            (nx == 0 && ny == 0) && continue
                            push_event!(buf, cp, p, (col - 1) * n_repeat + s, nx, ny)
                        end
                    end
                end
            end
        end
    end

    # ── Far field: uniform background, random particle stamps ────────────────
    if D_far > 0
        cranges = chunk_ranges(ncell, nchunks)
        fbufs   = [EventBuf() for _ in cranges]
        mX, mY  = κ * D_far, λ * D_far
        @sync for (c, cr) in enumerate(cranges)
            Threads.@spawn begin
                rng = Xoshiro(hash((seed, :far, c)))
                buf = fbufs[c]
                for cp in cr, d in 1:ndom
                    for _ in 1:rand(rng, Poisson(mX))
                        push_event!(buf, cp, rand(rng, 1:Npar), d, 1, 0)
                    end
                    for _ in 1:rand(rng, Poisson(mY))
                        push_event!(buf, cp, rand(rng, 1:Npar), d, 0, 1)
                    end
                end
            end
        end
        append!(bufs, fbufs)
    end

    # ── Merge into CSR sorted by (cell, particle) ────────────────────────────
    cpos = reduce(vcat, (b.cpos for b in bufs))
    pv   = reduce(vcat, (b.p    for b in bufs))
    dv   = reduce(vcat, (b.dom  for b in bufs))
    nxv  = reduce(vcat, (b.nx   for b in bufs))
    nyv  = reduce(vcat, (b.ny   for b in bufs))
    bufs = nothing
    perm = sortperm((UInt64.(cpos) .<< 32) .| UInt64.(pv))
    cpos, pv, dv, nxv, nyv = cpos[perm], pv[perm], dv[perm], nxv[perm], nyv[perm]

    offs = zeros(Int, ncell + 1); offs[1] = 1
    for c in cpos; offs[c + 1] += 1; end
    cumsum!(offs, offs)

    nE = length(pv)
    @printf("  events=%d (%.1f MB)  X lesions/cell/Gy=%.1f (expected %.1f)  Y/cell/Gy=%.3f\n",
            nE, nE * 11 / 2^20, sum(Int, nxv) / ncell / D_max,
            κ * ndom * D_kernel / D_max, sum(Int, nyv) / ncell / D_max)

    return DamageEvents(Npar, D_max, ndom, ncell, offs, pv, dv, nxv, nyv)
end

# ============================================================================
# In-place SSA for one cell (same kinetics as compute_repair_domain, incl. au)
# ============================================================================
@inline wdom(x::Int, g::GSM2) = (g.r + g.a) * x + (x > 1 ? g.b * x * (x - 1) : 0.0)

mutable struct RepairState
    sumX::Int
    a0::Float64
end

"""
Advance repair for `Δt` hours. Returns `false` if a lethal event occurs.
Any non-repair event (misrepair `a`, interaction `b`) is lethal, as in
`compute_repair_domain`. Overshooting the window is discarded (memoryless).
"""
function repair_window!(X::Vector{Int}, st::RepairState, g::GSM2,
                        Δt::Float64, au::Float64, rng::AbstractRNG)::Bool
    t = 0.0
    while st.sumX > 0
        t += au * randexp(rng) / st.a0
        t >= Δt && return true
        u = rand(rng) * st.a0
        acc = 0.0; j = 0; jlast = 0
        @inbounds for i in eachindex(X)
            x = X[i]; x == 0 && continue
            jlast = i
            acc += wdom(x, g)
            if acc >= u; j = i; break; end
        end
        j == 0 && (j = jlast)
        x = X[j]
        if rand(rng) * wdom(x, g) < g.r * x
            st.a0 -= wdom(x, g) - wdom(x - 1, g)
            X[j] = x - 1
            st.sumX -= 1
            st.sumX == 0 && (st.a0 = 0.0)
        else
            return false
        end
    end
    return true
end

"""
Fills `surv[j]` with the survival probability of cell `c` for dose j
(= first `Ncut[j]` particles), then repair to completion.
"""
function simulate_cell!(surv, X::Vector{Int}, Yz::Vector{Int}, ev::DamageEvents, c::Int,
                        S::Vector{Float64}, λp::Float64, Ncut::Vector{Int},
                        g::GSM2, au::Float64, rng::AbstractRNG)
    fill!(X, 0)
    st = RepairState(0, 0.0)
    alive = true; tprev = 0.0; j = 1; nD = length(Ncut)
    @inbounds for e in ev.offs[c]:(ev.offs[c + 1] - 1)
        p = Int(ev.p[e])
        while j <= nD && Ncut[j] < p
            surv[j] = domain_GSM2(X, Yz, g); j += 1
        end
        j > nD && break
        tp = isinf(λp) ? 0.0 : S[p] / λp
        alive = repair_window!(X, st, g, tp - tprev, au, rng)
        tprev = tp
        alive || break
        if ev.ny[e] > 0
            alive = false; break
        end
        d = Int(ev.dom[e]); x = X[d]; n = Int(ev.nx[e])
        st.a0 += wdom(x + n, g) - wdom(x, g)
        X[d] = x + n
        st.sumX += n
    end
    while j <= nD
        surv[j] = alive ? domain_GSM2(X, Yz, g) : 0.0
        j += 1
    end
    return nothing
end

"""
Per-cell survival for one damage realisation: returns a vector (one entry per
dose rate) of `nD × ncell` matrices. Dose rate → particle rate without `au`
(the clock is in hours; `au` only rescales the SSA).
"""
function survival_per_cell(ev::DamageEvents, doses, rates; g::GSM2 = G1,
                           au::Float64 = AU, seed::Integer = 1)
    Ncut = [round(Int, d / ev.D_max * ev.Npar) for d in doses]
    S    = cumsum(randexp(Xoshiro(hash((seed, :arrivals))), ev.Npar))
    out  = Matrix{Float64}[]
    for Ḋ in rates
        λp = Ḋ >= 1e2 ? Inf : Ḋ * 3600.0 * ev.Npar / ev.D_max   # particles per hour
        cs = zeros(length(doses), ev.ncell)
        tsim = @elapsed @threads for c in 1:ev.ncell
            X  = zeros(Int, ev.ndom); Yz = zeros(Int, ev.ndom)
            rng = Xoshiro(hash((seed, :ssa, c)))          # same stream at every dose rate
            simulate_cell!(view(cs, :, c), X, Yz, ev, c, S, λp, Ncut, g, au, rng)
        end
        @printf("    Ḋ=%-7.0e Gy/s  SF(Dmax)=%.4f  (%.1f s)\n", Ḋ, mean(cs[end, :]), tsim)
        push!(out, cs)
    end
    return out
end

# ============================================================================
# Fits
# ============================================================================
"""
Weighted least squares on −ln S, constrained to α ≥ 0 and β ≥ 0 (exact
two-parameter NNLS: when the unconstrained solution has a negative component,
the optimum lies on a boundary, so the better of the β = 0 and α = 0 fits is
taken). Reported SEs are those of the unconstrained fit (a zero estimate has no
meaningful symmetric SE).
"""
function fit_lq(D::Vector{Float64}, S::Vector{Float64}, SE::Vector{Float64})
    m = S .> 0
    D, S, SE = D[m], S[m], SE[m]
    y = -log.(S)
    w = (S ./ max.(SE, 1e-6 .* S)) .^ 2              # 1 / var(ln S)
    A = hcat(D, D .^ 2)
    M = A' * Diagonal(w) * A
    θ = M \ (A' * (w .* y))
    se = sqrt.(max.(diag(inv(M)), 0.0))
    if any(θ .< 0)
        χ2(a, b) = sum(w .* (y .- a .* D .- b .* D .^ 2) .^ 2)
        a1 = max(sum(w .* D .* y) / sum(w .* D .^ 2), 0.0)         # β = 0
        b1 = max(sum(w .* D .^ 2 .* y) / sum(w .* D .^ 4), 0.0)    # α = 0
        θ = χ2(a1, 0.0) <= χ2(0.0, b1) ? [a1, 0.0] : [0.0, b1]
    end
    return θ[1], θ[2], se[1], se[2]
end

"""
β with α fixed (α does not depend on dose rate in GSM2/LPL kinetics: it comes
from single-track and direct-lethal events), constrained to β ≥ 0. With one
parameter the constrained optimum is the unconstrained one clipped at 0.
Returns (β, SE of the unconstrained estimate).
"""
function fit_beta_fixed_alpha(D, S, SE, α)
    m = S .> 0
    D, S, SE = D[m], S[m], SE[m]
    y = -log.(S) .- α .* D
    w = (S ./ max.(SE, 1e-6 .* S)) .^ 2
    sw = sum(w .* D .^ 4)
    return max(sum(w .* D .^ 2 .* y) / sw, 0.0), 1 / sqrt(sw)
end

G_LC(x) = x < 1e-8 ? 1.0 : 2 / x^2 * (x - 1 + exp(-x))

"""Lea–Catcheside survival with effective repair rate (r + a)/au."""
S_LC(D, Ḋ, α0, β0; g::GSM2 = G1, au::Float64 = AU) =
    Ḋ >= 1e2 ? exp(-α0 * D - β0 * D^2) :
    exp(-α0 * D - β0 * G_LC((g.r + g.a) / au * D / (Ḋ * 3600)) * D^2)

"""Pooled SF and SE over all cells of all replicates (cells are i.i.d. given dose)."""
pooled(cs::Matrix{Float64}) = (vec(mean(cs, dims = 2)),
                               vec(std(cs, dims = 2)) ./ sqrt(size(cs, 2)))


# ============================================================================
# Self-test of the in-place SSA against the analytic GSM2 completion
# ============================================================================
function self_test(; n = 20_000, ndom = 657, mean_per_dom = 0.18, seed = 7)
    rng = Xoshiro(seed)
    X0  = rand(rng, Poisson(mean_per_dom), ndom)
    p_exact = domain_GSM2(X0, zeros(Int, ndom), G1)
    alive = 0
    for i in 1:n
        X  = copy(X0)
        st = RepairState(sum(X), sum(wdom(x, G1) for x in X))
        alive += repair_window!(X, st, G1, Inf, AU, Xoshiro(hash((seed, i))))
    end
    p_ssa = alive / n
    se = sqrt(p_exact * (1 - p_exact) / n)
    @printf("[self-test] SSA survival %.4f vs domain_GSM2 %.4f (%.1f σ)\n",
            p_ssa, p_exact, (p_ssa - p_exact) / se)
    abs(p_ssa - p_exact) > 4se && @warn "SSA disagrees with analytic GSM2 survival"
end


# ============================================================================
# Run one condition
# ============================================================================
function run_condition(cond)
    println("\n", "="^70, "\nCONDITION ", cond.tag, "\n", "="^70)
    doses = Float64.(cond.doses)
    D_max = maximum(doses)
    nR    = length(DOSE_RATES)
    iref  = argmax(DOSE_RATES)

    env = setup(cond.E, cond.particle, D_max, cond.tumor_radius; X_box = cond.X_box)
    cell_df     = env.cell_df
    ion         = env.ion
    R_beam      = env.R_beam
    irrad_cond  = Base.invokelatest(getfield, Main, :irrad_cond)
    df_center_x = Base.invokelatest(getfield, Main, :df_center_x)
    df_center_y = Base.invokelatest(getfield, Main, :df_center_y)
    type_AT     = Base.invokelatest(getfield, Main, :type_AT)
    a = irrad_cond[1]                                   # track segment: one LET

    pooled_cs = [zeros(length(doses), 0) for _ in 1:nR]
    rep_rows  = DataFrame(tag = String[], rep = Int[], dose_rate_Gys = Float64[],
                          beta = Float64[], beta_se = Float64[])
    rep = 0
    while rep < MAX_REP
        rep += 1
        println("  replicate $rep")
        tgen = @elapsed ev = generate_damage_events(a, ion, cell_df, df_center_x, df_center_y,
                                                    G1, type_AT, R_beam, D_max; seed = rep)
        @printf("  damage generated in %.1f s\n", tgen)
        cs = survival_per_cell(ev, doses, DOSE_RATES; seed = rep)
        ev = nothing; GC.gc()

        # per-replicate β (α fixed to this replicate's acute α) → spread across realisations
        Sr, SEr = pooled(cs[iref]); αr = fit_lq(doses, Sr, SEr)[1]
        for k in 1:nR
            S, SE = pooled(cs[k]); β, sβ = fit_beta_fixed_alpha(doses, S, SE, αr)
            push!(rep_rows, (cond.tag, rep, DOSE_RATES[k], β, sβ))
        end
        for k in 1:nR
            pooled_cs[k] = hcat(pooled_cs[k], cs[k])
        end
        n_surv = sum(pooled_cs[iref][end, :])
        @printf("  pooled cells = %d   expected survivors at D_max (acute) = %.0f / %d\n",
                size(pooled_cs[iref], 2), n_surv, cond.min_survivors)
        n_surv >= cond.min_survivors && break
    end
    rep == MAX_REP && @warn "$(cond.tag): MAX_REP reached before min_survivors"

    # ── Pooled fits ──────────────────────────────────────────────────────────
    S0, SE0 = pooled(pooled_cs[iref])
    α0, β0, sα0, sβ0 = fit_lq(doses, S0, SE0)
    @printf("\n  acute: α = %.3f ± %.3f Gy⁻¹   β = %.4f ± %.4f Gy⁻²\n", α0, sα0, β0, sβ0)

    surv_rows = DataFrame(tag = String[], dose_Gy = Float64[], dose_rate_Gys = Float64[],
                          SF = Float64[], SE = Float64[], SF_LQ = Float64[], SF_LC = Float64[])
    lq_rows   = DataFrame(tag = String[], dose_rate_Gys = Float64[], alpha = Float64[],
                          beta = Float64[], beta_se = Float64[], beta_rep_sd = Float64[],
                          beta_LC = Float64[], n_cells = Int[],
                          alpha_free = Float64[], alpha_free_se = Float64[],
                          beta_free = Float64[], beta_free_se = Float64[])
    for k in 1:nR
        Ḋ = DOSE_RATES[k]
        S, SE = pooled(pooled_cs[k])
        β, sβ = k == iref ? (β0, sβ0) : fit_beta_fixed_alpha(doses, S, SE, α0)
        Slc   = [S_LC(D, Ḋ, α0, β0) for D in doses]
        βlc   = fit_beta_fixed_alpha(doses, Slc, SE ./ max.(S, eps()) .* Slc, α0)[1]
        rsd   = rep > 1 ? std(rep_rows.beta[rep_rows.dose_rate_Gys .== Ḋ]) : NaN
        αf, βf, sαf, sβf = fit_lq(doses, S, SE)         # both fitted (≥ 0) at this dose rate
        push!(lq_rows, (cond.tag, Ḋ, α0, β, sβ, rsd, βlc, size(pooled_cs[k], 2),
                        αf, sαf, βf, sβf))
        for (i, D) in enumerate(doses)
            push!(surv_rows, (cond.tag, D, Ḋ, S[i], SE[i], exp(-α0 * D - β * D^2), Slc[i]))
        end
    end

    CSV.write(joinpath(OUTDIR, "survival_$(cond.tag).csv"), surv_rows)
    CSV.write(joinpath(OUTDIR, "lq_$(cond.tag).csv"), lq_rows)
    CSV.write(joinpath(OUTDIR, "beta_replicates_$(cond.tag).csv"), rep_rows)

    println("\n  Ḋ (Gy/s)    β (α fixed)        β_LC     cells")
    for r in eachrow(lq_rows)
        @printf("  %-9.0e  %.4f ± %.4f   %.4f   %d\n",
                r.dose_rate_Gys, r.beta, r.beta_se, r.beta_LC, r.n_cells)
    end
    return surv_rows, lq_rows
end

# ============================================================================
# Plots
# ============================================================================
rate_label(Ḋ) = Ḋ >= 1e2 ? "acute" : @sprintf("%.0e Gy/s", Ḋ)

"""SF vs dose for one ion: simulation ± SE (points), LQ fit (solid), Lea–Catcheside (dashed)."""
function plot_sf(surv::DataFrame, tag::String)
    p = plot(yscale = :log10, xlabel = "Dose (Gy)", ylabel = "Surviving fraction",
             title = tag, legend = :bottomleft, size = (650, 480))
    cols = palette(:viridis, length(DOSE_RATES) + 1)
    for (k, Ḋ) in enumerate(sort(unique(surv.dose_rate_Gys)))
        s = sort(surv[surv.dose_rate_Gys .== Ḋ, :], :dose_Gy)
        D = vcat(0.0, s.dose_Gy)
        lo = min.(s.SE, 0.9 .* s.SF)                    # keep error bars > 0 on log axis
        scatter!(p, s.dose_Gy, s.SF; yerror = (lo, s.SE), color = cols[k], ms = 4,
                 label = rate_label(Ḋ))
        plot!(p, D, vcat(1.0, s.SF_LQ); color = cols[k], lw = 1.5, label = "")
        plot!(p, D, vcat(1.0, s.SF_LC); color = cols[k], ls = :dash, lw = 1, label = "")
    end
    plot!(p, [NaN], [NaN]; color = :black, lw = 1.5, label = "LQ fit (α fixed)")
    plot!(p, [NaN], [NaN]; color = :black, ls = :dash, label = "Lea–Catcheside")
    return p
end

# ============================================================================
# Main
# ============================================================================
println("threads = ", nthreads(), "   QUICK = ", QUICK, "   au = ", AU)
self_test()

all_surv = DataFrame(); all_lq = DataFrame()
for cond in CONDITIONS
    s, l = run_condition(cond)
    append!(all_surv, s); append!(all_lq, l)
    savefig(plot_sf(s, cond.tag), joinpath(OUTDIR, "sf_$(cond.tag).png"))
end
CSV.write(joinpath(OUTDIR, "survival_all.csv"), all_surv)
CSV.write(joinpath(OUTDIR, "lq_all.csv"), all_lq)

# β(Ḋ): simulation ± SE vs Lea–Catcheside (acute reference column excluded)
let
    p = plot(xscale = :log10, xlabel = "Dose rate (Gy/s)", ylabel = "β (Gy⁻²)",
             legend = :topleft, size = (750, 450))
    for (i, g) in enumerate(groupby(all_lq, :tag))
        s = sort(g[g.dose_rate_Gys .< 1e2, :], :dose_rate_Gys)
        scatter!(p, s.dose_rate_Gys, s.beta; yerror = s.beta_se, color = i, label = "$(g.tag[1]) ABM")
        plot!(p, s.dose_rate_Gys, s.beta_LC; color = i, ls = :dash, label = "$(g.tag[1]) Lea–Catcheside")
    end
    hline!(p, [0.0]; color = :gray, lw = 0.5, label = "")
    savefig(p, joinpath(OUTDIR, "beta_vs_doserate.png"))

    # α(Ḋ) and β(Ḋ) from constrained LQ fits at each dose rate; dashed = acute reference
    pa = plot(xscale = :log10, xlabel = "Dose rate (Gy/s)", ylabel = "α (Gy⁻¹)",
              legend = :best, size = (750, 450))
    pb = plot(xscale = :log10, xlabel = "Dose rate (Gy/s)", ylabel = "β (Gy⁻²)",
              legend = :best, size = (750, 450))
    for (i, g) in enumerate(groupby(all_lq, :tag))
        s  = sort(g[g.dose_rate_Gys .< 1e2, :], :dose_rate_Gys)
        ac = g[g.dose_rate_Gys .>= 1e2, :]
        scatter!(pa, s.dose_rate_Gys, s.alpha_free; yerror = s.alpha_free_se, color = i,
                 label = g.tag[1])
        scatter!(pb, s.dose_rate_Gys, s.beta_free; yerror = s.beta_free_se, color = i,
                 label = g.tag[1])
        if !isempty(ac)
            hline!(pa, [ac.alpha_free[1]]; color = i, ls = :dash, label = "")
            hline!(pb, [ac.beta_free[1]];  color = i, ls = :dash, label = "")
        end
    end
    hline!(pa, [0.0]; color = :gray, lw = 0.5, label = "")
    hline!(pb, [0.0]; color = :gray, lw = 0.5, label = "")
    savefig(pa, joinpath(OUTDIR, "alpha_vs_doserate.png"))
    savefig(pb, joinpath(OUTDIR, "beta_free_vs_doserate.png"))
    savefig(plot(pa, pb; layout = (1, 2), size = (1500, 450)),
            joinpath(OUTDIR, "alpha_beta_vs_doserate.png"))

    sf_plots = [plot_sf(all_surv[all_surv.tag .== c.tag, :], c.tag) for c in CONDITIONS]
    savefig(plot(sf_plots...; layout = (1, length(sf_plots)), size = (1800, 500)),
            joinpath(OUTDIR, "sf_all.png"))
end
println("\nResults in ", OUTDIR)
