# ============================================================================
# results/plot_spheroid_multipanel.jl
#
# Multipanel figure of one irradiated spheroid, from the CSVs written by
# scripts/spheroid_ions.jl (scripts/output/spheroid_ions/). Nothing is rerun.
#
#   top row    (a) survival probability, log scale (3-D, half spheroid)
#              (b) cell dose (3-D, half spheroid)
#              (c) distribution of the repair-completion time, per phase at irradiation
#              (d) distribution of the time of radiation-induced death, per phase
#   bottom row (e)–(g) cell-cycle phase at TIMES (3-D, doomed cells in black)
#
# Folder layout:  repo/src/plot_style.jl   repo/scripts/output/spheroid_ions/   repo/results/ (this file)
# Outputs: results/figures/spheroid_multipanel/multipanel_<tag>.pdf and single panels
#
# Run: julia results/plot_spheroid_multipanel.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))
using CSV, DataFrames, Statistics, Printf, Plots
gr()

const DATA_DIR = joinpath(@__DIR__, "..", "scripts", "output", "spheroid_ions")
const FIG_DIR  = joinpath(@__DIR__, "figures", "spheroid_multipanel"); mkpath(FIG_DIR)
const TAGS     = ["1H_100MeV", "12C_80MeVu"]     # one figure per condition
const TIMES    = [6, 24, 72]                       # h, bottom row
const CAMERA   = (-30, 20)                         # azimuth 0 = looking at the cut face
const MS       = 2.6                               # marker size (3-D panels)
const EVERY    = 1                                 # plot every n-th cell (lighter PDFs)
const SP_FLOOR = 1e-3                              # survival probabilities below this (incl. 0) shown at the floor
const R_CELL   = 15.0                              # µm, for the axis margin

default(guidefontsize = 10, tickfontsize = 8, legendfontsize = 8, legendtitlefontsize = 9,
        colorbar_titlefontsize = 9, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
        right_margin = 6Plots.mm, top_margin = 2Plots.mm)
paper_style!()

isdir(DATA_DIR) || error("data folder not found: $DATA_DIR (run scripts/spheroid_ions.jl first)")
pos = CSV.read(joinpath(DATA_DIR, "lattice_positions.csv"), DataFrame)
const POS = Dict(Int(r.index) => (Float64(r.x), Float64(r.y), Float64(r.z)) for r in eachrow(pos))

const PHASES = ["G0", "G1", "S", "G2", "M"]

"""Half-width of the plotted cube: the largest cell distance in the given index sets (+ one cell)."""
function extent(index_sets)
    r = 0.0
    for idx in index_sets, i in idx
        p = POS[Int(i)]; r = max(r, sqrt(p[1]^2 + p[2]^2 + p[3]^2))
    end
    return ceil((r + R_CELL) / 25) * 25
end

"""Positions of the cells, keeping the half y ≥ 0, ordered back to front."""
function half(idx)
    P = [POS[Int(i)] for i in idx]
    x = first.(P); y = getindex.(P, 2); z = last.(P)
    keep = findall(y .>= 0)
    keep = keep[sortperm(y[keep]; rev = true)][1:EVERY:end]
    return keep, x, y, z
end

axes3d(L; kw...) = plot(; xlabel = tex("x (\$\\mu\$m)"), ylabel = tex("y (\$\\mu\$m)"),
                        zlabel = tex("z (\$\\mu\$m)"), xlims = (-L, L), ylims = (0, L), zlims = (-L, L),
                        camera = CAMERA, kw...)

"""3-D panel coloured by a continuous value."""
function value_panel(d, v, L, cmap, ctitle; clims = :auto, cticks = :auto)
    keep, x, y, z = half(d.index)
    p = axes3d(L; legend = false, colorbar = true, colorbar_title = tex(ctitle), colorbar_ticks = cticks)
    scatter3d!(p, x[keep], y[keep], z[keep]; marker_z = v[keep], color = cmap, clims = clims,
               ms = MS, msw = 0, label = "")
    return p
end

"""3-D panel coloured by cell-cycle phase, doomed cells in black."""
function phase_panel(snap, L, title_txt)
    keep, x, y, z = half(snap.index)
    ph = String.(snap.phase); dm = Bool.(snap.doomed)
    p = axes3d(L; legend = :outerright, legendtitle = tex(title_txt))
    for c in vcat(PHASES, "doomed")
        sel = c == "doomed" ? keep[dm[keep]] : keep[(ph[keep] .== c) .& .!dm[keep]]
        isempty(sel) && continue
        scatter3d!(p, x[sel], y[sel], z[sel]; color = c == "doomed" ? :black : PHASE_COLOURS[c],
                   ms = MS, msw = 0, label = c)
    end
    return p
end

"""
Smooth probability density of non-negative times (Gaussian kernel, Silverman
bandwidth, reflected at 0 so that no probability is lost below t = 0).
"""
function smooth_density(t::AbstractVector{<:Real}; n = 400, tmax = nothing)
    t = Float64.(t); m = length(t)
    s = min(std(t), (quantile(t, 0.75) - quantile(t, 0.25)) / 1.34)
    s = s > 0 ? s : std(t) + eps()
    h = 0.9 * s * m^(-1 / 5)
    g = collect(range(0, tmax === nothing ? maximum(t) + 3h : tmax; length = n))
    f = [sum(exp(-0.5 * ((u - ti) / h)^2) + exp(-0.5 * ((u + ti) / h)^2) for ti in t) for u in g]
    return g, f ./ (m * h * sqrt(2π))
end

"""
Distribution of a time over all cells (black) and split by phase at irradiation;
the phase curves are weighted by their share of cells, so they add up to the black one.
"""
function time_panel(t, phase, xlab)
    ok = isfinite.(t) .& (t .>= 0)
    p = plot(; xlabel = xlab, ylabel = tex("Probability density (h\$^{-1}\$)"), legend = :topright)
    count(ok) < 20 && return p
    tmax = quantile(t[ok], 0.995)
    g, f = smooth_density(t[ok]; tmax = tmax)
    plot!(p, g, f; color = :black, lw = 2.2, label = "all")
    for c in PHASES
        sel = ok .& (phase .== c)
        count(sel) >= 20 || continue
        g, f = smooth_density(t[sel]; tmax = tmax)
        plot!(p, g, (count(sel) / count(ok)) .* f; color = PHASE_COLOURS[c], lw = 1.5, label = c)
    end
    return p
end

letter!(p, k) = plot!(p; title = "($(Char('a' + k - 1)))", titlelocation = :left, titlefontsize = 11)

for tag in TAGS
    f0 = joinpath(DATA_DIR, "cells_t0_$(tag).csv")
    isfile(f0) || (println("missing $f0 — rerun scripts/spheroid_ions.jl"); continue)
    d = CSV.read(f0, DataFrame)
    snaps = Dict(h => CSV.read(joinpath(DATA_DIR, "snap_$(tag)_t$(h)h.csv"), DataFrame)
                 for h in TIMES if isfile(joinpath(DATA_DIR, "snap_$(tag)_t$(h)h.csv")))
    L = extent(vcat([d.index], [s.index for s in values(snaps)]))

    phase = String.(d.phase)
    sp    = Float64.(d.sp)
    logsp = log10.(max.(sp, SP_FLOOR))
    dose  = Float64.(d.dose)
    rec   = [isfinite(t) && t > 0 ? t : NaN for t in Float64.(d.recover_time)]
    dth   = [isfinite(t) ? t : NaN for t in Float64.(d.death_time)]
    @printf("%s: %d cells | half-width %.0f µm | mean dose %.2f Gy | mean SP %.3f | SP < %.0e: %d | repaired %d | doomed %d\n",
            tag, nrow(d), L, mean(dose), mean(sp), SP_FLOOR, count(<(SP_FLOOR), sp),
            count(isfinite, rec), count(isfinite, dth))

    lo = Int(log10(SP_FLOOR))
    pa = value_panel(d, logsp, L, :viridis, "Survival probability"; clims = (lo, 0),
                     cticks = (collect(lo:0), [tex("10\$^{$k}\$") for k in lo:0]))
    pb = value_panel(d, dose, L, :inferno, "Dose (Gy)")
    pc = time_panel(rec, phase, "Repair time (h)")
    pd = time_panel(dth, phase, "Time of death (h)")

    bottom = [haskey(snaps, h) ? phase_panel(snaps[h], L, "$(ion_label(tag)), t = $h h") :
              plot(framestyle = :none) for h in TIMES]

    for (p, name) in zip((pa, pb, pc, pd), ("survival", "dose", "repair_time", "death_time"))
        savefig(p, joinpath(FIG_DIR, "panel_$(tag)_$(name).pdf"))
    end
    for (p, h) in zip(bottom, TIMES); savefig(p, joinpath(FIG_DIR, "panel_$(tag)_t$(h)h.pdf")); end

    panels = vcat([pa, pb, pc, pd], bottom)
    foreach(((k, p),) -> letter!(p, k), enumerate(panels))
    l = @layout [grid(1, 4); grid(1, length(TIMES))]
    savefig(plot(panels...; layout = l, size = (2400, 1150)), joinpath(FIG_DIR, "multipanel_$(tag).pdf"))
    println("  saved ", joinpath(FIG_DIR, "multipanel_$(tag).pdf"))
end
