# ============================================================================
# example/plot_spheroid_multipanel.jl
#
# Multipanel 3-D figure of one irradiated spheroid, from the CSVs written by
# example/spheroid_ions.jl (nothing is rerun). Half spheroid (y ≥ 0), cut face
# towards the viewer. Vector PDF, paper style, panels labelled (a)–(g).
#
#   top row    (a) GSM2 survival probability   (b) cell dose (Gy)
#              (c) repair completion time (h)  (d) time of radiation-induced death (h)
#              — all right after irradiation and fate assignment
#   bottom row (e)–(g) cell-cycle phase at TIMES (doomed cells in black)
#
# Inputs (example/output/spheroid_ions/):
#   cells_t0_<tag>.csv, snap_<tag>_t<h>h.csv, lattice_positions.csv
# Outputs: multipanel_<tag>.pdf and the single panels panel_<tag>_<name>.pdf
#
# Run: julia example/plot_spheroid_multipanel.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))
using CSV, DataFrames, Statistics, Printf, Plots
gr()

const DIR   = joinpath(@__DIR__, "output", "spheroid_ions")
const TAGS  = ["1H_100MeV", "12C_80MeVu"]       # one figure per condition
const TIMES = [6, 24, 72]                         # h, bottom row
const CAMERA = (-30, 20)                          # azimuth 0 = looking at the cut face
const MS     = 2.0                                # marker size
const EVERY  = 1                                  # plot every n-th cell (lighter PDFs)

default(guidefontsize = 10, tickfontsize = 8, legendfontsize = 8, legendtitlefontsize = 9,
        colorbar_titlefontsize = 9, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
        right_margin = 6Plots.mm, top_margin = 2Plots.mm)
paper_style!()

pos = CSV.read(joinpath(DIR, "lattice_positions.csv"), DataFrame)
const POS = Dict(Int(r.index) => (Float64(r.x), Float64(r.y), Float64(r.z)) for r in eachrow(pos))
const LIM = let r = maximum(sqrt(p[1]^2 + p[2]^2 + p[3]^2) for p in values(POS)); r end

"""Positions of the cells with the given lattice indices, keeping the half y ≥ 0, back to front."""
function half(idx)
    P = [POS[Int(i)] for i in idx]
    x = first.(P); y = getindex.(P, 2); z = last.(P)
    keep = findall(y .>= 0)
    keep = keep[sortperm(y[keep]; rev = true)][1:EVERY:end]
    return keep, x, y, z
end

"""Empty 3-D axes in the common view."""
function axes3d(; kw...)
    L = ceil(LIM / 50) * 50
    plot(; xlabel = tex("x (\$\\mu\$m)"), ylabel = tex("y (\$\\mu\$m)"), zlabel = tex("z (\$\\mu\$m)"),
         xlims = (-L, L), ylims = (0, L), zlims = (-L, L), camera = CAMERA, kw...)
end

"""
Cells coloured by a continuous value. Cells with `missing`/non-finite values (e.g.
undamaged cells for the repair time) are drawn in light grey underneath.
"""
function value_panel(df, v, cmap, ctitle; clims = :auto)
    keep, x, y, z = half(df.index)
    ok = [isfinite(v[k]) for k in keep]
    p = axes3d(; legend = false, colorbar = true, colorbar_title = tex(ctitle))
    bg = keep[.!ok]
    isempty(bg) || scatter3d!(p, x[bg], y[bg], z[bg]; color = :gray85, ms = MS, msw = 0, label = "")
    fg = keep[ok]
    isempty(fg) || scatter3d!(p, x[fg], y[fg], z[fg]; marker_z = v[fg], color = cmap, clims = clims,
                              ms = MS, msw = 0, label = "")
    return p
end

const PHASES = ["G0", "G1", "S", "G2", "M"]
function phase_panel(snap, title_txt)
    keep, x, y, z = half(snap.index)
    ph = String.(snap.phase); dm = Bool.(snap.doomed)
    p = axes3d(; legend = :outerright, legendtitle = tex(title_txt))
    for c in vcat(PHASES, "doomed")
        sel = c == "doomed" ? keep[dm[keep]] : keep[(ph[keep] .== c) .& .!dm[keep]]
        isempty(sel) && continue
        scatter3d!(p, x[sel], y[sel], z[sel]; color = c == "doomed" ? :black : PHASE_COLOURS[c],
                   ms = MS, msw = 0, label = c)
    end
    return p
end

# panel letters as left-aligned titles (3-D axes do not support relative annotations)
label!(p, k) = plot!(p; title = "($(Char('a' + k - 1)))", titlelocation = :left, titlefontsize = 11)

for tag in TAGS
    f0 = joinpath(DIR, "cells_t0_$(tag).csv")
    isfile(f0) || (println("missing $f0 — rerun spheroid_ions.jl"); continue)
    d = CSV.read(f0, DataFrame)
    rec = [isfinite(t) && t > 0 ? t : NaN for t in Float64.(d.recover_time)]
    dth = [isfinite(t) ? t : NaN for t in Float64.(d.death_time)]
    sp  = Float64.(d.sp); dose = Float64.(d.dose)
    @printf("%s: %d cells | mean dose %.2f Gy | mean SP %.3f | repaired %d | doomed %d\n",
            tag, nrow(d), mean(dose), mean(sp), count(isfinite, rec), count(isfinite, dth))

    pa = value_panel(d, sp,   :viridis, "Survival probability"; clims = (0, 1))
    pb = value_panel(d, dose, :inferno, "Dose (Gy)")
    pc = value_panel(d, rec,  :plasma,  "Repair time (h)")
    pd = value_panel(d, dth,  :magma,   "Death time (h)")

    bottom = map(TIMES) do h
        fs = joinpath(DIR, "snap_$(tag)_t$(h)h.csv")
        isfile(fs) || return plot(framestyle = :none)
        phase_panel(CSV.read(fs, DataFrame), "$(ion_label(tag)), t = $h h")
    end

    for (p, name) in zip((pa, pb, pc, pd), ("survival", "dose", "repair_time", "death_time"))
        savefig(p, joinpath(DIR, "panel_$(tag)_$(name).pdf"))
    end
    for (p, h) in zip(bottom, TIMES); savefig(p, joinpath(DIR, "panel_$(tag)_t$(h)h.pdf")); end

    panels = vcat([pa, pb, pc, pd], bottom)
    foreach(((k, p),) -> label!(p, k), enumerate(panels))
    l = @layout [grid(1, 4); grid(1, length(TIMES))]
    fig = plot(panels...; layout = l, size = (2400, 1150))
    savefig(fig, joinpath(DIR, "multipanel_$(tag).pdf"))
    println("  saved multipanel_$(tag).pdf")
end
