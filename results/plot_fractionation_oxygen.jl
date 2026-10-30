# ============================================================================
# results/plot_fractionation_oxygen.jl
#
# Figures for scripts/spheroid_fractionation_oxygen.jl
# (data in scripts/output/spheroid_fractionation_oxygen/). Vector PDF, paper style,
# no titles; saved in results/figures/fractionation_oxygen/.
#
#   clonogenic.pdf        clonogenic cells / N0 vs time: oxygen profile (solid) vs uniform (dashed),
#                         1H (left) and 12C (right), mean ± SD; dashed verticals: fractions
#   fraction_sf.pdf       surviving fraction of each fraction, profile vs uniform, both ions
#   hypoxia.pdf           hypoxic (< 1% O2) and anoxic fraction of the cells at each fraction
#   oxygen_radius.pdf     O2 vs distance from the centre at fractions 1–3 (realisation 1)
#   survival_radius.pdf   mean survival probability vs distance from the centre at fraction 1,
#                         profile vs uniform, both ions
#   oxygen_all.pdf        the panels above in one figure, labelled (a)–(f)
#
# Run: julia results/plot_fractionation_oxygen.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))
using CSV, DataFrames, Statistics, Printf, Plots, StatsPlots
gr()

const DIR  = joinpath(@__DIR__, "..", "scripts", "output", "spheroid_fractionation_oxygen")
const FDIR = joinpath(@__DIR__, "figures", "fractionation_oxygen"); mkpath(FDIR)
isfile(joinpath(DIR, "hourly.csv")) || error("hourly.csv not found in $DIR")

h  = CSV.read(joinpath(DIR, "hourly.csv"), DataFrame)
fx = CSV.read(joinpath(DIR, "fractions.csv"), DataFrame)
for d in (h, fx); d.ion = String.(d.ion); d.oxygen = String.(d.oxygen); end
h.rel = h.clonogenic ./ h.N0
const IONS = ["1H_100MeV", "12C_80MeVu"]
const FXT  = sort(unique(fx.t))
const COL  = Dict("1H_100MeV" => "#3b6ea8", "12C_80MeVu" => "#c0392b", "control" => :gray40)
const LS   = Dict("diffusion" => :solid, "uniform" => :dash)
const OXLAB = Dict("diffusion" => "oxygen profile", "uniform" => "uniform oxygen")

default(lw = 2.0, guidefontsize = 12, tickfontsize = 10, legendfontsize = 9, legendtitlefontsize = 10,
        size = (640, 470), left_margin = 8Plots.mm, bottom_margin = 7Plots.mm, right_margin = 5Plots.mm,
        top_margin = 3Plots.mm, foreground_color_legend = nothing,
        background_color_legend = RGBA(1, 1, 1, 0.85))
paper_style!()
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  saved ", joinpath(FDIR, name)))
xl = "Time after the first fraction (h)"

function meansd(d)
    g = combine(groupby(d, :t), :rel => mean => :m, :rel => std => :s)
    g.s = coalesce.(replace(g.s, NaN => 0.0), 0.0); sort!(g, :t)
end

# ── 1. clonogenic population ────────────────────────────────────────────────
pc = map(IONS) do ion
    p = plot(xlabel = xl, ylabel = tex("Clonogenic cells / \$N_0\$"), yscale = :log10,
             legend = :bottomright, legendtitle = ion_label(ion), xticks = 0:24:Int(maximum(h.t)))
    g = meansd(h[h.ion .== "control", :])
    plot!(p, g.t, g.m; ribbon = g.s, fillalpha = 0.15, color = COL["control"], label = "control")
    for ox in ("diffusion", "uniform")
        g = meansd(h[(h.ion .== ion) .& (h.oxygen .== ox), :])
        y = max.(g.m, 1e-3); lo = max.(g.m .- g.s, 1e-3)
        plot!(p, g.t, y; ribbon = (y .- lo, g.m .+ g.s .- y), fillalpha = 0.15, color = COL[ion],
              ls = LS[ox], label = OXLAB[ox])
    end
    vline!(p, FXT; color = :black, ls = :dot, lw = 0.8, label = "")
end
for (ion, p) in zip(IONS, pc); save(p, "clonogenic_$(ion).pdf"); end
save(plot(deepcopy.(pc)...; layout = (1, 2), size = (1300, 480)), "clonogenic.pdf")

# ── 2. surviving fraction per fraction ──────────────────────────────────────
m = combine(groupby(fx, [:ion, :oxygen, :fraction]), :SF => mean => :SF, :SF => std => :sd,
            :frac_hypoxic => mean => :hyp, :frac_anoxic => mean => :anox, :R_spheroid => mean => :R)
sort!(m, [:ion, :oxygen, :fraction])
nfx = maximum(m.fraction)
psf = plot(xlabel = "Fraction", ylabel = "Surviving fraction", xticks = 1:nfx, xlims = (0.6, nfx + 0.4),
           legend = (0.5, 0.62))
for (j, ion) in enumerate(IONS), (k, ox) in enumerate(("diffusion", "uniform"))
    d = m[(m.ion .== ion) .& (m.oxygen .== ox), :]
    dx = 0.06 * (2(j - 1) + (k - 1) - 1.5)
    plot!(psf, d.fraction .+ dx, d.SF; yerror = d.sd, color = COL[ion], ls = LS[ox],
          marker = ox == "diffusion" ? :circle : :diamond, ms = 6,
          label = tex("$(ion_label(ion)), $(OXLAB[ox])"))
end
save(psf, "fraction_sf.pdf")

# ── 3. hypoxia at each fraction (oxygen profile only) ───────────────────────
ph = plot(xlabel = "Fraction", ylabel = "Fraction of cells", xticks = 1:nfx, xlims = (0.6, nfx + 0.4),
          ylims = (0, 1), legend = :topright)
for ion in IONS
    d = m[(m.ion .== ion) .& (m.oxygen .== "diffusion"), :]
    plot!(ph, d.fraction, d.hyp; color = COL[ion], marker = :circle, label = tex("$(ion_label(ion)), O\$_2\$ < 1\\%"))
    plot!(ph, d.fraction, d.anox; color = COL[ion], ls = :dash, marker = :diamond, label = tex("$(ion_label(ion)), O\$_2\$ < 0.2\\%"))
end
save(ph, "hypoxia.pdf")

# ── 4. radial profiles (realisation 1) ──────────────────────────────────────
cellfile(ion, ox, f) = joinpath(DIR, "cells_$(ion)_$(ox)_f$(f).csv")
"""Mean of `col` in radial bins of width `w` µm."""
function radial(d, col; w = 30.0)
    r = sqrt.(d.x .^ 2 .+ d.y .^ 2 .+ d.z .^ 2)
    b = floor.(Int, r ./ w)
    g = combine(groupby(DataFrame(b = b, v = d[!, col]), :b), :v => mean => :v, nrow => :n)
    g = g[g.n .>= 5, :]; sort!(g, :b)
    return (g.b .+ 0.5) .* w, g.v
end
po = plot(xlabel = tex("Distance from the centre (\$\\mu\$m)"), ylabel = tex("O\$_2\$ (\\%)"),
          legend = :topleft, legendtitle = tex("$(ion_label("12C_80MeVu")), oxygen profile"))
fcols = ["#1f3b5a", "#3b6ea8", "#8fb3d9"]
for f in 1:nfx
    isfile(cellfile("12C_80MeVu", "diffusion", f)) || continue
    d = CSV.read(cellfile("12C_80MeVu", "diffusion", f), DataFrame)
    r, v = radial(d, :O2)
    plot!(po, r, v; color = fcols[f], marker = :circle, ms = 3, label = "fraction $f")
end
hline!(po, [1.0]; color = :gray, ls = :dot, lw = 1, label = tex("1\\% O\$_2\$"))
save(po, "oxygen_radius.pdf")

ps = plot(xlabel = tex("Distance from the centre (\$\\mu\$m)"), ylabel = "Mean survival probability",
          ylims = (0, 1), legend = :bottomleft, legendtitle = "fraction 1")
for ion in IONS, ox in ("diffusion", "uniform")
    isfile(cellfile(ion, ox, 1)) || continue
    d = CSV.read(cellfile(ion, ox, 1), DataFrame)
    r, v = radial(d, :sp)
    plot!(ps, r, v; color = COL[ion], ls = LS[ox], label = tex("$(ion_label(ion)), $(OXLAB[ox])"))
end
save(ps, "survival_radius.pdf")

# ── 5. combined figure ──────────────────────────────────────────────────────
panels = deepcopy.(vcat(pc, [psf, ph, po, ps]))
foreach(((k, p),) -> plot!(p; annotations = ((0.02, 0.97), text("($(Char('a' + k - 1)))", 12, :left, :top))),
        enumerate(panels))
save(plot(panels...; layout = (3, 2), size = (1300, 1420)), "oxygen_all.pdf")

println("\nPer-fraction summary:"); show(stdout, m; allcols = true); println()
