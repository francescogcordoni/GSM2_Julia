# ============================================================================
# results/plot_spheroid_schedules.jl
#
# Figures for scripts/spheroid_schedules.jl (data in scripts/output/spheroid_schedules/).
# Vector PDF, paper style, no titles; saved in results/figures/spheroid_schedules/.
#
#   dose_response.pdf          total cells / N0 (top) and clonogenic cells / N0 (bottom) after
#                              0, 2, 4, 6 Gy, for 1H 100 MeV (left) and 12C 80 MeV/u (right)
#   schedules.pdf              the same for 6 Gy delivered as 1 × 6, 3 × 2 Gy / 24 h, 4 × 1.5 Gy / 10.75 h
#   schedule_summary.pdf       clonogenic cells / N0 at the end and at the minimum, per schedule and ion
#   single panels:             dose_<metric>_<ion>.pdf, schedule_<metric>_<ion>.pdf
#   schedule_summary.csv       the numbers of schedule_summary.pdf
#
# Run: julia results/plot_spheroid_schedules.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))
using CSV, DataFrames, Statistics, Printf, Plots, StatsPlots
gr()

const DIR  = joinpath(@__DIR__, "..", "scripts", "output", "spheroid_schedules")
const FDIR = joinpath(@__DIR__, "figures", "spheroid_schedules"); mkpath(FDIR)
isfile(joinpath(DIR, "hourly.csv")) || error("hourly.csv not found in $DIR (run scripts/spheroid_schedules.jl)")

h  = CSV.read(joinpath(DIR, "hourly.csv"), DataFrame)
fx = CSV.read(joinpath(DIR, "fractions.csv"), DataFrame)
for d in (h, fx); d.ion = String.(d.ion); d.schedule = String.(d.schedule); end
h.rel_total = h.total ./ h.N0
h.rel_clono = h.clonogenic ./ h.N0
const IONS = ["1H_100MeV", "12C_80MeVu"]
const T_END = maximum(h.t)
println("data: $DIR | realisations: $(length(unique(h.rep))) | follow-up $(T_END) h")

default(lw = 2.0, guidefontsize = 12, tickfontsize = 10, legendfontsize = 9, legendtitlefontsize = 10,
        size = (640, 470), left_margin = 8Plots.mm, bottom_margin = 7Plots.mm, right_margin = 5Plots.mm,
        top_margin = 3Plots.mm, foreground_color_legend = nothing,
        background_color_legend = RGBA(1, 1, 1, 0.85))
paper_style!()
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  saved ", joinpath(FDIR, name)))
const XT = 0:24:Int(T_END)

# condition styles
const DOSE_SC = [("control", "0 Gy", :gray40), ("1x2Gy", "2 Gy", "#3b6ea8"),
                 ("1x4Gy", "4 Gy", "#e9a23b"), ("1x6Gy", "6 Gy", "#c0392b")]
const FX_SC   = [("control", "control", :gray40), ("1x6Gy", "1 × 6 Gy", "#c0392b"),
                 ("3x2Gy_24h", "3 × 2 Gy, every 24 h", "#3b6ea8"),
                 ("4x1.5Gy_10.75h", "4 × 1.5 Gy, every 10.75 h", "#2a9d8f")]

"""Rows of one condition: the control is shared by both ions."""
cond(ion, sc) = sc == "control" ? h[(h.schedule .== "control"), :] : h[(h.ion .== ion) .& (h.schedule .== sc), :]

"""Mean ± SD over realisations of a column, per time."""
function meansd(d, col)
    g = combine(groupby(d, :t), col => mean => :m, col => std => :s)
    g.s = coalesce.(replace(g.s, NaN => 0.0), 0.0)
    sort!(g, :t)
end

"""One panel: `metric` = :rel_total (linear) or :rel_clono (log)."""
function panel(ion, styles, metric; legend = :topleft, fraction_marks = false)
    logy = metric === :rel_clono
    p = plot(xlabel = "Time after the first fraction (h)", xticks = XT, xlims = (0, T_END),
             ylabel = logy ? tex("Clonogenic cells / \$N_0\$") : tex("Total cells / \$N_0\$"),
             yscale = logy ? :log10 : :identity, legend = legend, legendtitle = ion_label(ion))
    for (sc, lab, c) in styles
        d = cond(ion, sc); isempty(d) && continue
        g = meansd(d, metric)
        y = logy ? max.(g.m, 1e-3) : g.m
        lo = logy ? max.(g.m .- g.s, 1e-3) : g.m .- g.s
        hi = g.m .+ g.s
        plot!(p, g.t, y; ribbon = (y .- lo, hi .- y), fillalpha = 0.18, color = c, label = lab)
        if fraction_marks && sc != "control"
            tf = sort(unique(fx.t[(fx.ion .== ion) .& (fx.schedule .== sc)]))
            vline!(p, tf; color = c, ls = :dash, lw = 0.9, alpha = 0.7, label = "")
        end
    end
    logy || hline!(p, [1.0]; color = :gray, lw = 0.6, label = "")
    return p
end

# ── 1. dose response ────────────────────────────────────────────────────────
dpan = Any[]
for (k, metric) in enumerate((:rel_total, :rel_clono)), ion in IONS
    p = panel(ion, DOSE_SC, metric; legend = metric === :rel_total ? :topleft : :bottomright)
    save(p, "dose_$(metric === :rel_total ? "total" : "clonogenic")_$(ion).pdf")
    push!(dpan, deepcopy(p))
end
foreach(((k, p),) -> plot!(p; annotations = ((0.02, 0.97), text("($(Char('a' + k - 1)))", 12, :left, :top))),
        enumerate(dpan))
save(plot(dpan...; layout = (2, 2), size = (1300, 960)), "dose_response.pdf")

# ── 2. schedules with the same total dose ───────────────────────────────────
span = Any[]
for metric in (:rel_total, :rel_clono), ion in IONS
    p = panel(ion, FX_SC, metric; legend = metric === :rel_total ? :topleft : :bottomright,
              fraction_marks = true)
    save(p, "schedule_$(metric === :rel_total ? "total" : "clonogenic")_$(ion).pdf")
    push!(span, deepcopy(p))
end
foreach(((k, p),) -> plot!(p; annotations = ((0.02, 0.97), text("($(Char('a' + k - 1)))", 12, :left, :top))),
        enumerate(span))
save(plot(span...; layout = (2, 2), size = (1300, 960)), "schedules.pdf")

# ── 3. summary of the schedules ─────────────────────────────────────────────
summ = DataFrame(ion = String[], schedule = String[], label = String[], end_mean = Float64[], end_sd = Float64[],
                 min_mean = Float64[], min_sd = Float64[], t_min = Float64[], t_regrow = Float64[])
for ion in IONS, (sc, lab, _) in FX_SC
    sc == "control" && continue
    d = cond(ion, sc)
    per = combine(groupby(d, :rep)) do g
        g = sort(g, :t)
        imin = argmin(g.rel_clono)
        after = findfirst(i -> i > imin && g.rel_clono[i] >= 1.0, eachindex(g.t))
        (end_ = g.rel_clono[end], min_ = g.rel_clono[imin], tmin = g.t[imin],
         treg = after === nothing ? NaN : g.t[after])
    end
    push!(summ, (ion, sc, lab, mean(per.end_), std(per.end_), mean(per.min_), std(per.min_),
                 mean(per.tmin), (r = filter(!isnan, per.treg); isempty(r) ? NaN : mean(r))))
end
CSV.write(joinpath(FDIR, "schedule_summary.csv"), summ)
show(stdout, summ; allcols = true); println()

labs = [l for (sc, l, _) in FX_SC if sc != "control"]
cols = permutedims([c for (sc, _, c) in FX_SC if sc != "control"])
M(col) = permutedims(reduce(hcat, [[summ[(summ.ion .== ion) .& (summ.label .== l), col][1] for l in labs] for ion in IONS]))
xt = (1:length(IONS), [ion_label(i) for i in IONS])          # LaTeX ion labels as ticks
pa = groupedbar(M(:min_mean); yerror = M(:min_sd), yscale = :log10, color = cols, label = permutedims(labs),
                xticks = xt, ylabel = tex("Minimum clonogenic cells / \$N_0\$"), legend = :bottomleft,
                bar_width = 0.8)
pb = groupedbar(M(:end_mean); yerror = M(:end_sd), color = cols, label = "", xticks = xt,
                ylabel = tex("Clonogenic cells / \$N_0\$ at $(Int(T_END)) h"), bar_width = 0.8)
save(pa, "schedule_minimum.pdf"); save(pb, "schedule_end.pdf")
save(plot(pa, pb; layout = (1, 2), size = (1300, 480)), "schedule_summary.pdf")
println("Figures in $FDIR")
