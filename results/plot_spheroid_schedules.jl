# ============================================================================
# results/plot_spheroid_schedules.jl
#
# Figures for scripts/spheroid_schedules.jl (data in scripts/output/spheroid_schedules/<study>/).
# Vector PDF, paper style, no titles; saved in results/figures/spheroid_schedules/.
#
# Study :dose (single shot of 0, 2, 4, 6 Gy):
#   dose_phases.pdf            grid ions × doses: total cells and cells in G0, G1, S, G2, M vs time
#   dose_phases_<ion>_<d>.pdf  the single panels
#   dose_total.pdf             total and clonogenic cells / N0 vs time, all doses, one panel per ion
# Study :fractionation (6 Gy as 1 × 6, 3 × 2 Gy / 24 h, 2 × 3 Gy / 48 h):
#   fx_total.pdf               total and clonogenic cells / N0 vs time, one panel per ion
#   fx_phases.pdf              grid ions × schedules: total cells and cell-cycle phases
#   fx_phases_<ion>_<sched>.pdf the single panels
#   fx_summary.pdf / .csv      clonogenic cells / N0: minimum and at the end, per schedule and ion
#
# Run: julia results/plot_spheroid_schedules.jl   (plots whatever study has data)
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))
using CSV, DataFrames, Statistics, Printf, Plots, StatsPlots
gr()

const BASE = joinpath(@__DIR__, "..", "scripts", "output", "spheroid_schedules")
const FDIR = joinpath(@__DIR__, "figures", "spheroid_schedules"); mkpath(FDIR)
const IONS = ["1H_100MeV", "12C_80MeVu"]
const PHASES = ["G0", "G1", "S", "G2", "M"]

default(lw = 2.0, guidefontsize = 12, tickfontsize = 10, legendfontsize = 9, legendtitlefontsize = 10,
        size = (640, 470), left_margin = 8Plots.mm, bottom_margin = 7Plots.mm, right_margin = 5Plots.mm,
        top_margin = 3Plots.mm, foreground_color_legend = nothing,
        background_color_legend = RGBA(1, 1, 1, 0.85))
paper_style!()
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  saved ", joinpath(FDIR, name)))
letter!(ps) = foreach(((k, p),) -> plot!(p; annotations = ((0.02, 0.97),
                      text("($(Char('a' + k - 1)))", 12, :left, :top))), enumerate(ps))

function load(study)
    d = joinpath(BASE, study)
    isfile(joinpath(d, "hourly.csv")) || return nothing
    h  = CSV.read(joinpath(d, "hourly.csv"), DataFrame)
    fx = CSV.read(joinpath(d, "fractions.csv"), DataFrame)
    for x in (h, fx); x.ion = String.(x.ion); x.schedule = String.(x.schedule); end
    h.rel_total = h.total ./ h.N0; h.rel_clono = h.clonogenic ./ h.N0
    println("$study: $(length(unique(h.rep))) realisations, follow-up $(maximum(h.t)) h")
    return h, fx
end

cond(h, ion, sc) = sc == "control" ? h[h.schedule .== "control", :] : h[(h.ion .== ion) .& (h.schedule .== sc), :]
function meansd(d, col)
    g = combine(groupby(d, :t), col => mean => :m, col => std => :s)
    g.s = coalesce.(replace(g.s, NaN => 0.0), 0.0); sort!(g, :t)
end
fxtimes(fx, ion, sc) = sort(unique(fx.t[(fx.ion .== ion) .& (fx.schedule .== sc)]))

"""Total cells (black) and cells in each phase (all cells, doomed included), mean over realisations."""
function phase_panel(h, fx, ion, sc, legtitle; legend = :topleft)
    d = cond(h, ion, sc)
    T = maximum(h.t)
    p = plot(xlabel = "Time after irradiation (h)", ylabel = "Number of cells", xticks = 0:24:Int(T),
             xlims = (0, T), legend = legend, legendtitle = tex(legtitle))
    g = meansd(d, :total)
    plot!(p, g.t, g.m; ribbon = g.s, fillalpha = 0.15, color = :black, lw = 2.4, label = "total")
    for ph in PHASES
        g = meansd(d, Symbol(ph, "_all"))
        plot!(p, g.t, g.m; color = PHASE_COLOURS[ph], lw = 1.6, ls = ph == "G0" ? :dash : :solid, label = ph)
    end
    sc == "control" || vline!(p, fxtimes(fx, ion, sc); color = :black, ls = :dot, lw = 0.8, label = "")
    return p
end

"""Total (solid) and clonogenic (dashed) cells / N0 for a set of conditions, one ion."""
function total_panel(h, fx, ion, styles; marks = false)
    T = maximum(h.t)
    p = plot(xlabel = "Time after the first fraction (h)", ylabel = tex("Cells / \$N_0\$"), yscale = :log10,
             xticks = 0:24:Int(T), xlims = (0, T), legend = :bottomright, legendtitle = ion_label(ion))
    for (sc, lab, c) in styles
        d = cond(h, ion, sc); isempty(d) && continue
        for (col, ls) in ((:rel_total, :solid), (:rel_clono, :dash))
            g = meansd(d, col); y = max.(g.m, 1e-3); lo = max.(g.m .- g.s, 1e-3)
            plot!(p, g.t, y; ribbon = (y .- lo, g.m .+ g.s .- y), fillalpha = 0.12, color = c, ls = ls,
                  label = col === :rel_total ? lab : "")
        end
        marks && sc != "control" && vline!(p, fxtimes(fx, ion, sc); color = c, ls = :dot, lw = 0.9, label = "")
    end
    plot!(p, [NaN], [NaN]; color = :black, ls = :solid, label = "total cells")
    plot!(p, [NaN], [NaN]; color = :black, ls = :dash, label = "clonogenic cells")
    return p
end

# ── study 1: single shot ────────────────────────────────────────────────────
dat = load("dose")
if dat !== nothing
    h, fx = dat
    DS = [("control", "0 Gy", :gray40), ("1x2Gy", "2 Gy", "#3b6ea8"), ("1x4Gy", "4 Gy", "#e9a23b"), ("1x6Gy", "6 Gy", "#c0392b")]
    pgrid = Any[]
    for ion in IONS, (sc, lab, _) in DS
        p = phase_panel(h, fx, ion, sc, "$(ion_label(ion)), $lab")
        save(p, "dose_phases_$(ion)_$(replace(lab, " " => "")).pdf"); push!(pgrid, deepcopy(p))
    end
    letter!(pgrid)
    save(plot(pgrid...; layout = (2, 4), size = (2500, 980), link = :y), "dose_phases.pdf")
    tp = [total_panel(h, fx, ion, DS) for ion in IONS]
    foreach(((ion, p),) -> save(p, "dose_total_$(ion).pdf"), zip(IONS, tp))
    tp2 = deepcopy.(tp); letter!(tp2)
    save(plot(tp2...; layout = (1, 2), size = (1300, 480)), "dose_total.pdf")
end

# ── study 2: fractionation ──────────────────────────────────────────────────
dat = load("fractionation")
if dat !== nothing
    h, fx = dat
    FS = [("control", "control", :gray40), ("1x6Gy", "1 × 6 Gy", "#c0392b"),
          ("3x2Gy_24h", "3 × 2 Gy, every 24 h", "#3b6ea8"), ("2x3Gy_48h", "2 × 3 Gy, 48 h apart", "#2a9d8f")]
    tp = [total_panel(h, fx, ion, FS; marks = true) for ion in IONS]
    foreach(((ion, p),) -> save(p, "fx_total_$(ion).pdf"), zip(IONS, tp))
    tp2 = deepcopy.(tp); letter!(tp2)
    save(plot(tp2...; layout = (1, 2), size = (1300, 480)), "fx_total.pdf")

    pgrid = Any[]
    for ion in IONS, (sc, lab, _) in FS[2:end]
        p = phase_panel(h, fx, ion, sc, "$(ion_label(ion)), $lab")
        save(p, "fx_phases_$(ion)_$(sc).pdf"); push!(pgrid, deepcopy(p))
    end
    letter!(pgrid)
    save(plot(pgrid...; layout = (2, 3), size = (1900, 980), link = :y), "fx_phases.pdf")

    summ = DataFrame(ion = String[], schedule = String[], label = String[], min_mean = Float64[],
                     min_sd = Float64[], end_mean = Float64[], end_sd = Float64[])
    for ion in IONS, (sc, lab, _) in FS[2:end]
        per = combine(groupby(cond(h, ion, sc), :rep)) do g
            g = sort(g, :t); (mn = minimum(g.rel_clono), en = g.rel_clono[end])
        end
        push!(summ, (ion, sc, lab, mean(per.mn), std(per.mn), mean(per.en), std(per.en)))
    end
    CSV.write(joinpath(FDIR, "fx_summary.csv"), summ); show(stdout, summ; allcols = true); println()
    labs = [l for (_, l, _) in FS[2:end]]; cols = permutedims([c for (_, _, c) in FS[2:end]])
    M(col) = permutedims(reduce(hcat, [[summ[(summ.ion .== ion) .& (summ.label .== l), col][1] for l in labs] for ion in IONS]))
    xt = (1:length(IONS), [ion_label(i) for i in IONS])
    pa = groupedbar(M(:min_mean); yerror = M(:min_sd), yscale = :log10, color = cols, label = permutedims(labs),
                    xticks = xt, ylabel = tex("Minimum clonogenic cells / \$N_0\$"), legend = :bottomleft, bar_width = 0.8)
    pb = groupedbar(M(:end_mean); yerror = M(:end_sd), color = cols, label = "", xticks = xt,
                    ylabel = tex("Clonogenic cells / \$N_0\$ at $(Int(maximum(h.t))) h"), bar_width = 0.8)
    save(pa, "fx_minimum.pdf"); save(pb, "fx_end.pdf")
    save(plot(pa, pb; layout = (1, 2), size = (1300, 480)), "fx_summary.pdf")
end
println("Figures in $FDIR")
