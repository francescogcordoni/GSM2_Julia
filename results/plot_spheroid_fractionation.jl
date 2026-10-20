# ============================================================================
# example/plot_spheroid_fractionation.jl
#
# Reads the CSVs written by spheroid_fractionation.jl and makes all figures as
# vector PDF, without titles (condition shown in the legend). Nothing is rerun.
#
#   single_<tag>.pdf          one realisation: all, clonogenic and doomed cells
#   phases_<tag>.pdf          one realisation: phase composition of clonogenic cells
#   single_phases_<tag>.pdf   the two above stacked
#   replicates_<tag>.pdf      all realisations (thin) + mean ± SD
#   comparison.pdf            mean ± SD of every condition
#   variability_sd.pdf        SD of the clonogenic count vs time
#   variability_cv.pdf        CV = SD / mean vs time
#   fraction_sf.pdf           surviving fraction of each fraction (ABM vs GSM2)
#   fractionation_all.pdf     replicates + comparison + variability + SF in one figure
#
# Run: julia example/plot_spheroid_fractionation.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using CSV, DataFrames, Statistics, Printf, Plots, StatsPlots
gr()

const DIR  = joinpath(@__DIR__, "output", "spheroid_fractionation")
const FDIR = joinpath(DIR, "figures"); mkpath(FDIR)
const REP_SHOWN = 1                     # realisation used for the single-trajectory plots

hourly = CSV.read(joinpath(DIR, "hourly_all.csv"), DataFrame)
fx     = CSV.read(joinpath(DIR, "fractions.csv"), DataFrame)
hourly.tag = String.(hourly.tag); fx.tag = String.(fx.tag)

# Same statistics as in the simulation script, recomputed from the raw rows
summ = combine(groupby(hourly, [:tag, :t]),
               :clonogenic => mean => :clono_mean, :clonogenic => std => :clono_sd,
               :total => mean => :total_mean, :total => std => :total_sd)
summ.clono_cv = summ.clono_sd ./ max.(summ.clono_mean, 1e-9)
sort!(summ, [:tag, :t])

order  = ["control", "1H_100MeV", "12C_80MeVu"]
tags   = vcat([t for t in order if t in unique(hourly.tag)],
              [t for t in unique(hourly.tag) if !(t in order)])
N0     = first(hourly.total[(hourly.t .== 0.0) .& (hourly.rep .== minimum(hourly.rep))])
FXT    = sort(unique(fx.t))
nrep   = length(unique(hourly.rep))
println("conditions: $(tags) | realisations: $nrep | N0 = $N0 | fractions at $(FXT) h")

# ── Style: large margins so axis labels and legends stay inside the figure ──
default(framestyle = :box, lw = 1.8, guidefontsize = 12, tickfontsize = 10,
        legendfontsize = 9, legendtitlefontsize = 10, dpi = 300, size = (640, 460),
        left_margin = 8Plots.mm, bottom_margin = 7Plots.mm, right_margin = 5Plots.mm,
        top_margin = 3Plots.mm, foreground_color_legend = nothing,
        background_color_legend = RGBA(1, 1, 1, 0.85))
paper_style!()
const COL = Dict("control" => :gray40, "1H_100MeV" => :steelblue, "12C_80MeVu" => :firebrick)
col(t)  = get(COL, t, :black)
nice(t) = ion_label(t)
const PH     = ["G0", "G1", "S", "G2", "M"]
const PH_COL = [:gray70 :steelblue :seagreen :darkorange :firebrick]
fxlines!(p) = vline!(p, FXT; color = :black, ls = :dash, lw = 0.8, label = "")
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  ", name))
xl = "Time (h)"

# ── 1. One realisation per condition ───────────────────────────────────────
single = Dict{String,Any}(); phases = Dict{String,Any}()
for t in tags
    r = sort(hourly[(hourly.tag .== t) .& (hourly.rep .== REP_SHOWN), :], :t)
    p = plot(xlabel = xl, ylabel = "Number of cells", legend = :topleft,
             legendtitle = tex("$(nice(t)), realisation $REP_SHOWN"))
    plot!(p, r.t, r.total; color = :black, label = "all cells")
    plot!(p, r.t, r.clonogenic; color = col(t), label = "clonogenic")
    plot!(p, r.t, r.doomed; color = col(t), ls = :dot, label = "doomed")
    fxlines!(p)
    single[t] = p; save(p, "single_$(t).pdf")

    comp = hcat([r[!, ph] for ph in PH]...) ./ max.(r.clonogenic, 1)
    q = areaplot(r.t, comp; label = permutedims(PH), seriescolor = PH_COL, fillalpha = 0.85,
                 lw = 0, xlabel = xl, ylabel = "Fraction of clonogenic cells", ylims = (0, 1),
                 legend = :outerright, legendtitle = nice(t), size = (760, 460),
                 right_margin = 4Plots.mm)
    fxlines!(q)
    phases[t] = q; save(q, "phases_$(t).pdf")
    save(plot(p, q; layout = (2, 1), size = (760, 860)), "single_phases_$(t).pdf")
end

# ── 2. All realisations + mean ± SD ────────────────────────────────────────
repl = Dict{String,Any}()
ymax = maximum(hourly.clonogenic) / N0 * 1.05
for t in tags
    p = plot(xlabel = xl, ylabel = tex("Clonogenic cells / \$N_0\$"), legend = :topleft,
             legendtitle = nice(t), ylims = (0, ymax))
    for rep in sort(unique(hourly.rep))
        r = sort(hourly[(hourly.tag .== t) .& (hourly.rep .== rep), :], :t)
        plot!(p, r.t, r.clonogenic ./ N0; color = col(t), alpha = 0.3, lw = 0.8,
              label = rep == minimum(hourly.rep) ? "realisations" : "")
    end
    s = summ[summ.tag .== t, :]
    plot!(p, s.t, s.clono_mean ./ N0; ribbon = s.clono_sd ./ N0, fillalpha = 0.25,
          color = col(t), lw = 2.4, label = tex("mean \$\\pm\$ SD"))
    fxlines!(p)
    repl[t] = p; save(p, "replicates_$(t).pdf")
end

pcmp = plot(xlabel = xl, ylabel = tex("Clonogenic cells / \$N_0\$"), legend = :topleft, ylims = (0, ymax))
for t in tags
    s = summ[summ.tag .== t, :]
    plot!(pcmp, s.t, s.clono_mean ./ N0; ribbon = s.clono_sd ./ N0, fillalpha = 0.25,
          color = col(t), lw = 2.2, label = nice(t))
end
fxlines!(pcmp); save(pcmp, "comparison.pdf")

# ── 3. Variability ─────────────────────────────────────────────────────────
psd = plot(xlabel = xl, ylabel = "SD of clonogenic cells", legend = :topleft)
pcv = plot(xlabel = xl, ylabel = "Coefficient of variation (SD / mean)", legend = :topright)
for t in tags
    s = summ[summ.tag .== t, :]
    plot!(psd, s.t, s.clono_sd; color = col(t), label = nice(t))
    plot!(pcv, s.t, s.clono_cv; color = col(t), label = nice(t))
end
fxlines!(psd); fxlines!(pcv)
save(psd, "variability_sd.pdf"); save(pcv, "variability_cv.pdf")

# ── 4. Surviving fraction of each fraction ─────────────────────────────────
irr  = [t for t in tags if any(fx.mean_dose[fx.tag .== t] .> 0)]
nfx  = maximum(fx.fraction)
psf  = plot(xlabel = "Fraction", ylabel = "Surviving fraction", xticks = 1:nfx,
            xlims = (0.6, nfx + 0.4), legend = :bottomright)
for (j, t) in enumerate(irr)
    g  = fx[fx.tag .== t, :]
    dx = 0.1 * (j - (length(irr) + 1) / 2)
    scatter!(psf, g.fraction .+ dx, g.SF; color = col(t), alpha = 0.45, ms = 4, msw = 0, label = "")
    m = combine(groupby(g, :fraction), :SF => mean => :SF, :SF => std => :sd,
                :mean_sp_GSM2 => mean => :sp)
    sort!(m, :fraction)
    plot!(psf, m.fraction .+ dx, m.SF; yerror = m.sd, color = col(t), marker = :diamond,
          ms = 6, lw = 2, label = tex("$(nice(t)), ABM"))
    scatter!(psf, m.fraction .+ dx, m.sp; color = :white, msc = col(t), msw = 1.5,
             marker = :circle, ms = 6, label = tex("$(nice(t)), GSM2 mean"))
end
save(psf, "fraction_sf.pdf")

# ── 5. Combined figure, panels labelled ────────────────────────────────────
panels = vcat([repl[t] for t in tags], [pcmp, psd, pcv, psf])
for (k, p) in enumerate(panels)
    plot!(p; annotations = ((0.03, 0.97), text("($(Char('a' + k - 1)))", 12, :left, :top)))
end
ncol = 4; nrw = cld(length(panels), ncol)
blank = [plot(framestyle = :none) for _ in 1:(ncol * nrw - length(panels))]
save(plot(panels..., blank...; layout = (nrw, ncol), size = (640 * ncol, 480 * nrw),
          left_margin = 10Plots.mm, bottom_margin = 9Plots.mm), "fractionation_all.pdf")
println("Figures in $FDIR")
