# ============================================================================
# plot_benchmark.jl
#
# Six-panel timing figure from scripts/output/benchmark/timings_summary.csv
# (written by benchmark_computation.jl). Nothing is rerun. Vector PDF, no titles.
#
#   top row    : total, dose and ABM time vs number of cells
#   bottom row : the same vs spheroid radius
#   symbols    : median over repeats; error bars: min–max; dashed: power-law fit
#                (fitted on the standard sizes, extended to the stress-test sizes);
#                open symbols: stress-test sizes
#
# Also writes scaling_exponents.csv (time ∝ cells^k and ∝ radius^m per ion, stage).
#
# Run: julia results/plot_benchmark.jl   (data in scripts/output/benchmark/)
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using CSV, DataFrames, Statistics, Printf, Plots
gr()

const DIR     = joinpath(@__DIR__, "..", "scripts", "output", "benchmark")   # data (scripts/output)
const FIG_DIR = joinpath(@__DIR__, "figures", "benchmark"); mkpath(FIG_DIR)   # figures (results/figures)
summ = CSV.read(joinpath(DIR, "timings_summary.csv"), DataFrame)
summ.ion = String.(summ.ion)
hasproperty(summ, :stress) || (summ.stress = falses(nrow(summ)))   # older timing files

const STAGES = [(:t_total, "Total time (s)"), (:t_dose, "Dose time (s)"), (:t_abm, "ABM time (s)")]
const XS     = [(:n_cells, "Number of cells"), (:radius_um, tex("Spheroid radius (\$\\mu\$m)"))]
const ORDER  = ["1H_100MeV", "1H_30MeV", "12C_80MeVu", "12C_10MeVu"]
ions = vcat([i for i in ORDER if i in unique(summ.ion)], [i for i in unique(summ.ion) if !(i in ORDER)])
const COL = Dict("1H_100MeV" => :steelblue, "1H_30MeV" => :deepskyblue,
                 "12C_80MeVu" => :firebrick, "12C_10MeVu" => :darkorange)
const MRK = Dict("1H_100MeV" => :circle, "1H_30MeV" => :utriangle,
                 "12C_80MeVu" => :diamond, "12C_10MeVu" => :rect)
col(i) = get(COL, i, :black); mrk(i) = get(MRK, i, :circle)
nice(t) = ion_label(t)

default(framestyle = :box, lw = 1.6, guidefontsize = 12, tickfontsize = 10,
        legendfontsize = 9, dpi = 300, left_margin = 9Plots.mm, bottom_margin = 8Plots.mm,
        right_margin = 4Plots.mm, top_margin = 3Plots.mm,
        background_color_legend = RGBA(1, 1, 1, 0.85), foreground_color_legend = nothing)
paper_style!()

"""Least-squares slope and intercept of log(y) vs log(x)."""
function powerfit(x, y)
    lx, ly = log.(x), log.(max.(y, 1e-9))
    k = cov(lx, ly) / var(lx)
    return k, mean(ly) - k * mean(lx)
end

expo = DataFrame(ion = String[], stage = String[], x = String[], exponent = Float64[])
panels = []
for (xc, xl) in XS, (st, yl) in STAGES
    p = plot(xscale = :log10, yscale = :log10, xlabel = xl, ylabel = yl, legend = :topleft)
    for i in ions
        g = sort(summ[summ.ion .== i, :], xc)
        x = Float64.(g[!, xc])
        med = g[!, Symbol(st, :_med)]
        lo  = med .- g[!, Symbol(st, :_min)]; hi = g[!, Symbol(st, :_max)] .- med
        scatter!(p, x, med; yerror = (lo, hi), color = col(i), marker = mrk(i), ms = 5,
                 msw = 0.5, label = nice(i))
        st_ = Bool.(g.stress)          # stress-test sizes: open symbols
        any(st_) && scatter!(p, x[st_], med[st_]; marker = mrk(i), ms = 7, mc = :white,
                             msc = col(i), msw = 1.5, label = "")
        if count(.!st_) >= 3
            k, c = powerfit(x[.!st_], med[.!st_])      # exponent from the standard sizes
            xf = exp.(range(log(minimum(x)), log(maximum(x)); length = 50))
            plot!(p, xf, exp.(c .+ k .* log.(xf)); color = col(i), ls = :dash, lw = 1.2, label = "")
            push!(expo, (i, String(st), String(xc), k))
        end
    end
    push!(panels, p)
end
for (k, p) in enumerate(panels)
    plot!(p; annotations = ((0.97, 0.06), text("($(Char('a' + k - 1)))", 12, :right, :bottom)))
end

fig = plot(panels...; layout = (2, 3), size = (1800, 1000))
savefig(fig, joinpath(FIG_DIR, "benchmark_6panel.pdf"))
CSV.write(joinpath(FIG_DIR, "scaling_exponents.csv"), expo)

println("Scaling exponents (time ∝ x^k):")
for r in eachrow(sort(expo, [:x, :stage, :ion]))
    @printf("  %-10s %-8s %-12s k = %.2f\n", r.x, r.stage, r.ion, r.exponent)
end
println("Saved ", joinpath(FIG_DIR, "benchmark_6panel.pdf"))
