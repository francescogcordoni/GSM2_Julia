# ============================================================================
# plot_doserate_relative.jl
#
# All dose-rate figures from the CSVs written by continuous_dr_events.jl
# (no simulation is rerun). Vector PDF, no titles (ion shown in the legend).
#
#   sf_<tag>.pdf            SF vs dose: ABM ± SE, LQ fit (α fixed), Lea–Catcheside
#   beta_fixed_alpha.pdf    β (α fixed to acute) vs dose rate, ABM vs Lea–Catcheside
#   alpha_free.pdf          α from the constrained free LQ fit vs dose rate
#   beta_free.pdf           β from the constrained free LQ fit vs dose rate
#   quadratic_share.pdf     β·D_max / (α + β·D_max) vs dose rate
#   sparing.pdf             SF(Ḋ) / SF(acute) at D_max vs dose rate
#   beta_normalised.pdf     β(Ḋ) / β(acute) (ions with acute β clearly > 0)
#   doserate_all.pdf        all panels in one figure, labelled (a)–(i)
#
# Run: julia scripts/plot_doserate_relative.jl
# ============================================================================
using CSV, DataFrames, Statistics, Printf, Plots
gr()

const DIR  = joinpath(@__DIR__, "..", "data", "continuous_dr_events")
const FDIR = joinpath(DIR, "figures"); mkpath(FDIR)
lq   = CSV.read(joinpath(DIR, "lq_all.csv"), DataFrame)
surv = CSV.read(joinpath(DIR, "survival_all.csv"), DataFrame)
tags = unique(lq.tag)

default(framestyle = :box, grid = false, lw = 1.8,
        guidefontsize = 12, tickfontsize = 10, legendfontsize = 9, legendtitlefontsize = 10,
        markerstrokewidth = 0.5, size = (600, 450), dpi = 300,
        left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, right_margin = 3Plots.mm)
const ION_COL = Dict("12C_10MeVu" => :darkorange, "12C_80MeVu" => :firebrick, "1H_100MeV" => :steelblue)
col(tag) = get(ION_COL, tag, :black)
nice(tag) = replace(tag, "12C_" => "¹²C ", "1H_" => "¹H ", "MeVu" => " MeV/u", "MeV" => " MeV")
rate_label(Ḋ) = Ḋ >= 1e2 ? "acute" : @sprintf("%.0e Gy/s", Ḋ)
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  ", name))

acute(g) = g[g.dose_rate_Gys .>= 1e2, :]
rates(g) = sort(g[g.dose_rate_Gys .< 1e2, :], :dose_rate_Gys)
xr = (xscale = :log10, xlabel = "Dose rate (Gy/s)")

# ── SF vs dose, one per ion ────────────────────────────────────────────────
sf_panels = Dict{String,Any}()
for tag in tags
    s = surv[surv.tag .== tag, :]
    p = plot(yscale = :log10, xlabel = "Dose (Gy)", ylabel = "Surviving fraction",
             legend = :bottomleft, legendtitle = nice(tag))
    Ḋs = sort(unique(s.dose_rate_Gys))
    cs = palette(:viridis, length(Ḋs) + 1)
    for (k, Ḋ) in enumerate(Ḋs)
        g = sort(s[s.dose_rate_Gys .== Ḋ, :], :dose_Gy)
        D = vcat(0.0, g.dose_Gy)
        lo = min.(g.SE, 0.9 .* g.SF)
        scatter!(p, g.dose_Gy, g.SF; yerror = (lo, g.SE), color = cs[k], ms = 4, label = rate_label(Ḋ))
        plot!(p, D, vcat(1.0, g.SF_LQ); color = cs[k], label = "")
        plot!(p, D, vcat(1.0, g.SF_LC); color = cs[k], ls = :dash, lw = 1.2, label = "")
    end
    plot!(p, [NaN], [NaN]; color = :black, label = "LQ fit (α fixed)")
    plot!(p, [NaN], [NaN]; color = :black, ls = :dash, label = "Lea–Catcheside")
    sf_panels[tag] = p
    save(p, "sf_$(tag).pdf")
end

# ── β (α fixed) vs dose rate ───────────────────────────────────────────────
pβ = plot(; xr..., ylabel = "β (Gy⁻²), α fixed", legend = :topleft)
for tag in tags
    r = rates(lq[lq.tag .== tag, :])
    scatter!(pβ, r.dose_rate_Gys, r.beta; yerror = r.beta_se, color = col(tag), label = nice(tag))
    plot!(pβ, r.dose_rate_Gys, r.beta_LC; color = col(tag), ls = :dash, label = "")
end
plot!(pβ, [NaN], [NaN]; color = :black, ls = :dash, label = "Lea–Catcheside")
hline!(pβ, [0.0]; color = :gray, lw = 0.5, label = "")
save(pβ, "beta_fixed_alpha.pdf")

# ── α and β from the constrained free fit ──────────────────────────────────
pαf = plot(; xr..., ylabel = "α (Gy⁻¹), free fit", legend = :best)
pβf = plot(; xr..., ylabel = "β (Gy⁻²), free fit", legend = :best)
for tag in tags
    g = lq[lq.tag .== tag, :]; r = rates(g); a = acute(g)
    scatter!(pαf, r.dose_rate_Gys, r.alpha_free; yerror = r.alpha_free_se, color = col(tag), label = nice(tag))
    scatter!(pβf, r.dose_rate_Gys, r.beta_free; yerror = r.beta_free_se, color = col(tag), label = nice(tag))
    if nrow(a) > 0
        hline!(pαf, [a.alpha_free[1]]; color = col(tag), ls = :dot, label = "")
        hline!(pβf, [a.beta_free[1]];  color = col(tag), ls = :dot, label = "")
    end
end
plot!(pαf, [NaN], [NaN]; color = :black, ls = :dot, label = "acute")
plot!(pβf, [NaN], [NaN]; color = :black, ls = :dot, label = "acute")
hline!(pβf, [0.0]; color = :gray, lw = 0.5, label = "")
save(pαf, "alpha_free.pdf"); save(pβf, "beta_free.pdf")

# ── relative views ─────────────────────────────────────────────────────────
pq = plot(; xr..., ylabel = "β·D_max / (α + β·D_max)", legend = :topleft, ylims = (0, 1))
ps = plot(; xr..., ylabel = "SF(Ḋ) / SF(acute) at D_max", legend = :topright)
pn = plot(; xr..., ylabel = "β(Ḋ) / β(acute)", legend = :topleft)
for tag in tags
    g = lq[lq.tag .== tag, :]; r = rates(g); a = acute(g)
    s = surv[surv.tag .== tag, :]; Dmax = maximum(s.dose_Gy)
    α0, β0 = a.alpha[1], a.beta[1]
    q(b) = b * Dmax / (α0 + b * Dmax)
    qse(b, sb) = sb * Dmax * α0 / (α0 + b * Dmax)^2
    scatter!(pq, r.dose_rate_Gys, q.(r.beta); yerror = qse.(r.beta, r.beta_se), color = col(tag),
             label = "$(nice(tag)), D_max = $(Dmax) Gy")
    plot!(pq, r.dose_rate_Gys, q.(r.beta_LC); color = col(tag), ls = :dash, label = "")
    hline!(pq, [q(β0)]; color = col(tag), ls = :dot, lw = 1, label = "")

    sD = s[s.dose_Gy .== Dmax, :]; sa = acute(sD); sr = rates(sD)
    ratio = sr.SF ./ sa.SF[1]
    rse   = ratio .* sqrt.((sr.SE ./ sr.SF) .^ 2 .+ (sa.SE[1] / sa.SF[1])^2)
    scatter!(ps, sr.dose_rate_Gys, ratio; yerror = rse, color = col(tag), label = nice(tag))
    plot!(ps, sr.dose_rate_Gys, sr.SF_LC ./ sa.SF_LC[1]; color = col(tag), ls = :dash, label = "")

    if β0 > 2 * a.beta_se[1]
        scatter!(pn, r.dose_rate_Gys, r.beta ./ β0; yerror = r.beta_se ./ β0, color = col(tag), label = nice(tag))
        plot!(pn, r.dose_rate_Gys, r.beta_LC ./ β0; color = col(tag), ls = :dash, label = "")
    else
        println("  $tag: acute β not significantly > 0, left out of beta_normalised.pdf")
    end
end
for p in (pq, ps, pn); plot!(p, [NaN], [NaN]; color = :black, ls = :dash, label = "Lea–Catcheside"); end
plot!(pq, [NaN], [NaN]; color = :black, ls = :dot, label = "acute")
hline!(ps, [1.0]; color = :gray, lw = 0.5, label = "")
save(pq, "quadratic_share.pdf"); save(ps, "sparing.pdf"); save(pn, "beta_normalised.pdf")

# ── combined figure, panels labelled (a)… ──────────────────────────────────
panels = vcat([sf_panels[t] for t in tags], [pβ, pαf, pβf, pq, ps, pn])
for (k, p) in enumerate(panels)
    plot!(p; annotations = ((0.02, 0.97), text("($(Char('a' + k - 1)))", 12, :left, :top)))
end
ncol = 3; nrow_ = cld(length(panels), ncol)
save(plot(panels...; layout = (nrow_, ncol), size = (600 * ncol, 460 * nrow_)), "doserate_all.pdf")
println("Figures in ", FDIR)
