# ============================================================================
# plot_split_repopulation.jl
#
# Split-dose figures from the CSVs in data/split_repopulation (nothing is rerun).
# Same style as the other figures: vector PDF, no titles, labels and legends kept
# inside the figure.
#
#   surv_vs_time.pdf     survival vs time between fractions (full ABM, G1 only,
#                        no repopulation)
#   alive_vs_time.pdf    mean alive cells before the second fraction
#   phases_vs_time.pdf   mean cells per phase before the second fraction (full ABM)
#   split_all.pdf        the three panels in one row, labelled (a)–(c)
#
# Run: julia scripts/plot_split_repopulation.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using CSV, DataFrames, Printf, Plots
gr()

# ── Load data ──────────────────────────────────────────────────────────────
const DIR  = joinpath(@__DIR__, "..", "data", "split_repopulation")
const FDIR = joinpath(DIR, "figures"); mkpath(FDIR)
const T_MAX = 30.0                     # h, longest interval shown

surv_df  = CSV.read(joinpath(DIR, "surv_curves.csv"),           DataFrame)
pre_full = CSV.read(joinpath(DIR, "phase_times_pre_full.csv"),  DataFrame)
pre_G1   = CSV.read(joinpath(DIR, "phase_times_pre_G1.csv"),    DataFrame)
meta     = CSV.read(joinpath(DIR, "metadata.csv"),              DataFrame)

surv_df  = sort(surv_df[surv_df.time .<= T_MAX, :], :time)
pre_full = sort(pre_full[pre_full.time .<= T_MAX, :], :time)
pre_G1   = sort(pre_G1[pre_G1.time .<= T_MAX, :], :time)
particle, E, dose = meta.particle[1], meta.E[1], meta.dose[1]
nice(p) = ion_label(p)
cond = tex("$(nice(particle)) $(E) MeV, $(dose) Gy + $(dose) Gy")
println("condition: $cond | intervals: $(length(surv_df.time)) up to $(T_MAX) h")

# ── Style (shared with the other figures) ──────────────────────────────────
default(framestyle = :box, lw = 1.8, guidefontsize = 12, tickfontsize = 10,
        legendfontsize = 9, legendtitlefontsize = 10, dpi = 300, size = (640, 460),
        markerstrokewidth = 0.5, ms = 4,
        left_margin = 8Plots.mm, bottom_margin = 7Plots.mm, right_margin = 5Plots.mm,
        top_margin = 3Plots.mm, foreground_color_legend = nothing,
        background_color_legend = RGBA(1, 1, 1, 0.85))
paper_style!()
const C_FULL = :steelblue          # full ABM
const C_G1   = :darkorange         # radiosensitivity of G1 only
const C_NOAB = :seagreen           # standard GSM2, no repopulation
const PH_COL = Dict("G0" => :gray60, "G1" => :steelblue, "S" => :seagreen,
                    "G2" => :darkorange, "M" => :firebrick)
const XL = "Time between fractions (h)"
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  ", name))

# ── Survival vs inter-fraction time ────────────────────────────────────────
p1 = plot(xlabel = XL, ylabel = "Survival probability", legend = :topleft,
          legendtitle = tex(cond), xlims = (0, T_MAX))
plot!(p1, surv_df.time, surv_df.surv_abm_full; color = C_FULL, marker = :circle,  label = "ABM, full cycle")
plot!(p1, surv_df.time, surv_df.surv_abm_G1;   color = C_G1,   marker = :diamond, label = "ABM, G1 only")
plot!(p1, surv_df.time, surv_df.surv_noabm;    color = C_NOAB, marker = :rect,    label = "No repopulation")
save(p1, "surv_vs_time.pdf")

# ── Alive cells before the second fraction ─────────────────────────────────
p2 = plot(xlabel = XL, ylabel = "Mean alive cells before fraction 2", legend = :topleft,
          legendtitle = tex(cond), xlims = (0, T_MAX))
plot!(p2, pre_full.time, pre_full.Nalive; color = C_FULL, marker = :circle,  label = "Full cycle")
plot!(p2, pre_G1.time,   pre_G1.Nalive;   color = C_G1,   marker = :diamond, label = "G1 only")
save(p2, "alive_vs_time.pdf")

# ── Phase distribution before the second fraction (full ABM) ───────────────
p3 = plot(xlabel = XL, ylabel = "Mean cells per phase before fraction 2", legend = :topright,
          legendtitle = "Full cycle", xlims = (0, T_MAX))
for ph in ("G0", "G1", "S", "G2", "M")
    plot!(p3, pre_full.time, pre_full[!, ph]; color = PH_COL[ph], marker = :circle, ms = 3, label = ph)
end
save(p3, "phases_vs_time.pdf")

# ── Combined figure ────────────────────────────────────────────────────────
panels = [deepcopy(p) for p in (p1, p2, p3)]
for (k, p) in enumerate(panels)
    plot!(p; annotations = ((0.97, 0.06), text("($(Char('a' + k - 1)))", 12, :right, :bottom)))
end
save(plot(panels...; layout = (1, 3), size = (1920, 480), left_margin = 10Plots.mm,
          bottom_margin = 9Plots.mm), "split_all.pdf")
println("Figures in $FDIR")
