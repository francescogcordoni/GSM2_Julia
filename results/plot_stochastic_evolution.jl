using CSV, DataFrames
using Plots
using Statistics
using Printf

# ============================================================
# Configuration
# ============================================================
datadir = joinpath(@__DIR__, "..", "data", "stochastic_time_ev")
outdir  = datadir

t_repair     = 48.0
n_replicates = 100
Ntot         = nothing

const PLOT_DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

# ============================================================
# Load data
# ============================================================
repair_path   = joinpath(datadir, "ts_repair_frac1.csv")
summary_path  = joinpath(datadir, "ts_summary.csv")
combined_path = joinpath(datadir, "ts_all_replicates.csv")

isfile(repair_path)   || error("Missing: $repair_path")
isfile(summary_path)  || error("Missing: $summary_path")
isfile(combined_path) || error("Missing: $combined_path")

repair_df   = CSV.read(repair_path,   DataFrame)
summary_df  = CSV.read(summary_path,  DataFrame)
combined_df = CSV.read(combined_path, DataFrame)

rep_cols = [col for col in names(combined_df) if startswith(string(col), "rep_")]
n_reps   = length(rep_cols)
println("Loaded $n_reps replicates.")

t_frac2_start = t_repair
t_repair_plot = repair_df.time
t_post_plot   = summary_df.time .+ t_frac2_start

# Gradient palette: one color per replicate (light blue → purple → coral)
_grad      = cgrad([:deepskyblue, :mediumpurple, :lightsalmon])
rep_colors = [_grad[(k - 1) / max(n_reps - 1, 1)] for k in 1:n_reps]

# ============================================================
# PLOT 1: Full timeline — repair phase + stochastic replicates
# ============================================================
println("Building Plot 1 — full stochastic timeline...")

p1 = plot(;
    xlabel = "Time (h)",
    ylabel = "Total cells",
    legend = false,
    size   = (1000, 550),
    PLOT_DEFAULTS...)

# Repair phase
plot!(p1, t_repair_plot, repair_df.total_cells;
      lw    = 2.5,
      color = :steelblue)

# Individual replicates — each with its own gradient color
for (k, col) in enumerate(rep_cols)
    plot!(p1, t_post_plot, combined_df[!, col];
          lw    = 0.8,
          alpha = 0.35,
          color = rep_colors[k])
end

# Mean line on top
plot!(p1, t_post_plot, summary_df.mean_cells;
      lw    = 2.5,
      color = :black)

# Second fraction marker
vline!(p1, [t_frac2_start];
       color     = :black,
       lw        = 1.5,
       linestyle = :dash)

display(p1)
savefig(p1, joinpath(outdir, "stochastic_full_timeline.png"))
savefig(p1, joinpath(outdir, "stochastic_full_timeline.pdf"))
println("Saved: stochastic_full_timeline")

# ============================================================
# PLOT 2: Post-second-fraction only
# ============================================================
println("Building Plot 2 — post-frac2 stochastic spread...")

p2 = plot(;
    xlabel = "Time after 2nd fraction (h)",
    ylabel = "Total cells",
    legend = false,
    size   = (900, 500),
    PLOT_DEFAULTS...)

for (k, col) in enumerate(rep_cols)
    plot!(p2, summary_df.time, combined_df[!, col];
          lw    = 0.8,
          alpha = 0.35,
          color = rep_colors[k])
end

plot!(p2, summary_df.time, summary_df.mean_cells;
      lw    = 2.5,
      color = :black)

display(p2)
savefig(p2, joinpath(outdir, "stochastic_post_frac2.png"))
savefig(p2, joinpath(outdir, "stochastic_post_frac2.pdf"))
println("Saved: stochastic_post_frac2")

# ============================================================
# PLOT 3: Repair phase alone
# ============================================================
println("Building Plot 3 — repair phase...")

p3 = plot(;
    xlabel = "Time after 1st fraction (h)",
    ylabel = "Total cells",
    legend = false,
    size   = (800, 450),
    PLOT_DEFAULTS...)

plot!(p3, repair_df.time, repair_df.total_cells;
      lw    = 2.5,
      color = :steelblue)

display(p3)
savefig(p3, joinpath(outdir, "repair_phase.png"))
savefig(p3, joinpath(outdir, "repair_phase.pdf"))
println("Saved: repair_phase")

# ============================================================
# PLOT 4: Distribution of final cell count
# ============================================================
println("Building Plot 4 — final cell count distribution...")

final_counts = [combined_df[end, col] for col in rep_cols]
t_final      = round(Int, t_repair + summary_df.time[end])

p4 = histogram(final_counts;
    bins       = 20,
    xlabel     = "Final cell count (t = $(t_final) h)",
    ylabel     = "Number of replicates",
    legend     = :topright,
    color      = :orangered,
    alpha      = 0.75,
    linecolor  = :white,
    linewidth  = 0.5,
    size       = (800, 450),
    PLOT_DEFAULTS...)

vline!(p4, [mean(final_counts)];
       lw        = 2,
       color     = :darkred,
       linestyle = :dash,
       label     = "Mean = $(round(mean(final_counts), digits=1))")

display(p4)
savefig(p4, joinpath(outdir, "final_cell_distribution.png"))
savefig(p4, joinpath(outdir, "final_cell_distribution.pdf"))
println("Saved: final_cell_distribution")

println("\n", "="^60)
println("ALL PLOTS SAVED TO $outdir/")
println("="^60)
