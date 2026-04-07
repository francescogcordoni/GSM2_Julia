using CSV, DataFrames, Plots, Statistics, StatsBase, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data", "damage_temporal_evolution")

const DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

CONDITIONS = [
    (tag="1H_80MeV",   label="¹H 80 MeV/u",  color=:royalblue),
    (tag="1H_30MeV",   label="¹H 30 MeV/u",  color=:steelblue),
    (tag="12C_80MeVu", label="¹²C 80 MeV/u", color=:firebrick),
]

# Gradient shared across conditions for individual cell lines
_grad = cgrad([:deepskyblue, :mediumpurple, :lightsalmon])

# Maximum number of individual cell lines to draw (for rendering speed)
const MAX_LINES = 300

# ── Step-function interpolation ───────────────────────────────────────────────
# Given sorted event (times_cell, vals_cell), return value at each point in t_grid
# using last-observation-carried-forward.
function stepinterp(times_cell::AbstractVector, vals_cell::AbstractVector,
                    t_grid::AbstractVector)
    n = length(t_grid)
    out = zeros(Float64, n)
    for j in 1:n
        idx = searchsortedlast(times_cell, t_grid[j])
        out[j] = idx == 0 ? Float64(vals_cell[1]) : Float64(vals_cell[idx])
    end
    return out
end

# ── Main loop ─────────────────────────────────────────────────────────────────
for cond in CONDITIONS
    traj_path = joinpath(datadir, "$(cond.tag)_trajectories.csv")
    isfile(traj_path) || (@warn "Missing: $traj_path"; continue)

    println("Loading $(cond.tag)...")
    df = CSV.read(traj_path, DataFrame)

    # Group by cell_id for fast per-cell access
    gdf      = groupby(df, :cell_id)
    n_cells  = length(gdf)
    println("  $n_cells cells, $(nrow(df)) total events")

    # Keep only cells with X_init > 0 (damaged cells) so the mean reflects
    # actual repair dynamics and isn't diluted by undamaged cells.
    damaged_groups = [g for g in gdf if first(sort(DataFrame(g), :time)).X_total > 0]
    n_damaged = length(damaged_groups)
    println("  $n_damaged damaged cells (X_init > 0)")

    # Common time grid over damaged-cell events only
    t_max  = maximum(df.time)
    t_grid = collect(range(0.0, t_max, length=600))

    # Build interpolated matrices for damaged cells only
    X_mat = Matrix{Float64}(undef, length(t_grid), n_damaged)
    Y_mat = Matrix{Float64}(undef, length(t_grid), n_damaged)

    for (j, g) in enumerate(damaged_groups)
        sub = sort(DataFrame(g), :time)
        X_mat[:, j] = stepinterp(sub.time, sub.X_total, t_grid)
        Y_mat[:, j] = stepinterp(sub.time, sub.Y_total, t_grid)
    end

    # Subsample from damaged cells for plotting; mean is over the same subset
    plot_idx = n_damaged <= MAX_LINES ? collect(1:n_damaged) :
               sort(sample(1:n_damaged, MAX_LINES; replace=false))
    n_plot   = length(plot_idx)
    cell_colors = [_grad[(k - 1) / max(n_plot - 1, 1)] for k in 1:n_plot]

    X_mean = vec(mean(X_mat[:, plot_idx], dims=2))
    Y_mean = vec(mean(Y_mat[:, plot_idx], dims=2))

    # ── Panel: X damage over time ─────────────────────────────────────────────
    px = plot(;
        xlabel = "Time (h)",
        ylabel = "Total X lesions",
        legend = false,
        size   = (700, 480),
        DEFAULTS...,
    )
    for (k, j) in enumerate(plot_idx)
        plot!(px, t_grid, X_mat[:, j];
              lw    = 0.6,
              alpha = 0.18,
              color = cell_colors[k])
    end
    plot!(px, t_grid, X_mean;
          lw    = 2.5,
          color = :black)

    # ── Panel: Y damage over time ─────────────────────────────────────────────
    py = plot(;
        xlabel = "Time (h)",
        ylabel = "Total Y lesions",
        legend = false,
        size   = (700, 480),
        DEFAULTS...,
    )
    for (k, j) in enumerate(plot_idx)
        plot!(py, t_grid, Y_mat[:, j];
              lw    = 0.6,
              alpha = 0.18,
              color = cell_colors[k])
    end
    plot!(py, t_grid, Y_mean;
          lw    = 2.5,
          color = :black)

    # ── Combine and save ──────────────────────────────────────────────────────
    p = plot(px, py;
             layout = (1, 2),
             size   = (1400, 500),
             dpi    = 600,
             plot_title = "")

    display(p)
    savefig(p, joinpath(datadir, "damage_evolution_$(cond.tag).png"))
    savefig(p, joinpath(datadir, "damage_evolution_$(cond.tag).pdf"))
    println("Saved: damage_evolution_$(cond.tag)")
end

println("\nAll plots saved to $datadir")
