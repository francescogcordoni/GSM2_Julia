using CSV, DataFrames, Plots, LaTeXStrings

# ── Paths ─────────────────────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data", "benchmark_spheroid_scaling")

const DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

df = CSV.read(joinpath(datadir, "scaling_1H_100MeV_1Gy.csv"), DataFrame)
sort!(df, :N_cells)

# ── PLOT 1: Timing vs number of cells ─────────────────────────────────────────
p1 = plot(;
    xlabel = "Number of cells",
    ylabel = "Wall-clock time (s)",
    legend = :topleft,
    size   = (800, 500),
    DEFAULTS...,
)

plot!(p1, df.N_cells, df.t_irrad_s;
    label     = "Irradiation",
    color     = :steelblue,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)
plot!(p1, df.N_cells, df.t_abm_s;
    label     = "ABM",
    color     = :firebrick,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)
plot!(p1, df.N_cells, df.t_total_s;
    label     = "Total",
    color     = :black,
    linewidth = 2,
    linestyle = :dash,
    marker    = :diamond,
    markersize        = 6,
    markerstrokewidth = 0.5,
)

display(p1)
savefig(p1, joinpath(datadir, "timing_vs_cells.png"))
savefig(p1, joinpath(datadir, "timing_vs_cells.pdf"))
println("Saved: timing_vs_cells")

# ── PLOT 2: Timing vs tumor radius ────────────────────────────────────────────
p2 = plot(;
    xlabel = "Tumor radius (µm)",
    ylabel = "Wall-clock time (s)",
    legend = :topleft,
    size   = (800, 500),
    DEFAULTS...,
)

plot!(p2, df.tumor_radius_um, df.t_irrad_s;
    label     = "Irradiation",
    color     = :steelblue,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)
plot!(p2, df.tumor_radius_um, df.t_abm_s;
    label     = "ABM",
    color     = :firebrick,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)
plot!(p2, df.tumor_radius_um, df.t_total_s;
    label     = "Total",
    color     = :black,
    linewidth = 2,
    linestyle = :dash,
    marker    = :diamond,
    markersize        = 6,
    markerstrokewidth = 0.5,
)

display(p2)
savefig(p2, joinpath(datadir, "timing_vs_radius.png"))
savefig(p2, joinpath(datadir, "timing_vs_radius.pdf"))
println("Saved: timing_vs_radius")

# ── PLOT 3: Fraction of time per component ────────────────────────────────────
irrad_frac = df.t_irrad_s ./ df.t_total_s
abm_frac   = df.t_abm_s   ./ df.t_total_s

p3 = plot(;
    xlabel = "Number of cells",
    ylabel = "Fraction of total time",
    legend = :right,
    ylims  = (0, 1),
    size   = (800, 500),
    DEFAULTS...,
)

plot!(p3, df.N_cells, irrad_frac;
    label     = "Irradiation",
    color     = :steelblue,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)
plot!(p3, df.N_cells, abm_frac;
    label     = "ABM",
    color     = :firebrick,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)

display(p3)
savefig(p3, joinpath(datadir, "time_fraction_vs_cells.png"))
savefig(p3, joinpath(datadir, "time_fraction_vs_cells.pdf"))
println("Saved: time_fraction_vs_cells")

# ── PLOT 4: Cells vs radius ───────────────────────────────────────────────────
p4 = plot(;
    xlabel = "Tumor radius (µm)",
    ylabel = "Number of cells",
    legend = false,
    size   = (700, 450),
    DEFAULTS...,
)

plot!(p4, df.tumor_radius_um, df.N_cells;
    color     = :seagreen,
    linewidth = 2,
    marker    = :circle,
    markersize        = 6,
    markerstrokewidth = 0.5,
)

display(p4)
savefig(p4, joinpath(datadir, "cells_vs_radius.png"))
savefig(p4, joinpath(datadir, "cells_vs_radius.pdf"))
println("Saved: cells_vs_radius")

println("\nAll plots saved to $datadir")
