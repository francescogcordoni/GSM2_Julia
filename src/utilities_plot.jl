#! ============================================================================
#! utilities_plots.jl
#!
#! FUNCTIONS
#! ---------
#~ Dose, Survival & Damage
#?   plot_scalar_cell(cell_df, col; layer_plot) -> Plot
#       Generic two-panel figure for any scalar column col (e.g. :dose_cell, :sp).
#       Panel 1: density (all active cells, or grouped by energy_step when layer_plot=true).
#       Panel 2: 3D scatter of (x,y,z) colored by col.
#       Replaces the old plot_dose_cell and plot_survival_probability_cell.
#?   plot_damage(cell_df; layer_plot) -> Plot
#       Density of total X-damage per active cell.
#       layer_plot=true: one curve per energy_step. layer_plot=false: single density + mean line.
#
#~ Timing
#?   plot_times(cell_df; show_means, summary, verbose, ...) -> Plot
#       Four-panel figure: death_time, recover_time, cycle_time densities,
#       and total X-damage density. Active cells only. Finite values only.
#?   plot_initial_distributions(cell_df; bins, linewidth, size, ...) -> Plot
#       Two-panel figure: death_time and recover_time densities at initialization.
#
#~ Population Dynamics (SimulationTimeSeries)
#?   plot_cell_dynamics(ts) -> Plot
#       Total cell count vs time.
#?   plot_phase_dynamics(ts) -> Plot
#       G1/S/G2/M cell counts vs time.
#?   plot_stem_dynamics(ts) -> Plot
#       Stem vs non-stem cell counts vs time.
#       Returns placeholder if no stem data available.
#?   plot_simulation_results(ts) -> Plot
#       Combined 3-panel figure: total cells + phases + stem/non-stem.
#
#~ Cell Cycle Distribution
#?   plot_phase_proportions_alive(cell_df; title_text) -> Plot
#       Color-coded bar chart of phase percentages for alive cells.
#       Bars: G0=black, G1=green, S=orange, G2=purple, M=red.
#?   print_phase_distribution(cell_df; label) -> Nothing
#       Prints phase counts and percentages to stdout. No plot produced.
#?   plot_cell_cycle_distribution(cell_df; phase_plot, half_sphere) -> Plot
#       Two panels: phase bar chart (counts or %) + 3D scatter colored by phase.
#       half_sphere=true filters to x≥0 for the 3D panel.
#?   plot_cell_cycle_snapshots(snapshots; times, half_sphere) -> Plot
#       Multi-row figure with one row per snapshot time (bar + 3D).
#       Accepts Dict{Int, DataFrame} or Dict{Int, CellPopulation}.
#?   animate_cell_cycle_3d(snapshots; output_file, fps, half_sphere) -> Animation
#       GIF animation of 3D scatter colored by cell-cycle phase over time.
#       Accepts Dict{Int, DataFrame} or Dict{Int, CellPopulation}.
#
#~ Utilities
#?   create_snapshot(pop::CellPopulation) -> CellPopulation
#       Returns a new CellPopulation containing only alive cells (is_cell==1).
#! ============================================================================

using Plots
using Statistics
using DataFrames

"""
    plot_scalar_cell(cell_df, col::Symbol = :dose_cell; layer_plot=false) -> Plot

Generic two-panel figure for any per-cell scalar column `col`.
- Panel 1: density of `col` for active cells (`is_cell==1`, positive values only).
    `layer_plot=false` → single density + mean line.
    `layer_plot=true`  → one curve per `energy_step`.
- Panel 2: 3D scatter of `(x, y, z)` colored by `col` (half-sphere `x≥0`).

Returns `nothing` if no active cells or no positive values.

Required columns: `:x, :y, :z, :is_cell, col`.
Layer mode also requires `:energy_step`.

# Example
```julia
plot_scalar_cell(cell_df, :dose_cell)
plot_scalar_cell(cell_df, :sp; layer_plot=true)
```
"""
function plot_scalar_cell(cell_df, col::Symbol = :dose_cell; layer_plot::Bool = false)
    for c in [:x, :y, :z, :is_cell, col]
        hasproperty(cell_df, c) || error("cell_df must contain column :$c")
    end

    df_active = cell_df[cell_df.is_cell .== 1, :]
    isempty(df_active) && (@warn "No active cells (is_cell == 1)."; return nothing)

    active_vals = filter(>(0), df_active[!, col])
    isempty(active_vals) && (@warn "No positive values in column :$col."; return nothing)

    col_str = string(col)

    # Panel 1 — density
    if !layer_plot
        p1 = density(active_vals;
                        title="$col_str Density (all active cells)",
                        xlabel=col_str, ylabel="Density",
                        linewidth=2, legend=false)
        vline!(p1, [mean(active_vals)], color=:red, linestyle=:dash)
    else
        hasproperty(cell_df, :energy_step) ||
            error("layer_plot=true requires column :energy_step")
        p1 = plot(title="$col_str Density grouped by energy_step",
                  xlabel=col_str, ylabel="Density",
                  legend=:topright, linewidth=2)
        for E in sort(unique(df_active.energy_step))
            sub_pos = filter(>(0), df_active[df_active.energy_step .== E, col])
            isempty(sub_pos) && continue
            density!(p1, sub_pos, label="energy_step $E")
        end
    end

    # Panel 2 — 3D scatter (half-sphere x≥0)
    df3 = df_active[df_active.x .>= 0, :]
    p2 = scatter(df3.x, df3.y, df3.z;
                    markersize=4, markerstrokewidth=0.1,
                    marker_z=df3[!, col], colorbar=true,
                    xlabel="x (µm)", ylabel="y (µm)", zlabel="z",
                    title="3D $col_str Distribution", legend=false,
                    aspect_ratio=:equal, seriescolor=:viridis,
                    size=(900, 700), camera=(320, 30))

    return plot(p1, p2, layout=(1, 2), size=(1200, 500))
end

"""
    plot_damage(cell_df; layer_plot=false) -> Plot

Density of total X-damage (`sum(dam_X_dom)`) per active cell.
`layer_plot=true`: one density curve per `energy_step`.
`layer_plot=false`: single density with mean line.
Returns `nothing` if no valid data.

Required columns: `:is_cell, :dam_X_dom`. Layer mode also requires `:energy_step`.

# Example
```julia
plot_damage(cell_df)
plot_damage(cell_df; layer_plot=true)
```
"""
function plot_damage(cell_df::DataFrame; layer_plot::Bool = false)

    if layer_plot
        for c in (:is_cell, :dam_X_dom, :energy_step)
            hasproperty(cell_df, c) ||
                error("Layer-grouped damage plot requires column :$c")
        end

        plt = plot(title="X-Damage Distribution Grouped by energy_step",
                    xlabel="Total X-Damage", ylabel="Density",
                    legend=:topright, linewidth=2)

        for E in sort(unique(cell_df.energy_step))
            idx = findall(i -> cell_df.is_cell[i] == 1 && cell_df.energy_step[i] == E,
                            1:nrow(cell_df))
            isempty(idx) && (@warn "energy_step $E has no active cells."; continue)

            damage_vals = [sum(cell_df.dam_X_dom[i])
                            for i in idx if cell_df.dam_X_dom[i] isa AbstractVector]
            isempty(damage_vals) && (@warn "energy_step $E has no valid damage vectors."; continue)

            density!(plt, damage_vals, label="energy_step $E")
        end
        return plt
    end

    # Single distribution
    active_idx = findall(row -> row.is_cell == 1, eachrow(cell_df))
    isempty(active_idx) && (@warn "No active cells (is_cell == 1)."; return nothing)

    damage_values = Float64[]
    for i in active_idx
        dam = cell_df.dam_X_dom[i]
        if dam isa AbstractVector
            push!(damage_values, sum(dam))
        else
            @warn "dam_X_dom at row $i is not a vector. Skipping."
        end
    end
    isempty(damage_values) && (@warn "No valid damage vectors."; return nothing)

    mean_damage = mean(damage_values)
    println("Mean X-damage per active cell = $(round(mean_damage, digits=4))")

    plt = density(damage_values;
                  title="Density of X-Damage per Active Cell",
                  xlabel="Total X-Damage", ylabel="Density",
                  linewidth=2, legend=:topright)
    vline!(plt, [mean_damage], color=:red, linestyle=:dash, label="Mean")
    return plt
end

"""
    plot_times(cell_df; show_means=true, summary=true, verbose=false, ...) -> Plot

Four-panel figure for active cells (`is_cell==1`):
1. `death_time` density   2. `recover_time` density
3. `cycle_time` density   4. total X-damage density

Finite values only. If `dam_X_total` is absent, computes `sum(dam_X_dom[i])` on the fly.
Missing columns produce annotated empty panels rather than errors.

# Example
```julia
plot_times(cell_df; show_means=true, summary=true)
```
"""
function plot_times(cell_df::DataFrame;
                    show_means::Bool     = true,
                    summary::Bool        = true,
                    verbose::Bool        = false,
                    color_main           = :D55E00,
                    color_alt            = :steelblue,
                    layout_tuple::Tuple{Int,Int} = (2, 2),
                    size_px::Tuple{Int,Int}      = (1200, 800))

    :is_cell in propertynames(cell_df) || error("plot_times: cell_df must contain :is_cell.")

    df_active = cell_df[cell_df.is_cell .== 1, :]
    if nrow(df_active) == 0
        println("[plot_times] No active cells. Nothing to plot.")
        return nothing
    end

    get_finite = function(df, col)
        !(col in propertynames(df)) && (verbose && println("[plot_times] Column $col not found."); return Float64[])
        [Float64(x) for x in df[!, col] if x isa Real && isfinite(Float64(x))]
    end

    death_vals   = get_finite(df_active, :death_time)
    recover_vals = get_finite(df_active, :recover_time)
    cycle_vals   = get_finite(df_active, :cycle_time)

    damX_vals = if :dam_X_total in propertynames(df_active)
        [x for x in Float64.(df_active.dam_X_total) if isfinite(x)]
    elseif :dam_X_dom in propertynames(df_active)
        [sum(row[:dam_X_dom]) for row in eachrow(df_active)
         if row[:dam_X_dom] isa AbstractVector]
    else
        verbose && println("[plot_times] Neither :dam_X_total nor :dam_X_dom found.")
        Float64[]
    end

    function _panel(title, xlabel, vals)
        p = plot(title=title, xlabel=xlabel, ylabel="Density", legend=false)
        if !isempty(vals)
            density!(p, vals; lw=2)
            show_means && vline!(p, [mean(vals)]; color=:red, ls=:dash)
        else
            annotate!(p, 0.5, 0.5, text("No finite values", 10, :gray))
        end
        return p
    end

    p1 = _panel("Death time",    "time",   death_vals)
    p2 = _panel("Recovery time", "time",   recover_vals)
    p3 = _panel("Cycle time",    "time",   cycle_vals)
    p4 = _panel("Total X damage","damage", damX_vals)

    if summary
        for (name, vals) in [("death_time", death_vals), ("recover_time", recover_vals),
                              ("cycle_time", cycle_vals), ("dam_X_total", damX_vals)]
            print("[plot_times] $name : $(length(vals))")
            isempty(vals) || print(" | mean=$(round(mean(vals), digits=4))")
            println()
        end
    end

    return plot(p1, p2, p3, p4; layout=layout_tuple, size=size_px)
end

"""
    plot_initial_distributions(cell_df; bins=50, linewidth=2, size=(1000,400), ...) -> Plot

Two-panel figure: `death_time` and `recover_time` density for alive cells.
Missing or all-Inf columns produce labeled placeholder panels.

# Example
```julia
plot_initial_distributions(cell_df)
```
"""
function plot_initial_distributions(cell_df::DataFrame;
                                    bins::Int              = 50,
                                    linewidth::Int         = 2,
                                    size::Tuple{Int,Int}   = (1000, 400),
                                    titlefont              = 12,
                                    labelfont              = 10)
    alive_mask = cell_df.is_cell .== 1

    finite_vals(vec) = [v for v in vec[alive_mask] if !(ismissing(v) || isinf(v))]

    finite_death = finite_vals(cell_df.death_time)
    p_death = isempty(finite_death) ?
        plot(title="No finite death times", titlefont=titlefont) :
        density(finite_death; title="Death Time Distribution",
                xlabel="Time (h)", ylabel="Density", label="Death Times",
                linewidth=linewidth, titlefont=titlefont, guidefont=labelfont)

    have_recover  = hasproperty(cell_df, :recover_time)
    finite_recover = have_recover ? finite_vals(cell_df.recover_time) : Float64[]

    p_recover = !have_recover ?
        plot(title="Column `recover_time` not found", titlefont=titlefont) :
        isempty(finite_recover) ?
        plot(title="No finite recovery times", titlefont=titlefont) :
        density(finite_recover; title="Recovery Time Distribution",
                xlabel="Time (h)", ylabel="Density", label="Recovery Times",
                linewidth=linewidth, titlefont=titlefont, guidefont=labelfont)

    return plot(p_death, p_recover; layout=(1, 2), size=size)
end

#! ============================================================================
#! Population Dynamics
#! ============================================================================

"""
    plot_cell_dynamics(ts::SimulationTimeSeries) -> Plot

Total cell count vs time.

# Example
```julia
plot_cell_dynamics(ts)
```
"""
function plot_cell_dynamics(ts::SimulationTimeSeries)
    return plot(ts.time, ts.total_cells;
                xlabel="Time (h)", ylabel="Number of Cells",
                title="Total Cell Population Dynamics",
                label="Total Cells", linewidth=2,
                legend=:best, grid=true, framestyle=:box)
end

"""
    plot_phase_dynamics(ts::SimulationTimeSeries) -> Plot

G1/S/G2/M cell counts vs time.

# Example
```julia
plot_phase_dynamics(ts)
```
"""
function plot_phase_dynamics(ts::SimulationTimeSeries)
    p = plot(xlabel="Time (h)", ylabel="Number of Cells",
             title="Cell Cycle Phase Distribution",
             legend=:best, grid=true, framestyle=:box)
    plot!(p, ts.time, ts.g1_cells, label="G1", linewidth=2)
    plot!(p, ts.time, ts.s_cells,  label="S",  linewidth=2)
    plot!(p, ts.time, ts.g2_cells, label="G2", linewidth=2)
    plot!(p, ts.time, ts.m_cells,  label="M",  linewidth=2)
    return p
end

"""
    plot_stem_dynamics(ts::SimulationTimeSeries) -> Plot

Stem vs non-stem cell counts vs time.
Returns a placeholder if no stem cell data (`all zeros`).

# Example
```julia
plot_stem_dynamics(ts)
```
"""
function plot_stem_dynamics(ts::SimulationTimeSeries)
    if all(ts.stem_cells .== 0) && all(ts.non_stem_cells .== 0)
        return plot(title="No stem cell data available", framestyle=:box, grid=true)
    end
    p = plot(xlabel="Time (h)", ylabel="Number of Cells",
             title="Stem vs Non-Stem Cell Dynamics",
             legend=:best, grid=true, framestyle=:box)
    plot!(p, ts.time, ts.stem_cells,     label="Stem Cells",     linewidth=2)
    plot!(p, ts.time, ts.non_stem_cells, label="Non-Stem Cells", linewidth=2)
    return p
end

"""
    plot_simulation_results(ts::SimulationTimeSeries) -> Plot

Combined 3-panel figure: total cells + phases + stem/non-stem (3×1 layout).

# Example
```julia
plot_simulation_results(ts)
savefig(plot_simulation_results(ts), "summary.png")
```
"""
function plot_simulation_results(ts::SimulationTimeSeries)
    return plot(plot_cell_dynamics(ts), plot_phase_dynamics(ts), plot_stem_dynamics(ts);
                layout=(3, 1), size=(1000, 1200))
end

#! ============================================================================
#! Cell Cycle Distribution
#! ============================================================================

"""
    plot_phase_proportions_alive(cell_df; title_text="Cell Cycle Distribution (Alive Cells)")
        -> Plot

Color-coded bar chart of phase percentages for alive cells (`is_cell==1`).
G0=black, G1=green, S=orange, G2=purple, M=red.
Percentage labels shown on bars >2%.

# Example
```julia
plot_phase_proportions_alive(cell_df; title_text="Final Proportions")
```
"""
function plot_phase_proportions_alive(cell_df::DataFrame;
                                      title_text::String = "Cell Cycle Distribution (Alive Cells)")
    alive_phases = cell_df.cell_cycle[cell_df.is_cell .== 1]
    isempty(alive_phases) && return plot(title="No alive cells", grid=false, showaxis=false)

    phases     = ["G0", "G1", "S", "G2", "M"]
    ph_colors  = [:black, :green, :orange, :purple, :red]
    counts     = Dict(p => count(==(String(p)), alive_phases) for p in phases)
    total      = sum(values(counts))
    percents   = [100.0 * counts[p] / total for p in phases]

    p = bar(phases, percents;
            xlabel="Cell Cycle Phase", ylabel="Percentage (%)",
            title="$title_text\n(n=$total alive cells)",
            legend=false, color=ph_colors, alpha=0.7, ylims=(0, 100))

    for (i, pct) in enumerate(percents)
        pct > 2 && annotate!(i, pct + 2, text("$(round(pct, digits=1))%", 8))
    end
    return p
end

"""
    print_phase_distribution(cell_df; label="Current") -> Nothing

Prints phase counts and percentages to stdout for alive cells (`is_cell==1`).
Also prints cycling (G1+S+G2+M) vs quiescent (G0) summary.

# Example
```julia
print_phase_distribution(cell_df; label="t = 12h")
```
"""
function print_phase_distribution(cell_df::DataFrame; label::String = "Current")
    alive_phases = cell_df.cell_cycle[cell_df.is_cell .== 1]
    isempty(alive_phases) && (println("[$label] No alive cells"); return)

    phases  = ["G0", "G1", "S", "G2", "M"]
    counts  = Dict(p => count(==(String(p)), alive_phases) for p in phases)
    total   = sum(values(counts))

    println("\n[$label] Phase Distribution (n=$total alive cells):")
    for p in phases
        pct = 100.0 * counts[p] / total
        println("  $p: $(counts[p]) ($(round(pct, digits=1))%)")
    end

    cycling   = counts["G1"] + counts["S"] + counts["G2"] + counts["M"]
    quiescent = counts["G0"]
    println("  Cycling:   $cycling ($(round(100.0*cycling/total, digits=1))%)")
    println("  Quiescent: $quiescent ($(round(100.0*quiescent/total, digits=1))%)")
end

"""
    plot_cell_cycle_distribution(cell_df; phase_plot=false, half_sphere=true) -> Plot

Two panels: phase bar chart (counts or percentages when `phase_plot=true`)
+ 3D scatter colored by phase.
`half_sphere=true` restricts 3D panel to `x ≥ 0`.
Returns `nothing` if no active cells.

Required columns: `:x, :y, :z, :cell_cycle, :is_cell`.

# Example
```julia
plot_cell_cycle_distribution(cell_df; phase_plot=false, half_sphere=true)
```
"""
function plot_cell_cycle_distribution(cell_df; phase_plot::Bool = false, half_sphere::Bool = true)
    required_cols = [:x, :y, :z, :cell_cycle, :is_cell]
    all(c -> c in propertynames(cell_df), required_cols) ||
        error("cell_df must contain: ", join(string.(required_cols), ", "))

    df_active = cell_df[cell_df.is_cell .== 1, :]
    isempty(df_active) && (@warn "No active cells."; return nothing)

    phases    = ["G0", "G1", "S", "G2", "M"]
    ph_colors = Dict("G0"=>:black, "G1"=>:green, "S"=>:orange, "G2"=>:purple, "M"=>:red)
    counts    = Dict(p => count(==(String(p)), df_active.cell_cycle) for p in phases)
    total     = sum(values(counts))

    if !phase_plot
        vals = [counts[p] for p in phases]
        p1 = bar(phases, vals;
                 title="Cell Cycle Distribution\n(n=$total active cells)",
                 xlabel="Cell Cycle Phase", ylabel="Number of Cells",
                 legend=false, color=[ph_colors[p] for p in phases],
                 alpha=0.7, bar_width=0.6)
        for (i, (ph, c)) in enumerate(zip(phases, vals))
            c > 0 && annotate!(p1, [(i, c + total*0.02,
                                    text("$(round(100.0*c/total, digits=1))%", 8, :center))])
        end
    else
        percents = [100.0 * counts[p] / total for p in phases]
        p1 = bar(phases, percents;
                 title="Cell Cycle Distribution (%)\n(n=$total active cells)",
                 xlabel="Cell Cycle Phase", ylabel="Percentage (%)",
                 legend=false, color=[ph_colors[p] for p in phases],
                 alpha=0.7, bar_width=0.6, ylims=(0, 100))
        for (i, pct) in enumerate(percents)
            pct > 2 && annotate!(p1, [(i, pct+2,
                                      text("$(round(pct, digits=1))%", 8, :center))])
        end
    end

    df3 = half_sphere ? df_active[df_active.x .>= 0, :] : df_active
    isempty(df3) && (@warn "No cells in selected region for 3D plot."; return p1)

    phase_to_num  = Dict("G0"=>0, "G1"=>1, "S"=>2, "G2"=>3, "M"=>4)
    color_values  = [get(phase_to_num, String(p), -1) for p in df3.cell_cycle]
    custom_colors = [RGB(0.,0.,0.), RGB(0.,0.8,0.), RGB(1.,0.65,0.),
                     RGB(0.6,0.,0.8), RGB(1.,0.,0.)]

    p2 = scatter(df3.x, df3.y, df3.z;
                 markersize=4, markerstrokewidth=0.1,
                 marker_z=color_values, colorbar=false,
                 xlabel="x (µm)", ylabel="y (µm)", zlabel="z (µm)",
                 title="3D Cell Cycle Distribution" * (half_sphere ? " (x ≥ 0)" : ""),
                 legend=:topright, aspect_ratio=:equal,
                 seriescolor=cgrad(custom_colors, 5, categorical=true),
                 label="", size=(900, 700), camera=(320, 30), clims=(-0.5, 4.5))

    for phase in phases
        n = count(==(String(phase)), df3.cell_cycle)
        scatter!(p2, Float64[], Float64[], Float64[];
                 markersize=6, color=ph_colors[phase], label="$phase (n=$n)")
    end

    return plot(p1, p2, layout=(1, 2), size=(1400, 600))
end

"""
    plot_cell_cycle_snapshots(snapshots; times=nothing, half_sphere=true) -> Plot

Multi-row figure with one row per snapshot time (phase bar + 3D scatter).
`times=nothing` plots all keys in sorted order.
Accepts `Dict{Int, DataFrame}` or `Dict{Int, CellPopulation}`.

# Example
```julia
plot_cell_cycle_snapshots(snapshots; times=[0, 6, 12, 24])
```
"""
function plot_cell_cycle_snapshots(snapshots::Dict;
                                   times::Union{Nothing, Vector{Int}} = nothing,
                                   half_sphere::Bool = true)
    plot_times_keys = isnothing(times) ? sort(collect(keys(snapshots))) : times
    isempty(plot_times_keys) && (@warn "No snapshots to plot"; return nothing)

    plots = []
    for t in plot_times_keys
        haskey(snapshots, t) || (@warn "Snapshot at t=$t not found"; continue)

        cell_df = isa(snapshots[t], CellPopulation) ?
            to_dataframe(snapshots[t], alive_only=true) : snapshots[t]

        p = plot_cell_cycle_distribution(cell_df; phase_plot=false, half_sphere=half_sphere)
        plot!(p[1], title="Cell Cycle at t=$(t)h\n(n=$(nrow(cell_df)) cells)")
        plot!(p[2], title="3D Distribution t=$(t)h" * (half_sphere ? " (x ≥ 0)" : ""))
        push!(plots, p)
    end

    isempty(plots) && return plot(title="No valid plots created")
    return plot(plots...; layout=(length(plots), 1), size=(1400, 600*length(plots)))
end

"""
    animate_cell_cycle_3d(snapshots; output_file="cell_cycle_3d.gif", fps=2, half_sphere=true)
        -> Animation

GIF animation of 3D scatter colored by cell-cycle phase (G0→gray, G1→green,
S→orange, G2→purple, M→red). One frame per snapshot. Requires ≥ 2 snapshots.
Accepts `Dict{Int, DataFrame}` or `Dict{Int, CellPopulation}`.

# Example
```julia
anim = animate_cell_cycle_3d(snapshots; output_file="cc.gif", fps=3)
```
"""
function animate_cell_cycle_3d(snapshots::Dict;
                                output_file::String = "cell_cycle_3d.gif",
                                fps::Int            = 2,
                                half_sphere::Bool   = true)
    times = sort(collect(keys(snapshots)))
    length(times) < 2 && (@warn "Need at least 2 snapshots for animation"; return nothing)

    ph_colors    = Dict("G0"=>:gray, "G1"=>:green, "S"=>:orange, "G2"=>:purple, "M"=>:red)
    phase_to_num = Dict("G0"=>0, "G1"=>1, "S"=>2, "G2"=>3, "M"=>4)
    custom_colors = [RGB(0.5,0.5,0.5), RGB(0.,0.8,0.), RGB(1.,0.65,0.),
                     RGB(0.6,0.,0.8),  RGB(1.,0.,0.)]

    anim = @animate for t in times
        cell_df = isa(snapshots[t], CellPopulation) ?
            to_dataframe(snapshots[t], alive_only=true) :
            snapshots[t][snapshots[t].is_cell .== 1, :]

        df3 = half_sphere ? cell_df[cell_df.x .>= 0, :] : cell_df
        color_values = [get(phase_to_num, String(p), -1) for p in df3.cell_cycle]

        p = scatter(df3.x, df3.y, df3.z;
                    markersize=4, markerstrokewidth=0.1,
                    marker_z=color_values, colorbar=false,
                    xlabel="x (µm)", ylabel="y (µm)", zlabel="z (µm)",
                    title="Cell Cycle at t=$(t)h (n=$(nrow(df3)))",
                    legend=:topright, aspect_ratio=:equal,
                    seriescolor=cgrad(custom_colors, 5, categorical=true),
                    size=(900, 700), camera=(320, 30), clims=(-0.5, 4.5))

        for phase in ["G0", "G1", "S", "G2", "M"]
            n = count(==(String(phase)), df3.cell_cycle)
            scatter!(p, [], [], []; markersize=6, color=ph_colors[phase], label="$phase (n=$n)")
        end
    end

    gif(anim, output_file, fps=fps)
    println("Animation saved to: $output_file")
    return anim
end

#! ============================================================================
#! Snapshot Utility
#! ============================================================================

"""
    create_snapshot(pop::CellPopulation) -> CellPopulation

Returns a new `CellPopulation` containing only alive cells (`is_cell==1`).
All fields are copied; modifications to the result do not affect the original.

# Example
```julia
snap = create_snapshot(pop)
```
"""
function create_snapshot(pop::CellPopulation)::CellPopulation
    alive = findall(pop.is_cell .== 1)
    n     = length(alive)
    return CellPopulation(
        pop.is_cell[alive], pop.can_divide[alive],
        isnothing(pop.is_stem)      ? nothing : pop.is_stem[alive],
        isnothing(pop.is_death_rad) ? nothing : pop.is_death_rad[alive],
        pop.death_time[alive], pop.cycle_time[alive], pop.recover_time[alive],
        pop.cell_cycle[alive], pop.number_nei[alive],
        [copy(pop.nei[i]) for i in alive],
        isnothing(pop.x) ? nothing : pop.x[alive],
        isnothing(pop.y) ? nothing : pop.y[alive],
        Int32(n), Int32(n), pop.indices[alive]
    )
end
