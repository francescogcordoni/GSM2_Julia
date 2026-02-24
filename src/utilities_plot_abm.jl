#! ============================================================================
#! utilities_ABM_plots.jl
#!
#! FUNCTIONS
#! ---------
#~ Population Dynamics (SimulationTimeSeries)
#?   plot_cell_dynamics(ts) -> Plot
#       Total cell count vs time (blue line).
#?   plot_phase_dynamics(ts) -> Plot
#       Absolute counts per phase (G0/G1/S/G2/M) vs time.
#       Colors: G0=black, G1=green, S=orange, G2=purple, M=red.
#?   plot_phase_proportions(ts) -> Plot
#       Phase percentages (0–100%) vs time. Same color scheme.
#?   plot_phase_stacked(ts) -> Plot
#       Stacked area chart of phase counts vs time.
#?   plot_cycling_vs_quiescent(ts) -> Plot
#       Cycling (G1+S+G2+M) vs quiescent (G0) counts vs time.
#?   plot_growth_rate(ts; window_size=10) -> Plot
#       Sliding-window finite-difference growth rate (cells/h) vs time.
#       Returns placeholder if < window_size+1 time points.
#?   plot_phase_duration_distribution(ts) -> Plot
#       Bar chart of mean phase occupancy across the full simulation.
#       Approximate proxy for relative phase durations.
#?   plot_doubling_time(ts) -> Plot
#       Scatter of first-passage doubling events (2×, 4×, 8×, ...) vs time.
#?   plot_stem_dynamics(ts) -> Plot
#       Stem vs non-stem counts vs time. Placeholder if all zeros.
#?   plot_simulation_results(ts) -> Plot
#       3-panel figure: total cells + phases + stem/non-stem.
#?   plot_analysis_dashboard(ts) -> Plot
#       5-panel dashboard (3×2): dynamics + phases + proportions + growth + cycling.
#
#~ Snapshot Comparisons
#?   plot_snapshot_comparison(snapshots; metric, times) -> Plot
#       Grid of per-time bar charts for :cell_cycle or :can_divide.
#       Accepts Dict{Int, DataFrame|CellPopulation}.
#?   plot_phase_comparison_before_after(cell_df_initial, cell_df_final) -> Plot
#       Side-by-side phase proportion bars: initial vs final snapshot.
#?   plot_phase_proportions_timeseries(ts; title_text) -> Plot
#       Phase proportions (%) from SimulationTimeSeries (same as plot_phase_proportions).
#       Kept as separate entry point with optional title override.
#
#~ Spatial Visualization
#?   plot_spatial_distribution(snapshot; color_by, title_text) -> Plot
#       2D scatter of (x, y) colored by :cell_cycle, :can_divide, or plain blue.
#       Accepts DataFrame or CellPopulation.
#?   create_spatial_animation(snapshots; output_file, fps, color_by) -> Animation
#       GIF of 2D spatial distribution over time via plot_spatial_distribution.
#
#~ Diagnostics
#?   print_simulation_summary(ts) -> Nothing
#       Prints population stats, phase distribution, cycling/quiescent fractions,
#       and stem cell info to stdout.
#?   export_timeseries_csv(ts, filename) -> DataFrame
#       Writes all SimulationTimeSeries fields to a CSV file via CSV.write.
#! ============================================================================

"""
    plot_cell_dynamics(ts::SimulationTimeSeries) -> Plot

Total cell count vs time (blue line).

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
                legend=:best, color=:blue)
end

"""
    plot_phase_dynamics(ts::SimulationTimeSeries) -> Plot

Absolute cell counts per phase (G0/G1/S/G2/M) vs time.
G0=black, G1=green, S=orange, G2=purple, M=red.

# Example
```julia
plot_phase_dynamics(ts)
```
"""
function plot_phase_dynamics(ts::SimulationTimeSeries)
    p = plot(xlabel="Time (h)", ylabel="Number of Cells",
                title="Cell Cycle Phase Distribution",
                legend=:best, size=(800, 400))
    plot!(p, ts.time, ts.g0_cells, label="G0", linewidth=2, color=:black)
    plot!(p, ts.time, ts.g1_cells, label="G1", linewidth=2, color=:green)
    plot!(p, ts.time, ts.s_cells,  label="S",  linewidth=2, color=:orange)
    plot!(p, ts.time, ts.g2_cells, label="G2", linewidth=2, color=:purple)
    plot!(p, ts.time, ts.m_cells,  label="M",  linewidth=2, color=:red)
    return p
end

"""
    plot_phase_proportions(ts::SimulationTimeSeries) -> Plot

Phase percentages (0–100%) vs time. Division-by-zero safe.
Same color scheme as plot_phase_dynamics.

# Example
```julia
plot_phase_proportions(ts)
```
"""
function plot_phase_proportions(ts::SimulationTimeSeries)
    total = ts.g0_cells .+ ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells
    pct(v) = ifelse.(total .> 0, 100 .* v ./ total, 0.0)

    p = plot(xlabel="Time (h)", ylabel="Percentage (%)",
                title="Cell Phase Distribution (%)", legend=:best)
    plot!(p, ts.time, pct(ts.g0_cells), label="G0", linewidth=2, color=:black)
    plot!(p, ts.time, pct(ts.g1_cells), label="G1", linewidth=2, color=:green)
    plot!(p, ts.time, pct(ts.s_cells),  label="S",  linewidth=2, color=:orange)
    plot!(p, ts.time, pct(ts.g2_cells), label="G2", linewidth=2, color=:purple)
    plot!(p, ts.time, pct(ts.m_cells),  label="M",  linewidth=2, color=:red)
    return p
end

"""
    plot_phase_stacked(ts::SimulationTimeSeries) -> Plot

Stacked area chart of phase counts (G0/G1/S/G2/M) vs time.

# Example
```julia
plot_phase_stacked(ts)
```
"""
function plot_phase_stacked(ts::SimulationTimeSeries)
    return areaplot(ts.time,
                    [ts.g0_cells ts.g1_cells ts.s_cells ts.g2_cells ts.m_cells];
                    labels=["G0" "G1" "S" "G2" "M"],
                    xlabel="Time (h)", ylabel="Number of Cells",
                    title="Cell Phase Distribution (Stacked)",
                    fillalpha=0.7, linewidth=0)
end

"""
    plot_cycling_vs_quiescent(ts::SimulationTimeSeries) -> Plot

Cycling (G1+S+G2+M, blue) vs quiescent (G0, black) counts vs time.

# Example
```julia
plot_cycling_vs_quiescent(ts)
```
"""
function plot_cycling_vs_quiescent(ts::SimulationTimeSeries)
    cycling = ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells
    p = plot(xlabel="Time (h)", ylabel="Number of Cells",
                title="Cycling vs Quiescent Cells", legend=:best, linewidth=2)
    plot!(p, ts.time, cycling,      label="Cycling (G1/S/G2/M)", color=:blue)
    plot!(p, ts.time, ts.g0_cells,  label="Quiescent (G0)",      color=:black)
    return p
end

"""
    plot_growth_rate(ts::SimulationTimeSeries; window_size=10) -> Plot

Sliding-window finite-difference growth rate (cells/h) vs time.
`growth_rate[i] = Δcells / Δt` over `window_size` samples.
Returns placeholder if fewer than `window_size+1` time points.

# Example
```julia
plot_growth_rate(ts; window_size=12)
```
"""
function plot_growth_rate(ts::SimulationTimeSeries; window_size::Int = 10)
    if length(ts.time) < window_size + 1
        return plot(title="Insufficient data for growth rate", grid=false, showaxis=false)
    end

    rates  = Float64[]
    t_pts  = Float64[]
    for i in window_size+1:length(ts.time)
        dt = ts.time[i] - ts.time[i - window_size]
        dt > 0 || continue
        push!(rates, (ts.total_cells[i] - ts.total_cells[i - window_size]) / dt)
        push!(t_pts, ts.time[i])
    end

    p = plot(t_pts, rates;
                xlabel="Time (h)", ylabel="Growth Rate (cells/h)",
                title="Population Growth Rate",
                label="Growth Rate", linewidth=2, color=:darkblue)
    hline!([0]; linestyle=:dash, color=:black, label="", alpha=0.5)
    return p
end

"""
    plot_phase_duration_distribution(ts::SimulationTimeSeries) -> Plot

Bar chart of mean phase occupancy across the full simulation.
Approximate population-level proxy for relative phase durations.
⚠ Uses mean counts, not single-cell tracking.

# Example
```julia
plot_phase_duration_distribution(ts)
```
"""
function plot_phase_duration_distribution(ts::SimulationTimeSeries)
    avgs   = [mean(ts.g0_cells), mean(ts.g1_cells), mean(ts.s_cells),
                mean(ts.g2_cells), mean(ts.m_cells)]
    p = bar(["G0","G1","S","G2","M"], avgs;
            title="Cell Phase Distribution Over Simulation",
            xlabel="Cell Cycle Phase", ylabel="Average Cell Count",
            legend=false, color=[:black :green :orange :purple :red],
            alpha=0.7, bar_width=0.6)
    return p
end

"""
    plot_doubling_time(ts::SimulationTimeSeries) -> Plot

First-passage times to successive population doublings (2×, 4×, 8×, ...).
Returns placeholder if < 10 time points or no doublings observed.

# Example
```julia
plot_doubling_time(ts)
```
"""
function plot_doubling_time(ts::SimulationTimeSeries)
    length(ts.time) >= 10 || return plot(title="Insufficient data for doubling time",
                                            grid=false, showaxis=false)

    N0       = ts.total_cells[1]
    d_times  = Float64[]
    d_mult   = Int[]
    multiple = 2
    while multiple * N0 <= maximum(ts.total_cells)
        idx = findfirst(x -> x >= multiple * N0, ts.total_cells)
        if !isnothing(idx)
            push!(d_times, ts.time[idx])
            push!(d_mult,  multiple)
        end
        multiple *= 2
    end

    isempty(d_times) && return plot(title="No population doublings observed",
                                    grid=false, showaxis=false)

    return scatter(d_mult, d_times;
                    xlabel="Population Multiple", ylabel="Time (h)",
                    title="Population Doubling Times",
                    label="Doubling Events", markersize=8, color=:blue,
                    xscale=:log2,
                    xticks=(d_mult, string.(d_mult) .* "×"))
end

"""
    plot_stem_dynamics(ts::SimulationTimeSeries) -> Plot

Stem (green) vs non-stem (orange) counts vs time.
Returns placeholder if all values are zero.

# Example
```julia
plot_stem_dynamics(ts)
```
"""
function plot_stem_dynamics(ts::SimulationTimeSeries)
    if all(ts.stem_cells .== 0) && all(ts.non_stem_cells .== 0)
        return plot(title="No stem cell data available", grid=false, showaxis=false)
    end
    p = plot(xlabel="Time (h)", ylabel="Number of Cells",
                title="Stem vs Non-Stem Cell Dynamics", legend=:best, linewidth=2)
    plot!(p, ts.time, ts.stem_cells,     label="Stem Cells",     color=:green)
    plot!(p, ts.time, ts.non_stem_cells, label="Non-Stem Cells", color=:orange)
    return p
end

"""
    plot_simulation_results(ts::SimulationTimeSeries) -> Plot

3-panel figure (3×1): total cells + phase dynamics + stem/non-stem.

# Example
```julia
plot_simulation_results(ts)
```
"""
function plot_simulation_results(ts::SimulationTimeSeries)
    return plot(plot_cell_dynamics(ts), plot_phase_dynamics(ts), plot_stem_dynamics(ts);
                layout=(3, 1), size=(1000, 1200))
end

"""
    plot_analysis_dashboard(ts::SimulationTimeSeries) -> Plot

5-panel dashboard (3×2): total cells, phase dynamics, phase proportions,
growth rate, cycling vs quiescent.

# Example
```julia
plot_analysis_dashboard(ts)
```
"""
function plot_analysis_dashboard(ts::SimulationTimeSeries)
    return plot(plot_cell_dynamics(ts),
                plot_phase_dynamics(ts),
                plot_phase_proportions(ts),
                plot_growth_rate(ts),
                plot_cycling_vs_quiescent(ts);
                layout=(3, 2), size=(1400, 1200),
                plot_title="Simulation Analysis Dashboard")
end

#! ============================================================================
#! Snapshot Comparisons
#! ============================================================================

"""
    plot_snapshot_comparison(snapshots; metric=:cell_cycle, times=nothing) -> Plot

Grid of bar charts (up to 4 per row) comparing snapshots at selected times.
`metric=:cell_cycle` → phase counts. `metric=:can_divide` → can divide vs blocked.
Accepts `Dict{Int, DataFrame|CellPopulation}`.

# Example
```julia
plot_snapshot_comparison(snaps; metric=:cell_cycle, times=[0, 6, 12, 24])
```
"""
function plot_snapshot_comparison(snapshots::Dict;
                                  metric::Symbol                   = :cell_cycle,
                                  times::Union{Nothing,Vector{Int}} = nothing)
    plot_times = isnothing(times) ? sort(collect(keys(snapshots))) : times
    length(plot_times) >= 2 || return plot(title="Insufficient snapshots for comparison")

    plots = []
    for t in plot_times
        haskey(snapshots, t) || (@warn "Snapshot at t=$t not found"; continue)

        data = isa(snapshots[t], DataFrame)     ? snapshots[t] :
               isa(snapshots[t], CellPopulation) ? to_dataframe(snapshots[t], alive_only=true) :
               (@warn "Unknown type: $(typeof(snapshots[t]))"; continue)

        if metric == :cell_cycle
            counts = Dict(p => count(==(String(p)), data.cell_cycle)
                          for p in ["G0","G1","S","G2","M"])
            p = bar(["G0","G1","S","G2","M"],
                    [counts[ph] for ph in ["G0","G1","S","G2","M"]];
                    title="t=$(t)h (n=$(nrow(data)))", ylabel="Count",
                    legend=false, color=[:black :green :orange :purple :red], alpha=0.7)

        elseif metric == :can_divide && hasproperty(data, :can_divide)
            cd = sum(data.can_divide)
            p  = bar(["Can Divide","Blocked"], [cd, nrow(data)-cd];
                     title="t=$(t)h (n=$(nrow(data)))", ylabel="Count",
                     legend=false, color=[:green :red], alpha=0.7)
        else
            @warn "Unknown or inapplicable metric: $metric"; continue
        end
        push!(plots, p)
    end

    isempty(plots) && return plot(title="No valid plots created")
    n_cols = min(4, length(plots))
    n_rows = ceil(Int, length(plots) / n_cols)
    return plot(plots...; layout=(n_rows, n_cols), size=(300*n_cols, 250*n_rows))
end

"""
    plot_phase_comparison_before_after(cell_df_initial, cell_df_final) -> Plot

Side-by-side phase proportion bars: initial (left) vs final (right) snapshot.

# Example
```julia
plot_phase_comparison_before_after(df_start, df_end)
```
"""
function plot_phase_comparison_before_after(cell_df_initial::DataFrame,
                                            cell_df_final::DataFrame)
    p1 = plot_phase_proportions_alive(cell_df_initial; title_text="Initial Distribution")
    p2 = plot_phase_proportions_alive(cell_df_final;   title_text="Final Distribution")
    return plot(p1, p2; layout=(1, 2), size=(1000, 400))
end

"""
    plot_phase_proportions_timeseries(ts; title_text="Cell Cycle Distribution Over Time")
        -> Plot

Phase percentages (0–100%) from SimulationTimeSeries vs time.
Same computation as `plot_phase_proportions`; exposed separately for title override.

# Example
```julia
plot_phase_proportions_timeseries(ts; title_text="My Sim")
```
"""
function plot_phase_proportions_timeseries(ts::SimulationTimeSeries;
                                           title_text::String = "Cell Cycle Distribution Over Time")
    total = ts.g0_cells .+ ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells
    pct(v) = ifelse.(total .> 0, 100 .* v ./ total, 0.0)

    p = plot(xlabel="Time (h)", ylabel="Percentage (%)",
             title=title_text, legend=:best, ylims=(0, 100))
    plot!(p, ts.time, pct(ts.g0_cells), label="G0", linewidth=2, color=:black)
    plot!(p, ts.time, pct(ts.g1_cells), label="G1", linewidth=2, color=:green)
    plot!(p, ts.time, pct(ts.s_cells),  label="S",  linewidth=2, color=:orange)
    plot!(p, ts.time, pct(ts.g2_cells), label="G2", linewidth=2, color=:purple)
    plot!(p, ts.time, pct(ts.m_cells),  label="M",  linewidth=2, color=:red)
    return p
end

#! ============================================================================
#! Spatial Visualization
#! ============================================================================

"""
    plot_spatial_distribution(snapshot; color_by=:cell_cycle, title_text="Spatial Distribution")
        -> Plot

2D scatter of (x, y) colored by `:cell_cycle` (fixed phase palette + legend),
`:can_divide` (green/red), or plain blue if column absent.
Accepts `DataFrame` or `CellPopulation`.

# Example
```julia
plot_spatial_distribution(df; color_by=:cell_cycle, title_text="t = 12h")
plot_spatial_distribution(pop; color_by=:can_divide)
```
"""
function plot_spatial_distribution(snapshot;
                                   color_by::Symbol   = :cell_cycle,
                                   title_text::String = "Spatial Distribution")
    if isa(snapshot, CellPopulation)
        (isnothing(snapshot.x) || isnothing(snapshot.y)) &&
            return plot(title="No spatial coordinates available", grid=false, showaxis=false)
        data = to_dataframe(snapshot, alive_only=true)
    elseif isa(snapshot, DataFrame)
        data = snapshot
    else
        @warn "Unknown snapshot type"; return plot()
    end

    (hasproperty(data, :x) && hasproperty(data, :y)) ||
        return plot(title="No spatial coordinates in data", grid=false, showaxis=false)

    ph_colors = Dict("G0"=>:black, "G1"=>:green, "S"=>:orange, "G2"=>:purple, "M"=>:red)

    if color_by == :cell_cycle && hasproperty(data, :cell_cycle)
        colors = [get(ph_colors, string(ph), :black) for ph in data.cell_cycle]
        p = scatter(data.x, data.y; color=colors,
                    xlabel="X Position", ylabel="Y Position", title=title_text,
                    label="", markersize=6, markerstrokewidth=0, aspect_ratio=:equal)
        for (ph, c) in ph_colors
            scatter!(p, [], []; color=c, label=ph, markersize=6)
        end

    elseif color_by == :can_divide && hasproperty(data, :can_divide)
        colors = [d == 1 ? :green : :red for d in data.can_divide]
        p = scatter(data.x, data.y; color=colors,
                    xlabel="X Position", ylabel="Y Position", title=title_text,
                    markersize=6, markerstrokewidth=0, aspect_ratio=:equal)
        scatter!(p, [], []; color=:green, label="Can Divide", markersize=6)
        scatter!(p, [], []; color=:red,   label="Blocked",    markersize=6)

    else
        p = scatter(data.x, data.y;
                    xlabel="X Position", ylabel="Y Position", title=title_text,
                    label="Cells", markersize=6, color=:blue,
                    markerstrokewidth=0, aspect_ratio=:equal)
    end
    return p
end

"""
    create_spatial_animation(snapshots; output_file="simulation.gif", fps=2, color_by=:cell_cycle)
        -> Animation

GIF of 2D spatial distribution over time via `plot_spatial_distribution`.
Requires ≥ 2 snapshots. Accepts `Dict{Int, DataFrame|CellPopulation}`.

# Example
```julia
anim = create_spatial_animation(snaps; output_file="spatial.gif", fps=3)
```
"""
function create_spatial_animation(snapshots::Dict;
                                  output_file::String = "simulation.gif",
                                  fps::Int            = 2,
                                  color_by::Symbol    = :cell_cycle)
    times = sort(collect(keys(snapshots)))
    length(times) >= 2 || (@warn "Need at least 2 snapshots for animation"; return nothing)

    anim = @animate for t in times
        plot_spatial_distribution(snapshots[t];
                                  color_by=color_by,
                                  title_text="Cell Distribution at t=$(t)h")
    end

    gif(anim, output_file, fps=fps)
    println("Animation saved to: $output_file")
    return anim
end

#! ============================================================================
#! Diagnostics
#! ============================================================================

"""
    print_simulation_summary(ts::SimulationTimeSeries) -> Nothing

Prints to stdout: time range, population stats (initial/final/peak/mean/fold-change),
average phase distribution (%), cycling vs quiescent fractions, and stem cell info
if available.

# Example
```julia
print_simulation_summary(ts)
```
"""
function print_simulation_summary(ts::SimulationTimeSeries)
    println("\n" * "="^70)
    println("SIMULATION SUMMARY STATISTICS")
    println("="^70)

    println("Time Range:  $(round(ts.time[1], digits=2))h – $(round(ts.time[end], digits=2))h")
    println("Time Points: $(length(ts.time))")

    println("\nPopulation Statistics:")
    println("  Initial : $(ts.total_cells[1])")
    println("  Final   : $(ts.total_cells[end])")
    println("  Peak    : $(maximum(ts.total_cells))")
    println("  Min     : $(minimum(ts.total_cells))")
    println("  Average : $(round(mean(ts.total_cells), digits=1))")

    if ts.total_cells[end] > ts.total_cells[1]
        println("  Fold change  : $(round(ts.total_cells[end] / ts.total_cells[1], digits=2))×")
    else
        pct = 100 * (1 - ts.total_cells[end] / ts.total_cells[1])
        println("  Decline      : $(round(pct, digits=1))%")
    end

    println("\nAverage Phase Distribution:")
    total_avg = mean(ts.g0_cells .+ ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells)
    if total_avg > 0
        for (label, vec) in [("G0", ts.g0_cells), ("G1", ts.g1_cells), ("S ", ts.s_cells),
                              ("G2", ts.g2_cells), ("M ", ts.m_cells)]
            println("  $label: $(round(100*mean(vec)/total_avg, digits=1))%")
        end
    end

    println("\nCycling vs Quiescent:")
    avg_cyc  = mean(ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells)
    avg_qui  = mean(ts.g0_cells)
    tot      = avg_cyc + avg_qui
    if tot > 0
        println("  Cycling (G1/S/G2/M) : $(round(100*avg_cyc/tot, digits=1))%")
        println("  Quiescent (G0)      : $(round(100*avg_qui/tot, digits=1))%")
    end

    if any(ts.stem_cells .> 0)
        println("\nStem Cell Statistics:")
        println("  Initial : $(ts.stem_cells[1])")
        println("  Final   : $(ts.stem_cells[end])")
        println("  Avg fraction : $(round(100*mean(ts.stem_cells ./ ts.total_cells), digits=1))%")
    end

    println("="^70 * "\n")
end

"""
    export_timeseries_csv(ts::SimulationTimeSeries, filename::String) -> DataFrame

Writes all SimulationTimeSeries fields to a CSV file and prints the path.
Returns the DataFrame that was written.

# Example
```julia
df = export_timeseries_csv(ts, "results.csv")
```
"""
function export_timeseries_csv(ts::SimulationTimeSeries, filename::String)
    df = DataFrame(
        time           = ts.time,
        total_cells    = ts.total_cells,
        g0_cells       = ts.g0_cells,
        g1_cells       = ts.g1_cells,
        s_cells        = ts.s_cells,
        g2_cells       = ts.g2_cells,
        m_cells        = ts.m_cells,
        stem_cells     = ts.stem_cells,
        non_stem_cells = ts.non_stem_cells
    )
    CSV.write(filename, df)
    println("Time series exported to: $filename")
    return df
end
