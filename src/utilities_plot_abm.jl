"""
Plot total cell count over time
"""
function plot_cell_dynamics(ts::SimulationTimeSeries)
    p = plot(ts.time, ts.total_cells,
                xlabel="Time (h)",
                ylabel="Number of Cells",
                title="Total Cell Population Dynamics",
                label="Total Cells",
                linewidth=2,
                legend=:best,
                color=:blue)
    return p
end

"""
Plot cell cycle phase distribution over time
"""
function plot_phase_dynamics(ts::SimulationTimeSeries)
    p = plot(xlabel="Time (h)",
                ylabel="Number of Cells",
                title="Cell Cycle Phase Distribution",
                legend=:best,
                size=(800, 400))
    
    plot!(p, ts.time, ts.g0_cells, label="G0", linewidth=2, color=:black)
    plot!(p, ts.time, ts.g1_cells, label="G1", linewidth=2, color=:green)
    plot!(p, ts.time, ts.s_cells, label="S", linewidth=2, color=:orange)
    plot!(p, ts.time, ts.g2_cells, label="G2", linewidth=2, color=:purple)
    plot!(p, ts.time, ts.m_cells, label="M", linewidth=2, color=:red)
    
    return p
end

"""
Plot stem vs non-stem cell dynamics
"""
function plot_stem_dynamics(ts::SimulationTimeSeries)
    if all(ts.stem_cells .== 0) && all(ts.non_stem_cells .== 0)
        return plot(title="No stem cell data available",
                    grid=false,
                    showaxis=false)
    end
    
    p = plot(xlabel="Time (h)",
                ylabel="Number of Cells",
                title="Stem vs Non-Stem Cell Dynamics",
                legend=:best,
                linewidth=2)
    
    plot!(p, ts.time, ts.stem_cells, label="Stem Cells", color=:green)
    plot!(p, ts.time, ts.non_stem_cells, label="Non-Stem Cells", color=:orange)
    
    return p
end

"""
Create comprehensive visualization of simulation results
"""
function plot_simulation_results(ts::SimulationTimeSeries)
    p1 = plot_cell_dynamics(ts)
    p2 = plot_phase_dynamics(ts)
    p3 = plot_stem_dynamics(ts)
    
    return plot(p1, p2, p3, layout=(3, 1), size=(1000, 1200))
end

# ============================================================================
# Additional Plotting Functions
# ============================================================================

"""
Plot stacked area chart of cell phases over time
"""
function plot_phase_stacked(ts::SimulationTimeSeries)
    p = areaplot(ts.time, 
                [ts.g0_cells ts.g1_cells ts.s_cells ts.g2_cells ts.m_cells],
                labels=["G0" "G1" "S" "G2" "M"],
                xlabel="Time (h)",
                ylabel="Number of Cells",
                title="Cell Phase Distribution (Stacked)",
                fillalpha=0.7,
                linewidth=0)
    return p
end

"""
Plot phase proportions (percentages) over time
"""
function plot_phase_proportions(ts::SimulationTimeSeries)
    # Calculate proportions
    total = ts.g0_cells .+ ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells
    
    # Handle division by zero
    g0_prop = ifelse.(total .> 0, 100 .* ts.g0_cells ./ total, 0.0)
    g1_prop = ifelse.(total .> 0, 100 .* ts.g1_cells ./ total, 0.0)
    s_prop = ifelse.(total .> 0, 100 .* ts.s_cells ./ total, 0.0)
    g2_prop = ifelse.(total .> 0, 100 .* ts.g2_cells ./ total, 0.0)
    m_prop = ifelse.(total .> 0, 100 .* ts.m_cells ./ total, 0.0)
    
    p = plot(xlabel="Time (h)",
                ylabel="Percentage (%)",
                title="Cell Phase Distribution (%)",
                legend=:best)
    
    plot!(p, ts.time, g0_prop, label="G0", linewidth=2, color=:black)
    plot!(p, ts.time, g1_prop, label="G1", linewidth=2, color=:green)
    plot!(p, ts.time, s_prop, label="S", linewidth=2, color=:orange)
    plot!(p, ts.time, g2_prop, label="G2", linewidth=2, color=:purple)
    plot!(p, ts.time, m_prop, label="M", linewidth=2, color=:red)
    
    return p
end

"""
Plot cycling vs quiescent cells over time
"""
function plot_cycling_vs_quiescent(ts::SimulationTimeSeries)
    cycling = ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells
    quiescent = ts.g0_cells
    
    p = plot(xlabel="Time (h)",
                ylabel="Number of Cells",
                title="Cycling vs Quiescent Cells",
                legend=:best,
                linewidth=2)
    
    plot!(p, ts.time, cycling, label="Cycling (G1/S/G2/M)", color=:blue)
    plot!(p, ts.time, quiescent, label="Quiescent (G0)", color=:black)
    
    return p
end

"""
Plot growth rate over time (derivative of cell count)
"""
function plot_growth_rate(ts::SimulationTimeSeries; window_size::Int=10)
    if length(ts.time) < window_size + 1
        return plot(title="Insufficient data for growth rate",
                    grid=false,
                    showaxis=false)
    end
    
    # Calculate growth rate using sliding window
    growth_rates = Float64[]
    time_points = Float64[]
    
    for i in window_size+1:length(ts.time)
        dt = ts.time[i] - ts.time[i-window_size]
        dcells = ts.total_cells[i] - ts.total_cells[i-window_size]
        
        if dt > 0
            growth_rate = dcells / dt  # cells per hour
            push!(growth_rates, growth_rate)
            push!(time_points, ts.time[i])
        end
    end
    
    p = plot(time_points, growth_rates,
                xlabel="Time (h)",
                ylabel="Growth Rate (cells/h)",
                title="Population Growth Rate",
                label="Growth Rate",
                linewidth=2,
                color=:darkblue)
    
    hline!([0], linestyle=:dash, color=:black, label="", alpha=0.5)
    
    return p
end

"""
Plot cell cycle phase duration histogram (from time series)
Estimates time spent in each phase
"""
function plot_phase_duration_distribution(ts::SimulationTimeSeries)
    # This is approximate - tracking individual cells would be better
    # But we can estimate from population-level data
    
    p = plot(title="Cell Phase Distribution Over Simulation",
                xlabel="Cell Cycle Phase",
                ylabel="Average Cell Count",
                legend=false,
                bar_width=0.6)
    
    # Calculate average cells in each phase
    avg_g0 = mean(ts.g0_cells)
    avg_g1 = mean(ts.g1_cells)
    avg_s = mean(ts.s_cells)
    avg_g2 = mean(ts.g2_cells)
    avg_m = mean(ts.m_cells)
    
    bar!(["G0", "G1", "S", "G2", "M"], 
            [avg_g0, avg_g1, avg_s, avg_g2, avg_m],
            color=[:black :green :orange :purple :red],
            alpha=0.7)
    
    return p
end

"""
Plot population doubling time analysis
"""
function plot_doubling_time(ts::SimulationTimeSeries)
    if length(ts.time) < 10
        return plot(title="Insufficient data for doubling time",
                    grid=false,
                    showaxis=false)
    end
    
    # Find times when population doubles from initial
    initial_pop = ts.total_cells[1]
    doubling_times = Float64[]
    doubling_multiples = Int[]
    
    multiple = 2
    while multiple * initial_pop <= maximum(ts.total_cells)
        target = multiple * initial_pop
        # Find first time >= target
        idx = findfirst(x -> x >= target, ts.total_cells)
        if !isnothing(idx)
            push!(doubling_times, ts.time[idx])
            push!(doubling_multiples, multiple)
        end
        multiple *= 2
    end
    
    if isempty(doubling_times)
        return plot(title="No population doublings observed",
                    grid=false,
                    showaxis=false)
    end
    
    p = scatter(doubling_multiples, doubling_times,
                xlabel="Population Multiple",
                ylabel="Time (h)",
                title="Population Doubling Times",
                label="Doubling Events",
                markersize=8,
                color=:blue,
                xscale=:log2,
                xticks=(doubling_multiples, string.(doubling_multiples) .* "×"))
    
    return p
end

"""
Create comprehensive analysis dashboard
"""
function plot_analysis_dashboard(ts::SimulationTimeSeries)
    p1 = plot_cell_dynamics(ts)
    p2 = plot_phase_dynamics(ts)
    p3 = plot_phase_proportions(ts)
    p4 = plot_growth_rate(ts)
    p5 = plot_cycling_vs_quiescent(ts)
    
    return plot(p1, p2, p3, p4, p5,
                layout=(3, 2), 
                size=(1400, 1200),
                plot_title="Simulation Analysis Dashboard")
end

# ============================================================================
# Snapshot Visualization Functions
# ============================================================================

"""
Plot comparison of snapshots at different time points
Works with both DataFrame and CellPopulation snapshots
"""
function plot_snapshot_comparison(snapshots::Dict; 
                                    metric::Symbol=:cell_cycle,
                                    times::Union{Nothing, Vector{Int}}=nothing)
    
    # Determine which times to plot
    plot_times = isnothing(times) ? sort(collect(keys(snapshots))) : times
    
    if length(plot_times) < 2
        @warn "Need at least 2 snapshots for comparison"
        return plot(title="Insufficient snapshots for comparison")
    end
    
    plots = []
    
    for t in plot_times
        if !haskey(snapshots, t)
            @warn "Snapshot at t=$t not found"
            continue
        end
        
        snapshot = snapshots[t]
        
        # Handle both DataFrame and CellPopulation
        if isa(snapshot, DataFrame)
            data = snapshot
        elseif isa(snapshot, CellPopulation)
            data = to_dataframe(snapshot, alive_only=true)
        else
            @warn "Unknown snapshot type: $(typeof(snapshot))"
            continue
        end
        
        # Create plot based on metric
        if metric == :cell_cycle
            phases = data.cell_cycle
            phase_counts = Dict("G0"=>0, "G1"=>0, "S"=>0, "G2"=>0, "M"=>0)
            for phase in phases
                if haskey(phase_counts, string(phase))
                    phase_counts[string(phase)] += 1
                end
            end
            
            p = bar(["G0", "G1", "S", "G2", "M"], 
                    [phase_counts["G0"], phase_counts["G1"], 
                        phase_counts["S"], phase_counts["G2"], phase_counts["M"]],
                    title="t = $(t)h (n=$(nrow(data)))",
                    ylabel="Count",
                    legend=false,
                    color=[:black :green :orange :purple :red],
                    alpha=0.7)
            
        elseif metric == :can_divide && hasproperty(data, :can_divide)
            can_divide_count = sum(data.can_divide)
            cannot_divide_count = nrow(data) - can_divide_count
            
            p = bar(["Can Divide", "Blocked"],
                    [can_divide_count, cannot_divide_count],
                    title="t = $(t)h (n=$(nrow(data)))",
                    ylabel="Count",
                    legend=false,
                    color=[:green :red],
                    alpha=0.7)
        else
            @warn "Unknown metric: $metric"
            continue
        end
        
        push!(plots, p)
    end
    
    if isempty(plots)
        return plot(title="No valid plots created")
    end
    
    # Arrange plots in grid
    n_plots = length(plots)
    n_cols = min(4, n_plots)
    n_rows = ceil(Int, n_plots / n_cols)
    
    return plot(plots..., 
                layout=(n_rows, n_cols), 
                size=(300*n_cols, 250*n_rows))
end

"""
Plot spatial distribution if x, y coordinates are available
"""
function plot_spatial_distribution(snapshot; 
                                    color_by::Symbol=:cell_cycle,
                                    title_text::String="Spatial Distribution")
    
    # Handle both DataFrame and CellPopulation
    if isa(snapshot, DataFrame)
        data = snapshot
    elseif isa(snapshot, CellPopulation)
        if isnothing(snapshot.x) || isnothing(snapshot.y)
            return plot(title="No spatial coordinates available",
                        grid=false,
                        showaxis=false)
        end
        data = to_dataframe(snapshot, alive_only=true)
    else
        @warn "Unknown snapshot type"
        return plot()
    end
    
    if !hasproperty(data, :x) || !hasproperty(data, :y)
        return plot(title="No spatial coordinates in data",
                    grid=false,
                    showaxis=false)
    end
    
    # Create scatter plot colored by specified attribute
    if color_by == :cell_cycle && hasproperty(data, :cell_cycle)
        phase_colors = Dict(
            "G0" => :black,
            "G1" => :green,
            "S" => :orange,
            "G2" => :purple,
            "M" => :red
        )
        
        colors = [get(phase_colors, string(phase), :black) for phase in data.cell_cycle]
        
        p = scatter(data.x, data.y,
                    color=colors,
                    xlabel="X Position",
                    ylabel="Y Position",
                    title=title_text,
                    label="",
                    markersize=6,
                    markerstrokewidth=0,
                    aspect_ratio=:equal)
        
        # Add legend manually
        scatter!([], [], color=:black, label="G0", markersize=6)
        scatter!([], [], color=:green, label="G1", markersize=6)
        scatter!([], [], color=:orange, label="S", markersize=6)
        scatter!([], [], color=:purple, label="G2", markersize=6)
        scatter!([], [], color=:red, label="M", markersize=6)
        
    elseif color_by == :can_divide && hasproperty(data, :can_divide)
        colors = [d == 1 ? :green : :red for d in data.can_divide]
        
        p = scatter(data.x, data.y,
                    color=colors,
                    xlabel="X Position",
                    ylabel="Y Position",
                    title=title_text,
                    markersize=6,
                    markerstrokewidth=0,
                    aspect_ratio=:equal)
        
        scatter!([], [], color=:green, label="Can Divide", markersize=6)
        scatter!([], [], color=:red, label="Blocked", markersize=6)
        
    else
        p = scatter(data.x, data.y,
                    xlabel="X Position",
                    ylabel="Y Position",
                    title=title_text,
                    label="Cells",
                    markersize=6,
                    color=:blue,
                    markerstrokewidth=0,
                    aspect_ratio=:equal)
    end
    
    return p
end

"""
Create animation of spatial distribution over time
"""
function create_spatial_animation(snapshots::Dict; 
                                    output_file::String="simulation.gif",
                                    fps::Int=2,
                                    color_by::Symbol=:cell_cycle)
    
    times = sort(collect(keys(snapshots)))
    
    if length(times) < 2
        @warn "Need at least 2 snapshots for animation"
        return nothing
    end
    
    anim = @animate for t in times
        plot_spatial_distribution(snapshots[t], 
                                    color_by=color_by,
                                    title_text="Cell Distribution at t = $(t)h")
    end
    
    gif(anim, output_file, fps=fps)
    println("Animation saved to: $output_file")
    
    return anim
end

# ============================================================================
# Statistical Summary Functions
# ============================================================================

"""
Print statistical summary of simulation results
"""
function print_simulation_summary(ts::SimulationTimeSeries)
    println("\n" * "="^70)
    println("SIMULATION SUMMARY STATISTICS")
    println("="^70)
    
    # Time range
    println("Time Range: $(round(ts.time[1], digits=2))h - $(round(ts.time[end], digits=2))h")
    println("Time Points: $(length(ts.time))")
    
    # Population statistics
    println("\nPopulation Statistics:")
    println("  Initial cells: $(ts.total_cells[1])")
    println("  Final cells: $(ts.total_cells[end])")
    println("  Peak cells: $(maximum(ts.total_cells))")
    println("  Min cells: $(minimum(ts.total_cells))")
    println("  Average cells: $(round(mean(ts.total_cells), digits=1))")
    
    # Growth statistics
    if ts.total_cells[end] > ts.total_cells[1]
        fold_change = ts.total_cells[end] / ts.total_cells[1]
        println("  Fold change: $(round(fold_change, digits=2))×")
    else
        percent_decline = 100 * (1 - ts.total_cells[end] / ts.total_cells[1])
        println("  Population decline: $(round(percent_decline, digits=1))%")
    end
    
    # Phase distribution (average)
    println("\nAverage Phase Distribution:")
    total_phase_cells = mean(ts.g0_cells .+ ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells)
    if total_phase_cells > 0
        println("  G0: $(round(100*mean(ts.g0_cells)/total_phase_cells, digits=1))%")
        println("  G1: $(round(100*mean(ts.g1_cells)/total_phase_cells, digits=1))%")
        println("  S:  $(round(100*mean(ts.s_cells)/total_phase_cells, digits=1))%")
        println("  G2: $(round(100*mean(ts.g2_cells)/total_phase_cells, digits=1))%")
        println("  M:  $(round(100*mean(ts.m_cells)/total_phase_cells, digits=1))%")
    end
    
    # Cycling vs Quiescent
    println("\nCycling vs Quiescent:")
    avg_cycling = mean(ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells)
    avg_quiescent = mean(ts.g0_cells)
    total_avg = avg_cycling + avg_quiescent
    if total_avg > 0
        println("  Cycling (G1/S/G2/M): $(round(100*avg_cycling/total_avg, digits=1))%")
        println("  Quiescent (G0):      $(round(100*avg_quiescent/total_avg, digits=1))%")
    end
    
    # Stem cell statistics (if available)
    if any(ts.stem_cells .> 0)
        println("\nStem Cell Statistics:")
        println("  Initial stem cells: $(ts.stem_cells[1])")
        println("  Final stem cells: $(ts.stem_cells[end])")
        println("  Average stem fraction: $(round(100*mean(ts.stem_cells./ts.total_cells), digits=1))%")
    end
    
    println("="^70 * "\n")
end

"""
Export time series data to CSV
"""
function export_timeseries_csv(ts::SimulationTimeSeries, filename::String)
    
    df = DataFrame(
        time = ts.time,
        total_cells = ts.total_cells,
        g0_cells = ts.g0_cells,
        g1_cells = ts.g1_cells,
        s_cells = ts.s_cells,
        g2_cells = ts.g2_cells,
        m_cells = ts.m_cells,
        stem_cells = ts.stem_cells,
        non_stem_cells = ts.non_stem_cells
    )
    
    CSV.write(filename, df)
    println("Time series data exported to: $filename")
    
    return df
end