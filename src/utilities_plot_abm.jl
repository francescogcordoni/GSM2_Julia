"""
    plot_cell_dynamics(ts::SimulationTimeSeries) -> Plot

Plot the **total number of alive cells** over time.  
This is the simplest and most direct visualization of population‑level behavior
in the simulation.

# Arguments
- `ts::SimulationTimeSeries`: A time‑series container providing:
    - `time` — vector of time points (e.g., hours)
    - `total_cells` — vector of alive‑cell counts at each time point  
    These vectors must have equal length.

# Behavior
- Produces a line plot of `total_cells` vs. `time`.
- Adds axis labels, a descriptive title, and a legend.
- Uses a blue line (`linewidth=2`) for clear visibility.

# Returns
- A `Plot` object (from `Plots.jl`) showing the evolution of the total cell
    population across the simulation.

# Notes
- This plot is typically the first panel in summary dashboards such as
    `plot_simulation_results` and `plot_analysis_dashboard`.
- If you want to highlight exponential or logistic regions, consider adding
    optional log‑scaling or growth‑rate overlays.

# Example
```julia
p = plot_cell_dynamics(ts)
display(p)
```
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
    plot_phase_dynamics(ts::SimulationTimeSeries) -> Plot

Plot the **absolute number of cells in each cell‑cycle phase** (G0, G1, S, G2, M)
over time. This visualization shows how the population shifts between phases
throughout the simulation.

# Arguments
- `ts::SimulationTimeSeries`: A time‑series container providing:
    - `time` — vector of time points (e.g., hours)
    - `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells` — vectors of phase counts  
    All vectors must have the same length.

# Behavior
- Creates a line plot with:
  - **G0** → black  
  - **G1** → green  
  - **S**  → orange  
  - **G2** → purple  
  - **M**  → red  
- Shows how the absolute number of cells in each phase evolves over time.
- Adds a legend, axis labels, and a descriptive title.

# Returns
- A `Plot` object (from `Plots.jl`) displaying the per‑phase dynamics.

# Notes
- For percentage‑based trends, use `plot_phase_proportions(ts)` instead.
- This plot is often included in dashboards such as `plot_simulation_results`
    and `plot_analysis_dashboard`.

# Example
```julia
p = plot_phase_dynamics(ts)
display(p)
savefig(p, "phase_dynamics.png")
```
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
    plot_stem_dynamics(ts::SimulationTimeSeries) -> Plot

Plot the time evolution of **stem** and **non‑stem** cell populations from a
simulation time series. This visualization highlights how stem cells and the
rest of the population grow, decline, or diverge over the course of the
simulation.

# Arguments
- `ts::SimulationTimeSeries`: Must contain the time‑aligned vectors:
    - `time` — sampling times (e.g., hours)
    - `stem_cells` — number of stem‑like cells at each time point
    - `non_stem_cells` — number of non‑stem cells at each time point

# Behavior
- If both `stem_cells` and `non_stem_cells` contain only zeros, the function
    returns a placeholder plot with the title `"No stem cell data available"`.
- Otherwise:
  - Plots **stem cells** over time (green line)
  - Plots **non‑stem cells** over time (orange line)
    - Adds descriptive axis labels, a legend, and a title:
    `"Stem vs Non-Stem Cell Dynamics"`

# Returns
- A `Plot` object (from `Plots.jl`) showing stem vs non‑stem cell trajectories.
- If no stem information exists, returns a placeholder empty plot.

# Notes
- This function assumes the fields `stem_cells` and `non_stem_cells` are present.
    If your simulation does not support stem tracking, consider extending the
    `SimulationTimeSeries` type or adding safe fallbacks.
- Works well as part of dashboards such as `plot_simulation_results`.

# Example
```julia
p = plot_stem_dynamics(ts)
display(p)
```
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
    plot_simulation_results(ts::SimulationTimeSeries) -> Plot

Create a compact, vertically stacked summary of the key simulation results,
combining:

1. **Cell dynamics** over time (`plot_cell_dynamics`)
2. **Cell‑cycle phase dynamics** (`plot_phase_dynamics`)
3. **Stem‑cell dynamics** (`plot_stem_dynamics`)

This function acts as a high‑level visualization wrapper, suitable for quick
inspection of a simulation run.

# Arguments
- `ts::SimulationTimeSeries`: Time‑series container containing the fields required by
    the three helper functions:
    - `time`
    - `total_cells`
    - `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells`
    - `stem_cells` (if available; `plot_stem_dynamics` should handle missing data gracefully)

# Behavior
- Calls the helper plotting functions:
    - `plot_cell_dynamics(ts)`
    - `plot_phase_dynamics(ts)`
    - `plot_stem_dynamics(ts)`
- Arranges them in a **3×1 vertical layout**.
- Sets the figure size to `(1000, 1200)` for a readable dashboard‑style output.

# Returns
- A `Plot` object (from `Plots.jl`) containing all three subplots stacked vertically.

# Notes
- Acts as a lightweight version of a full dashboard (e.g., `plot_analysis_dashboard`).
- If any underlying helper function returns `nothing` or a placeholder plot,
    the layout will still be created with the available outputs.
- Useful for quick checks during model development or debugging.

# Example
```julia
ts, snapshots = run_simulation_abm!(pop, 0.01; terminal_time=72)
p = plot_simulation_results(ts)
display(p)
```
"""
function plot_simulation_results(ts::SimulationTimeSeries)
    p1 = plot_cell_dynamics(ts)
    p2 = plot_phase_dynamics(ts)
    p3 = plot_stem_dynamics(ts)
    
    return plot(p1, p2, p3, layout=(3, 1), size=(1000, 1200))
end

"""
    plot_phase_stacked(ts::SimulationTimeSeries) -> Plot

Plot the **stacked area chart** of cell‑cycle phase counts over time.  
This visualization shows how the *absolute number* of cells in each phase
(G0, G1, S, G2, M) evolves, and how their contributions stack to form the
total population size.

The stacked representation makes it easy to see:
- shifts in dominance between phases,
- changes in proliferation vs quiescence,
- treatment‑induced dynamics,
- total population trends.

# Arguments
- `ts::SimulationTimeSeries`: Must contain time‑aligned vectors:
    - `time`
    - `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells`

All vectors must have equal length.

# Behavior
- Creates an area plot using:
areaplot(ts.time,
[ts.g0_cells ts.g1_cells ts.s_cells ts.g2_cells ts.m_cells],
labels=["G0" "G1" "S" "G2" "M"])
- Each phase is stacked on top of the previous, forming a cumulative total.
- Uses a semi‑transparent fill (`fillalpha=0.7`) and no borders (`linewidth=0`)
for a clean stacked‑area visualization.

# Returns
- A `Plot` object showing stacked phase occupancy vs. time.
Useful for:
- visualizing absolute phase transitions,
- correlating total population growth with phase composition,
- high‑level summaries in dashboards.

# Example
```julia
p = plot_phase_stacked(ts)
display(p)
```
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
    plot_phase_proportions(ts::SimulationTimeSeries) -> Plot

Plot the **percentage** of cells in each cell‑cycle phase (G0, G1, S, G2, M)
over time. The figure shows how the population redistributes among phases
throughout the simulation, normalized to 0–100% at each time point.

# Arguments
- `ts::SimulationTimeSeries`: Time‑series container with fields:
    - `time`
    - `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells`  
    (vectors of equal length giving the number of cells in each phase).

# Behavior
- Computes the total number of phase‑annotated cells at each time:
total = g0 + g1 + s + g2 + m
- Computes per‑phase percentages using:
The `ifelse.(...)` form avoids division-by-zero errors for empty populations.
- Plots the resulting trajectories:
- **G0** → black  
- **G1** → green  
- **S**  → orange  
- **G2** → purple  
- **M**  → red
- Adds axis labels, percentage scaling, a descriptive title, and a legend.

# Returns
- A `Plot` object (from `Plots.jl`) showing time vs. per‑phase percentage.
Useful for identifying:
- transitions to quiescence (higher G0),
- proliferation waves (higher G1/S/G2/M),
- treatment‑induced shifts in phase composition.

# Notes
- This plot complements `plot_phase_dynamics`, which displays **absolute** counts.
- For smoother trajectories, you can apply moving averages to the phase vectors
before plotting.

# Example
```julia
p = plot_phase_proportions(ts)
display(p)
```
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
    plot_cycling_vs_quiescent(ts::SimulationTimeSeries) -> Plot

Plot the time evolution of **cycling** versus **quiescent** cells from a
simulation time series.

- **Cycling** is computed as the sum of G1, S, G2, and M phase counts:
    `g1_cells + s_cells + g2_cells + m_cells`.
- **Quiescent** is taken as the G0 phase count: `g0_cells`.

# Arguments
- `ts::SimulationTimeSeries`: Time series with at least the fields:
    - `time`: vector of time points (e.g., hours),
    - `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells`: vectors of per-time
    phase counts (same length as `time`).

# Behavior
- Computes two series:
    - `cycling = g1_cells + s_cells + g2_cells + m_cells`,
    - `quiescent = g0_cells`.
- Produces a line plot of both series against `time` with labels:
    - `"Cycling (G1/S/G2/M)"` (blue)
    - `"Quiescent (G0)"` (black)

# Returns
- A `Plot` object (from `Plots.jl`) with time on the x‑axis and cell counts on
    the y‑axis, showing cycling vs quiescent populations over time.

# Notes
- This function assumes all vectors in `ts` have equal length and aligned time points.
- For percentage trends instead of raw counts, you could normalize each series by
    `ts.total_cells` before plotting.

# Example
```julia
p = plot_cycling_vs_quiescent(ts)
display(p)
```
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
    plot_growth_rate(ts::SimulationTimeSeries; window_size::Int = 10) -> Plot

Compute and plot the **population growth rate** (cells per hour) over time using a
sliding-window finite difference on total cell counts.

For each index `i = window_size+1 : end`, this function computes:
`growth_rate[i] = (total_cells[i] - total_cells[i - window_size]) / (time[i] - time[i - window_size])`,
and plots the result against `time[i]`.

# Arguments
- `ts::SimulationTimeSeries`: Time series with at least
    - `time` — sampling times (e.g., hours),
    - `total_cells` — total alive-cell counts at each sampled time.
- `window_size::Int = 10`: Width (in number of samples) of the sliding window used for
    the finite-difference estimate. Larger values smooth noise but reduce temporal resolution.

# Behavior
- Requires at least `window_size + 1` time points; otherwise returns a placeholder plot
    titled `"Insufficient data for growth rate"`.
- Iterates from `i = window_size + 1` to `length(ts.time)`, computing:
    - `dt = time[i] - time[i - window_size]`
    - `dcells = total_cells[i] - total_cells[i - window_size]`
    - `growth_rate = dcells / dt` (if `dt > 0`)
- Plots `growth_rate` vs `time[i]` as a line, and adds a horizontal zero line for reference.

# Returns
- A `Plot` object (`Plots.jl`) showing growth rate over time, with:
    - x-axis: time (h),
    - y-axis: growth rate (cells/h),
    - a dashed zero line to indicate expansion (>0) vs contraction (<0).

# Notes
- This is a **coarse derivative** over `window_size` samples, not a pointwise derivative.
    If you need per-step rates, set `window_size = 1`. For smoother estimates, increase it.
- If your sampling intervals are irregular, this formulation still accounts for `dt`
    explicitly in each window.

# Example
```julia
p = plot_growth_rate(ts; window_size=12)  # ~12-sample window smoothing
display(p)
```
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
    plot_phase_duration_distribution(ts::SimulationTimeSeries) -> Plot

Plot the **average number of cells** observed in each cell‑cycle phase
(G0, G1, S, G2, M) across the entire simulation.  
This provides an approximate estimate of relative phase durations at the
population level.

⚠️ **Important Note:**  
This method uses *population‑level averages* (`mean(ts.g*_cells)`) and does
**not** track individual cell trajectories.  
It therefore measures **average occupancy**, not true single‑cell phase
durations. For precise durations, individual cell histories would need to be
tracked.

# Arguments
- `ts::SimulationTimeSeries`: Must contain the fields  
    `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells`  
    — vectors representing per‑timepoint counts of cells in each phase.

# Behavior
- Computes the mean count in each phase across all recorded timepoints:
    - `avg_g0 = mean(ts.g0_cells)`
    - `avg_g1 = mean(ts.g1_cells)`
    - etc.
- Produces a bar chart with fixed phase ordering:
    `["G0", "G1", "S", "G2", "M"]`
- Uses a consistent color palette:
    - G0 → black  
    - G1 → green  
    - S → orange  
    - G2 → purple  
    - M → red

# Returns
- A `Plot` object showing the average number of cells in each phase.
- This plot is useful for identifying:
    - prolonged S‑phase occupancy,
    - rapid G2/M transitions,
    - population‑wide quiescence,
    - shifts due to treatments or interventions.

# Example
```julia
p = plot_phase_duration_distribution(ts)
display(p)
```
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
    plot_doubling_time(ts::SimulationTimeSeries) -> Plot

Plot the times at which the population reaches successive doublings relative to
its initial size. The x‑axis shows the **population multiple** (2×, 4×, 8×, …),
and the y‑axis shows the **time (hours)** when each multiple is first reached.

# Arguments
- `ts::SimulationTimeSeries`: Time series containing at least:
    - `time` — sampling times (e.g., hours),
    - `total_cells` — total alive cells at each sampled time.

# Behavior
- Requires at least 10 time points; otherwise returns a placeholder plot labeled
    `"Insufficient data for doubling time"`.
- Starting from the **initial population** `N₀ = ts.total_cells[1]`, iteratively
    checks targets `2·N₀`, `4·N₀`, `8·N₀`, … up to `maximum(ts.total_cells)`.
- For each target, finds the **first index** `idx` where `total_cells[idx] ≥ target`
    and records the corresponding time `ts.time[idx]`.
- If no targets are reached, returns a placeholder plot labeled
    `"No population doublings observed"`.
- Otherwise, returns a scatter plot of `(multiple, time)` with a log₂ x‑axis and
    labeled ticks (`"2×"`, `"4×"`, …).

# Returns
- A `Plot` object (from `Plots.jl`) showing doubling events over time, or a
    placeholder plot if data are insufficient or no doublings occurred.

# Notes
- This method uses the **first‑passage** time to each doubling threshold; it does
    not interpolate between time samples. If your sampling is coarse, consider
    densifying the time series or adding interpolation for more precise estimates.
- If the initial population is zero, no doubling can be computed; ensure your
    simulation starts with `total_cells[1] > 0`.

# Example
```julia
p = plot_doubling_time(ts)
display(p)
```
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
    plot_analysis_dashboard(ts::SimulationTimeSeries) -> Plot

Build a compact, multi-panel dashboard that summarizes key simulation
readouts over time. The dashboard aggregates plots from several helper
functions:

1. `plot_cell_dynamics(ts)` — total (alive) cell count vs time.
2. `plot_phase_dynamics(ts)` — absolute counts per cell-cycle phase vs time.
3. `plot_phase_proportions(ts)` — phase proportions (0–100%) vs time.
4. `plot_growth_rate(ts)` — instantaneous or smoothed growth rate over time.
5. `plot_cycling_vs_quiescent(ts)` — cycling (G1/S/G2/M) vs quiescent (G0) fractions.

The five subplots are arranged in a 3×2 grid for a quick, at-a-glance overview.

# Arguments
- `ts::SimulationTimeSeries`: Time-series container that should provide, at minimum:
    - `time` (vector of times),
    - `total_cells` (vector of alive-cell counts),
    - phase counts (`g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells`).
    Additional fields may be used by the helper plotting functions (e.g., for growth rate).

# Returns
- A single `Plot` object (from `Plots.jl`) with a `(3, 2)` layout sized `(1400, 1200)`.
    Subplots are produced by the respective helpers and combined into one dashboard.

# Notes
- This function assumes the helper functions `plot_cell_dynamics`, `plot_phase_dynamics`,
    `plot_phase_proportions`, `plot_growth_rate`, and `plot_cycling_vs_quiescent`
    are defined and compatible with `ts`.
- If any helper returns `nothing` or an empty plot (e.g., due to missing fields),
    the overall layout may show a blank panel for that position.
- Customize styles (colors, labels, smoothing) within the helper functions to keep
    the dashboard logic simple here.

# Example
```julia
p = plot_analysis_dashboard(ts)
display(p)
```
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

"""
    plot_snapshot_comparison(
        snapshots::Dict;
        metric::Symbol = :cell_cycle,
        times::Union{Nothing,Vector{Int}} = nothing
    ) -> Plot

Compare multiple snapshots side by side by plotting a chosen metric for each
selected time. Supports both `CellPopulation` snapshots and `DataFrame` snapshots.

# Arguments
- `snapshots::Dict`: Mapping from time keys (typically integer hours) to either
    - a `CellPopulation` (converted with `to_dataframe(..., alive_only=true)`), or
    - a `DataFrame` with relevant columns (e.g., `:cell_cycle`, optionally `:can_divide`).
- `metric::Symbol = :cell_cycle`:
    - `:cell_cycle` → bar plot of counts per phase (`G0`, `G1`, `S`, `G2`, `M`) with a fixed color palette.
    - `:can_divide` → bar plot of counts for `Can Divide` vs `Blocked` (requires `:can_divide` column).
- `times::Union{Nothing,Vector{Int}} = nothing`:
    - If `nothing`, all available keys in `snapshots` are plotted in sorted order.
    - Otherwise, plots only the specified times (skipping missing ones with a warning).

# Behavior
- Determines the list of time points to plot (`times` or all keys sorted).
- Requires at least **two** snapshots; otherwise returns a placeholder plot with a warning.
- For each selected time:
    - Converts `CellPopulation` to `DataFrame` of alive cells if needed.
    - Builds a bar plot according to `metric` and annotates the title as
    `"t = <time>h (n=<row count>)"`.
    - Skips unknown snapshot types or incompatible metrics with a warning.
- Arranges the produced plots into a grid:
    - Up to 4 columns per row (`n_cols = min(4, n_plots)`), as many rows as needed.

# Returns
- A combined `Plot` object containing the grid of per‑time subplots.
- If no valid plots can be created, returns a placeholder plot titled `"No valid plots created"`.

# Notes
- For `:cell_cycle`, phases not in `["G0","G1","S","G2","M"]` are ignored.
- For `:can_divide`, values are interpreted as integers/bools; green = can divide, red = blocked.
- To compare **percentages** instead of counts, consider adding a keyword (e.g., `as_percent=true`)
    and normalize the bars by `nrow(data)`.

# Example
```julia
p = plot_snapshot_comparison(snaps; metric=:cell_cycle, times=[0, 6, 12, 24])
display(p)

p2 = plot_snapshot_comparison(snaps; metric=:can_divide)
display(p2)
```
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
    plot_spatial_distribution(
        snapshot;
        color_by::Symbol = :cell_cycle,
        title_text::String = "Spatial Distribution"
    ) -> Plot

Render a 2D scatter plot of cell positions from a snapshot, optionally coloring
points by a chosen attribute. The function accepts either a `DataFrame` or a
`CellPopulation` and adapts accordingly.

- If `snapshot isa DataFrame`, it is used directly.
- If `snapshot isa CellPopulation`, it is converted with `to_dataframe(snapshot, alive_only=true)`.
- If spatial coordinates are unavailable, a placeholder plot is returned.

# Arguments
- `snapshot`: Either
    - a `DataFrame` containing at least `:x` and `:y` columns (and optionally the
    column indicated by `color_by`), or
    - a `CellPopulation`, which will be converted to a `DataFrame` of alive cells.
- `color_by::Symbol = :cell_cycle`: Attribute used to color the points.
    Supported options include:
    - `:cell_cycle` → categorical colors for phases `G0` (black), `G1` (green),
    `S` (orange), `G2` (purple), `M` (red) with a manual legend.
    - `:can_divide` → green for divisible (`1`) and red for blocked (`0`).
    - Any other value (or missing column) defaults to a single-color scatter.
- `title_text::String = "Spatial Distribution"`: Title for the plot.

# Behavior
- Validates presence of `:x` and `:y`; if missing, returns a placeholder plot
    titled `"No spatial coordinates in data"`.
- For `color_by == :cell_cycle` and when `:cell_cycle` exists, assigns a fixed
    phase palette and adds a manual legend with phase labels.
- For `color_by == :can_divide` and when `:can_divide` exists, colors by
    binary state and adds a legend (“Can Divide”, “Blocked”).
- Otherwise, produces a single-color scatter (blue) labeled “Cells”.
- Uses equal aspect ratio, no marker stroke, and moderate marker size for clarity.

# Returns
- A `Plot` object from `Plots.jl`. If inputs are invalid or coordinates are
    missing, returns a placeholder plot. If a `CellPopulation` lacks coordinates,
    returns a minimal “No spatial coordinates available” plot.

# Example
```julia
# DataFrame snapshot
p1 = plot_spatial_distribution(df; color_by=:cell_cycle, title_text="t = 12h")

# CellPopulation snapshot
p2 = plot_spatial_distribution(pop; color_by=:can_divide, title_text="Divisibility")
```
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
    create_spatial_animation(
        snapshots::Dict;
        output_file::String = "simulation.gif",
        fps::Int = 2,
        color_by::Symbol = :cell_cycle
    ) -> Animation

Create a time‑lapse GIF of the 3D spatial distribution of cells from a series
of snapshots. Each frame is produced by `plot_spatial_distribution`, optionally
coloring by a chosen attribute (e.g. `:cell_cycle`).

# Arguments
- `snapshots::Dict`: A mapping from time keys (e.g., integer hours) to snapshot
    objects. Each value is passed to `plot_spatial_distribution` and may be
    either a `CellPopulation` or a `DataFrame` supported by that plotting
    function.
- `output_file::String="simulation.gif"`: Path where the animated GIF will be saved.
- `fps::Int=2`: Frames per second of the output animation.
- `color_by::Symbol=:cell_cycle`: Attribute used for coloring in
    `plot_spatial_distribution` (e.g., `:cell_cycle`, `:is_cell`, or another
    supported column/field).

# Behavior
- Sorts the snapshot keys to define the chronological frame order.
- Requires at least two snapshots; otherwise prints a warning and returns `nothing`.
- For each time `t`, calls:
```julia
plot_spatial_distribution(snapshots[t]; color_by=color_by,
                            title_text="Cell Distribution at t = $(t)h")
                            Writes the GIF to output_file with the given fps and prints the save path.

Returns

The Animation object created by Plots.@animate. Returns nothing early if
fewer than two snapshots are provided.

Notes

Assumes Plots.jl is active and plot_spatial_distribution is available and
compatible with the values stored in snapshots.
If your snapshot keys are floating-point times, they will be sorted numerically
and displayed verbatim in the title.

Example
anim = create_spatial_animation(snaps; output_file="spatial.gif", fps=3, color_by=:cell_cycle)
```
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

"""
    print_simulation_summary(ts::SimulationTimeSeries)

Print a formatted textual summary of high‑level statistics from a completed
agent‑based simulation. The summary includes time range, population dynamics,
phase composition, cycling/quiescent fractions, and (when available) stem‑cell
information.

This function is intended as a quick terminal‑friendly diagnostic after running
a simulation with `run_simulation_abm!`.

# Arguments
- `ts::SimulationTimeSeries`: A time‑series container holding:
    - `time` — recorded time points,
    - `total_cells` — total alive cells at each time,
    - `g0_cells`, `g1_cells`, `s_cells`, `g2_cells`, `m_cells` — phase counts,
    - optionally `stem_cells` — number of stem‑like cells per time point.

# Output
A structured, human‑readable report printed to standard output, for example:
======================================================================
SIMULATION SUMMARY STATISTICS
Time Range: 0.0h - 48.0h
Time Points: 1201
Population Statistics:
Initial cells: 350
Final cells: 820
Peak cells: 950
Min cells: 340
Average cells: 612.3
Fold change: 2.34×
Average Phase Distribution:
G0: 12.3%
G1: 38.1%
S:  29.4%
G2: 14.7%
M:   5.5%
Cycling vs Quiescent:
Cycling (G1/S/G2/M): 87.7%
Quiescent (G0):       12.3%
Stem Cell Statistics:
Initial stem cells: 15
Final stem cells: 24
Average stem fraction: 2.9%
# Behavior
- Prints the simulation time range and number of recorded time points.
- Summarizes population behavior: initial, final, min/max, average, and growth
    trend (fold‑change or decline).
- Computes **average** phase composition across the full time series.
- Reports average cycling vs quiescent fractions based on phase means.
- If `ts.stem_cells` exists and contains nonzero values, prints stem‑cell
    statistics as well.
- Does **not** return anything; output is printed directly.

# Notes
- This function assumes that time‑series vectors in `ts` all have equal length.
- Average phase proportions are computed using the mean counts across the full
    simulation, not only the final state.
- Suitable for logging, debugging, and quick model‑behavior inspection.

# Example
```julia
ts, snaps = run_simulation_abm!(pop, 0.01; terminal_time=48)
print_simulation_summary(ts)
# Behavior
- Prints the simulation time range and number of recorded time points.
- Summarizes population behavior: initial, final, min/max, average, and growth
    trend (fold‑change or decline).
- Computes **average** phase composition across the full time series.
- Reports average cycling vs quiescent fractions based on phase means.
- If `ts.stem_cells` exists and contains nonzero values, prints stem‑cell
    statistics as well.
- Does **not** return anything; output is printed directly.

# Notes
- This function assumes that time‑series vectors in `ts` all have equal length.
- Average phase proportions are computed using the mean counts across the full
    simulation, not only the final state.
- Suitable for logging, debugging, and quick model‑behavior inspection.

# Example
```julia
ts, snaps = run_simulation_abm!(pop, 0.01; terminal_time=48)
print_simulation_summary(ts)
```
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

"""
    plot_phase_proportions_timeseries(
        ts::SimulationTimeSeries;
        title_text::String = "Cell Cycle Distribution Over Time"
    ) -> Plot

Plot the time evolution of cell‑cycle **phase proportions** for alive cells
recorded in a `SimulationTimeSeries`. Each phase (G0, G1, S, G2, M) is plotted as
a percentage of the total alive population at each recorded time point.

# Arguments
- `ts::SimulationTimeSeries`: A time‑series record produced by the ABM simulation.
    It must provide fields:
    - `time`
    - `g0_cells`
    - `g1_cells`
    - `s_cells`
    - `g2_cells`
    - `m_cells`
    Each field is expected to be a vector of the same length, representing counts
    of alive cells in each phase at each recorded time.
- `title_text::String = "Cell Cycle Distribution Over Time"`:
    Optional title for the plot.

# Behavior
- Computes the total alive cells at each time point as  
    `g0 + g1 + s + g2 + m`.
- Converts raw phase counts into percentages, handling `total == 0` safely
    using `ifelse.(...)` to avoid division by zero.
- Plots each phase as a line over time, using a fixed color scheme:
    - `G0` → gray
    - `G1` → green
    - `S`  → orange
    - `G2` → purple
    - `M`  → red
- Sets y‑limits to `(0, 100)` and labels axes as `Time (h)` and `Percentage (%)`.

# Returns
- A `Plot` object containing the full phase‑proportion time series.

# Example
```julia
ts, _ = run_simulation_abm!(pop, 0.01)
p = plot_phase_proportions_timeseries(ts)
display(p)
```
"""
function plot_phase_proportions_timeseries(ts::SimulationTimeSeries; 
                                            title_text::String="Cell Cycle Distribution Over Time")
    # Calculate proportions (already only counts alive cells)
    total = ts.g0_cells .+ ts.g1_cells .+ ts.s_cells .+ ts.g2_cells .+ ts.m_cells
    
    # Handle division by zero
    g0_prop = ifelse.(total .> 0, 100 .* ts.g0_cells ./ total, 0.0)
    g1_prop = ifelse.(total .> 0, 100 .* ts.g1_cells ./ total, 0.0)
    s_prop = ifelse.(total .> 0, 100 .* ts.s_cells ./ total, 0.0)
    g2_prop = ifelse.(total .> 0, 100 .* ts.g2_cells ./ total, 0.0)
    m_prop = ifelse.(total .> 0, 100 .* ts.m_cells ./ total, 0.0)
    
    p = plot(xlabel="Time (h)",
                ylabel="Percentage (%)",
                title=title_text,
                legend=:best,
                ylims=(0, 100))
    
    plot!(p, ts.time, g0_prop, label="G0", linewidth=2, color=:black)
    plot!(p, ts.time, g1_prop, label="G1", linewidth=2, color=:green)
    plot!(p, ts.time, s_prop, label="S", linewidth=2, color=:orange)
    plot!(p, ts.time, g2_prop, label="G2", linewidth=2, color=:purple)
    plot!(p, ts.time, m_prop, label="M", linewidth=2, color=:red)
    
    return p
end

"""
    plot_phase_comparison_before_after(
        cell_df_initial::DataFrame,
        cell_df_final::DataFrame
    ) -> Plot

Create a side‑by‑side comparison of the **alive‑cell phase proportions**
between an initial and a final population snapshot.

This function is a lightweight helper that applies
`plot_phase_proportions_alive` to both time points and arranges the resulting
plots horizontally for quick visual comparison.

# Arguments
- `cell_df_initial::DataFrame`: Snapshot representing the **starting** state.
    Must include `:cell_cycle` and `:is_cell` columns, where `is_cell == 1`
    indicates alive cells.
- `cell_df_final::DataFrame`: Snapshot representing the **ending** state, with
    the same column requirements.

# Behavior
- Computes the cell‑cycle phase proportions for alive cells in each snapshot.
- Creates two bar‑plots (or equivalent) via `plot_phase_proportions_alive`,
    one titled “Initial Distribution” and one titled “Final Distribution”.
- Returns a combined figure with a `(1, 2)` horizontal layout.

# Returns
- A single `Plot` object showing the two distributions side by side.

# Example
```julia
p = plot_phase_comparison_before_after(df_start, df_end)
display(p)
```
"""
function plot_phase_comparison_before_after(cell_df_initial::DataFrame, 
                                                cell_df_final::DataFrame)
    p1 = plot_phase_proportions_alive(cell_df_initial, title_text="Initial Distribution")
    p2 = plot_phase_proportions_alive(cell_df_final, title_text="Final Distribution")
    
    return plot(p1, p2, layout=(1, 2), size=(1000, 400))
end