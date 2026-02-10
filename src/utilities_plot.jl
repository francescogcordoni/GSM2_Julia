"""
    plot_dose_cell(cell_df; layer_plot::Bool = false)

Crea una figura con **due pannelli** per visualizzare la distribuzione delle dosi a livello di cellula.

# Pannelli
- **Pannello sinistro (Density)**
  - Se `layer_plot == false` → densità di `dose_cell` per **tutte le celle attive** (`is_cell == 1`), considerando solo valori positivi.
  - Se `layer_plot == true`  → densità **raggruppata per `energy_step`**: una curva per ciascun valore distinto di `energy_step` presente tra le celle attive.

- **Pannello destro (3D scatter)**
  - Rappresentazione 3D di `(x, y, dose_cell)` per le celle attive.
  - I punti sono colorati in funzione di `dose_cell`.
  - (Opzionale) filtro `x ≥ 0` per una vista “semi-sfera”.

# Requisiti colonne in `cell_df`
- Obbligatorie: `:x`, `:y`, `:dose_cell`, `:is_cell`
- Ulteriore richiesta se `layer_plot == true`: `:energy_step`

# Argomenti
- `cell_df::DataFrame`: tabella con le informazioni per cellula.
- `layer_plot::Bool = false`: 
    - `false` → singola densità su tutte le celle attive.
    - `true`  → densità suddivisa per `energy_step`.

# Comportamento
- Le densità considerano solo `dose_cell > 0`.
- Le celle non attive (`is_cell != 1`) vengono escluse in entrambe le visualizzazioni.
- Il pannello 3D utilizza sempre `dose_cell` come asse z e come colormap.

# Ritorno
- Restituisce l’oggetto `Plots.Plot` contenente i due pannelli affiancati.
- Restituisce `nothing` se non sono presenti celle attive o valori positivi di `dose_cell`.

# Esempi
```julia
# Densità complessiva (celle attive) + 3D scatter
plot_dose_cell(cell_df)

# Densità per energy_step + 3D scatter
plot_dose_cell(cell_df; layer_plot=true)
```
"""
function plot_dose_cell(cell_df; layer_plot::Bool = false)

    # ============================================================
    # 0. Extract active cells and check DataFrame integrity
    # ============================================================
    if !all(c -> c in propertynames(cell_df), [:x, :y, :dose_cell, :is_cell])
        error("cell_df must contain columns :x, :y, :dose_cell, :is_cell")
    end

    df_active = cell_df[cell_df.is_cell .== 1, :]

    if isempty(df_active)
        @warn "No active cells found (is_cell == 1)."
        return nothing
    end

    # Vector of dose values for density plots
    doses = df_active.dose_cell
    active_doses = filter(>(0), doses)

    if isempty(active_doses)
        @warn "No positive dose values available."
        return nothing
    end

    # ============================================================
    # PANEL 1 → Density
    # ============================================================

    if !layer_plot
        # Standard density on all active cells
        p1 = density(
            active_doses,
            title = "Dose Density (all active cells)",
            xlabel = "dose_cell",
            ylabel = "Density",
            linewidth = 2,
            legend = false,
        )

        vline!(p1, [mean(active_doses)], color=:red, linestyle=:dash)
    else
        # Grouped by energy_step
        if !(:energy_step in propertynames(cell_df))
            error("Grouped density requires column :energy_step")
        end

        energy_steps = sort(unique(df_active.energy_step))

        p1 = plot(
            title = "Dose Density grouped by energy_step",
            xlabel = "dose_cell",
            ylabel = "Density",
            legend = :topright,
            linewidth = 2,
        )

        for E in energy_steps
            sub = df_active[df_active.energy_step .== E, :].dose_cell
            sub_pos = filter(>(0), sub)
            if isempty(sub_pos)
                continue
            end
            density!(p1, sub_pos, label="energy_step $E")
        end
    end

    # ============================================================
    # PANEL 2 → 3D scatter of x,y,dose_cell
    # ============================================================

    df3 = df_active[df_active.x .>= 0, :]   # half-sphere selection (optional)

    p2 = scatter(
        df3.x,
        df3.y,
        df3.z;
        markersize = 4,
        markerstrokewidth = 0.1,
        marker_z = df3.dose_cell,
        colorbar = true,
        xlabel = "x (µm)",
        ylabel = "y (µm)",
        zlabel = "z",
        title = "3D Dose Distribution",
        legend = false,
        aspect_ratio = :equal,
        seriescolor = :viridis,
        size = (900, 700),
        camera = (320, 30),
    )

    # ============================================================
    # Combine both panels into one figure
    # ============================================================
    return plot(p1, p2, layout=(1, 2), size=(1200, 500))
end

"""
    plot_damage(cell_df; layer_plot::Bool = false)

Plot the distribution of X-type damage per active cell (is_cell == 1),  
with optional grouping by `layer`.

# Modes

## 1. Standard mode (`layer_plot = false`)
- Selects active cells (is_cell == 1)
- Extracts damage vectors from `dam_X_dom`
- Computes total damage per cell (`sum(dam_X_dom[i])`)
- Plots a single density distribution
- Draws a vertical dashed red line indicating the mean

## 2. Layer-grouped mode (`layer_plot = true`)
- For each layer in `cell_df.layer`:
    * filters active cells belonging to that layer
    * computes total X-damage per cell
    * plots one density curve per layer with different colors
- Useful for comparing X-damage distributions across layers

# Required columns
- Standard mode: `:is_cell`, `:dam_X_dom`
- Layer-grouped mode: additionally requires `:layer`

# Returns
- A Plots.jl plot object  
- Nothing if no valid data available
"""
function plot_damage(cell_df::DataFrame; layer_plot::Bool = false)

    # ============================================================
    # MODE 1 → GROUPED DENSITY PLOT BY LAYER
    # ============================================================
    if layer_plot
        
        required_cols = (:is_cell, :dam_X_dom, :layer)
        for c in required_cols
            if !(c in propertynames(cell_df))
                error("Layer-grouped damage plot requires column :$c in cell_df.")
            end
        end

        layers = unique(cell_df.layer)
        sort!(layers)

        plt = plot(
            title = "X-Damage Distribution Grouped by Layer",
            xlabel = "Total X-Damage",
            ylabel = "Density",
            legend = :topright,
            linewidth = 2,
        )

        for L in layers
            # Select active cells from this layer
            idx = findall(i -> cell_df.is_cell[i] == 1 && cell_df.energy_step[i] == L,
                            1:nrow(cell_df))

            if isempty(idx)
                @warn "Layer $L has no active cells. Skipping."
                continue
            end

            # Compute damage values
            damage_vals = Float64[]
            for i in idx
                dam = cell_df.dam_X_dom[i]
                if dam isa AbstractVector
                    push!(damage_vals, sum(dam))
                end
            end

            if isempty(damage_vals)
                @warn "Layer $L has no valid damage vectors. Skipping."
                continue
            end

            density!(
                plt,
                damage_vals,
                label = "Layer $L",
            )
        end

        return plt
    end

    # ============================================================
    # MODE 2 → ORIGINAL SINGLE DISTRIBUTION
    # ============================================================

    # Filter active cells
    active_idx = findall(row -> row.is_cell == 1, eachrow(cell_df))

    if isempty(active_idx)
        @warn "No active cells found (is_cell == 1)."
        return nothing
    end

    # Compute total damage per active cell
    damage_values = Float64[]
    for i in active_idx
        dam = cell_df.dam_X_dom[i]
        if dam isa AbstractVector
            push!(damage_values, sum(dam))
        else
            @warn "dam_X_dom at row $i is not a vector. Skipping."
        end
    end

    if isempty(damage_values)
        @warn "No valid damage vectors found."
        return nothing
    end

    # Compute mean damage
    mean_damage = mean(damage_values)
    println("Mean X-damage per active cell = $(round(mean_damage, digits=4))")

    # Create density plot
    plt = density(
        damage_values,
        title = "Density of X-Damage per Active Cell",
        xlabel = "Total X-Damage",
        ylabel = "Density",
        linewidth = 2,
        legend = :topright
    )

    # Add vertical mean line
    vline!(plt, [mean_damage], color=:red, linestyle=:dash, label="Mean")

    return plt
end
"""
    plot_survival_probability_cell(cell_df; layer_plot::Bool = false)

Crea una figura con **due pannelli** per visualizzare la distribuzione delle dosi a livello di cellula.

# Pannelli
- **Pannello sinistro (Density)**
  - Se `layer_plot == false` → densità di `dose_cell` per **tutte le celle attive** (`is_cell == 1`), considerando solo valori positivi.
  - Se `layer_plot == true`  → densità **raggruppata per `energy_step`**: una curva per ciascun valore distinto di `energy_step` presente tra le celle attive.

- **Pannello destro (3D scatter)**
    - Rappresentazione 3D di `(x, y, dose_cell)` per le celle attive.
    - I punti sono colorati in funzione di `dose_cell`.
    - (Opzionale) filtro `x ≥ 0` per una vista “semi-sfera”.

# Requisiti colonne in `cell_df`
- Obbligatorie: `:x`, `:y`, `:dose_cell`, `:is_cell`
- Ulteriore richiesta se `layer_plot == true`: `:energy_step`

# Argomenti
- `cell_df::DataFrame`: tabella con le informazioni per cellula.
- `layer_plot::Bool = false`: 
    - `false` → singola densità su tutte le celle attive.
    - `true`  → densità suddivisa per `energy_step`.

# Comportamento
- Le densità considerano solo `dose_cell > 0`.
- Le celle non attive (`is_cell != 1`) vengono escluse in entrambe le visualizzazioni.
- Il pannello 3D utilizza sempre `dose_cell` come asse z e come colormap.

# Ritorno
- Restituisce l’oggetto `Plots.Plot` contenente i due pannelli affiancati.
- Restituisce `nothing` se non sono presenti celle attive o valori positivi di `dose_cell`.

# Esempi
```julia
# Densità complessiva (celle attive) + 3D scatter
plot_dose_cell(cell_df)

# Densità per energy_step + 3D scatter
plot_dose_cell(cell_df; layer_plot=true)
```
"""
function plot_survival_probability_cell(cell_df; layer_plot::Bool = false)

    # ============================================================
    # 0. Extract active cells and check DataFrame integrity
    # ============================================================
    if !all(c -> c in propertynames(cell_df), [:x, :y, :dose_cell, :is_cell])
        error("cell_df must contain columns :x, :y, :sp, :is_cell")
    end

    df_active = cell_df[cell_df.is_cell .== 1, :]

    if isempty(df_active)
        @warn "No active cells found (is_cell == 1)."
        return nothing
    end

    # Vector of dose values for density plots
    doses = df_active.sp
    active_doses = filter(>(0), doses)

    if isempty(active_doses)
        @warn "No positive dose values available."
        return nothing
    end

    # ============================================================
    # PANEL 1 → Density
    # ============================================================

    if !layer_plot
        # Standard density on all active cells
        p1 = density(
            active_doses,
            title = "Survival probability",
            xlabel = "survival probability",
            ylabel = "Density",
            linewidth = 2,
            legend = false,
        )

        vline!(p1, [mean(active_doses)], color=:red, linestyle=:dash)
    else
        # Grouped by energy_step
        if !(:energy_step in propertynames(cell_df))
            error("Grouped density requires column :energy_step")
        end

        energy_steps = sort(unique(df_active.energy_step))

        p1 = plot(
            title = "Survival probability grouped by energy_step",
            xlabel = "survival probability",
            ylabel = "Density",
            legend = :topright,
            linewidth = 2,
        )

        for E in energy_steps
            sub = df_active[df_active.energy_step .== E, :].sp
            sub_pos = filter(>(0), sub)
            if isempty(sub_pos)
                continue
            end
            density!(p1, sub_pos, label="energy_step $E")
        end
    end

    # ============================================================
    # PANEL 2 → 3D scatter of x,y,dose_cell
    # ============================================================

    df3 = df_active[df_active.x .>= 0, :]   # half-sphere selection (optional)

    p2 = scatter(
        df3.x,
        df3.y,
        df3.z;
        markersize = 4,
        markerstrokewidth = 0.1,
        marker_z = df3.sp,
        colorbar = true,
        xlabel = "x (µm)",
        ylabel = "y (µm)",
        zlabel = "z",
        title = "3D Dose Distribution",
        legend = false,
        aspect_ratio = :equal,
        seriescolor = :viridis,
        size = (900, 700),
        camera = (320, 30),
    )

    # ============================================================
    # Combine both panels into one figure
    # ============================================================
    return plot(p1, p2, layout=(1, 2), size=(1200, 500))
end

using Plots
using Statistics
using DataFrames

"""
    plot_times(cell_df::DataFrame;
               show_means::Bool = true,
               summary::Bool = true,
               verbose::Bool = false,
               color_main = :D55E00,               # rust orange accent
               color_alt  = :steelblue,            # secondary for variety
               layout_tuple::Tuple{Int,Int} = (2, 2),
               size_px::Tuple{Int,Int} = (1200, 800))

Create a **four‑panel figure** showing densities of time-related outcomes and total X damage,
using only rows where `is_cell == 1`.

# Panels (top-left to bottom-right)
1. **Death time** — density of `death_time` restricted to finite values.
2. **Recovery time** — density of `recover_time` restricted to finite values.
3. **Cycle time** — density of `cycle_time` restricted to finite values.
4. **Total X damage** — density of `dam_X_total`. If `dam_X_total` does not exist,
   but `dam_X_dom` does and contains vectors, the function computes it on the fly as
   `sum(dam_X_dom[i])` for the selected (active) rows.

# Arguments
- `cell_df::DataFrame`: Input table.
- `show_means::Bool = true`: If `true`, draw a vertical dashed red line at the mean for each panel (when data is available).
- `summary::Bool = true`: If `true`, print a short summary of how many values were plotted and their means.
- `verbose::Bool = false`: If `true`, print extra details (e.g., when columns are missing or empty).
- `color_main`: Primary color for the first three panels (default rust orange `:D55E00`).
- `color_alt`: Secondary color for the damage panel.
- `layout_tuple = (2,2)`: Grid layout for subplots.
- `size_px = (1200,800)`: Overall figure size in pixels.

# Requirements in `cell_df`
- Must contain: `:is_cell` (Int or Bool), and preferably `:death_time`, `:recover_time`, `:cycle_time`.
- For the 4th panel, either `:dam_X_total` **or** `:dam_X_dom` (vector per row).

# Returns
- A `Plots.Plot` object with the 4 panels.  
- Returns `nothing` if there are no active cells.

# Notes
- Values equal to `Inf` are excluded from the time densities.
- Missing columns are handled gracefully; the corresponding panel will show a message.
"""
function plot_times(cell_df::DataFrame;
                    show_means::Bool = true,
                    summary::Bool = true,
                    verbose::Bool = false,
                    color_main = :D55E00,
                    color_alt  = :steelblue,
                    layout_tuple::Tuple{Int,Int} = (2, 2),
                    size_px::Tuple{Int,Int} = (1200, 800))

    # -----------------------------
    # 0) Select active cells
    # -----------------------------
    if !(:is_cell in propertynames(cell_df))
        error("plot_times: cell_df must contain :is_cell.")
    end

    df_active = cell_df[cell_df.is_cell .== 1, :]
    if nrow(df_active) == 0
        println("[plot_times] No active cells (is_cell == 1). Nothing to plot.")
        return nothing
    end

    # Helper to extract finite values safely
    get_finite = function(df::DataFrame, col::Symbol)
        if !(col in propertynames(df))
            if verbose
                println("[plot_times] Column $(col) not found.")
            end
            return Float64[]
        end
        v = df[!, col]
        # Convert to Float64 where possible; filter finite values only
        vals = Float64[]
        for x in v
            if x isa Real
                fx = Float64(x)
                if isfinite(fx)
                    push!(vals, fx)
                end
            end
        end
        return vals
    end

    # -----------------------------
    # 1) Prepare vectors for panels
    # -----------------------------
    death_vals   = get_finite(df_active, :death_time)
    recover_vals = get_finite(df_active, :recover_time)
    cycle_vals   = get_finite(df_active, :cycle_time)

    # dam_X_total: prefer existing column, otherwise try computing from dam_X_dom
    damX_vals = Float64[]
    if :dam_X_total in propertynames(df_active)
        damX_vals = Float64.(df_active.dam_X_total)
    elseif :dam_X_dom in propertynames(df_active)
        # compute sum(dam_X_dom[i]) when dam_X_dom[i] is a vector
        damX_vals = Float64[]
        for row in eachrow(df_active)
            v = row[:dam_X_dom]
            if v isa AbstractVector
                push!(damX_vals, sum(v))
            elseif verbose
                println("[plot_times] Row index $(row.row) has non-vector dam_X_dom; skipped.")
            end
        end
    else
        if verbose
            println("[plot_times] Neither :dam_X_total nor :dam_X_dom found; damage panel will be empty.")
        end
    end

    # Remove non-finite from damX if any (usually all finite)
    damX_vals = [x for x in damX_vals if isfinite(x)]

    # -----------------------------
    # 2) Create subplots
    # -----------------------------
    p1 = plot(title = "Death time", xlabel = "time", ylabel = "Density", legend = false)
    if !isempty(death_vals)
        density!(p1, death_vals; lw=2)
        if show_means
            vline!(p1, [mean(death_vals)]; color=:red, ls=:dash)
        end
    else
        annotate!(p1, 0.5, 0.5, text("No finite values", 10, :gray))
    end

    p2 = plot(title = "Recovery time", xlabel = "time", ylabel = "Density", legend = false)
    if !isempty(recover_vals)
        density!(p2, recover_vals; lw=2)
        if show_means
            vline!(p2, [mean(recover_vals)]; color=:red, ls=:dash)
        end
    else
        annotate!(p2, 0.5, 0.5, text("No finite values", 10, :gray))
    end

    p3 = plot(title = "Cycle time", xlabel = "time", ylabel = "Density", legend = false)
    if !isempty(cycle_vals)
        density!(p3, cycle_vals; lw=2)
        if show_means
            vline!(p3, [mean(cycle_vals)]; color=:red, ls=:dash)
        end
    else
        annotate!(p3, 0.5, 0.5, text("No finite values", 10, :gray))
    end

    p4 = plot(title = "Total X damage", xlabel = "damage", ylabel = "Density", legend = false)
    if !isempty(damX_vals)
        density!(p4, damX_vals; lw=2)
        if show_means
            vline!(p4, [mean(damX_vals)]; color=:red, ls=:dash)
        end
    else
        annotate!(p4, 0.5, 0.5, text("No values", 10, :gray))
    end

    plt = plot(p1, p2, p3, p4; layout=layout_tuple, size=size_px)

    # -----------------------------
    # 3) Optional printing
    # -----------------------------
    if summary
        println("[plot_times] Active cells used     : $(nrow(df_active))")
        println("[plot_times] death_time  (finite) : $(length(death_vals))",
                isempty(death_vals) ? "" : " | mean=$(round(mean(death_vals), digits=4))")
        println("[plot_times] recover_time(finite) : $(length(recover_vals))",
                isempty(recover_vals) ? "" : " | mean=$(round(mean(recover_vals), digits=4))")
        println("[plot_times] cycle_time  (finite) : $(length(cycle_vals))",
                isempty(cycle_vals) ? "" : " | mean=$(round(mean(cycle_vals), digits=4))")
        println("[plot_times] dam_X_total          : $(length(damX_vals))",
                isempty(damX_vals) ? "" : " | mean=$(round(mean(damX_vals), digits=4))")
    end
    if verbose
        missing_cols = Symbol[]
        for c in (:death_time, :recover_time, :cycle_time)
            if !(c in propertynames(cell_df))
                push!(missing_cols, c)
            end
        end
        if !(:dam_X_total in propertynames(cell_df)) && !(:dam_X_dom in propertynames(cell_df))
            push!(missing_cols, :dam_X_total)
            push!(missing_cols, :dam_X_dom)
        end
        if !isempty(missing_cols)
            println("[plot_times] Missing columns in input: $(unique(missing_cols))")
        end
    end

    return plt
end