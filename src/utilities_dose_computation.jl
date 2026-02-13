"""
MC_dose_fast!(ion, Npar, x_cb, y_cb, R_beam, irrad_cond,
                    cell_df_copy, df_center_x, df_center_y, at,
                    gsm2_cycle, type_AT, track_seg)

High-level Monte Carlo wrapper that:
1. Selects representative cells from DataFrames,
2. Converts them into matrix form,
3. Executes optimized Monte Carlo kernels,
4. Converts results back to DataFrames, and
5. Copies dose values into the full simulation domain.

This function is a **drop‑in replacement** for `MC_dose_fast!` and requires
**no changes** to user code. It only adds optimized matrix-based MC kernels
under the hood.

Arguments
---------
- `ion`          :: Ion species used in the simulation
- `Npar`         :: Expected number of incident particles
- `x_cb, y_cb`   :: Beam center coordinates
- `R_beam`       :: Beam radius
- `irrad_cond`   :: Irradiation conditions per energy step
- `cell_df_copy` :: Full domain DataFrame
- `df_center_x`  :: X coordinates per cell
- `df_center_y`  :: Y coordinates per cell
- `at`           :: Output dose/track structure DataFrame
- `gsm2_cycle`   :: Cycle of GSM2 parameters
- `type_AT`      :: Transport type string
- `track_seg`    :: Enables track-segment mode if true

Notes
-----
- No physical or mathematical behavior is changed.
- Only printing and clarity improvements have been added.
"""
function MC_dose_fast!(
    ion::Ion, Npar::Int64, R_beam::Float64,
    irrad_cond::Vector{AT}, cell_df_copy::DataFrame,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
     x_cb::Float64 = 0., y_cb::Float64 = 0.
)
    println("\n───────────────────────────────────────────────")
    println("🔧  Running MC_dose_fast!   (track_seg = $track_seg)")
    println("───────────────────────────────────────────────")

    t_start = time()
    gsm2 = gsm2_cycle[1]

    if track_seg
        println("• Mode: Track-Segment Matrix Optimization")
        println("• Selecting representative cells...")

        cell_df_is = filter(row -> row.is_cell == 1, cell_df_copy)
        if nrow(cell_df_is) == 0
            @warn "No cells with is_cell = 1 → skipping."
            return
        end

        grouped_df = combine(groupby(cell_df_is, [:x, :y]),
                                :index => first => :representative_index)
        rep_indices_set = Set(grouped_df.representative_index)

        println("  → Found $(length(rep_indices_set)) representative cells")

        cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
        cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
        at_single        = filter(row -> row.index in rep_indices_set, at)

        println("• Converting DataFrames → matrices...")
        mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

        println("• Running optimized TSC Monte Carlo kernel...")
        MC_loop_ions_domain_tsc_matrix!(
            Npar, x_cb, y_cb, [irrad_cond[1]], gsm2,
            mat_x, mat_y, mat_at, R_beam, type_AT, ion
        )

        println("• Converting matrices → DataFrames...")
        matrix_to_dataframe!(at_single, mat_at)

        println("• Copying dose values back to full domain...")
        MC_loop_copy_dose_domain_fast!(cell_df_copy, at_single, at)

    else
        println("• Mode: Layered (non-TSC) Matrix Optimization")

        Np = rand(Poisson(Npar))
        println("• Sampling $Np particle hits...")

        x_list = Vector{Float64}(undef, Np)
        y_list = Vector{Float64}(undef, Np)
        Threads.@threads for ip in 1:Np
            x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
        end

        println("• Processing layers:")
        for id in unique(cell_df_copy.energy_step)
            println("  → Layer $id")
            cell_df_is = filter(row -> (row.is_cell == 1) && (row.energy_step == id), cell_df_copy)
            if nrow(cell_df_is) == 0
                println("    (empty → skip)")
                continue
            end

            grouped_df = combine(groupby(cell_df_is, [:x, :y]),
                                    :index => first => :representative_index)
            rep_indices_set = Set(grouped_df.representative_index)

            cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
            cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
            at_single        = filter(row -> row.index in rep_indices_set, at)

            println("    • Converting to matrices...")
            mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

            println("    • Running MC kernel for this layer...")
            MC_loop_ions_domain_matrix!(
                x_list, y_list, [irrad_cond[id]], gsm2,
                mat_x, mat_y, mat_at, type_AT, ion
            )

            matrix_to_dataframe!(at_single, mat_at)

            println("    • Copying dose values back...")
            MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, at_single, at, id)
        end
    end

    dt = round(time() - t_start; digits=3)
    println("───────────────────────────────────────────────")
    println("🎉 MC_dose_fast! finished. Total time: $(dt)s")
    println("───────────────────────────────────────────────\n")
end

"""
dataframes_to_matrices(df_x, df_y, df_at)

Convert the numerical columns of three DataFrames into dense matrices.

- The column `:index` is automatically excluded.
- All remaining columns must be numeric and consistently ordered.
- Returns `(mat_x, mat_y, mat_at)` as `Matrix{Float64}`.

This is used to feed the optimized Monte Carlo kernels that operate on
matrix-domain representations instead of DataFrames.

Printing/logging has been added for clarity; **no computational changes**.
"""
function dataframes_to_matrices(df_x::DataFrame, df_y::DataFrame, df_at::DataFrame)
    println("• Converting DataFrames → matrices")
    println("  → Column selection: excluding :index")

    # Extract domain columns (everything except :index)
    domain_cols = names(df_x, Not(:index))

    println("  → Using $(length(domain_cols)) domain columns: ", join(domain_cols, ", "))

    # Convert to matrices
    mat_x  = Matrix{Float64}(df_x[:, domain_cols])
    mat_y  = Matrix{Float64}(df_y[:, domain_cols])
    mat_at = Matrix{Float64}(df_at[:, domain_cols])

    println("  → Conversion complete (sizes: X=$(size(mat_x)), Y=$(size(mat_y)), AT=$(size(mat_at)))")

    return mat_x, mat_y, mat_at
end


"""
matrix_to_dataframe!(df, mat)

Write matrix values back into an existing DataFrame `df`.

- The matrix must correspond exactly to all DataFrame columns except `:index`.
- Column order is preserved.
- Updates happen in place.

This is the inverse operation of `dataframes_to_matrices`.
"""
function matrix_to_dataframe!(df::DataFrame, mat::Matrix{Float64})
    println("• Writing matrices → DataFrame")
    domain_cols = names(df, Not(:index))

    println("  → Updating $(length(domain_cols)) columns")

    @assert size(mat, 2) == length(domain_cols) "Matrix column count does not match DataFrame"

    for (j, col) in enumerate(domain_cols)
        df[!, col] = mat[:, j]
    end

    println("  → Update complete (DataFrame rows = $(nrow(df)))")
end

"""
MC_loop_ions_domain_tsc_matrix!(...)

Monte Carlo Track Structure Calculation (TSC mode - matrix version).

This function simulates ion-induced energy deposition over a matrix of spatial
domains using a Monte Carlo approach. It:

1. Builds a radial dose lookup table (log-spaced sampling).
2. Simulates a Poisson-distributed number of particle hits.
3. Distributes deposited dose into each domain using:
    - Core region (constant core dose)
    - Mid region (lookup + linear interpolation)
    - Penumbra region (inverse-square decay)
4. Accumulates dose in a thread-safe manner.
5. Writes the final result into `mat_at` (in-place).

Arguments:
-----------
- `Npar`  : Expected number of primary particles
- `x_cb`, `y_cb` : Beam center coordinates
- `irrad_cond` : Irradiation conditions
- `gsm2` : Geometry/scaling model
- `mat_x`, `mat_y` : Domain coordinate matrices
- `mat_at` : Output matrix (modified in-place)
- `R_beam` : Beam radius
- `type_AT` : Track type identifier
- `ion` : Ion parameters

This function modifies `mat_at` in-place.
"""
function MC_loop_ions_domain_tsc_matrix!(
    Npar::Int, x_cb::Float64, y_cb::Float64,
    irrad_cond::Vector{AT}, gsm2::GSM2,
    mat_x::Matrix{Float64}, mat_y::Matrix{Float64}, mat_at::Matrix{Float64},
    R_beam::Float64, type_AT::String, ion::Ion
)

    println("\n============================================================")
    println(" Monte Carlo Loop - Ions Domain (TSC MODE - Matrix)")
    println("============================================================")

    # Extract parameters
    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2
    mid_radius_sq  = (gsm2.rd + 150 * Rc)^2
    penumbra_radius_sq = Rp^2

    println("→ Physical parameters:")
    println("   Rp = $Rp, Rc = $Rc, Kp = $Kp")
    println("   Core radius²     = $core_radius_sq")
    println("   Mid radius²      = $mid_radius_sq")
    println("   Penumbra radius² = $penumbra_radius_sq")

    # Build lookup table
    sim_ = 1000
    impact_p = 10 .^ range(log10(lower_bound_log),
                           stop=log10(gsm2.rd + 150 * Rc),
                            length=sim_)

    dose_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    println("\n→ Building radial dose lookup table ($sim_ samples)...")

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd,
                                            track, irrad_cond[1], type_AT)
        dose_lookup_threads[tid][i] = Gyr
    end

    dose_vec = sum(dose_lookup_threads)
    impact_vec = impact_p
    core_dose = dose_vec[1]

    println("   ✔ Lookup table completed")
    println("   Core dose = $core_dose Gy")

    # Matrix dimensions
    num_cells, num_domains_per_cell = size(mat_x)
    total_domains = num_cells * num_domains_per_cell

    println("\n→ Geometry:")
    println("   Cells             = $num_cells")
    println("   Domains per cell  = $num_domains_per_cell")
    println("   Total domains     = $total_domains")

    # Flatten matrices to vectors (column-major order)
    dom_x = vec(mat_x')
    dom_y = vec(mat_y')

    # Thread-local accumulators
    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    # Monte Carlo
    Np = rand(Poisson(Npar))

    println("\n→ Monte Carlo sampling:")
    println("   Expected particles = $Npar")
    println("   Sampled particles  = $Np")
    println("   Threads used       = $(Threads.maxthreadid())")

    Threads.@threads for _ in 1:Np
        tid = Threads.threadid()
        local_store = at_acc[tid]

        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        @inbounds for k in 1:total_domains
            dist_sq = (dom_x[k] - x)^2 + (dom_y[k] - y)^2

            if dist_sq <= core_radius_sq
                local_store[k] += core_dose

            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq)
                idx_l = searchsortedfirst(impact_vec, dist)

                if idx_l == 1
                    local_store[k] += core_dose

                elseif idx_l > sim_
                    local_store[k] += dose_vec[end]

                else
                    x1, x2 = impact_vec[idx_l-1], impact_vec[idx_l]
                    y1, y2 = dose_vec[idx_l-1], dose_vec[idx_l]
                    local_store[k] += y1 + (y2 - y1) *
                                        (dist - x1) / (x2 - x1)
                end

            elseif dist_sq < penumbra_radius_sq
                local_store[k] += Kp / dist_sq
            end
        end
    end

    final_at_row = sum(at_acc)

    # Reshape back to matrix (transpose back)
    mat_at .= reshape(final_at_row,
                        num_domains_per_cell,
                        num_cells)'

    println("\n✔ TSC simulation completed successfully")
    println("============================================================\n")
end

"""
    MC_loop_ions_domain_matrix!(...)

Full Monte Carlo Track Structure simulation (matrix version).

This function computes ion-induced energy deposition across a matrix
of spatial domains using explicit particle positions (`x_list`, `y_list`).

Workflow:
-----------
1. Builds a radial dose lookup table (log-spaced sampling).
2. Loops over all provided particle coordinates.
3. For each domain, deposits dose according to:
    - Core region (constant core dose)
    - Mid region (lookup + linear interpolation)
    - Penumbra region (inverse-square decay)
4. Uses thread-local accumulators for parallel safety.
5. Writes final accumulated dose into `mat_at` (in-place).

Arguments:
-----------
- `x_list`, `y_list` : Particle impact coordinates
- `irrad_cond`       : Irradiation conditions
- `gsm2`             : Geometry/scaling model
- `mat_x`, `mat_y`   : Domain coordinate matrices
- `mat_at`           : Output matrix (modified in-place)
- `type_AT`          : Track type identifier
- `ion`              : Ion parameters

This function modifies `mat_at` in-place.
"""
function MC_loop_ions_domain_matrix!(
    x_list::Vector{Float64}, y_list::Vector{Float64},
    irrad_cond::Vector{AT}, gsm2::GSM2,
    mat_x::Matrix{Float64}, mat_y::Matrix{Float64}, mat_at::Matrix{Float64},
    type_AT::String, ion::Ion
)

    println("\n============================================================")
    println(" Monte Carlo Loop - Ions Domain (FULL MC - Matrix)")
    println("============================================================")

    # Extract parameters
    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp

    lower_bound_log = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq = (gsm2.rd - 10Rc)^2
    mid_radius_sq  = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    println("→ Physical parameters:")
    println("   Rp = $Rp, Rc = $Rc, Kp = $Kp")
    println("   Core radius²     = $core_radius_sq")
    println("   Mid radius²      = $mid_radius_sq")
    println("   Penumbra radius² = $penumbra_radius_sq")

    # Build lookup table
    sim_ = 1000
    impact_vec = 10 .^ range(log10(lower_bound_log),
                                stop=log10(gsm2.rd + 150Rc),
                                length=sim_)

    lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    println("\n→ Building radial dose lookup table ($sim_ samples)...")

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_vec[i], 0.0, Rk)
        _d, _r, Gyr = distribute_dose_domain(
            0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT
        )
        lookup_threads[tid][i] = Gyr
    end

    dose_vec = sum(lookup_threads)
    core_dose = dose_vec[1]

    println("   ✔ Lookup table completed")
    println("   Core dose = $core_dose Gy")

    # Matrix dimensions
    num_cells, num_domains_per_cell = size(mat_x)
    total_domains = num_cells * num_domains_per_cell

    println("\n→ Geometry:")
    println("   Cells             = $num_cells")
    println("   Domains per cell  = $num_domains_per_cell")
    println("   Total domains     = $total_domains")

    # Flatten matrices
    dom_x = vec(mat_x')
    dom_y = vec(mat_y')

    # Thread-local accumulators
    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    Np = length(x_list)

    println("\n→ Monte Carlo sampling:")
    println("   Particle count = $Np")
    println("   Threads used   = $(Threads.maxthreadid())")

    Threads.@threads for ip in 1:Np
        tid = Threads.threadid()
        local_acc = at_acc[tid]

        x = x_list[ip]
        y = y_list[ip]

        @inbounds for k in 1:total_domains
            dx = dom_x[k] - x
            dy = dom_y[k] - y
            dist_sq = dx*dx + dy*dy

            if dist_sq <= core_radius_sq
                local_acc[k] += core_dose

            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq)
                idx_l = searchsortedfirst(impact_vec, dist)

                if idx_l == 1
                    local_acc[k] += core_dose

                elseif idx_l > sim_
                    local_acc[k] += dose_vec[end]

                else
                    x1, x2 = impact_vec[idx_l-1], impact_vec[idx_l]
                    y1, y2 = dose_vec[idx_l-1], dose_vec[idx_l]
                    local_acc[k] += y1 + (y2-y1) *
                                    (dist-x1)/(x2-x1)
                end

            elseif dist_sq < penumbra_radius_sq
                local_acc[k] += Kp / dist_sq
            end
        end
    end

    final_at_row = sum(at_acc)

    # Reshape back to matrix
    mat_at .= reshape(final_at_row,
                        num_domains_per_cell,
                        num_cells)'

    println("\n✔ Full MC simulation completed successfully")
    println("============================================================\n")
end

"""
MC_loop_ions_domain_tsc_fast!(
        Npar::Int, x_cb::Float64, y_cb::Float64,
        irrad_cond::Vector{AT}, gsm2::GSM2,
        df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
        R_beam::Float64, type_AT::String, ion::Ion
    )

Performs a fast Monte Carlo calculation using the Track‑Structure approximation
(TSC) on a representative cellular domain.

# Description
This function:
1. Precomputes microdosimetric dose as a function of radial distance from
    the ion track (`dose_cell_lookup`);
2. Flattens the cell-domain coordinates into a single domain list;
3. Draws the number of primary particles from a Poisson(Npar);
4. For each particle:
    - Generates a random impact point within the beam;
    - Computes the deposited dose to each subdomain using:
        - core region dose,
        - lookup table interpolation,
        - Kp / r² behaviour in the penumbra;
5. Accumulates dose contributions in `at` (modified in place).

# Output
The function prints:
    • geometry summary  
    • particle count  
    • progress bar during the MC loop  
    • final completion message  

The DataFrame `at` is modified in place.

"""
function MC_loop_ions_domain_tsc_fast!(
    Npar::Int, x_cb::Float64, y_cb::Float64,
    irrad_cond::Vector{AT}, gsm2::GSM2,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    R_beam::Float64, type_AT::String, ion::Ion
)

    println("\n-----------------------------------------------------------")
    println("      MC_loop_ions_domain_tsc_fast! (Track Segment Mode)")
    println("-----------------------------------------------------------")
    println("Ion species: ", ion.ion)
    println("Beam center: (", x_cb, ", ", y_cb, ")")
    println("Beam radius: ", R_beam)
    println("Computing microdosimetric lookup tables…")

    # --- Extract irradiation parameters ---
    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp

    # --- Boundaries ---
    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2
    mid_radius_sq  = (gsm2.rd + 150 * Rc)^2
    penumbra_radius_sq = Rp^2

    # --- Microdosimetric Lookup Table ---
    sim_ = 1000
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)
    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    # Compute lookup table (parallel)
    @showprogress for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_vec = sum(dose_cell_lookup_threads)
    impact_vec = impact_p

    core_dose = dose_vec[1]

    # --- Prepare domain geometry ---
    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    println("Total cells: ", num_cells)
    println("Domains per cell: ", num_domains_per_cell)
    println("Total domains: ", total_domains)

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    # --- Allocate per-thread accumulators ---
    at_row_accumulators = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    # --- Monte Carlo particle count ---
    Np = rand(Poisson(Npar))
    println("\nDrawing number of primaries from Poisson(Npar)…")
    println(" → Number of primary particles: ", Np)
    println("Starting Monte Carlo loop…\n")

    # --- Progress bar for MC loop ---
    p = Progress(Np, 1, "Simulating particles… ", barlen = 40)

    Threads.@threads for _ in 1:Np
        tid = Threads.threadid()
        local_store = at_row_accumulators[tid]

        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        for k in 1:total_domains
            dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

            if dist_sq <= core_radius_sq
                local_store[k] += core_dose

            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq)
                idx_lookup = searchsortedfirst(impact_vec, dist)

                if idx_lookup == 1
                    local_store[k] += core_dose
                elseif idx_lookup > sim_
                    local_store[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_lookup-1], impact_vec[idx_lookup]
                    y1, y2 = dose_vec[idx_lookup-1], dose_vec[idx_lookup]
                    local_store[k] += y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                end

            elseif dist_sq < penumbra_radius_sq
                local_store[k] += Kp / dist_sq
            end
        end

        next!(p)
    end

    println("\nAccumulating partial thread results…")
    final_at_row = sum(at_row_accumulators)

    # Copy to DataFrame
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            at[r, c] = final_at_row[idx]
            idx += 1
        end
    end

    println("\n✔ Monte Carlo Track-Segment simulation completed successfully.\n")
end


"""
MC_loop_ions_domain_tsc_fast!(
        Npar::Int, x_cb::Float64, y_cb::Float64,
        irrad_cond::Vector{AT}, gsm2::GSM2,
        df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
        R_beam::Float64, type_AT::String, ion::Ion
    )

Fast Monte Carlo dose computation for a Track-Structure (TSC)
representative layer. It constructs a microdosimetric lookup table,
then simulates N~Poisson(Npar) primary particles and computes
energy deposition into all subcellular domains.

# Features
- Thread-parallel lookup table computation with progress bar.
- Thread-parallel Monte Carlo simulation with progress bar.
- Clean, readable printing output.
- The result is written directly into the DataFrame `at`.

# Arguments
- `Npar` : Mean number of primaries.
- `x_cb, y_cb` : Beam center.
- `R_beam` : Beam radius.
- `irrad_cond` : Irradiation parameters (Rp, Rc, Kp…).
- `gsm2` : Microdosimetric geometry model.
- `df_center_x`, `df_center_y` : Geometry tables of domain centers.
- `at` : Output table (modified in place).
- `type_AT` : Advanced microdosimetry model name.
- `ion` : Ion species.

# Notes
- Uses thread-local accumulators to avoid locking.
- Uses lookup table interpolation for mid-distance region.

"""
function MC_loop_ions_domain_tsc_fast!(
    Npar::Int, x_cb::Float64, y_cb::Float64,
    irrad_cond::Vector{AT}, gsm2::GSM2,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    R_beam::Float64, type_AT::String, ion::Ion
)

    println("\n----------------------------------------------------")
    println("     MC_loop_ions_domain_tsc_fast!  (TSC MODE)")
    println("----------------------------------------------------")
    println("Ion species       : ", ion.ion)
    println("Beam center       : (", x_cb, ", ", y_cb, ")")
    println("Beam radius       : ", R_beam)
    println("Irrad condition   : Using irrad_cond[1]")
    println("----------------------------------------------------\n")

    # --- Extract parameters ---
    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2
    mid_radius_sq  = (gsm2.rd + 150 * Rc)^2
    penumbra_radius_sq = Rp^2

    # --- LOOKUP TABLE ---
    println("Step 1/3 : Building microdosimetric lookup table...")
    sim_ = 1000

    if lower_bound_log <= 0
        error("Lower bound for impact_p is non-positive.")
    end

    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)
    dose_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    @showprogress for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        dose_lookup_threads[tid][i] = Gyr
    end

    dose_vec = sum(dose_lookup_threads)
    impact_vec = impact_p
    core_dose = dose_vec[1]

    println(" → Lookup table completed.\n")

    # --- DOMAIN GEOMETRY FLATTENING ---
    println("Step 2/3 : Preparing domain geometry...")

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    println("Cells              : ", num_cells)
    println("Domains per cell   : ", num_domains_per_cell)
    println("Total domains      : ", total_domains)

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)

    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    println(" → Domain geometry ready.\n")

    # --- THREAD-LOCAL ACCUMULATORS ---
    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    # --- MONTE CARLO LOOP ---
    println("Step 3/3 : Monte Carlo simulation")

    Np = rand(Poisson(Npar))
    println("Number of particles: ", Np, "\n")

    @showprogress for _ in 1:Np
        tid = Threads.threadid()
        local_store = at_acc[tid]

        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        for k in 1:total_domains
            dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

            if dist_sq <= core_radius_sq
                local_store[k] += core_dose

            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq)
                idx_l = searchsortedfirst(impact_vec, dist)

                if idx_l == 1
                    local_store[k] += core_dose
                elseif idx_l > sim_
                    local_store[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_l-1], impact_vec[idx_l]
                    y1, y2 = dose_vec[idx_l-1], dose_vec[idx_l]
                    local_store[k] += y1 + (y2 - y1)*(dist - x1)/(x2 - x1)
                end

            elseif dist_sq < penumbra_radius_sq
                local_store[k] += Kp / dist_sq
            end
        end
    end

    println("\nAccumulating thread results...")

    final_at_row = sum(at_acc)

    # --- WRITE BACK TO at ---
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            at[r, c] = final_at_row[idx]
            idx += 1
        end
    end

    println("----------------------------------------------------")
    println("✔ Monte Carlo Track‑Segment simulation completed.")
    println("----------------------------------------------------\n")
end

"""
MC_loop_ions_domain_fast!(
        x_list::Vector{Float64},
        y_list::Vector{Float64},
        irrad_cond::Vector{AT},
        gsm2::GSM2,
        df_center_x::DataFrame,
        df_center_y::DataFrame,
        at::DataFrame,
        type_AT::String,
        ion::Ion
    )

Fast Monte Carlo dose calculation for all domains of a given energy layer,
without Track-Segment reduction. Each particle hit position is explicitly
sampled from `(x_list, y_list)`.

# Description

This routine:

1. Builds a microdosimetric lookup table:
    - Computes dose vs radial distance from the ion track
    - Parallel computation using thread-local accumulators
    - Includes a progress bar

2. Flattens the domain geometry:
    - Converts df_center_x / df_center_y into one-dimensional coordinate arrays

3. Runs a full Monte Carlo simulation:
    - One iteration per particle hit
    - Computes dose deposition per domain using:
        • core region (constant dose)  
        • mid region (lookup + interpolation)  
        • penumbra region (Kp / r²)
    - Features a thread-safe progress bar

4. Accumulates dose into `at` (modified in place).

# Output
- The `at` DataFrame is filled with total dose per domain.
- The function prints detailed execution logs and progress indicators.

"""
function MC_loop_ions_domain_fast!(
    x_list::Vector{Float64}, 
    y_list::Vector{Float64}, 
    irrad_cond::Vector{AT},
    gsm2::GSM2,
    df_center_x::DataFrame, 
    df_center_y::DataFrame, 
    at::DataFrame,
    type_AT::String, 
    ion::Ion
)

    println("\n-------------------------------------------------------")
    println("        MC_loop_ions_domain_fast!  (FULL MC MODE)")
    println("-------------------------------------------------------")
    println("Ion species      : ", ion.ion)
    println("Particles input  : ", length(x_list))
    println("Threads used     : ", Threads.maxthreadid())
    println("-------------------------------------------------------\n")

    # -------------------------------------------------------------
    # Extract irradiation parameters
    # -------------------------------------------------------------
    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp

    lower_bound_log = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq = (gsm2.rd - 10Rc)^2
    mid_radius_sq  = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    if lower_bound_log <= 0
        error("Lower bound for lookup calculation is non-positive.")
    end

    # -------------------------------------------------------------
    # STEP 1 — Build microdosimetric lookup table
    # -------------------------------------------------------------
    println("Step 1/3 : Building lookup table...")

    sim_ = 1000
    impact_vec = 10 .^ range(log10(lower_bound_log),
                                stop=log10(gsm2.rd + 150Rc),
                                length=sim_)

    lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    @showprogress for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_vec[i], 0.0, Rk)
        _d, _r, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd,
                                                track, irrad_cond[1], type_AT)
        lookup_threads[tid][i] = Gyr
    end

    dose_vec = sum(lookup_threads)
    core_dose = dose_vec[1]

    println(" → Lookup table OK.\n")

    # -------------------------------------------------------------
    # STEP 2 — Flattening domain geometry
    # -------------------------------------------------------------
    println("Step 2/3 : Flattening domain geometry...")

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    println("Cells              : ", num_cells)
    println("Domains per cell   : ", num_domains_per_cell)
    println("Total domains      : ", total_domains)

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)

    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    println(" → Domain geometry prepared.\n")

    # -------------------------------------------------------------
    # STEP 3 — Full Monte Carlo loop
    # -------------------------------------------------------------
    println("Step 3/3 : Running Monte Carlo loop...")
    Np = length(x_list)
    println("Simulating ", Np, " primary particles.\n")

    # Thread-local accumulators
    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    @showprogress for ip in 1:Np
        tid = Threads.threadid()
        local_acc = at_acc[tid]

        x = x_list[ip]
        y = y_list[ip]

        for k in 1:total_domains
            dx = dom_x_row[k] - x
            dy = dom_y_row[k] - y
            dist_sq = dx*dx + dy*dy

            if dist_sq <= core_radius_sq
                local_acc[k] += core_dose

            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq)
                idx_l = searchsortedfirst(impact_vec, dist)

                if idx_l == 1
                    local_acc[k] += core_dose
                elseif idx_l > sim_
                    local_acc[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_l-1], impact_vec[idx_l]
                    y1, y2 = dose_vec[idx_l-1], dose_vec[idx_l]
                    local_acc[k] += y1 + (y2-y1)*(dist-x1)/(x2-x1)
                end

            elseif dist_sq < penumbra_radius_sq
                local_acc[k] += Kp / dist_sq
            end
        end
    end

    println("\nAccumulating partial results...")

    final_at_row = sum(at_acc)

    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            at[r, c] = final_at_row[idx]
            idx += 1
        end
    end

    println("-------------------------------------------------------")
    println("✔ Full Monte Carlo simulation completed successfully.")
    println("-------------------------------------------------------\n")
end

"""
MC_loop_copy_dose_domain_layer_fast_notsc!(
        cell_df::DataFrame,
        at_single::DataFrame,
        at::DataFrame,
        energy_step_to_match::Int64
    )

Copy domain doses from a set of **representative cells** (`at_single`) into the
full set of active cells in `cell_df` and the main `at` DataFrame for a specific
`energy_step`.

This function is used when **track_seg = false** (full MC mode), where `at_single`
contains doses computed for representative cells of a specific `energy_step`.
It then propagates those doses:
- into `cell_df.dose` (vector per cell),
- into `cell_df.dose_cell` (scalar per cell, mean of the domains),
- and into the corresponding rows of `at` (domain columns only).

If `track_seg = true`, this logic is normally handled by
`MC_loop_copy_dose_domain_fast!` (not this function), which matches across all layers.

# Behaviour
1. Validates required columns.
2. Ensures the presence and shape of `:dose` (Vector{Float64}) and `:dose_cell` (Float64).
3. Builds a map `(x, y) => (domain_doses_vector, scalar_dose_cell)` from `at_single`.
4. Iterates over **active cells** in `cell_df` for the given `energy_step_to_match` and copies:
    - `:dose[i] .= domain_doses_vector`
    - `:dose_cell[i] = mean(domain_doses_vector)`
    - `at[row, domain_cols] .= domain_doses_vector` (for the matched global row)
5. Prints progress and a final summary.

# Arguments
- `cell_df::DataFrame`  
    Full set of cells. Must contain: `:index, :x, :y, :layer, :is_cell, :energy_step`.
  It will be **modified in place** (`:dose` and `:dose_cell`).
- `at_single::DataFrame`  
  Doses for **representative cells** at the requested energy step. Must have `:index`.
  All other columns are treated as **domain columns** and are copied as vectors.
- `at::DataFrame`  
    Main target DataFrame (same structure as `at_single` but for all cells).
  Will be **modified in place** for rows belonging to `energy_step_to_match`.
- `energy_step_to_match::Int64`  
    The energy step whose cells will be updated.

# Notes
- This function assumes that representatives are matched by `(x, y)` coordinates.
- If a cell’s `(x, y)` is not present in the representative map, it receives **zero dose** and a warning is emitted.
- If the number of domain columns differs between `at_single` and `at`, the row update into `at` is skipped with an error message, but `cell_df` still receives its doses.

"""
function MC_loop_copy_dose_domain_layer_fast_notsc!(
    cell_df::DataFrame,
    at_single::DataFrame,
    at::DataFrame,
    energy_step_to_match::Int64
)

    println("\n-----------------------------------------------------------")
    println("  MC_loop_copy_dose_domain_layer_fast_notsc!  (COPY DOSES)")
    println("-----------------------------------------------------------")
    println("Target energy_step : ", energy_step_to_match)
    println("Total cells        : ", nrow(cell_df))
    println("Threads available  : ", Threads.maxthreadid())
    println("-----------------------------------------------------------\n")

    t_start = time()

    # -------------------------------
    # 1) Input validation
    # -------------------------------
    req_cols_cell = (:index, :x, :y, :layer, :is_cell, :energy_step)
    cell_cols = Set(propertynames(cell_df))

    for c in req_cols_cell
        if !(c in cell_cols)
            error("cell_df is missing required column :$c")
        end
    end

    if !(:index in propertynames(at_single))
        error("at_single is missing required column :index")
    end
    if !(:index in propertynames(at))
        error("at is missing required column :index")
    end

    num_cells = nrow(cell_df)

    # Determine domain columns from at_single
    domain_cols = names(at_single, Not(:index))
    num_domains = length(domain_cols)

    if num_domains == 0 && nrow(at_single) > 0
        @warn "No domain columns found in 'at_single' (excluding :index). Proceeding with empty dose vectors."
    elseif nrow(at_single) == 0
        @warn "'at_single' is empty for energy_step $energy_step_to_match. No doses to copy. Initializing columns and exiting."
    end

    # -------------------------------
    # 2) Initialize dose columns
    # -------------------------------
    if !(:dose in propertynames(cell_df)) ||
        !(eltype(cell_df.dose) <: AbstractVector)
        println("Initializing cell_df.dose as Vector{Float64}[$num_domains] per cell…")
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    end

    if !(:dose_cell in propertynames(cell_df)) ||
        !(eltype(cell_df.dose_cell) <: AbstractFloat)
        println("Initializing cell_df.dose_cell as Float64…")
        cell_df.dose_cell = zeros(Float64, num_cells)
    end

    if nrow(at_single) == 0
        println("\nFinished copy: 'at_single' was empty. Nothing to do for energy_step $energy_step_to_match.")
        println("Elapsed: $(round(time() - t_start, digits=3)) s\n")
        return
    end

    # -------------------------------
    # 3) Build (x,y) -> dose map from at_single
    # -------------------------------
    println("Step 1/2 : Building representative (x,y) → dose map…")
    xy_to_dose_data = Dict{Tuple{Float64,Float64},Tuple{Vector{Float64},Float64}}()
    sizehint!(xy_to_dose_data, nrow(at_single))

    # inverse index map for cell_df.index -> row
    cell_df_index_to_row = Dict(idx => r for (r, idx) in enumerate(cell_df.index))

    @showprogress for row_single in eachrow(at_single)
        representative_index = row_single.index
        cell_df_row_idx = get(cell_df_index_to_row, representative_index, 0)
        if cell_df_row_idx == 0
            @warn "Representative index $representative_index not found in cell_df. Skipping."
            continue
        end

        if cell_df.energy_step[cell_df_row_idx] != energy_step_to_match
            @warn "Representative index $representative_index belongs to energy_step $(cell_df.energy_step[cell_df_row_idx]) (expected $energy_step_to_match). Skipping."
            continue
        end

        rep_x = Float64(cell_df.x[cell_df_row_idx])
        rep_y = Float64(cell_df.y[cell_df_row_idx])

        domain_doses_vector = Vector{Float64}(undef, num_domains)
        @inbounds for (j, c) in enumerate(domain_cols)
            domain_doses_vector[j] = Float64(row_single[c])
        end
        scalar_dose_cell = isempty(domain_doses_vector) ? 0.0 : mean(domain_doses_vector)

        xy_to_dose_data[(rep_x, rep_y)] = (domain_doses_vector, scalar_dose_cell)
    end
    println(" → Map size: ", length(xy_to_dose_data), " representative coordinates.\n")

    # -------------------------------
    # 4) Copy doses to target cells
    # -------------------------------
    println("Step 2/2 : Copying doses into cell_df and at…")

    at_index_to_row = Dict(idx => r for (r, idx) in enumerate(at.index))

    target_idx = findall(i -> (cell_df.energy_step[i] == energy_step_to_match) &&
                                (cell_df.is_cell[i] == 1), 1:num_cells)

    n_targets = length(target_idx)
    println("Active cells in this energy_step: ", n_targets)

    p = Progress(n_targets; dt=1, desc="Copying doses…", barlen=40)

    lk = ReentrantLock()
    Threads.@threads for t in 1:n_targets
        i = target_idx[t]
    
        coords = (Float64(cell_df.x[i]), Float64(cell_df.y[i]))
        dose_data = get(xy_to_dose_data, coords, nothing)
    
        local domain_doses_vector::Vector{Float64}
        local scalar_dose_cell::Float64
    
        if dose_data === nothing
            @warn "Coordinates $coords (cell index $(cell_df.index[i])) not found among representatives. Assigning zero dose."
            domain_doses_vector = zeros(Float64, num_domains)
            scalar_dose_cell = 0.0
        else
            domain_doses_vector, scalar_dose_cell = dose_data
        end
    
        if length(cell_df.dose[i]) != num_domains
            cell_df.dose[i] = zeros(Float64, num_domains)
        end
    
        cell_df.dose[i] .= domain_doses_vector
        cell_df.dose_cell[i] = scalar_dose_cell
    
        cell_idx = cell_df.index[i]
        at_row_idx = get(at_index_to_row, cell_idx, 0)
    
        if at_row_idx != 0
            if length(names(at, Not(:index))) == num_domains && !isempty(domain_cols)
                at[at_row_idx, domain_cols] .= domain_doses_vector
            else
                @error "Domain column count mismatch in 'at' (row for index=$cell_idx). Skipping 'at' update."
            end
        end
    
        # progress bar update (thread-safe)
        lock(lk) do
            next!(p)
        end
    end

    println("\n✔ Finished copying doses for energy_step $energy_step_to_match.")
    println("Elapsed: $(round(time() - t_start, digits=3)) s\n")
end

"""
MC_loop_copy_dose_domain_fast!(
        cell_df::DataFrame,
        at_single::DataFrame,
        at::DataFrame;
        verbose::Bool = false
    )

Copy domain-level doses from a **representative plane** (`at_single`) into  
**all layers** in the full cell grid (`cell_df`, `at`).  
This function is used in **Track-Segment mode**, where only a single (x,y)
slice is simulated microdosimetrically and the resulting domain doses must be
propagated to all cells that share the same transverse coordinates.

# Workflow

1. **Input validation**
    - Ensures that required columns exist in all DataFrames.

2. **Initialization**
    - Creates (if missing):
        - `cell_df.dose`      : Vector of domain doses for each cell.
        - `cell_df.dose_cell` : Scalar mean dose for each cell.

3. **Coordinate grouping**
    - Creates dictionary:  
        `(x, y) → list of all cell indices across all layers`

4. **Dose propagation**
    - For each representative cell in `at_single`:
        - Find its `(x, y)` in `cell_df`
        - Retrieve all matching cells at all depths
        - Copy:
         * domain dose vector → `at`
         * domain dose vector → `cell_df.dose`
         * scalar mean → `cell_df.dose_cell`
        - Inactive cells (`is_cell = 0`) receive zero dose.

# Arguments
- `cell_df::DataFrame`  
    Full 3D cell dataframe. Will be updated in-place.

- `at_single::DataFrame`  
    Domain doses for representative layer (Track-Segment slice).
    Must contain an `:index` column.

- `at::DataFrame`  
    Full domain-level dataframe. Updated in-place.

# Keywords
- `verbose::Bool=false`  
    If `true`, prints detailed per-index mapping and copy actions.

# Returns
- Nothing. All updates are done **in-place**.

"""
function MC_loop_copy_dose_domain_fast!(
    cell_df::DataFrame,
    at_single::DataFrame,
    at::DataFrame;
    verbose::Bool = false
)

    vprintln(args...) = (verbose ? println("[DEBUG] ", args...) : nothing)

    println("\n-----------------------------------------------------------")
    println("         MC_loop_copy_dose_domain_fast!  (TSC COPY)")
    println("-----------------------------------------------------------")

    # ---------------------------------------------------------
    # 1. VALIDATION
    # ---------------------------------------------------------
    if !hasproperty(cell_df, :index) || !hasproperty(at_single, :index) || !hasproperty(at, :index)
        error("Missing :index column in one or more DataFrames.")
    end
    if !hasproperty(cell_df, :x) || !hasproperty(cell_df, :y)
        error("cell_df must contain :x and :y columns.")
    end

    num_cells = nrow(cell_df)
    domain_cols = names(at_single, Not(:index))
    num_domains = length(domain_cols)

    println("Representative rows (at_single): ", nrow(at_single))
    println("Domain columns                : ", num_domains)
    println("Total cells                   : ", num_cells)

    # ---------------------------------------------------------
    # 2. Initialize dose fields
    # ---------------------------------------------------------
    if !hasproperty(cell_df, :dose)
        println("Initializing cell_df.dose column...")
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    elseif !(eltype(cell_df.dose) <: AbstractVector)
        println("Reinitializing invalid cell_df.dose column...")
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    end

    if !hasproperty(cell_df, :dose_cell)
        println("Initializing cell_df.dose_cell column...")
        cell_df.dose_cell = zeros(Float64, num_cells)
    elseif !(eltype(cell_df.dose_cell) <: AbstractFloat)
        println("Reinitializing invalid cell_df.dose_cell column...")
        cell_df.dose_cell = zeros(Float64, num_cells)
    end

    if nrow(at_single) == 0
        println("WARNING: at_single is empty → nothing to copy.")
        return
    end

    # ---------------------------------------------------------
    # 3. Build mapping (x,y) → all cell indices
    # ---------------------------------------------------------
    println("Building coordinate-to-index map...")
    xy_to_indices = Dict{Tuple{Float64,Float64}, Vector{Int}}()

    for row in eachrow(cell_df)
        coords = (Float64(row.x), Float64(row.y))
        push!(get!(xy_to_indices, coords, Vector{Int}()), row.index)
    end

    println("Map contains ", length(keys(xy_to_indices)), " unique (x,y) positions.\n")

    # ---------------------------------------------------------
    # 4. Main copy loop
    # ---------------------------------------------------------
    println("Copying doses from representative plane to all layers...\n")

    for row_single in eachrow(at_single)

        representative_index = row_single.index
        vprintln("Processing representative index: ", representative_index)

        # Domain vector for this representative cell
        domain_vec = Vector(row_single[domain_cols])
        scalar_dose = isempty(domain_vec) ? 0.0 : mean(domain_vec)

        # Locate representative cell coordinates
        rep_pos = findfirst(==(representative_index), cell_df.index)
        if rep_pos === nothing
            @warn "Representative index $representative_index not found in cell_df. Skipping."
            continue
        end

        rep_coords = (Float64(cell_df.x[rep_pos]), Float64(cell_df.y[rep_pos]))
        vprintln("Coordinates: ", rep_coords)

        # Look up all cells aligned vertically
        target_ids = get(xy_to_indices, rep_coords, nothing)
        if target_ids === nothing
            @warn "Coordinates $(rep_coords) not found in xy_to_indices. Skipping representative index $representative_index"
            continue
        end

        # Copy into each corresponding cell
        for idx in target_ids
            at_row   = findfirst(==(idx), at.index)
            cell_row = findfirst(==(idx), cell_df.index)

            if at_row === nothing || cell_row === nothing
                @warn "Index $idx missing in either 'at' or 'cell_df'. Skipping."
                continue
            end

            # --- Copy to at ---
            if num_domains > 0
                at[at_row, domain_cols] .= domain_vec
            end

            # --- Copy to cell_df ---
            if cell_df.is_cell[cell_row] == 1
                cell_df.dose[cell_row]      = copy(domain_vec)
                cell_df.dose_cell[cell_row] = scalar_dose
            else
                # inactive cells → zero dose
                cell_df.dose[cell_row]      = zeros(Float64, num_domains)
                cell_df.dose_cell[cell_row] = 0.0
            end
        end
    end

    println("✔ Finished copying doses.\n")
end
