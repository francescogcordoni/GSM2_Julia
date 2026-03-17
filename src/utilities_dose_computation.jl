#! ============================================================================
#! utilities_dose_computation.jl
#!
#! FUNCTIONS
#! ---------
#~ High-Level MC Wrapper
#?   MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x,
#                df_center_y, at, gsm2_cycle, type_AT, track_seg; x_cb, y_cb)
#       Top-level MC wrapper. Dispatches to TSC (track_seg=true) or layered
#       (track_seg=false) matrix kernels. Handles representative cell selection,
#       DataFrame↔matrix conversion, and dose copy-back to full domain.
#       If single_particle = true then it computes the dose for a single particle.
#?   MC_precompute_lut!(ion, , irrad_cond, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg) -> Dict
#
#~ DataFrame ↔ Matrix Conversion
#?   dataframes_to_matrices(df_x, df_y, df_at) -> (mat_x, mat_y, mat_at)
#       Extracts domain columns (excluding :index) → dense Matrix{Float64}.
#?   matrix_to_dataframe!(df, mat) -> Nothing
#       Writes matrix columns back into DataFrame domain columns in-place.
#
#~ MC Kernels — Matrix-Based (fast, no DataFrame overhead)
#?   MC_loop_ions_domain_tsc_matrix!(Npar, x_cb, y_cb, irrad_cond, gsm2,
#                                    mat_x, mat_y, mat_at, R_beam, type_AT, single_particle)
#       TSC Monte Carlo on flattened matrix domain. Poisson-sampled particle
#       count. Thread-local accumulators. Mutates mat_at in-place.
#?   MC_loop_ions_domain_matrix!(x_list, y_list, irrad_cond, gsm2,
#                                mat_x, mat_y, mat_at, type_AT, ion)
#       Full-MC (explicit particle positions) on matrix domain.
#       Thread-local accumulators. Mutates mat_at in-place.
#
#~ MC Kernels — DataFrame-Based (with progress bars)
#?   MC_loop_ions_domain_tsc_fast!(Npar, x_cb, y_cb, irrad_cond, gsm2,
#                                  df_center_x, df_center_y, at,
#                                  R_beam, type_AT, ion)
#       TSC Monte Carlo directly on DataFrame domain geometry.
#       Builds lookup table, Poisson-samples particles, accumulates dose.
#       Writes result into at in-place.
#       NOTE: two overloads exist — original (Progress bar) and revised (@showprogress).
#?   MC_loop_ions_domain_fast!(x_list, y_list, irrad_cond, gsm2,
#                              df_center_x, df_center_y, at, type_AT, ion)
#       Full-MC (explicit particle positions) on DataFrame domain geometry.
#       Writes result into at in-place.
#
#~ Dose Copy-Back
#?   MC_loop_copy_dose_domain_fast!(cell_df, at_single, at; verbose) -> Nothing
#       TSC mode: propagates domain doses from a representative plane (at_single)
#       to all z-layers in cell_df and at, matched by (x, y) coordinates.
#?   MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df, at_single, at,
#                                               energy_step_to_match) -> Nothing
#       Full-MC mode: copies domain doses from representative cells of a specific
#       energy_step into cell_df.dose, cell_df.dose_cell, and at in-place.
#! ============================================================================

"""
    MC_dose_CPU!(
        ion::Ion, Npar::Int64, R_beam::Float64,
        irrad_cond::Vector{AT}, cell_df_copy::DataFrame,
        df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
        gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
        x_cb::Float64 = 0.0, y_cb::Float64 = 0.0, single_particle::Bool = false
    ) -> Nothing

CPU Monte Carlo (MC) dose-deposition driver with two execution modes:
1) **Track-Segment (TSC) matrix optimization** (`track_seg = true`)
2) **Layered (non‑TSC) matrix optimization** (`track_seg = false`)

The routine orchestrates sampling of particle impact points, reduction of the
cell domain to representative indices, matrix assembly for geometry/AT state,
MC transport/update loops, and scatter of the updated dose/AT fields back to
the full-resolution DataFrames.

It **mutates** `cell_df_copy` (dose/energy fields) and `at` (AT state) in place
and prints progress/timing info.

# Arguments

## Positional
- `ion::Ion`  
  Ion species descriptor for the transport kernels.

- `Npar::Int64`  
  Expected number of primaries crossing the target (mean for Poisson sampling
  in layered mode).

- `R_beam::Float64`  
  Beam radius for sampling impact coordinates on a circle (2D domain).

- `irrad_cond::Vector{AT}`  
  Irradiation / AT-condition objects.  
  *TSC mode*: a vector of length 1 is expected (`[irrad_cond[1]]` is used).  
  *Layered mode*: one element per energy layer (indexed by `energy_step`).

- `cell_df_copy::DataFrame`  
  Working copy of the cell lattice & dose table. Must include (at minimum)
  the following columns:
  - `:index::Int`: unique cell/voxel identifier
  - `:x, :y::Float64`: planar coordinates
  - `:is_cell::Int`: 1 for cell voxels, 0 otherwise
  - `:energy_step::Int`: layer identifier used in layered mode
  Additional dose/track/AT fields required by your downstream kernels.

- `df_center_x::DataFrame`, `df_center_y::DataFrame`  
  DataFrames aligned by `:index` that contain geometry center information for
  x and y respectively. They are reduced to representative subsets and then
  converted into matrices for fast kernels.

- `at::DataFrame`  
  AT-state table aligned by `:index`. Updated in place based on transport.

- `gsm2_cycle::Vector{GSM2}`  
  Cycle of GSM2 models/settings. This routine uses `gsm2_cycle[1]`.

- `type_AT::String`  
  Identifier of the AT model used by matrix kernels (e.g., `"KC"`).

- `track_seg::Bool`  
  Selects execution mode.
  - `true`  → TSC matrix optimization (`MC_loop_ions_domain_tsc_matrix!`)
  - `false` → layered domain matrix optimization (`MC_loop_ions_domain_matrix!`)

## Keywords
- `x_cb::Float64 = 0.0`, `y_cb::Float64 = 0.0`  
  Beam center coordinates used when sampling particle impacts.

- `single_particle::Bool = false`  
  If `true`, forces a single primary (`Np = 1`) in layered mode (useful for
  debugging/diagnostics). Otherwise, `Np ~ Poisson(Npar)`.

# Workflow
## Common
- Start timer and pick `gsm2 = gsm2_cycle[1]`.
- Print a concise header and mode.
## Track-Segment mode (`track_seg = true`)
1. Filter active cells: `is_cell == 1`. If none, warns and returns early.
2. Group by `(:x, :y)` to determine **representative indices**, reducing the
   domain (one index per unique `(x,y)` pair).
3. Filter `df_center_x`, `df_center_y`, and `at` to these representative indices
   and convert to matrices via `dataframes_to_matrices(...)` → `(mat_x, mat_y, mat_at)`.
4. Run the TSC kernel:
MC_loop_ions_domain_tsc_matrix!(
Npar, x_cb, y_cb, [irrad_cond[1]], gsm2,
mat_x, mat_y, mat_at, R_beam, type_AT, ion,
single_particle
)
5. Propagate matrix updates back to DataFrames (`matrix_to_dataframe!`) and copy
dose onto the full `cell_df_copy` (`MC_loop_copy_dose_domain_fast!`).
## Layered mode (`track_seg = false`)
1. Determine number of primaries:
- `Np = 1` if `single_particle = true`
- else `Np ~ Poisson(Npar)`
2. Sample `Np` impact points `(x_list, y_list)` on the circle with center
`(x_cb, y_cb)` and radius `R_beam` using `GenerateHit_Circle`.
3. For each `id` in `unique(cell_df_copy.energy_step)`:
- Filter active cells in that layer (`is_cell == 1 && energy_step == id`),
  skip empty layers.
- Determine representative indices by grouping `(:x, :y)` and select the
  reduced `df_center_x`, `df_center_y`, and `at` slices.
- Convert to matrices and call the layered kernel:
  MC_loop_ions_domain_matrix!(
      x_list, y_list, [irrad_cond[id]], gsm2,
      mat_x, mat_y, mat_at, type_AT, ion
  )
  ```
- Scatter back matrix → DataFrame (`matrix_to_dataframe!`) and copy layer
  dose into `cell_df_copy` via
  `MC_loop_copy_dose_domain_layer_fast_notsc!(..., id)`.
## Finalization
- Print total elapsed time (s) and return `nothing`.

# Returns
`nothing` — results are written in place to:
- `cell_df_copy` (dose/energy accumulation)
- `at` (updated AT state)

# Side Effects
- Mutates `cell_df_copy` and `at`.
- Allocates temporary reduced DataFrames and matrices.
- Prints progress and timing information.
- Not thread‑safe w.r.t. shared DataFrames unless externally synchronized.

# Assumptions / Requirements
- `df_center_x`, `df_center_y`, and `at` are **aligned by `:index`** with
`cell_df_copy`.
- `irrad_cond` is indexed consistently with `energy_step` in layered mode, and
has length 1 in TSC mode.
- Helper functions are available in scope:
- `dataframes_to_matrices`, `matrix_to_dataframe!`
- `MC_loop_ions_domain_tsc_matrix!`, `MC_loop_ions_domain_matrix!`
- `MC_loop_copy_dose_domain_fast!`,
 `MC_loop_copy_dose_domain_layer_fast_notsc!`
- `GenerateHit_Circle`
- `Poisson` distribution is available (e.g., `using Distributions`).

# Errors / Early exits
- TSC mode: if no rows with `is_cell == 1`, logs a warning and returns early.
- Layered mode: layers with no active cells are skipped with a message.

# Notes
- In layered mode, the log line “Sampling Np particle hits...” prints the
actual sampled `Np`.
- Group‑by reduction on `(:x, :y)` can significantly decrease matrix sizes and
speed up the kernels; ensure coordinates are properly discretized to avoid
spurious uniqueness from floating‑point noise.
- For reproducibility, seed the RNG before Poisson sampling and hit generation
if needed (e.g., `Random.seed!(...)`).

# Example
```julia
# Assuming all inputs are prepared and aligned by :index
MC_dose_CPU!(
 ion, Npar, R_beam,
 irrad_cond, cell_df_copy,
 df_center_x, df_center_y, at,
 gsm2_cycle, "KC", true;       # track_seg = true (TSC)
 x_cb=0.0, y_cb=0.0, single_particle=false
)

# Layered (non-TSC) mode
MC_dose_CPU!(
 ion, Npar, R_beam,
 irrad_cond, cell_df_copy,
 df_center_x, df_center_y, at,
 gsm2_cycle, "KC", false;      # track_seg = false (layered)
 x_cb=0.0, y_cb=0.0, single_particle=true
)
```
"""
function MC_dose_CPU!(
    ion::Ion, Npar::Int64, R_beam::Float64,
    irrad_cond::Vector{AT}, cell_df_copy::DataFrame,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
    x_cb::Float64 = 0., y_cb::Float64 = 0., single_particle = false
    )
    println("\n───────────────────────────────────────────────")
    println("🔧  Running MC_dose_fast!   (track_seg = $track_seg)")
    println("───────────────────────────────────────────────")

    t_start = time()
    gsm2 = gsm2_cycle[1]

    if track_seg
        println("• Mode: Track-Segment Matrix Optimization")

        cell_df_is = filter(row -> row.is_cell == 1, cell_df_copy)
        nrow(cell_df_is) == 0 && (@warn "No cells with is_cell = 1 → skipping."; return)

        grouped_df      = combine(groupby(cell_df_is, [:x, :y]),
                                    :index => first => :representative_index)
        rep_indices_set = Set(grouped_df.representative_index)
        println("  → Found $(length(rep_indices_set)) representative cells")

        cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
        cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
        at_single        = filter(row -> row.index in rep_indices_set, at)

        mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

        MC_loop_ions_domain_tsc_matrix!(
            Npar, x_cb, y_cb, [irrad_cond[1]], gsm2,
            mat_x, mat_y, mat_at, R_beam, type_AT, ion,
            single_particle
        )

        matrix_to_dataframe!(at_single, mat_at)
        MC_loop_copy_dose_domain_fast!(cell_df_copy, at_single, at)

    else
        println("• Mode: Layered (non-TSC) Matrix Optimization")

        Np::Int = single_particle ? 1 : rand(Poisson(Npar))
        println("• Sampling $Np particle hits...")

        x_list = Vector{Float64}(undef, Np)
        y_list = Vector{Float64}(undef, Np)
        Threads.@threads for ip in 1:Np
            x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
        end

        for id in unique(cell_df_copy.energy_step)
            println("  → Layer $id")
            cell_df_is = filter(row -> (row.is_cell == 1) && (row.energy_step == id), cell_df_copy)
            if nrow(cell_df_is) == 0
                println("    (empty → skip)")
                continue
            end

            grouped_df      = combine(groupby(cell_df_is, [:x, :y]),
                                        :index => first => :representative_index)
            rep_indices_set = Set(grouped_df.representative_index)

            cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
            cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
            at_single        = filter(row -> row.index in rep_indices_set, at)

            mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

            MC_loop_ions_domain_matrix!(
                x_list, y_list, [irrad_cond[id]], gsm2,
                mat_x, mat_y, mat_at, type_AT, ion
            )

            matrix_to_dataframe!(at_single, mat_at)
            MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, at_single, at, id)
        end
    end

    dt = round(time() - t_start; digits=3)
    println("───────────────────────────────────────────────")
    println("🎉 MC_dose! finished. Total time: $(dt)s")
    println("───────────────────────────────────────────────\n")
end

#! ============================================================================
#! DataFrame ↔ Matrix Conversion
#! ============================================================================

"""
    dataframes_to_matrices(df_x, df_y, df_at) -> (mat_x, mat_y, mat_at)

Extracts domain columns (all except `:index`) from three DataFrames and returns
them as `Matrix{Float64}`. Inverse of `matrix_to_dataframe!`.

# Example
```julia
mat_x, mat_y, mat_at = dataframes_to_matrices(df_center_x, df_center_y, at)
```
"""
function dataframes_to_matrices(df_x::DataFrame, df_y::DataFrame, df_at::DataFrame)
    domain_cols = names(df_x, Not(:index))
    mat_x  = Matrix{Float64}(df_x[:, domain_cols])
    mat_y  = Matrix{Float64}(df_y[:, domain_cols])
    mat_at = Matrix{Float64}(df_at[:, domain_cols])
    return mat_x, mat_y, mat_at
end

"""
    matrix_to_dataframe!(df::DataFrame, mat::Matrix{Float64}) -> Nothing

Writes matrix columns back into DataFrame domain columns in-place.
Column count must match exactly. Inverse of `dataframes_to_matrices`.

# Example
```julia
matrix_to_dataframe!(at_single, mat_at)
```
"""
function matrix_to_dataframe!(df::DataFrame, mat::Matrix{Float64})
    domain_cols = names(df, Not(:index))
    @assert size(mat, 2) == length(domain_cols) "Matrix column count does not match DataFrame"
    for (j, col) in enumerate(domain_cols)
        df[!, col] = mat[:, j]
    end
end

#! ============================================================================
#! MC Kernels — Matrix-Based
#! ============================================================================

"""
    MC_loop_ions_domain_tsc_matrix!(Npar, x_cb, y_cb, irrad_cond, gsm2,
                                    mat_x, mat_y, mat_at,
                                    R_beam, type_AT, single_particle)

TSC Monte Carlo on a flattened matrix domain.
Builds a radial dose lookup table, samples N~Poisson(Npar) particles,
deposits dose into each domain via core/mid/penumbra regions.
Thread-local accumulators. Mutates `mat_at` in-place.

Regions:
- core      : dist² ≤ (rd - 10Rc)²        → constant core_dose
- mid       : core < dist² ≤ (rd+150Rc)²  → lookup table interpolation
- penumbra  : mid < dist² < Rp²            → Kp / dist²

# Example
```julia
MC_loop_ions_domain_tsc_matrix!(Npar, 0.0, 0.0, irrad_cond, gsm2,
                                    mat_x, mat_y, mat_at, R_beam, "KC", single_particle)
```
"""
function MC_loop_ions_domain_tsc_matrix!(
    Npar::Int, x_cb::Float64, y_cb::Float64,
    irrad_cond::Vector{AT}, gsm2::GSM2,
    mat_x::Matrix{Float64}, mat_y::Matrix{Float64}, mat_at::Matrix{Float64},
    R_beam::Float64, type_AT::String, single_particle::Bool = false
)
    println("\n============================================================")
    println(" MC Loop - TSC Matrix")
    println("============================================================")

    Rp = irrad_cond[1].Rp; Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp; Rk = Rp

    lower_bound_log    = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq     = (gsm2.rd - 10Rc)^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    # Radial dose lookup table
    sim_ = 1000
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150Rc), length=sim_)
    dose_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        dose_lookup_threads[tid][i] = Gyr
    end

    dose_vec  = sum(dose_lookup_threads)
    impact_vec = impact_p
    core_dose  = dose_vec[1]

    num_cells, num_domains_per_cell = size(mat_x)
    total_domains = num_cells * num_domains_per_cell

    dom_x = vec(mat_x')
    dom_y = vec(mat_y')

    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    Np::Int = single_particle ? 1 : rand(Poisson(Npar))
    println("Expected: $Npar  |  Sampled: $Np  |  Threads: $(Threads.maxthreadid())")

    Threads.@threads for _ in 1:Np
        tid = Threads.threadid()
        local_store = at_acc[tid]
        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        @inbounds for k in 1:total_domains
            dist_sq = (dom_x[k] - x)^2 + (dom_y[k] - y)^2

            if dist_sq <= core_radius_sq
                local_store[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist  = sqrt(dist_sq)
                idx_l = searchsortedfirst(impact_vec, dist)
                if idx_l == 1
                    local_store[k] += core_dose
                elseif idx_l > sim_
                    local_store[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_l-1], impact_vec[idx_l]
                    y1, y2 = dose_vec[idx_l-1], dose_vec[idx_l]
                    local_store[k] += y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                end
            elseif dist_sq < penumbra_radius_sq
                local_store[k] += Kp / dist_sq
            end
        end
    end

    mat_at .= reshape(sum(at_acc), num_domains_per_cell, num_cells)'
    println("✔ TSC matrix simulation complete.\n")
end

"""
    MC_loop_ions_domain_matrix!(x_list, y_list, irrad_cond, gsm2,
                                mat_x, mat_y, mat_at, type_AT, ion)

Full-MC (explicit particle positions) on a matrix domain.
Same dose deposition logic as the TSC variant but accepts pre-generated
hit positions `(x_list, y_list)` instead of sampling them internally.
Thread-local accumulators. Mutates `mat_at` in-place.

# Example
```julia
MC_loop_ions_domain_matrix!(x_list, y_list, [irrad_cond[id]], gsm2,
                                mat_x, mat_y, mat_at, "KC", ion)
```
"""
function MC_loop_ions_domain_matrix!(
    x_list::Vector{Float64}, y_list::Vector{Float64},
    irrad_cond::Vector{AT}, gsm2::GSM2,
    mat_x::Matrix{Float64}, mat_y::Matrix{Float64}, mat_at::Matrix{Float64},
    type_AT::String, ion::Ion
)
    println("\n============================================================")
    println(" MC Loop - Full MC Matrix")
    println("============================================================")

    Rp = irrad_cond[1].Rp; Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp; Rk = Rp

    lower_bound_log    = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq     = (gsm2.rd - 10Rc)^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    sim_ = 1000
    impact_vec = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150Rc), length=sim_)
    lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_vec[i], 0.0, Rk)
        _d, _r, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        lookup_threads[tid][i] = Gyr
    end

    dose_vec  = sum(lookup_threads)
    core_dose = dose_vec[1]

    num_cells, num_domains_per_cell = size(mat_x)
    total_domains = num_cells * num_domains_per_cell
    dom_x = vec(mat_x'); dom_y = vec(mat_y')

    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]
    Np     = length(x_list)
    println("Particles: $Np  |  Threads: $(Threads.maxthreadid())")

    Threads.@threads for ip in 1:Np
        tid       = Threads.threadid()
        local_acc = at_acc[tid]
        x = x_list[ip]; y = y_list[ip]

        @inbounds for k in 1:total_domains
            dx = dom_x[k] - x; dy = dom_y[k] - y
            dist_sq = dx*dx + dy*dy

            if dist_sq <= core_radius_sq
                local_acc[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist  = sqrt(dist_sq)
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

    mat_at .= reshape(sum(at_acc), num_domains_per_cell, num_cells)'
    println("✔ Full MC matrix simulation complete.\n")
end

#! ============================================================================
#! MC Kernels — DataFrame-Based
#! ============================================================================

"""
    MC_loop_ions_domain_tsc_fast!(Npar, x_cb, y_cb, irrad_cond, gsm2,
                                    df_center_x, df_center_y, at,
                                    R_beam, type_AT, ion)

TSC Monte Carlo directly on DataFrame domain geometry.
Builds a lookup table, samples N~Poisson(Npar) particles, accumulates domain
dose using core/mid/penumbra logic. Writes result into `at` in-place.

NOTE: this function has two overloads with slightly different progress bar
implementations (Progress vs @showprogress). The second overload (below) is
the revised version with cleaner step logging.

# Example
```julia
MC_loop_ions_domain_tsc_fast!(Npar, 0.0, 0.0, irrad_cond, gsm2,
                                df_center_x, df_center_y, at, R_beam, "KC", ion)
```
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

    Rp = irrad_cond[1].Rp; Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp; Rk = Rp

    lower_bound_log    = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq     = (gsm2.rd - 10Rc)^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    sim_ = 1000
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150Rc), length=sim_)
    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    @showprogress for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_vec   = sum(dose_cell_lookup_threads)
    impact_vec = impact_p
    core_dose  = dose_vec[1]

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells            = size(df_center_x, 1)
    total_domains        = num_cells * num_domains_per_cell

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells, c in 1:num_domains_per_cell
        dom_x_row[idx] = df_center_x[r, c]
        dom_y_row[idx] = df_center_y[r, c]
        idx += 1
    end

    at_row_accumulators = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]
    Np = rand(Poisson(Npar))
    println("Particles: $Np")

    p = Progress(Np, 1, "Simulating particles… ", barlen=40)
    Threads.@threads for _ in 1:Np
        tid         = Threads.threadid()
        local_store = at_row_accumulators[tid]
        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        for k in 1:total_domains
            dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

            if dist_sq <= core_radius_sq
                local_store[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist      = sqrt(dist_sq)
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

    final_at_row = sum(at_row_accumulators)
    idx = 1
    for r in 1:num_cells, c in 1:num_domains_per_cell
        at[r, c] = final_at_row[idx]; idx += 1
    end

    println("✔ TSC DataFrame simulation complete.\n")
end

"""
    MC_loop_ions_domain_fast!(x_list, y_list, irrad_cond, gsm2,
                                df_center_x, df_center_y, at, type_AT, ion)

Full-MC (explicit particle positions) on DataFrame domain geometry.
Builds lookup table, deposits dose from each hit in `(x_list, y_list)`.
Writes result into `at` in-place.

# Example
```julia
MC_loop_ions_domain_fast!(x_list, y_list, [irrad_cond[id]], gsm2,
                            df_center_x, df_center_y, at, "KC", ion)
```
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

    Rp = irrad_cond[1].Rp; Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp; Rk = Rp

    lower_bound_log    = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq     = (gsm2.rd - 10Rc)^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    lower_bound_log <= 0 && error("Lower bound for lookup is non-positive.")

    # Lookup table
    sim_ = 1000
    impact_vec = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150Rc), length=sim_)
    lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    @showprogress for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_vec[i], 0.0, Rk)
        _d, _r, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        lookup_threads[tid][i] = Gyr
    end
    dose_vec  = sum(lookup_threads)
    core_dose = dose_vec[1]

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells            = size(df_center_x, 1)
    total_domains        = num_cells * num_domains_per_cell

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells, c in 1:num_domains_per_cell
        dom_x_row[idx] = df_center_x[r, c]
        dom_y_row[idx] = df_center_y[r, c]
        idx += 1
    end

    at_acc = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]
    Np     = length(x_list)
    println("Particles: $Np  |  Threads: $(Threads.maxthreadid())")

    @showprogress for ip in 1:Np
        tid       = Threads.threadid()
        local_acc = at_acc[tid]
        x = x_list[ip]; y = y_list[ip]

        for k in 1:total_domains
            dx = dom_x_row[k] - x; dy = dom_y_row[k] - y
            dist_sq = dx*dx + dy*dy

            if dist_sq <= core_radius_sq
                local_acc[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist  = sqrt(dist_sq)
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

    final_at_row = sum(at_acc)
    idx = 1
    for r in 1:num_cells, c in 1:num_domains_per_cell
        at[r, c] = final_at_row[idx]; idx += 1
    end

    println("✔ Full MC DataFrame simulation complete.\n")
end

#! ============================================================================
#! Dose Copy-Back
#! ============================================================================

"""
    MC_loop_copy_dose_domain_fast!(cell_df, at_single, at; verbose=false) -> Nothing

TSC mode: propagates domain doses from a representative plane (`at_single`)
to all z-layers in `cell_df` and `at`, matched by `(x, y)` coordinates.
Inactive cells (`is_cell=0`) receive zero dose.

# Example
```julia
MC_loop_copy_dose_domain_fast!(cell_df, at_single, at)
```
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

    (!hasproperty(cell_df, :index) || !hasproperty(at_single, :index) || !hasproperty(at, :index)) &&
        error("Missing :index column in one or more DataFrames.")
    (!hasproperty(cell_df, :x) || !hasproperty(cell_df, :y)) &&
        error("cell_df must contain :x and :y columns.")

    num_cells   = nrow(cell_df)
    domain_cols = names(at_single, Not(:index))
    num_domains = length(domain_cols)

    # Initialize dose fields if missing
    if !hasproperty(cell_df, :dose) || !(eltype(cell_df.dose) <: AbstractVector)
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    end
    if !hasproperty(cell_df, :dose_cell) || !(eltype(cell_df.dose_cell) <: AbstractFloat)
        cell_df.dose_cell = zeros(Float64, num_cells)
    end

    nrow(at_single) == 0 && (println("WARNING: at_single is empty → nothing to copy."); return)

    # Build (x,y) → [all cell indices across layers]
    xy_to_indices = Dict{Tuple{Float64,Float64}, Vector{Int}}()
    for row in eachrow(cell_df)
        push!(get!(xy_to_indices, (Float64(row.x), Float64(row.y)), Vector{Int}()), row.index)
    end

    for row_single in eachrow(at_single)
        rep_idx    = row_single.index
        domain_vec = Vector(row_single[domain_cols])
        scalar_dose = isempty(domain_vec) ? 0.0 : mean(domain_vec)

        rep_pos = findfirst(==(rep_idx), cell_df.index)
        rep_pos === nothing && (@warn "Representative index $rep_idx not found. Skipping."; continue)

        rep_coords  = (Float64(cell_df.x[rep_pos]), Float64(cell_df.y[rep_pos]))
        target_ids  = get(xy_to_indices, rep_coords, nothing)
        target_ids === nothing && (@warn "Coords $rep_coords not found. Skipping index $rep_idx."; continue)

        for idx in target_ids
            at_row   = findfirst(==(idx), at.index)
            cell_row = findfirst(==(idx), cell_df.index)
            (at_row === nothing || cell_row === nothing) && (@warn "Index $idx missing. Skipping."; continue)

            num_domains > 0 && (at[at_row, domain_cols] .= domain_vec)

            if cell_df.is_cell[cell_row] == 1
                cell_df.dose[cell_row]      = copy(domain_vec)
                cell_df.dose_cell[cell_row] = scalar_dose
            else
                cell_df.dose[cell_row]      = zeros(Float64, num_domains)
                cell_df.dose_cell[cell_row] = 0.0
            end
        end
    end

    println("✔ Finished copying doses.\n")
end

"""
    MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df, at_single, at,
                                                energy_step_to_match) -> Nothing

Full-MC mode: copies domain doses from representative cells of a specific
`energy_step` into `cell_df.dose`, `cell_df.dose_cell`, and `at` in-place.
Representatives matched by `(x, y)` coordinates. Threaded copy loop.

# Example
```julia
MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df, at_single, at, 3)
```
"""
function MC_loop_copy_dose_domain_layer_fast_notsc!(
    cell_df::DataFrame,
    at_single::DataFrame,
    at::DataFrame,
    energy_step_to_match::Int64
)
    println("\n-----------------------------------------------------------")
    println("  MC_loop_copy_dose_domain_layer_fast_notsc!  (COPY DOSES)")
    println("  Target energy_step : ", energy_step_to_match)
    println("-----------------------------------------------------------\n")

    t_start = time()

    for c in (:index, :x, :y, :layer, :is_cell, :energy_step)
        hasproperty(cell_df, c) || error("cell_df is missing required column :$c")
    end
    hasproperty(at_single, :index) || error("at_single missing :index")
    hasproperty(at, :index)        || error("at missing :index")

    num_cells   = nrow(cell_df)
    domain_cols = names(at_single, Not(:index))
    num_domains = length(domain_cols)

    if !hasproperty(cell_df, :dose) || !(eltype(cell_df.dose) <: AbstractVector)
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    end
    if !hasproperty(cell_df, :dose_cell) || !(eltype(cell_df.dose_cell) <: AbstractFloat)
        cell_df.dose_cell = zeros(Float64, num_cells)
    end

    nrow(at_single) == 0 && (println("at_single empty → nothing to do."); return)

    # Build (x,y) → dose map from at_single
    cell_df_index_to_row = Dict(idx => r for (r, idx) in enumerate(cell_df.index))
    xy_to_dose_data = Dict{Tuple{Float64,Float64}, Tuple{Vector{Float64},Float64}}()
    sizehint!(xy_to_dose_data, nrow(at_single))

    @showprogress for row_single in eachrow(at_single)
        rep_idx      = row_single.index
        cell_row_idx = get(cell_df_index_to_row, rep_idx, 0)
        cell_row_idx == 0 && (@warn "Representative index $rep_idx not in cell_df."; continue)
        cell_df.energy_step[cell_row_idx] != energy_step_to_match && continue

        rep_x = Float64(cell_df.x[cell_row_idx])
        rep_y = Float64(cell_df.y[cell_row_idx])

        domain_vec  = [Float64(row_single[c]) for c in domain_cols]
        scalar_dose = isempty(domain_vec) ? 0.0 : mean(domain_vec)
        xy_to_dose_data[(rep_x, rep_y)] = (domain_vec, scalar_dose)
    end

    # Copy to target cells
    at_index_to_row = Dict(idx => r for (r, idx) in enumerate(at.index))
    target_idx = findall(i -> cell_df.energy_step[i] == energy_step_to_match &&
                                cell_df.is_cell[i] == 1, 1:num_cells)
    n_targets = length(target_idx)
    println("Active cells in this step: $n_targets")

    p  = Progress(n_targets; dt=1, desc="Copying doses…", barlen=40)
    lk = ReentrantLock()

    Threads.@threads for t in 1:n_targets
        i      = target_idx[t]
        coords = (Float64(cell_df.x[i]), Float64(cell_df.y[i]))
        dose_data = get(xy_to_dose_data, coords, nothing)

        domain_vec, scalar_dose = if dose_data === nothing
            @warn "Coords $coords not found. Assigning zero dose."
            zeros(Float64, num_domains), 0.0
        else
            dose_data
        end

        length(cell_df.dose[i]) != num_domains && (cell_df.dose[i] = zeros(Float64, num_domains))
        cell_df.dose[i] .= domain_vec
        cell_df.dose_cell[i] = scalar_dose

        cell_idx   = cell_df.index[i]
        at_row_idx = get(at_index_to_row, cell_idx, 0)
        if at_row_idx != 0
            if length(names(at, Not(:index))) == num_domains && !isempty(domain_cols)
                at[at_row_idx, domain_cols] .= domain_vec
            else
                @error "Domain column count mismatch for index=$cell_idx. Skipping at update."
            end
        end

        lock(lk) do; next!(p); end
    end

    println("\n✔ Copy complete for energy_step $energy_step_to_match.")
    println("Elapsed: $(round(time() - t_start, digits=3)) s\n")
end

"""
    MC_precompute_lut!(
        ion::Ion, Npar::Int, R_beam::Float64,
        irrad_cond::Vector{AT}, cell_df::DataFrame,
        df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
        gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
        x_cb::Float64 = 0.0, y_cb::Float64 = 0.0,
        chunk_size::Int = 10_000
    ) where {AT}

Construct a **Monte Carlo lookup table (LUT)** for **continuous dose‑rate irradiation** by
sampling particle impact positions in the beam field and computing the **per-domain dose**
induced on all representative cell domains.  
The function returns a vector `lut` such that:
    lut[p][rep_idx] = dose_vector::Vector{Float64}
where each `dose_vector` contains the **raw absorbed dose per domain** for particle `p` at the
representative cell index `rep_idx`.  
Subsequent radiobiology steps (e.g., `MC_loop_damage!`) will consume this LUT.

# Overview

1. **Geometry assembly (done once):**
   - Extracts all *representative indices* for distinct `(x,y)` positions (one per column of
     domains).
   - Loads domain center coordinates from `df_center_x` / `df_center_y`.
   - Flattens domain coordinates so each particle can compute dose on all domains by a single
     loop over `total_domains`.

2. **Radial dose lookup table (done once):**
   - Builds a high-resolution 1D table of `dose(dist)` by calling
     `distribute_dose_domain` inside a threaded loop.
   - Covers three regions:
       - **Core** (very small radius): constant `core_dose`.
       - **Mid / in-target region**: bilinear interpolation on the pre-tabulated vector.
       - **Penumbra**: falloff ∝ `Kp / dist²`.
   - This lookup considerably speeds up per-particle dose deposition.

3. **Monte Carlo particle loop (chunked):**
   - The particle hits `(x, y)` are randomly sampled in a disk of radius `R_beam`.
   - For each particle:
       - Compute squared distances to all domain centers (flat arrays `dom_x`, `dom_y`).
       - Convert each distance into dose using:
         - core value,
         - table interpolation,
         - or penumbra formula.
   - Domains are then **grouped by representative cell index** and only the non-zero segments
     are stored.

4. **Chunking and threading:**
   - Particles are processed in chunks (`chunk_size`) to control peak memory usage.
   - Each chunk is threaded over particles.
   - The function provides progress prints and performs GC per chunk.

# Arguments

- `ion::Ion`: Particle species information (not directly used here but consistent with the
  larger simulation pipeline).
- `Npar::Int`: Number of incident particles to simulate.
- `R_beam::Float64`: Beam radius (for uniform circular particle sampling).
- `irrad_cond::Vector{AT}`: Irradiation physics settings (first element used for
  Rp, Rc, Kp, etc.).
- `cell_df::DataFrame`: Full cell geometry; used to extract which grid sites contain cells.
- `df_center_x`, `df_center_y`: DataFrames containing domain center coordinates
  (one row per representative index).
- `at::DataFrame`: Additional parameters (passed for consistency; not used here directly).
- `gsm2_cycle::Vector{GSM2}`: Only the first entry is used for geometric radii computations.
- `type_AT::String`: Specifies the radial dose distribution model for `distribute_dose_domain`.
- `track_seg::Bool`: Unused flag, but preserved for interface compatibility.
- `x_cb`, `y_cb`: Beam center coordinates.
- `chunk_size::Int`: Number of particles processed per batch.

# Returned value

- `lut::Vector{Dict{Int, Vector{Float64}}}` of length `Npar`.  
  For each particle `p`, `lut[p]` maps:
  rep_idx  →  dose_per_domain::Vector{Float64}
Only representative indices with at least one non-zero domain are kept to reduce memory use.

# Notes and assumptions

- All geometry matrices (`dom_x`, `dom_y`, representative index vector) are built only once.
- Radial dose is tabulated using `sim_ = 1000` logarithmically spaced distances.
- Dose interpolation is linear between adjacent table points.
- Particles are independent, so the inner loops are safely threaded.
- A domain's absorbed dose depends solely on its distance to the particle impact position.
- Because this function computes *raw per-domain dose*, downstream functions must perform
  the biological damage conversion.
- Removes all-zero domain groups per particle to reduce storage footprint.
- `GC.gc()` is explicitly called after each chunk to bound memory spikes.

# Example

```julia
lut = MC_precompute_lut!(
    ion, 100_000, 50.0,
    irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, "AT_Katz", false;
    x_cb = 0.0, y_cb = 0.0,
    chunk_size = 20_000
)
```
"""
function MC_precompute_lut!(
    ion::Ion, Npar::Int, R_beam::Float64,
    irrad_cond::Vector{AT}, cell_df::DataFrame,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
    x_cb::Float64 = 0.0, y_cb::Float64 = 0.0,
    chunk_size::Int = 10_000  # process in chunks to avoid memory explosion
)
    println("\n══════════════════════════════════════════════════════")
    println(" MC_precompute_lut!  —  building per-particle LUT")
    println(" Npar = $Npar  |  chunk_size = $chunk_size")
    println("══════════════════════════════════════════════════════")

    gsm2 = gsm2_cycle[1]

    # ── 1. Build geometry (once) ──────────────────────────────────────────────
    cell_df_is = filter(row -> row.is_cell == 1, cell_df)
    grouped_df = combine(groupby(cell_df_is, [:x, :y]),
                            :index => first => :representative_index)
    rep_indices_set = Set(grouped_df.representative_index)

    cell_df_sx = filter(row -> row.index in rep_indices_set, df_center_x)
    cell_df_sy = filter(row -> row.index in rep_indices_set, df_center_y)

    domain_cols          = names(cell_df_sx, Not(:index))
    num_cells            = nrow(cell_df_sx)
    num_domains_per_cell = length(domain_cols)
    total_domains        = num_cells * num_domains_per_cell

    # Flatten geometry vectors (once)
    mat_x = Matrix{Float64}(cell_df_sx[:, domain_cols])
    mat_y = Matrix{Float64}(cell_df_sy[:, domain_cols])
    dom_x = vec(mat_x')   # length = total_domains
    dom_y = vec(mat_y')

    # Representative index for each domain position
    rep_indices_vec = repeat(cell_df_sx.index, inner = num_domains_per_cell)

    # ── 2. Build radial dose lookup table (once) ──────────────────────────────
    Rp = irrad_cond[1].Rp; Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp; Rk = Rp

    lower_bound_log    = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq     = (gsm2.rd - 10Rc)^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    sim_ = 1000
    impact_vec = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150Rc), length=sim_)
    lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid   = Threads.threadid()
        track = Track(impact_vec[i], 0.0, Rk)
        _, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        lookup_threads[tid][i] = Gyr
    end
    dose_vec  = sum(lookup_threads)
    core_dose = dose_vec[1]

    println("✔ Lookup table built. Geometry: $num_cells cells × $num_domains_per_cell domains")

    lut = Vector{Dict{Int, Vector{Float64}}}(undef, Npar)

    n_chunks = ceil(Int, Npar / chunk_size)

    for chunk in 1:n_chunks
        p_start = (chunk - 1) * chunk_size + 1
        p_end   = min(chunk * chunk_size, Npar)
        Np_chunk = p_end - p_start + 1

        println("  Chunk $chunk / $n_chunks  (particles $p_start:$p_end)")

        # Sample hit positions for this chunk
        x_hits = Vector{Float64}(undef, Np_chunk)
        y_hits = Vector{Float64}(undef, Np_chunk)
        Threads.@threads for i in 1:Np_chunk
            x_hits[i], y_hits[i] = GenerateHit_Circle(x_cb, y_cb, R_beam)
        end

        # Per-particle dose accumulation (thread-safe: each particle is independent)
        chunk_results = Vector{Dict{Int, Vector{Float64}}}(undef, Np_chunk)

        Threads.@threads for i in 1:Np_chunk
            x = x_hits[i]; y = y_hits[i]
            dose_p = zeros(Float64, total_domains)

            @inbounds for k in 1:total_domains
                dx = dom_x[k] - x; dy = dom_y[k] - y
                dist_sq = dx*dx + dy*dy

                if dist_sq <= core_radius_sq
                    dose_p[k] = core_dose
                elseif dist_sq <= mid_radius_sq
                    dist  = sqrt(dist_sq)
                    idx_l = searchsortedfirst(impact_vec, dist)
                    if idx_l == 1
                        dose_p[k] = core_dose
                    elseif idx_l > sim_
                        dose_p[k] = dose_vec[end]
                    else
                        x1, x2 = impact_vec[idx_l-1], impact_vec[idx_l]
                        y1, y2 = dose_vec[idx_l-1], dose_vec[idx_l]
                        dose_p[k] = y1 + (y2-y1)*(dist-x1)/(x2-x1)
                    end
                elseif dist_sq < penumbra_radius_sq
                    dose_p[k] = Kp / dist_sq
                end
            end

            # Group by representative index, skip all-zero groups
            d = Dict{Int, Vector{Float64}}()
            for j in 1:num_cells
                seg = dose_p[(j-1)*num_domains_per_cell+1 : j*num_domains_per_cell]
                if any(!iszero, seg)
                    d[cell_df_sx.index[j]] = seg
                end
            end
            chunk_results[i] = d
        end

        # Write chunk into lut
        for i in 1:Np_chunk
            lut[p_start + i - 1] = chunk_results[i]
        end

        # Free chunk memory
        chunk_results = nothing
        GC.gc()
    end

    println("\n✔ LUT complete. $(Npar) particles stored.")
    println("══════════════════════════════════════════════════════\n")
    return lut
end

