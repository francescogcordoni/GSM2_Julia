#! ============================================================================
#! utilities_GSM2.jl
#!
#! FUNCTIONS
#! ---------
#~ Domain Geometry
#?   calculate_centers(x0, y0, radius, nucleus) -> (center_x, center_y)
#       Computes domain center positions on a circular lattice inside the nucleus.
#?   create_domain_dataframes(cell_df, rel_center_x, rel_center_y)
#           -> (df_center_x, df_center_y, at)
#       Builds absolute domain center DataFrames for all cells and initializes
#       the AT (Amorphous Track) results table.
#
#~ Damage Computation
#?   MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle; verbose, show_progress) -> Nothing
#       Computes per-domain X/Y DNA damage for each cell using a Poisson process.
#       Updates dam_X_dom, dam_Y_dom, dam_X_total, dam_Y_total in-place.
#?   calculate_OER(LET, O) -> Float64
#       Oxygen Enhancement Ratio from LET and oxygenation level O.
#?   calculate_kappa(ion_name, LET, O; OER_bool) -> Float64
#       Damage yield per Gy, ion-species dependent, with optional OER correction.
#?   precompute_damage_lut!(lut, cell_df, irrad_cond, gsm2_cycle, ion, chunk_size) -> Dict
#
#~ Survival
#?   compute_cell_survival_GSM2!(cell_df, gsm2_cycle; NFrac) -> Nothing
#       Per-cell survival probabilities via GSM2. Resets timing fields and
#       updates cell_df.sp in-place. Supports multi-fraction dosing (NFrac).
#?   domain_GSM2(X, Y, gsm2) -> Float64
#       GSM2 survival probability for a single cell from per-domain X/Y damage vectors.
#       Returns 0 immediately if any Y-type (lethal) damage is present.
#?   survival_ci(surv; alpha) -> (p_hat, lower, upper)
#       Wilson score confidence interval for survival from a vector of
#       per-trial probabilities (simulates Bernoulli outcomes internally).
#! ============================================================================

"""
    calculate_centers(x0::Float64, y0::Float64, radius::Float64, nucleus::Float64)
        -> (center_x, center_y)

Domain center positions on a circular lattice inside a nucleus of radius `nucleus`.
Domains have radius `radius` and are packed in concentric rings.

# Example
```julia
cx, cy = calculate_centers(0.0, 0.0, 0.5, 4.0)
```
"""
function calculate_centers(x0::Float64, y0::Float64, radius::Float64, nucleus::Float64)
    center_x = Vector{Float64}(undef, 0)
    center_y = Vector{Float64}(undef, 0)

    # Central domain
    push!(center_x, x0)
    push!(center_y, y0)

    # Concentric rings
    for rho in 1:(floor(Int64, nucleus / (2 * radius)))
        center_rho = rho * 2 * radius
        n_circle   = floor(Int64, (pi * center_rho) / radius)
        theta      = range(0, 2π, length=n_circle)

        xi = x0 .+ center_rho .* cos.(theta[1:(end-1)])
        yi = y0 .+ center_rho .* sin.(theta[1:(end-1)])

        center_x = vcat(center_x, xi)
        center_y = vcat(center_y, yi)
    end

    return (center_x, center_y)
end

"""
    create_domain_dataframes(cell_df, rel_center_x, rel_center_y)
        -> (df_center_x, df_center_y, at)

Absolute domain center DataFrames for all cells and initialized AT results table.

Shifts the relative domain template `(rel_center_x, rel_center_y)` by each cell's
`(x, y)` position. Returns:
- `df_center_x`, `df_center_y` — one row per cell, one column per domain center
- `at` — zero-initialized AT table (same indexing)

# Example
```julia
df_cx, df_cy, at = create_domain_dataframes(cell_df, rel_cx, rel_cy)
```
"""
function create_domain_dataframes(cell_df::DataFrame,
                                    rel_center_x::Vector{Float64},
                                    rel_center_y::Vector{Float64})
    println("... Creating domain dataframes ...")

    num_cols = length(rel_center_x)

    # Absolute coordinates: rows = cells, cols = domains
    mat_center_x = cell_df.x .+ transpose(rel_center_x)
    mat_center_y = cell_df.y .+ transpose(rel_center_y)

    df_center_x = DataFrame(mat_center_x, Symbol.("center_$i" for i in 1:num_cols))
    df_center_x.index = cell_df.index

    df_center_y = DataFrame(mat_center_y, Symbol.("center_$i" for i in 1:num_cols))
    df_center_y.index = cell_df.index

    # AT table: zero-initialized, (num_cols - 1) columns (historical convention)
    at = DataFrame(zeros(size(df_center_y, 1), (size(df_center_y, 2) - 1)), :auto)
    rename!(at, Symbol.("center_$i" for i in 1:(size(df_center_y, 2) - 1)))
    at.index = df_center_y.index

    return df_center_x, df_center_y, at
end

"""
    MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle; verbose, show_progress) -> Nothing

Per-domain X/Y DNA damage for each cell via Poisson sampling.
Updates `dam_X_dom`, `dam_Y_dom`, `dam_X_total`, `dam_Y_total` in `cell_df` in-place.

Damage rates per domain:
- `λX = kappa_base * d`  (double-strand breaks)
- `λY = lambda_base * d` (complex/lethal lesions, ~1e-3 × λX)

where `kappa_base = 9 * kappa_yield / (n_repeat * N_domains)`.
LET and O are taken per-cell from `irrad_cond[energy_step].LET` and `cell_df.O`.

# Example
```julia
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle; show_progress=true)
```
"""
function MC_loop_damage!(
    ion::Ion,
    cell_df::DataFrame,
    irrad_cond::Vector{AT},
    gsm2_cycle::Vector{GSM2};
    verbose::Bool = false,
    show_progress::Bool = true
) where {AT}

    gsm2 = gsm2_cycle[1]
    vprintln(args...) = (verbose ? println("[DEBUG] ", args...) : nothing)

    println("\n-----------------------------------------------------------")
    println("     MC_loop_damage! (FAST DAMAGE MODEL)")
    println("-----------------------------------------------------------")

    num_cells = nrow(cell_df)
    n_repeat  = floor(Int, gsm2.Rn / gsm2.rd)

    println("Cells       : $num_cells")
    println("n_repeat    : $n_repeat\n")

    @assert hasproperty(cell_df, :dose)        "Missing column: dose"
    @assert hasproperty(cell_df, :energy_step) "Missing column: energy_step"
    @assert hasproperty(cell_df, :O)           "Missing column: O"
    @assert hasproperty(cell_df, :is_cell)     "Missing column: is_cell"

    # Expected output vector length
    first_nonempty = findfirst(x -> !isempty(x), cell_df.dose)
    expected_len = first_nonempty === nothing ? 0 : length(cell_df.dose[first_nonempty]) * n_repeat

    # Initialize damage vector columns
    for cname in (:dam_X_dom, :dam_Y_dom)
        if !hasproperty(cell_df, cname)
            cell_df[!, cname] = [zeros(Int, expected_len) for _ in 1:num_cells]
        else
            col = cell_df[!, cname]
            if !(col isa Vector{Vector{Int}})
                @warn "$cname had wrong type ($(typeof(col))), reinitializing"
                cell_df[!, cname] = [zeros(Int, expected_len) for _ in 1:num_cells]
            end
        end
    end

    # Initialize scalar damage totals
    hasproperty(cell_df, :dam_X_total) || (cell_df[!, :dam_X_total] = zeros(Int, num_cells))
    hasproperty(cell_df, :dam_Y_total) || (cell_df[!, :dam_Y_total] = zeros(Int, num_cells))

    show_progress && (p = Progress(num_cells, 1, "Damage: "))

    # Pre-allocated Poisson sample buffers
    buffer_X = Vector{Int}(undef, n_repeat)
    buffer_Y = Vector{Int}(undef, n_repeat)

    for i in 1:num_cells
        show_progress && next!(p)
        vprintln("Processing cell $i")

        # Skip inactive or zero-dose cells
        if cell_df.is_cell[i] != 1 || isempty(cell_df.dose[i]) ||
           (hasproperty(cell_df, :dose_cell) && cell_df.dose_cell[i] <= 0.0)
            fill!(cell_df.dam_X_dom[i], 0)
            fill!(cell_df.dam_Y_dom[i], 0)
            cell_df.dam_X_total[i] = 0
            cell_df.dam_Y_total[i] = 0
            continue
        end

        dose_vector = cell_df.dose[i]
        L   = length(dose_vector)
        O   = cell_df.O[i]
        LET = irrad_cond[cell_df.energy_step[i]].LET

        kappa_yield = calculate_kappa(ion.ion, LET, O)
        kappa_base  = 9.0 * kappa_yield / (n_repeat * L)
        lambda_base = kappa_base * 1e-3

        vprintln("  LET=", LET, "  O=", O,
                 "  kappa_base=", kappa_base, "  lambda_base=", lambda_base)

        row_len = L * n_repeat
        Xrow = cell_df.dam_X_dom[i]
        Yrow = cell_df.dam_Y_dom[i]

        length(Xrow) != row_len && (resize!(Xrow, row_len); cell_df.dam_X_dom[i] = Xrow)
        length(Yrow) != row_len && (resize!(Yrow, row_len); cell_df.dam_Y_dom[i] = Yrow)

        pos = 1
        @inbounds for d in dose_vector
            λX = max(0.0, kappa_base * d)
            λY = max(0.0, lambda_base * d)

            λX > 0.0 ? rand!(Poisson(λX), buffer_X) : fill!(buffer_X, 0)
            λY > 0.0 ? rand!(Poisson(λY), buffer_Y) : fill!(buffer_Y, 0)

            copyto!(Xrow, pos, buffer_X, 1, n_repeat)
            copyto!(Yrow, pos, buffer_Y, 1, n_repeat)
            pos += n_repeat
        end

        cell_df.dam_X_total[i] = sum(Xrow)
        cell_df.dam_Y_total[i] = sum(Yrow)

        vprintln("  dam_X_total=", cell_df.dam_X_total[i],
                    "  dam_Y_total=", cell_df.dam_Y_total[i])
    end

    println("\n✔ Finished damage computation.")
    println("  Total X damages across all cells: ", sum(cell_df.dam_X_total))
    println("  Total Y damages across all cells: ", sum(cell_df.dam_Y_total))
    println()

    return nothing
end

"""
    precompute_damage_lut!(
        lut::Vector{Dict{Int, Vector{Float64}}},
        cell_df::DataFrame,
        irrad_cond::Vector{AT},
        gsm2_cycle::Vector{GSM2},
        ion::Ion;
        chunk_size::Int = 10_000
    ) where {AT}

Build a **particle-wise damage lookup table** (LUT) for a **continuous dose-rate** irradiation,
re-using per-(x,y) dose vectors across all z-layers and sampling **stochastic damage events**
(X, Y) per cell and per domain repetition.

# Summary

- **Input**: `lut[p][rep_idx] => dose_vector::Vector{Float64}` gives the per-domain dose time
  series for particle `p` at a *representative* cell index `rep_idx` (one per unique `(x,y)`).
- **Output**: `Vector{Dict{Int, Tuple{Vector{Int}, Vector{Int}}}}` of length `Npar`, where
  each entry `damage_lut[p][cell_idx] = (Xrow, Yrow)` stores the **sampled counts** of two
  lesion types (e.g., complex/simple) along the repeated dose timeline for **all cells**
  sharing the representative `(x,y)`.

# What it does

1. **(x,y) → indices mapping**: Groups all cell indices that lie on the same `(x,y)` in the
   z-stack so that a single `dose_vector` per `(x,y)` can be broadcast to all z-layers.

2. **Kinetic parameters per cell**:
   - Derives `n_repeat = floor(Int, gsm2.Rn / gsm2.rd)` from the GSM2 cycle (number of domain
     repetitions in the full irradiation).
   - Computes `κ` (scaled lesion-rate factor) from `LET` through `calculate_kappa`, normalized
     by `n_repeat * num_domains_per_cell`. Also sets `λ = κ * 1e-3` (second lesion class).
   - Stores `(energy_step, O)` and the derived `κ`, `λ` per cell.

3. **Chunked, threaded sampling**:
   - Processes particles in chunks (`chunk_size`) to control memory pressure.
   - For each particle and each representative cell index `rep_idx`:
     - Retrieves all cells sharing `(x,y)`, copies the same `dose_vector`.
     - For every dose point `d` and each repetition, samples
       `X ~ Poisson(max(0, κ*d))`, `Y ~ Poisson(max(0, λ*d))` into `Xrow`, `Yrow`.
     - If any counts are non-zero for a cell, records `(Xrow, Yrow)`.

4. **Returns** the full `damage_lut`, along with console summaries of coverage.

# Arguments

- `lut::Vector{Dict{Int, Vector{Float64}}}`:
  For each particle `p = 1:Npar`, a dictionary from **representative** cell index `rep_idx`
  (unique per `(x,y)`) to a **dose time series** `dose_vector`. All z-layers at the same
  `(x,y)` will re-use this dose vector.

- `cell_df::DataFrame`:
  Must contain at least the columns:
  - `:index::Int` (unique cell index),
  - `:x, :y::Real` (planar coordinates),
  - `:is_cell::Int` (1 if a cell; 0 otherwise),
  - `:energy_step::Int` (indexing `irrad_cond`),
  - `:O::Real` (oxygenation or other scalar affecting κ).

- `irrad_cond::Vector{AT}`:
  Per-energy-step irradiation conditions. Each `irrad_cond[energy_step]` must provide `LET`.

- `gsm2_cycle::Vector{GSM2}`:
  Only the first element is used (`gsm2 = gsm2_cycle[1]`) to derive
  `n_repeat = floor(Int, gsm2.Rn / gsm2.rd)`.

- `ion::Ion`:
  Particle species info; passed to `calculate_kappa(ion.ion, LET, O)`.

- `chunk_size::Int=10_000`:
  Number of particles to process per chunk. Increase to reduce overhead; decrease to limit RAM.

# Returns
- `damage_lut::Vector{Dict{Int, Tuple{Vector{Int}, Vector{Int}}}}`:
  For each particle `p`, maps `cell_idx` to `(Xrow, Yrow)`, the time-expanded lesion counts.
  The length of `Xrow`/`Yrow` equals `length(dose_vector) * n_repeat`. Cells with all-zero
  counts are omitted to save memory.

# Assumptions & Notes

- **Continuous dose-rate** handling is achieved by repeating each `dose_vector` `n_repeat` times
  to match the GSM2 cycle duration.
- `num_domains_per_cell` is inferred from the first **non-empty** LUT entry and used to scale κ.
- `λ = κ * 1e-3` is a fixed proportional relation; adjust as needed for your model.
- Only rows with `row.is_cell == 1` are considered as biological targets.
- Uses `Threads.@threads` over particles within each chunk; ensure RNG behavior is acceptable
  for your reproducibility needs (e.g., set thread-safe seeds if required).
- Skips storing `(Xrow, Yrow)` if both are all zeros for a given cell and particle—this keeps
  memory bounded when events are rare.
- Console prints summarize progress and the fraction of particles producing non-zero damage.

# Performance tips

- Choose `chunk_size` to balance throughput and memory. Larger chunks reduce dispatch/GC
  overhead but may peak RAM.
- Pre-allocations (`buf_X`, `buf_Y`) amortize Poisson sampling costs across repetitions.
- If `lut` is very sparse, you benefit from the all-zero suppression per cell.

# Example

```julia
damage_lut = precompute_damage_lut!(
    lut,            # Vector{Dict{Int, Vector{Float64}}}
    cell_df,        # DataFrame with :index, :x, :y, :is_cell, :energy_step, :O
    irrad_cond,     # Vector{AT} where AT has field LET
    gsm2_cycle,     # Vector{GSM2} where first element has fields Rn, rd
    ion;            # Ion with field `ion`, used by calculate_kappa
    chunk_size = 20_000
)
```
"""
function precompute_damage_lut!(
    lut::Vector{Dict{Int, Vector{Float64}}},
    cell_df::DataFrame,
    irrad_cond::Vector{AT},
    gsm2_cycle::Vector{GSM2},
    ion::Ion;
    chunk_size::Int = 10_000
) where {AT}

    println("\n══════════════════════════════════════════════════════")
    println(" precompute_damage_lut!")
    println(" Npar = $(length(lut))  |  chunk_size = $chunk_size")
    println("══════════════════════════════════════════════════════")

    gsm2     = gsm2_cycle[1]
    n_repeat = floor(Int, gsm2.Rn / gsm2.rd)
    Npar     = length(lut)

    # ── 1. Build (x,y) → all cell indices map (z-layer copy) ─────────────────
    xy_to_indices = Dict{Tuple{Float64,Float64}, Vector{Int}}()
    for row in eachrow(cell_df)
        row.is_cell != 1 && continue
        key = (Float64(row.x), Float64(row.y))
        push!(get!(xy_to_indices, key, Int[]), row.index)
    end

    # Map representative index → all cell indices sharing same (x,y)
    index_to_row = Dict(idx => r for (r, idx) in enumerate(cell_df.index))
    rep_to_all_indices = Dict{Int, Vector{Int}}()
    for (p, dose_dict) in enumerate(lut)
        for rep_idx in keys(dose_dict)
            haskey(rep_to_all_indices, rep_idx) && continue
            row = get(index_to_row, rep_idx, nothing)
            row === nothing && continue
            key = (Float64(cell_df.x[row]), Float64(cell_df.y[row]))
            rep_to_all_indices[rep_idx] = get(xy_to_indices, key, Int[])
        end
    end

    # ── 2. Build cell info + kappa/lambda per cell index ──────────────────────
    num_domains_per_cell = length(first(values(first(filter(p -> !isempty(p), lut)))))

    cell_info  = Dict{Int, Tuple{Int, Float64}}()  # index => (energy_step, O)
    kappa_map  = Dict{Int, Float64}()
    lambda_map = Dict{Int, Float64}()

    for row in eachrow(cell_df)
        row.is_cell != 1 && continue
        LET = irrad_cond[row.energy_step].LET
        κ   = 9.0 * calculate_kappa(ion.ion, LET, Float64(row.O)) / (n_repeat * num_domains_per_cell)
        cell_info[row.index]  = (row.energy_step, Float64(row.O))
        kappa_map[row.index]  = κ
        lambda_map[row.index] = κ * 1e-3
    end

    println("✔ Built z-layer map: $(length(rep_to_all_indices)) representative cells")

    # ── 3. Chunked damage computation ─────────────────────────────────────────
    damage_lut = Vector{Dict{Int, Tuple{Vector{Int}, Vector{Int}}}}(undef, Npar)
    n_chunks   = ceil(Int, Npar / chunk_size)

    for chunk in 1:n_chunks
        p_start  = (chunk - 1) * chunk_size + 1
        p_end    = min(chunk * chunk_size, Npar)
        Np_chunk = p_end - p_start + 1
        println("  Chunk $chunk / $n_chunks  (particles $p_start:$p_end)")

        chunk_damage = Vector{Dict{Int, Tuple{Vector{Int}, Vector{Int}}}}(undef, Np_chunk)

        Threads.@threads for i in 1:Np_chunk
            p           = p_start + i - 1
            dose_dict   = lut[p]
            particle_damage = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()

            for (rep_idx, dose_vector) in dose_dict

                # Copy dose to all z-layers and compute damage for each
                all_indices = get(rep_to_all_indices, rep_idx, Int[])

                for cell_idx in all_indices
                    κ = get(kappa_map,  cell_idx, 0.0)
                    λ = get(lambda_map, cell_idx, 0.0)

                    row_len = length(dose_vector) * n_repeat
                    Xrow    = zeros(Int, row_len)
                    Yrow    = zeros(Int, row_len)
                    buf_X   = Vector{Int}(undef, n_repeat)
                    buf_Y   = Vector{Int}(undef, n_repeat)

                    pos = 1
                    @inbounds for d in dose_vector
                        λX = max(0.0, κ * d)
                        λY = max(0.0, λ * d)
                        λX > 0.0 ? rand!(Poisson(λX), buf_X) : fill!(buf_X, 0)
                        λY > 0.0 ? rand!(Poisson(λY), buf_Y) : fill!(buf_Y, 0)
                        copyto!(Xrow, pos, buf_X, 1, n_repeat)
                        copyto!(Yrow, pos, buf_Y, 1, n_repeat)
                        pos += n_repeat
                    end

                    if any(!iszero, Xrow) || any(!iszero, Yrow)
                        particle_damage[cell_idx] = (Xrow, Yrow)
                    end
                end
            end

            chunk_damage[i] = particle_damage
        end

        for i in 1:Np_chunk
            damage_lut[p_start + i - 1] = chunk_damage[i]
        end

        chunk_damage = nothing
        GC.gc()
    end

    println("\n✔ damage_lut complete.")
    n_nonzero = count(p -> !isempty(p), damage_lut)
    println("  Non-zero particles: $n_nonzero / $Npar ($(round(100*n_nonzero/Npar, digits=1))%)")
    println("══════════════════════════════════════════════════════\n")

    return damage_lut
end


"""
    calculate_OER(LET::Float64, O::Float64) -> Float64

Oxygen Enhancement Ratio from LET (keV/µm) and oxygenation level O.
Uses the empirical model: `OER = (b*(a*M0 + LET^g)/(a + LET^g) + O) / (b + O)`.

# Example
```julia
oer = calculate_OER(50.0, 5.0)
```
"""
function calculate_OER(LET::Float64, O::Float64)
    M0 = 3.4
    b  = 0.41
    a  = 8.27e5
    g  = 3.0
    return (b * (a*M0 + LET^g) / (a + LET^g) + O) / (b + O)
end

"""
    calculate_kappa(ion_name::String, LET::Float64, O::Float64; OER_bool=true) -> Float64

Damage yield per Gy for a given ion, LET, and oxygenation level.
Ion-specific parameters supported: `"12C"`, `"4He"`, `"3He"`, `"1H"`, `"2H"`, `"16O"`.
Applies OER correction by default (`OER_bool=true`).

# Example
```julia
κ = calculate_kappa("12C", 80.0, 5.0)
```
"""
function calculate_kappa(ion_name::String, LET::Float64, O::Float64; OER_bool::Bool = true)
    if ion_name == "12C"
        p1,p2,p3,p4,p5 = 6.8, 0.156, 0.9214, 0.005245, 1.395
    elseif ion_name == "4He" || ion_name == "3He"
        p1,p2,p3,p4,p5 = 6.8, 0.1471, 1.038, 0.006239, 1.582
    elseif ion_name == "1H" || ion_name == "2H"
        p1,p2,p3,p4,p5 = 6.8, 0.1773, 0.9314, 0.0, 1.0
    elseif ion_name == "16O"
        p1,p2,p3,p4,p5 = 6.8, 0.1749, 0.8722, 0.004987, 1.347
    else
        @warn "Unknown ion species '$ion_name'"
        return 0.0
    end

    yield = (p1 + (p2 * LET)^p3) / (1 + (p4 * LET)^p5)

    OER_bool && (yield /= calculate_OER(LET, O))

    return yield
end

8. * calculate_kappa("12C", 150., 21.)

"""
    compute_cell_survival_GSM2!(cell_df, gsm2_cycle; NFrac=1) -> Nothing

Per-cell survival probabilities via GSM2. Resets timing/death fields and updates
`cell_df.sp` in-place. Phase-dependent GSM2 params: G1/G0→[1], S→[2], G2/M→[3].
Supports multi-fraction dosing via `NFrac` (SP = SP_cell ^ NFrac).

# Example
```julia
compute_cell_survival_GSM2!(cell_df, gsm2_cycle; NFrac=5)
```
"""
function compute_cell_survival_GSM2!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2}; NFrac::Int64 = 1)
    cell_df.sp          .= 1.
    cell_df.apo_time    .= Inf
    cell_df.death_time  .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time  .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type  .= -1

    for i in cell_df.index[cell_df.is_cell .== 1]
        gsm2 = if cell_df.cell_cycle[i] == "G1" || cell_df.cell_cycle[i] == "G0"
            gsm2_cycle[1]
        elseif cell_df.cell_cycle[i] == "S"
            gsm2_cycle[2]
        elseif cell_df.cell_cycle[i] == "G2" || cell_df.cell_cycle[i] == "M"
            gsm2_cycle[3]
        else
            println("Cell cycle not found: cell $i has phase '$(cell_df.cell_cycle[i])'")
            gsm2_cycle[4]
        end

        SP_cell = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)
        cell_df.sp[i] = SP_cell ^ NFrac
    end
end

"""
    domain_GSM2(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2) -> Float64

GSM2 survival probability for a single cell from per-domain X/Y damage vectors.
Returns `0.0` immediately if any Y-type (lethal) damage is present (`sum(Y) > 0`).

For each domain `j` with `X[j]` lesions:
`p_j = ∏_{i=1}^{X[j]}  (i*r) / ((r+a)*i + b*i*(i-1))`

Total survival: `SP = ∏_j p_j`.

# Example
```julia
sp = domain_GSM2(X_vec, Y_vec, gsm2)
```
"""
function domain_GSM2(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2)
    sum(Y) > 0 && return 0.

    p_cell = 1.
    for j in 1:length(X)
        p = 1.0
        for i in X[j]:-1:1
            p *= (i * gsm2.r) /
                 ((gsm2.r + gsm2.a) * i + gsm2.b * i * (i - 1))
        end
        p_cell *= p
    end

    return p_cell
end

"""
    survival_ci(surv::AbstractVector{<:Real}; alpha=0.05) -> (p_hat, lower, upper)

Wilson score confidence interval for survival from a vector of per-trial probabilities.
Simulates one Bernoulli outcome per element, then computes the Wilson interval at
level `alpha` (default 0.05 → 95% CI). Stochastic: set `Random.seed!` for reproducibility.

# Example
```julia
p̂, lo, hi = survival_ci([0.7, 0.9, 0.5, 0.8]; alpha=0.05)
```
"""
function survival_ci(surv::AbstractVector{<:Real}; alpha=0.05)
    surv01 = rand.(Ref(Random.default_rng()), Bernoulli.(surv)) .|> Int

    n   = length(surv01)
    n_s = sum(surv01)
    n_f = n - n_s

    p_hat = n_s / n
    z     = quantile(Normal(), 1 - alpha/2)

    denom     = n + z^2
    center    = (n_s + 0.5*z^2) / denom
    halfwidth = (z/denom) * sqrt((n_s*n_f)/n + (z^2)/4)

    return p_hat, center - halfwidth, center + halfwidth
end
