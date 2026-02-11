"""
setup_GSM2!(r, a, b, rd, Rn; x0=-300.0, y0=-300.0) -> Nothing

Configure the GSM2 model **in the caller environment** (module `Main`), print a short
setup summary, and *define* the following globals without returning any value:

- `gsm2      :: GSM2`                  → constructed as `GSM2(r, a, b, rd, Rn)`
- `center_x  :: Vector{Float64}`       → x-coordinates of domain centers from `calculate_centers(x0, y0, rd, Rn)`
- `center_y  :: Vector{Float64}`       → y-coordinates of domain centers from `calculate_centers(x0, y0, rd, Rn)`
- `domain    :: Int`                   → number of domains per layer, i.e. `length(center_x)`

# Purpose
This function is intended for **interactive scripts** and **notebook-style workflows** where
you prefer **side effects** (variables defined in the global environment) over returned values.
It builds a `GSM2` object from scalar parameters, generates the reusable domain-center template,
computes the domain count, **prints a concise summary**, and exposes all outputs as globals.

# Arguments
- `r::Real`     : GSM2 scale/radius parameter
- `a::Real`     : GSM2 α-like parameter
- `b::Real`     : GSM2 β-like parameter
- `rd::Real`    : domain radius used to tile the nucleus
- `Rn::Real`    : nucleus radius
- `x0::Real`    : (keyword) x-origin for domain center template (default `-300.0`)
- `y0::Real`    : (keyword) y-origin for domain center template (default `-300.0`)

# Behavior
1. Constructs `gsm2 = GSM2(r, a, b, rd, Rn)`.
2. Calls `center_x, center_y = calculate_centers(x0, y0, rd, Rn)`.
3. Sets `domain = length(center_x)`.
4. Prints a human-readable summary of the configuration.
5. Injects `gsm2`, `center_x`, `center_y`, `domain` into **`Main`** via `@eval`.
6. Returns `nothing`.

# Side Effects
- **Defines/overwrites** `gsm2`, `center_x`, `center_y`, `domain` in the **global scope** (`Main`).
- Produces console output describing the setup.

# Reproducibility
Deterministic given the same inputs; no random components are used here. The resulting
`center_x`/`center_y` template can be reused by offsetting with each cell’s `(x, y)`.

# Example
```julia
# Parameters (units consistent with your model, e.g., µm)
r, a, b = 4.3, 0.01, 0.30
rd, Rn  = 0.8, 7.2

# Call the setup function
setup_GSM2!(r, a, b, rd, Rn; x0=-300.0, y0=-300.0)
```
"""
# Now available in the environment:
# gsm2, center_x, center_y, domain
function setup_GSM2!(r, a, b, rd, Rn; x0=-300.0, y0=-300.0)
    # Construct GSM2 and calculate centers
    gsm2_local = GSM2(r, a, b, rd, Rn)
    center_x_local, center_y_local = calculate_centers(x0, y0, rd, Rn)
    domain_local = length(center_x_local)

    # Inject into caller environment (Main)
    @eval Main begin
        gsm2      = $gsm2_local
        center_x  = $center_x_local
        center_y  = $center_y_local
        domain    = $domain_local
    end

    # Pretty printing
    println("========== GSM2 Model Setup ==========")
    println(" r  = $r")
    println(" a  = $a")
    println(" b  = $b")
    println(" rd = $rd")
    println(" Rn = $Rn")
    println(" Origin for center generation: ($x0, $y0)")
    println(" Number of domains per layer: $domain_local")
    println("======================================")

    return nothing
end

"""
    setup_IonIrrad!(dose::Float64, E::Float64, particle::String;
                    type_AT::String = "KC", kR::Float64 = 1.0,
                    oxygen::Float64 = 0.18) -> Nothing

Configure the ion and irradiation model in the caller environment, compute LET from
stopping-power tables, construct the `Ion` and `Irrad` objects, compute track-structure
parameters, and print a concise summary.  
This function **does not return anything** — instead it defines:

- `type_AT`   :: String             — track model ("KC" or "LEM")
- `dose`      :: Float64            — irradiation dose (Gy)
- `A`         :: Int                — mass number (extracted from particle string)
- `Z`         :: Int                — atomic number via `getZ`
- `LET`       :: Float64            — interpolated stopping power
- `ion`       :: Ion                — configured ion object
- `irrad`     :: Irrad              — irradiation object
- `Rc`,`Rp`,`Rk`,`Kp` :: Float64    — track radii and amplitude
- `DoseRate_h`,`F`,`D`,`T` :: Float64 — initialized irradiation metadata

All variables are injected into the **caller’s environment (`Main`)**.

# Arguments
- `dose`     : dose in Gy  
- `E`        : kinetic energy (MeV/u)  
- `particle` : ion label, e.g. `"1H"`, `"4He"`, `"12C"`  
- `type_AT`  : track model (`"KC"` default, `"LEM"` optional)  
- `kR`       : parameter kept for compatibility (default 1.0)  
- `oxygen`   : oxygen enhancement ratio / oxygen fraction parameter  

# Behavior
1. Extracts mass number `A` and computes nuclear charge `Z`.
2. Interpolates LET using the global stopping‑power dictionary `sp`.
3. Builds the ion object.
4. Computes track parameters via `ATRadius`.
5. Builds the irradiation object.
6. Prints a formatted summary.
7. Exposes all computed variables in `Main`.

# Example
```julia
setup_IonIrrad!(1.0, 100.0, "1H")         # KC model
setup_IonIrrad!(2.0, 150.0, "12C"; type_AT="LEM")
```
"""
function setup_IonIrrad!(dose::Float64, E::Float64, particle::String;
        type_AT::String = "KC", kR::Float64 = 1.0,
        dose_rate::Float64 = 0.18)
    # Extract mass number from ion string: "12C" → 12
    A_ = parse(Int, filter(x -> isdigit(x), particle))
    A = 1.

    # Determine Z from map
    Z = getZ(particle)

    # LET interpolation (needs global `sp`)
    LET_local = linear_interpolation(particle, E, sp)

    # Build Ion object
    ion_local = Ion(particle, E, A, Z, LET_local, 1.0)

    # Compute amorphous track radii
    Rc_local, Rp_local, Kp_local = ATRadius(ion_local, Irrad(dose, kR, dose_rate), type_AT)

    # Standard convention Rk = Rp
    Rk_local = Rp_local

    # Build Irrad object
    irrad_local = Irrad(dose, kR, dose_rate)

    # Extra irradiation parameters
    DoseRate_h_local = 0.0
    F_local = 0.0
    D_local = 0.0
    T_local = 0.0

    # Inject in caller environment (Main)
    @eval Main begin
        type_AT   = $type_AT
        dose      = $dose
        A_        = $A_
        A         = $A
        Z         = $Z
        LET       = $LET_local
        ion       = $ion_local
        irrad     = $irrad_local

        Rc = $Rc_local
        Rp = $Rp_local
        Rk = $Rk_local
        Kp = $Kp_local

        DoseRate_h = $DoseRate_h_local
        F = $F_local
        D = $D_local
        T = $T_local
    end

    # Print summary
    println("=========== Ion & Irradiation Setup ===========")
    println(" Particle:         $particle  (A = $A_, Z = $Z)")
    println(" Energy:           $E MeV/u")
    println(" LET:              $LET_local keV/μm")
    println(" Dose:             $dose Gy")
    println(" Track model:      $type_AT")
    println(" Rc = $(Rc_local), Rp = $(Rp_local), Rk = $(Rk_local), Kp = $(Kp_local)")
    println(" Dose rate.:       $dose_rate Gy/h")
    println("===============================================")

    return nothing
end

"""
setup_cell_lattice!(target_geom::String, X_box::Float64, R_cell::Float64,
                        N_sideVox::Int, N_CellsSide::Int;
                        ParIrr::String = "false") -> Nothing

Configure the full cell‑lattice initialization based on geometry and optional
partial‑irradiation mode. This function:

- defines global variables in `Main`:  
    `ParIrr`, `N`, `nodes_positions`, `N_CellsSide`, `cell_df`,
    `track_seg`, `full_cycle`
- generates cell positions using `generate_cells_positions_selector`
- builds the `cell_df` DataFrame with x, y, z coordinates
- computes `.layer` indices for analysis/plotting
- prints a short summary
- returns nothing (side‑effect function, REPL-friendly)

# Arguments
- `target_geom::String`  
    `"square"` or `"circle"` — geometric region for the cell layout.
- `X_box::Float64`  
    Half‑side of the simulation cube.
- `R_cell::Float64`  
    Cell radius (µm).
- `N_sideVox::Int`  
    Number of voxels per side.
- `N_CellsSide::Int`  
    Default number of cells per side for full‑irradiation mode.
- `ParIrr::String = "false"`  
    Optional — `"true"` enables partial irradiation (1‑voxel layer).  
    Default: `"false"`.

# Defined Globals (in `Main`)
- `ParIrr` — `"true"` or `"false"`
- `N` — number of generated cells
- `nodes_positions` — vector of `(x, y, z)` coordinates
- `N_CellsSide` — cells per side (possibly overwritten by generator)
- `cell_df` — DataFrame storing cell coordinates and layer IDs
- `track_seg` — set to `true`
- `full_cycle` — set to `true`

# Example
```julia
setup_cell_lattice!("square", X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="true")
```
"""
function setup_cell_lattice!(target_geom::String,
                                X_box::Float64,
                                R_cell::Float64,
                                N_sideVox::Int,
                                N_CellsSide::Int;
                                ParIrr::String = "false",
                                track_seg::Bool = true)
    # --- Generate positions ---
    N_local, nodes_positions_local, N_CellsSide_local =
        generate_cells_positions_selector(target_geom, ParIrr,
                                            X_box, R_cell,
                                            N_sideVox, N_CellsSide)

    # Build cell DataFrame
    cell_df_local = DataFrame(
        index = collect(eachindex(nodes_positions_local)),
        x     = [p[1] for p in nodes_positions_local],
        y     = [p[2] for p in nodes_positions_local],
        z     = [p[3] for p in nodes_positions_local],
    )

    # Assign layer IDs for visualization or analysis
    unique_z = sort(unique(cell_df_local.z))
    z_to_layer = Dict(z => i for (i, z) in enumerate(unique_z))
    cell_df_local.layer = [z_to_layer[z] for z in cell_df_local.z]

    # Simulation flags
    track_seg_local  = track_seg
    full_cycle_local = true

    # --- Inject into global environment (Main) ---
    @eval Main begin
        ParIrr        = $ParIrr
        N             = $N_local
        nodes_positions = $nodes_positions_local
        N_CellsSide   = $N_CellsSide_local
        cell_df       = $cell_df_local
        track_seg     = $track_seg_local
        full_cycle    = $full_cycle_local
    end

    # --- Print summary ---
    println("=========== Cell Lattice Setup ===========")
    println(" Geometry:            $target_geom")
    println(" Partial Irradiation: $ParIrr")
    println(" Total Cells:         $N_local")
    println(" Cells per side:      $N_CellsSide_local")
    println(" Layer count (z):     ", length(unique_z))
    println("==========================================")

    return nothing
end

"""
setup_cell_population!(
        target_geom::String,
        X_box::Float64,
        R_cell::Float64,
        N_sideVox::Int,
        N_CellsSide::Int;
        ParIrr::String = "false"
    ) -> Nothing

High‑level wrapper that prepares a complete cell population for simulation.

This function:
1. Calls `setup_cell_lattice!` → generates cell_df and spatial layout.
2. Calls `populate_cells_wrapper` → populates biological attributes (GSM2, domains, voxel indices…).
3. Computes GSM2 domain-center template (relative positions).
4. Builds absolute domain-center DataFrames and initializes the AT table.
5. Defines all variables in the caller’s environment (`Main`).
6. Prints a concise summary.
7. Returns nothing.

# Requirements
Before calling this function, you must have run:
- `setup_GSM2!`    (defines `gsm2`, needed for domain centers)
- `setup_IonIrrad!` (defines `ion`, `irrad`, `type_AT`, `domain`)

# Defined Globals (created in Main)
- From lattice:
    ParIrr, N, nodes_positions, N_CellsSide, cell_df,
    track_seg, full_cycle
- From population:
    updated cell_df with biological and voxel info
- From domain setup:
    rel_center_x, rel_center_y, num_cols
    df_center_x, df_center_y, at

# Example
```julia
setup_cell_population!("square", X_box, R_cell, N_sideVox, N_CellsSide)
```
"""
function setup_cell_population!(
    target_geom::String,
    X_box::Float64,
    R_cell::Float64,
    N_sideVox::Int,
    N_CellsSide::Int,
    gsm2::GSM2 = gsm2;
    ParIrr::String = "false"
)

    println(">>> Running full cell population setup...")

    # 'cell_df', 'nodes_positions', 'track_seg', etc. now exist in Main.
    # We capture them locally for safety:
    cell_df_local = Main.cell_df
    domain_local  = Main.domain 
    nodes_positions_local = Main.nodes_positions
    N_local               = Main.N
    N_CellsSide_local     = Main.N_CellsSide
    track_seg_local       = Main.track_seg
    full_cycle_local      = Main.full_cycle

    # ============================================================
    # 2) Populate biological attributes into cell_df
    # ============================================================
    println(">>> Populating biological properties...")

    @time populate_cells_wrapper(
        ParIrr,
        N_local,
        nodes_positions_local,
        R_cell,
        gsm2,              # GSM2 model from previous setup
        cell_df_local,
        domain,            # already defined by setup_GSM2!
        tumor_radius,
        full_cycle_local,
        target_geom,
        type_AT,           # KC or LEM
        N_sideVox,
        N_CellsSide_local,
        X_box,
        X_voxel
    )

    # Write updated df back to Main
    @eval Main cell_df = $cell_df_local

    # ============================================================
    # 3) Domain-center template (relative coordinates)
    # ============================================================
    rel_center_x_local, rel_center_y_local =
        calculate_centers(0.0, 0.0, gsm2.rd, gsm2.Rn)

    num_cols_local = length(rel_center_x_local)

    # ============================================================
    # 4) Build full domain-center DataFrames for all cells
    # ============================================================
    df_center_x_local, df_center_y_local, at_local =
        create_domain_dataframes(cell_df_local,
                                    rel_center_x_local,
                                    rel_center_y_local)

    # ============================================================
    # 5) Inject results into Main
    # ============================================================
    @eval Main begin
        rel_center_x = $rel_center_x_local
        rel_center_y = $rel_center_y_local
        df_center_x  = $df_center_x_local
        df_center_y  = $df_center_y_local
        at           = $at_local
        num_cols     = $num_cols_local
    end

    compute_possible_division_df!(cell_df_local)
    @eval Main cell_df = $cell_df_local
    
    # ============================================================
    # Summary
    # ============================================================
    println("=========== Cell Population Setup COMPLETE ===========")
    println(" Geometry:            $target_geom")
    println(" Partial irradiation: $ParIrr")
    println(" Total cells:         $N_local")
    println(" Domains per cell:    $num_cols_local")
    println(" DataFrames created:  df_center_x, df_center_y, at")
    println("=======================================================")

    return nothing
end

"""
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

High-level wrapper to compute irradiation conditions across the spheroid layers,
assign energy steps, and build the reduced irradiation-condition array.

This function performs the following steps:
1. Allocates `irrad_cond` (one AT object per distinct z-layer).
2. Calls `compute_energy_box!` to compute LET, E, Rk, Rc, etc. for each layer.
3. Extracts LET and energy vectors for later analysis.
4. Assigns `cell_df.layer` 
5. Saves an original copy of `irrad_cond` (one per layer).
6. Calls `set_energy_steps!` to determine which cells share the same energy step.
7. Allocates a reduced `irrad_cond` array indexed by energy step.
8. For each energy step, finds the corresponding layers and picks the mid-layer
    irradiation condition.
9. Prints a summary.
10. Returns nothing — all variables are injected into `Main`.

Arguments:
- `ion`          : Ion object (created previously by setup_IonIrrad!)
- `irrad`        : Irrad object
- `type_AT`      : Track-structure model ("KC" or "LEM")
- `cell_df`      : DataFrame of cells containing at least `z`, `is_cell`
- `track_seg`    : Boolean controlling track segmentation mode

Creates the following global variables in `Main`:
- `irrad_cond`
- `irrad_cond_original_layers`
- `lets`
- `energies`
- `cell_df`     (updated with `layer` and `energy_step`)
- `num_energy_steps`
"""
function setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    println(">>> Computing irradiation conditions per layer...")

    unique_z = sort(unique(cell_df.z))
    z_to_layer = Dict(z => i for (i, z) in enumerate(unique_z))

    # ===============================================================
    # 1) Allocate irrad_cond array (one entry per unique z-layer)
    # ===============================================================
    irrad_cond_local = Array{AT}(undef, length(unique_z))

    # ===============================================================
    # 2) Compute energy/LET boxes for each layer
    # ===============================================================
    compute_energy_box!(irrad_cond_local, ion, irrad, type_AT, cell_df, track_seg)

    # Extract LET and E vectors
    lets_local     = getfield.(irrad_cond_local, :LET)
    energies_local = getfield.(irrad_cond_local, :E)

    # ===============================================================
    # 3) Assign layers to cell_df
    # ===============================================================
    cell_df.layer = [z_to_layer[z] for z in cell_df.z]

    # Deep copy the original per-layer irrad data
    irrad_cond_original_local = deepcopy(irrad_cond_local)

    # ===============================================================
    # 4) Assign energy steps to each cell
    # ===============================================================
    println(">>> Assigning energy steps to each cell...")
    @time set_energy_steps!(cell_df, irrad_cond_local)

    # Identify how many steps exist
    unique_steps = unique(cell_df.energy_step)
    num_steps = length(unique_steps)

    # ===============================================================
    # 5) Build reduced irrad_cond by energy step
    # ===============================================================
    irrad_cond_reduced = Array{AT}(undef, num_steps)

    for step in unique_steps
        layers = unique(cell_df[cell_df.energy_step .== step, :layer])

        # Step corresponds to the middle layer among these
        mid_layer = round(Int, 0.5 * (minimum(layers) + maximum(layers)))

        irrad_cond_reduced[step] = irrad_cond_original_local[mid_layer]

        println("Energy step $step covers layers: $layers  → using mid-layer $mid_layer")
    end

    # ===============================================================
    # Inject results into global environment
    # ===============================================================
    @eval Main begin
        irrad_cond              = $irrad_cond_reduced
        irrad_cond_original_layers = $irrad_cond_original_local
        lets                    = $lets_local
        energies                = $energies_local
        cell_df                 = $cell_df
        num_energy_steps        = $num_steps
    end

    # ===============================================================
    # Summary
    # ===============================================================
    println("=========== Irradiation Condition Setup COMPLETE ===========")
    println(" Layers detected:           ", length(unique_z))
    println(" Energy steps computed:     ", num_steps)
    println(" LET range:                 ", minimum(lets_local), " → ", maximum(lets_local))
    println(" Energy range (MeV/u):      ", minimum(energies_local), " → ", maximum(energies_local))
    println("=============================================================")

    return nothing
end

"""
set_energy_steps!(cell_df::DataFrame, irrad_cond::Vector{AT}) -> Nothing

Assign a discrete "energy_step" to each cell based on the layer-dependent ion energy
stored in `irrad_cond`. The mapping uses logarithmic binning (log-spaced bins)
spanning several decades to provide finer resolution at low energies and coarser
resolution at high energies.

What this does:
1. Builds a log-spaced set of bin edges from min_val to max_val, with a
    fixed number of bins per decade.
2. For each cell, looks up its layer, fetches the corresponding energy from
    `irrad_cond[layer].E`, and assigns the bin index using `set_energy_bins`.
3. Calls `remap_bins` to produce compact, consecutive step indices suitable for
    downstream processing and visualization.
4. Writes the final step index into `cell_df.energy_step`.

Inputs:
- cell_df: DataFrame with at least a column `:layer` (Int) giving the layer index for each row.
- irrad_cond: Vector of AT objects (one per layer), each providing the energy `E` (MeV/u).

Notes:
- The log binning is defined by:
    - min_val = 1e-2 (MeV/u)
    - max_val = 1e3  (MeV/u)
    - bins_per_decade = 50
- The first bin includes its lower bound; subsequent bins are (lower, upper] intervals.
- Function mutates `cell_df` in place; no value is returned.
- Assumes that `remap_bins(::AbstractVector)` exists and returns a compact set of
    consecutive integers starting from 1 in the order of first appearance.

Example:
```julia
set_energy_steps!(cell_df, irrad_cond)
unique(cell_df.energy_step) |> sort
```
"""
function set_energy_steps!(cell_df::DataFrame, irrad_cond::Vector{AT})
    min_val = 10^-2
    max_val = 10^3
    bins_per_decade = 50
    num_decades = log10(max_val / min_val)
    num_bins = Int(round(num_decades * bins_per_decade))
    # Log-spaced bin edges, plus a zero at the start to catch exact zeros if any
    bins = vcat(0, 10 .^ range(log10(min_val), log10(max_val), length = num_bins + 1))

    # Compute a bin index for each cell based on its layer's energy
    bin_energy = zeros(Float64, size(cell_df, 1))
    for j in 1:size(cell_df, 1)
        layer = cell_df.layer[j]
        bin_energy[j] = set_energy_bins(irrad_cond[layer].E, bins)
    end

    # Remap to compact consecutive step indices
    cell_df.energy_step .= remap_bins(bin_energy)
    return nothing
end

"""
set_energy_bins(value::Float64, bins::Vector{Float64}) -> Int

Return the index of the bin in `bins` that contains the given `value`.

The vector `bins` represents a sequence of bin edges. For example, if
`bins = [0, 0.1, 1.0, 10.0]`, then:

* Bin 1 covers values equal to the first edge (value == bins[1]).
* Bin 2 covers values strictly greater than bins[1] and up to bins[2], including the upper edge.
* Bin 3 covers values strictly greater than bins[2] and up to bins[3], including the upper edge.

In other words:
- The very first bin includes its lower boundary.
- All other bins include their upper boundary but exclude their lower boundary.

The function scans through the bin edges and returns the index of the first
interval whose boundaries contain the input value. This is useful for assigning
energies to logarithmic or linear bins.

Arguments:
- `value` : the scalar to place into a bin.
- `bins`  : vector of bin edges sorted in ascending order.

Returns:
- The integer index of the bin that contains `value`.

Notes:
- The function does not handle values outside the provided bin range; ensure that
    `value` always fits inside the edges or extend `bins` accordingly.
"""
function set_energy_bins(value::Float64, bins::Vector{Float64})
    for i in 1:(length(bins) - 1)
        lower = bins[i]
        upper = bins[i + 1]

        # First bin: allow value == lower.
        if i == 1 && value == lower
            return i
        end

        # General case: lower < value ≤ upper
        if (value > lower) && (value ≤ upper)
            return i
        end
    end

    # No explicit return for out-of-range values (they should not happen)
    return nothing
end

"""
    remap_bins(bin_indices::AbstractVector{<:Real}) -> Vector{Int}

Remap an arbitrary sequence of bin identifiers to a compact, consecutive set of
integers starting from 1, preserving the **order of first appearance**.

This is useful when bin IDs come from heterogeneous sources (e.g., floating-point
labels or sparse integer labels) and you want stable, compact step indices for
grouping, plotting, or indexing.

Behavior:
- Scans `bin_indices` from left to right.
- The first time a bin label is seen, it is assigned the next available integer
    (1, 2, 3, ...).
- Subsequent occurrences of the same label receive the same assigned integer.
- The output length matches the input length.
- The mapping order is defined by first appearance in `bin_indices`.

Notes:
- If your input contains floating-point labels (e.g., 1.0, 2.0), they will be
    treated as distinct keys if their values differ. If the labels are actually
    integer-like floats, they will still map consistently (e.g., 1.0 -> 1, 2.0 -> 2).
- If you already have integer bin indices, pass them directly; this function
    will keep their encounter order but re-label to a compact range starting at 1.

Example:
```julia
remap_bins([10.0, 10.0, 25.0, 10.0, 40.0])  # -> [1, 1, 2, 1, 3]
remap_bins([3, 3, 7, 5, 7, 7])              # -> [1, 1, 2, 3, 2, 2]
```
"""
function remap_bins(bin_indices::AbstractVector{<:Real})::Vector{Int}
                    mapping = Dict{Float64, Int}()  # use Float64 to handle both Int and Float inputs uniformly
                    result = Vector{Int}(undef, length(bin_indices))
    next_id = 1
    @inbounds for i in eachindex(bin_indices)
        key = Float64(bin_indices[i])  # normalize key type to avoid Dict{Int,Int} vs Float64 issues
        if !haskey(mapping, key)
            mapping[key] = next_id
            next_id += 1
        end
        result[i] = mapping[key]
    end

    return result
end

"""
    compute_possible_division_df!(cell_df::DataFrame)

Determine whether each cell in the lattice can divide, based on its biological state
and the availability of empty neighboring positions. The function updates the
columns `can_divide` and `number_nei` of `cell_df` directly (in place).

How the rule works:
1. Each cell has a list of neighbors stored in `cell_df.nei`. These are the 3D
    neighbors in the lattice (up to 26 positions).
2. A cell is considered "real" when `is_cell == 1`. Empty lattice locations have
    `is_cell == 0`.
3. For each cell, we count how many of its neighboring positions are empty.
    This is the number of available sites into which the cell could expand
    during division. This count is stored in `number_nei`.
4. A cell is allowed to divide only if:
    - it is a real cell (is_cell == 1)
    - its cell cycle state is exactly "M", meaning the mitosis phase
    - it has at least one empty neighboring site
5. When these conditions are met, `can_divide` is set to 1. Otherwise it is 0.

Parallelization:
The loop is threaded using `Threads.@threads` for better performance when the
DataFrame contains many cells.

Arguments:
- `cell_df` : DataFrame containing at least the columns `is_cell`, `cell_cycle`,
    and `nei`, where `nei[i]` is a vector of neighbor indices for cell i.

Returns:
- Nothing. The function modifies the DataFrame in-place by setting the columns
    `can_divide` and `number_nei`.

Example:
```julia
compute_possible_division_df!(cell_df)
println(cell_df.can_divide)
```
"""
function compute_possible_division_df!(cell_df::DataFrame)
    N = nrow(cell_df)
    can_divide_vec = zeros(Int64, N)
    number_nei_vec = zeros(Int64, N)

    # Extract columns for speed (avoids repeated DataFrame indexing)
    nei_col = cell_df.nei
    is_cell_col = cell_df.is_cell
    cell_cycle_col = cell_df.cell_cycle

    Threads.@threads for i in 1:N
        # ------------------------------------------------------------
        # 1) Count empty neighbors
        # ------------------------------------------------------------
        current_nei_indices = nei_col[i]

        num_empty = 0
        for j in current_nei_indices
            # is_cell == 1 → filled, is_cell == 0 → empty
            num_empty += (1 - is_cell_col[j])
        end
        number_nei_vec[i] = num_empty

        # ------------------------------------------------------------
        # 2) Determine if cell can divide
        # ------------------------------------------------------------
        if (is_cell_col[i] == 1) && (cell_cycle_col[i] == "M")
            if num_empty > 0
                can_divide_vec[i] = 1
            end
        end
    end

    # Write results back into DataFrame
    cell_df.can_divide = can_divide_vec
    cell_df.number_nei = number_nei_vec

    return nothing
end