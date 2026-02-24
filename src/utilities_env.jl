#! ============================================================================
#! utilities_env.jl
#!
#! FUNCTIONS
#! ---------
#~ Environment / Global Setup (side-effect functions — inject into Main)
#?   setup_GSM2!(r, a, b, rd, Rn; x0, y0) -> Nothing
#       Builds GSM2 object and domain-center template; injects gsm2, center_x,
#       center_y, domain into Main.
#?   setup_IonIrrad!(dose, E, particle; type_AT, kR, dose_rate) -> Nothing
#       Interpolates LET, builds Ion/Irrad objects, computes AT track radii;
#       injects type_AT, ion, irrad, LET, Rc, Rp, Rk, Kp, A, Z into Main.
#?   setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr, track_seg)
#           -> Nothing
#       Generates cell positions, builds cell_df with x/y/z/layer columns;
#       injects N, nodes_positions, cell_df, track_seg, full_cycle into Main.
#?   setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2; ParIrr)
#           -> Nothing
#       Full population setup: calls setup_cell_lattice!, populate_cells_wrapper,
#       domain-center DataFrames; injects rel_center_x/y, df_center_x/y, at, num_cols into Main.
#       Requires setup_GSM2! and setup_IonIrrad! to have been run first.
#?   setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg) -> Nothing
#       Computes per-layer AT conditions, assigns energy steps, builds reduced
#       irrad_cond array; injects irrad_cond, lets, energies, num_energy_steps into Main.
#
#~ Energy Binning
#?   set_energy_steps!(cell_df, irrad_cond) -> Nothing
#       Assigns a discrete energy_step to each cell via log-spaced binning of layer energy.
#?   set_energy_bins(value, bins) -> Int
#       Returns the bin index for a scalar value given a vector of bin edges.
#?   remap_bins(bin_indices) -> Vector{Int}
#       Remaps arbitrary bin labels to compact consecutive integers (order of first appearance).
#
#~ Cell Division
#?   compute_possible_division_df!(cell_df) -> Nothing
#       Sets can_divide and number_nei for each cell based on empty neighbor count.
#       A cell can divide only if is_cell==1, cell_cycle=="M", and has empty neighbors.
#       Threaded.
#! ============================================================================

"""
    setup_GSM2!(r, a, b, rd, Rn; x0=-300.0, y0=-300.0) -> Nothing

Builds `GSM2` object and domain-center template; injects `gsm2`, `center_x`,
`center_y`, `domain` into `Main`. Prints a setup summary.

Side-effect function — no return value.

# Example
```julia
setup_GSM2!(4.3, 0.01, 0.30, 0.8, 7.2)
```
"""
function setup_GSM2!(r, a, b, rd, Rn; x0=-300.0, y0=-300.0)
    gsm2_local = GSM2(r, a, b, rd, Rn)
    center_x_local, center_y_local = calculate_centers(x0, y0, rd, Rn)
    domain_local = length(center_x_local)

    @eval Main begin
        gsm2     = $gsm2_local
        center_x = $center_x_local
        center_y = $center_y_local
        domain   = $domain_local
    end

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
    setup_IonIrrad!(dose, E, particle; type_AT="KC", kR=1.0, dose_rate=0.18) -> Nothing

Interpolates LET, builds `Ion`/`Irrad` objects, computes AT track radii.
Injects `type_AT`, `ion`, `irrad`, `LET`, `Rc`, `Rp`, `Rk`, `Kp`, `A`, `Z` into `Main`.

Requires global `sp` (stopping power dict) to be in scope.

# Example
```julia
setup_IonIrrad!(1.0, 100.0, "1H")
setup_IonIrrad!(2.0, 150.0, "12C"; type_AT="LEM")
```
"""
function setup_IonIrrad!(dose::Float64, E::Float64, particle::String;
        type_AT::String  = "KC",
        kR::Float64      = 1.0,
        dose_rate::Float64 = 0.18)

    A_ = parse(Int, filter(isdigit, particle))
    A  = 1.
    Z  = getZ(particle)

    LET_local  = linear_interpolation(particle, E, sp)
    ion_local  = Ion(particle, E, A, Z, LET_local, 1.0)
    irrad_local = Irrad(dose, kR, dose_rate)

    Rc_local, Rp_local, Kp_local = ATRadius(ion_local, irrad_local, type_AT)
    Rk_local = Rp_local

    @eval Main begin
        type_AT    = $type_AT
        dose       = $dose
        A_         = $A_
        A          = $A
        Z          = $Z
        LET        = $LET_local
        ion        = $ion_local
        irrad      = $irrad_local
        Rc         = $Rc_local
        Rp         = $Rp_local
        Rk         = $Rk_local
        Kp         = $Kp_local
        DoseRate_h = 0.0
        F          = 0.0
        D          = 0.0
        T          = 0.0
    end

    println("=========== Ion & Irradiation Setup ===========")
    println(" Particle:         $particle  (A = $A_, Z = $Z)")
    println(" Energy:           $E MeV/u")
    println(" LET:              $LET_local keV/μm")
    println(" Dose:             $dose Gy")
    println(" Track model:      $type_AT")
    println(" Rc = $(Rc_local), Rp = $(Rp_local), Rk = $(Rk_local), Kp = $(Kp_local)")
    println(" Dose rate:        $dose_rate Gy/h")
    println("===============================================")

    return nothing
end

"""
    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=true) -> Nothing

Generates cell positions and builds `cell_df` with `x/y/z/layer` columns.
Injects `ParIrr`, `N`, `nodes_positions`, `N_CellsSide`, `cell_df`,
`track_seg`, `full_cycle` into `Main`.

# Example
```julia
setup_cell_lattice!("square", 900.0, 15.0, 12, 60; ParIrr="true")
```
"""
function setup_cell_lattice!(target_geom::String,
                              X_box::Float64,
                              R_cell::Float64,
                              N_sideVox::Int,
                              N_CellsSide::Int;
                              ParIrr::String  = "false",
                              track_seg::Bool = true)

    N_local, nodes_positions_local, N_CellsSide_local =
        generate_cells_positions_selector(target_geom, ParIrr,
                                          X_box, R_cell,
                                          N_sideVox, N_CellsSide)

    cell_df_local = DataFrame(
        index = collect(eachindex(nodes_positions_local)),
        x     = [p[1] for p in nodes_positions_local],
        y     = [p[2] for p in nodes_positions_local],
        z     = [p[3] for p in nodes_positions_local],
    )

    unique_z    = sort(unique(cell_df_local.z))
    z_to_layer  = Dict(z => i for (i, z) in enumerate(unique_z))
    cell_df_local.layer = [z_to_layer[z] for z in cell_df_local.z]

    @eval Main begin
        ParIrr          = $ParIrr
        N               = $N_local
        nodes_positions = $nodes_positions_local
        N_CellsSide     = $N_CellsSide_local
        cell_df         = $cell_df_local
        track_seg       = $track_seg
        full_cycle      = true
    end

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
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2;
                           ParIrr="false") -> Nothing

Full population setup pipeline:
1. `setup_cell_lattice!` — spatial layout
2. `populate_cells_wrapper` — biological attributes (cycle, voxels, damage matrices…)
3. Domain-center DataFrames and AT table

Injects `rel_center_x/y`, `df_center_x/y`, `at`, `num_cols` into `Main`.
Requires `setup_GSM2!` and `setup_IonIrrad!` to have run first.

# Example
```julia
setup_cell_population!("square", 900.0, 15.0, 12, 60, gsm2)
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

    cell_df_local         = Main.cell_df
    nodes_positions_local = Main.nodes_positions
    N_local               = Main.N
    N_CellsSide_local     = Main.N_CellsSide
    full_cycle_local      = Main.full_cycle

    println(">>> Populating biological properties...")
    @time populate_cells_wrapper(
        ParIrr, N_local, nodes_positions_local, R_cell, gsm2,
        cell_df_local, domain, tumor_radius, full_cycle_local,
        target_geom, type_AT, N_sideVox, N_CellsSide_local, X_box, X_voxel
    )
    @eval Main cell_df = $cell_df_local

    # Domain-center template (relative coordinates)
    rel_center_x_local, rel_center_y_local = calculate_centers(0.0, 0.0, gsm2.rd, gsm2.Rn)
    num_cols_local = length(rel_center_x_local)

    # Absolute domain-center DataFrames for all cells
    df_center_x_local, df_center_y_local, at_local =
        create_domain_dataframes(cell_df_local, rel_center_x_local, rel_center_y_local)

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

    N_alive = length(cell_df_local.index[cell_df_local.is_cell .== 1])

    println("=========== Cell Population Setup COMPLETE ===========")
    println(" Geometry:            $target_geom")
    println(" Partial irradiation: $ParIrr")
    println(" Total cells:         $N_alive")
    println(" Domains per cell:    $num_cols_local")
    println(" DataFrames created:  df_center_x, df_center_y, at")
    println("=======================================================")

    return nothing
end

"""
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg) -> Nothing

Computes per-layer AT conditions, assigns energy steps, and builds a reduced
`irrad_cond` array indexed by energy step.

Injects `irrad_cond`, `irrad_cond_original_layers`, `lets`, `energies`,
`cell_df`, `num_energy_steps` into `Main`.

# Example
```julia
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
```
"""
function setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
    println(">>> Computing irradiation conditions per layer...")

    unique_z   = sort(unique(cell_df.z))
    z_to_layer = Dict(z => i for (i, z) in enumerate(unique_z))

    irrad_cond_local = Array{AT}(undef, length(unique_z))
    compute_energy_box!(irrad_cond_local, ion, irrad, type_AT, cell_df, track_seg)

    lets_local     = getfield.(irrad_cond_local, :LET)
    energies_local = getfield.(irrad_cond_local, :E)

    cell_df.layer = [z_to_layer[z] for z in cell_df.z]

    irrad_cond_original_local = deepcopy(irrad_cond_local)

    println(">>> Assigning energy steps to each cell...")
    @time set_energy_steps!(cell_df, irrad_cond_local)

    unique_steps = unique(cell_df.energy_step)
    num_steps    = length(unique_steps)

    # Reduced irrad_cond: one entry per energy step, using mid-layer condition
    irrad_cond_reduced = Array{AT}(undef, num_steps)
    for step in unique_steps
        layers    = unique(cell_df[cell_df.energy_step .== step, :layer])
        mid_layer = round(Int, 0.5 * (minimum(layers) + maximum(layers)))
        irrad_cond_reduced[step] = irrad_cond_original_local[mid_layer]
        println("Energy step $step covers layers: $layers  → using mid-layer $mid_layer")
    end

    @eval Main begin
        irrad_cond                 = $irrad_cond_reduced
        irrad_cond_original_layers = $irrad_cond_original_local
        lets                       = $lets_local
        energies                   = $energies_local
        cell_df                    = $cell_df
        num_energy_steps           = $num_steps
    end

    println("=========== Irradiation Condition Setup COMPLETE ===========")
    println(" Layers detected:       ", length(unique_z))
    println(" Energy steps:          ", num_steps)
    println(" LET range:             ", minimum(lets_local), " → ", maximum(lets_local))
    println(" Energy range (MeV/u):  ", minimum(energies_local), " → ", maximum(energies_local))
    println("=============================================================")

    return nothing
end

"""
    set_energy_steps!(cell_df::DataFrame, irrad_cond::Vector{AT}) -> Nothing

Assigns a discrete `energy_step` to each cell via log-spaced binning of layer energy.
Bins span 1e-2 to 1e3 MeV/u with 50 bins/decade. Mutates `cell_df` in-place.

# Example
```julia
set_energy_steps!(cell_df, irrad_cond)
sort(unique(cell_df.energy_step))
```
"""
function set_energy_steps!(cell_df::DataFrame, irrad_cond::Vector{AT})
    min_val         = 1e-2
    max_val         = 1e3
    bins_per_decade = 50
    num_bins        = Int(round(log10(max_val / min_val) * bins_per_decade))
    bins = vcat(0, 10 .^ range(log10(min_val), log10(max_val), length = num_bins + 1))

    bin_energy = zeros(Float64, nrow(cell_df))
    for j in 1:nrow(cell_df)
        bin_energy[j] = set_energy_bins(irrad_cond[cell_df.layer[j]].E, bins)
    end

    cell_df.energy_step .= remap_bins(bin_energy)
    return nothing
end

"""
    set_energy_bins(value::Float64, bins::Vector{Float64}) -> Int

Returns the bin index containing `value` from a sorted vector of bin edges.
First bin includes its lower bound; all others are `(lower, upper]`.

# Example
```julia
set_energy_bins(0.5, [0.0, 0.1, 1.0, 10.0])  # → 2
```
"""
function set_energy_bins(value::Float64, bins::Vector{Float64})
    for i in 1:(length(bins) - 1)
        lower = bins[i]
        upper = bins[i + 1]
        (i == 1 && value == lower) && return i
        (value > lower && value ≤ upper) && return i
    end
    return nothing
end

"""
    remap_bins(bin_indices::AbstractVector{<:Real}) -> Vector{Int}

Remaps arbitrary bin labels to compact consecutive integers starting from 1,
preserving order of first appearance.

# Example
```julia
remap_bins([10.0, 10.0, 25.0, 10.0, 40.0])  # → [1, 1, 2, 1, 3]
remap_bins([3, 3, 7, 5, 7, 7])              # → [1, 1, 2, 3, 2, 2]
```
"""
function remap_bins(bin_indices::AbstractVector{<:Real})::Vector{Int}
    mapping = Dict{Float64, Int}()
    result  = Vector{Int}(undef, length(bin_indices))
    next_id = 1

    @inbounds for i in eachindex(bin_indices)
        key = Float64(bin_indices[i])
        if !haskey(mapping, key)
            mapping[key] = next_id
            next_id += 1
        end
        result[i] = mapping[key]
    end

    return result
end

"""
    compute_possible_division_df!(cell_df::DataFrame) -> Nothing

Sets `can_divide` and `number_nei` for each cell based on empty neighbor count.
A cell can divide only if `is_cell==1`, `cell_cycle=="M"`, and has at least one
empty neighbor (`is_cell==0`). Threaded.

# Example
```julia
compute_possible_division_df!(cell_df)
```
"""
function compute_possible_division_df!(cell_df::DataFrame)
    N = nrow(cell_df)
    can_divide_vec = zeros(Int64, N)
    number_nei_vec = zeros(Int64, N)

    nei_col        = cell_df.nei
    is_cell_col    = cell_df.is_cell
    cell_cycle_col = cell_df.cell_cycle

    Threads.@threads for i in 1:N
        num_empty = 0
        for j in nei_col[i]
            num_empty += (1 - is_cell_col[j])
        end
        number_nei_vec[i] = num_empty

        if (is_cell_col[i] == 1) && (cell_cycle_col[i] == "M") && (num_empty > 0)
            can_divide_vec[i] = 1
        end
    end

    cell_df.can_divide = can_divide_vec
    cell_df.number_nei = number_nei_vec

    return nothing
end

"""
    setup_simulation(E, particle, dose, tumor_radius;
                        X_box        = 900.0,
                        X_voxel      = 300.0,
                        r_nucleus    = Rn,
                        R_cell       = 15.0,
                        target_geom  = "circle",
                        calc_type    = "full",
                        type_AT      = "KC",
                        ParIrr       = "false",
                        track_seg    = true,
                        plot_oxygen  = false,
                        verbose      = true)

Wrapper for the complete simulation‑setup pipeline for particle‑irradiation
studies. This routine prepares geometry, ion and irradiation parameters,
beam characteristics, cell‑lattice configuration, oxygenation, and
particle fluence. Only the four required arguments must be provided; all
other parameters have sensible defaults.

# Required Arguments
- `E`            :: Float64  
    Ion kinetic energy in MeV/u.

- `particle`     :: String  
    Ion species identifier, e.g. `"1H"`, `"4He"`, `"12C"`.

- `dose`         :: Float64  
    Total delivered dose in Gy.

- `tumor_radius` :: Float64  
    Radius of the target/tumor region in µm.

# Optional Keyword Arguments
- `X_box`        :: Float64 = 900.0  
    Half‑side of the computational domain (µm).

- `X_voxel`      :: Float64 = 300.0  
    Voxel side length used for grid discretization (µm).

- `r_nucleus`    :: Float64 = Rn  
    Nuclear radius of each cell (µm).

- `R_cell`       :: Float64 = 15.0  
    Cellular radius (µm), used in building the cell lattice.

- `target_geom`  :: String = "circle"  
    Target geometry: `"circle"` or `"square"` (or others if implemented).

- `calc_type`    :: String = "full"  
    Beam‑calculation mode, e.g. `"full"` or `"simple"`.

- `type_AT`      :: String = "KC"  
    Model type for amorphous track‑structure (AT) calculations.

- `ParIrr`       :: String = "false"  
    If `"true"`, enables parallel irradiation (model‑dependent).

- `track_seg`    :: Bool = true  
    Whether to enable track‑segmentation output.

- `plot_oxygen`  :: Bool = false  
    Whether to display oxygen‑distribution diagnostics.

- `verbose`      :: Bool = true  
    Toggles detailed printed output.

# Description
The function performs the following steps:
1. **Geometry setup:** Computes voxel counts and cell‑lattice dimensions.  
2. **Ion/irradiation initialization:** Sets ion properties and dose conditions.  
3. **Beam‑parameter calculation:** Determines beam radius and centroid.  
4. **Amorphous track‑structure initialization:** Computes AT radii and constructs an initial AT object.  
5. **Cell‑lattice generation and population:** Generates spatial arrangement of cells and assigns biological parameters.  
6. **Irradiation‑condition setup:** Prepares physical and radiobiological interaction parameters.  
7. **Oxygenation:** Assigns oxygen values per cell and computes average O₂.  
8. **Fluence and particle‑count estimation:** Converts dose to fluence and computes the number of required primaries.

# Returns
A tuple:
`(Npar, R_beam, x_beam, y_beam, at_start, O2_mean)`, where

- `Npar`    :: Int  
    Number of primary particles required to deliver the dose.

- `R_beam`  :: Float64  
    Beam radius (µm).

- `x_beam` / `y_beam` :: Float64  
    Beam center coordinates (µm).

- `at_start`  
    Initial amorphous‑track object for the given ion and energy.

- `O2_mean` :: Float64  
    Mean oxygen level among all valid cells.

"""
function setup_simulation(
    E            :: Float64,
    particle     :: String,
    dose         :: Float64,
    tumor_radius :: Float64;
    X_box        :: Float64  = 900.0,
    X_voxel      :: Float64  = 300.0,
    r_nucleus    :: Float64  = Rn,
    R_cell       :: Float64  = 15.0,
    target_geom  :: String   = "circle",
    calc_type    :: String   = "full",
    type_AT      :: String   = "KC",
    ParIrr       :: String   = "false",
    track_seg    :: Bool     = true,
    plot_oxygen  :: Bool     = false,
    verbose      :: Bool     = true
)
    # ── Geometry ──────────────────────────────────────────────────────────────
    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    verbose && println("X_box        : $X_box")
    verbose && println("X_voxel      : $X_voxel")
    verbose && println("N_sideVox    : $N_sideVox")
    verbose && println("r_nucleus    : $r_nucleus")
    verbose && println("R_cell       : $R_cell")

    # ── Ion & irradiation ─────────────────────────────────────────────────────
    setup_IonIrrad!(dose, E, particle)

    # ── Beam properties ───────────────────────────────────────────────────────
    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    # ── Amorphous track structure ─────────────────────────────────────────────
    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)  # Rk = Rp

    # ── Cell lattice & population ─────────────────────────────────────────────
    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr=ParIrr, track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
    verbose && println("Number of cells : $(sum(cell_df.is_cell .== 1))")

    # ── Irradiation conditions ────────────────────────────────────────────────
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    # ── Oxygen ───────────────────────────────────────────────────────────────
    set_oxygen!(cell_df; plot_oxygen=plot_oxygen)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])
    verbose && println("Mean O2         : $(round(O2_mean, digits=3))")

    # ── Particle fluence ──────────────────────────────────────────────────────
    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    verbose && println("Npar            : $Npar")
    verbose && println("R_beam          : $(round(R_beam, digits=2))")

    return (Npar, R_beam, x_beam, y_beam, at_start, O2_mean)
end
