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
#?   setup(E, particle, dose, tumor_radius; X_box, X_voxel, R_cell, target_geom, calc_type, type_AT, track_seg)
#           -> NamedTuple (ion, irrad, cell_df, at_start, R_beam, x_beam, y_beam, O2_mean, Npar, zF, D, T)
#       Full pipeline: setup_GSM2! → setup_IonIrrad! → setup_cell_lattice! →
#       setup_cell_population! → setup_irrad_conditions! → set_oxygen!.
#       Uses Base.invokelatest to avoid world-age errors in notebook/async contexts.
#       Requires gsm2 to be in Main (set by setup_GSM2! beforehand).
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
`A` is the mass number extracted from `particle` (e.g. `12` for `"12C"`).

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
    A  = Float64(A_)   # mass number in atomic mass units
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
                        ParIrr="false", track_seg=true, full_cycle=true) -> Nothing

Generates cell positions and builds `cell_df` with `x/y/z/layer` columns.
Injects `ParIrr`, `N`, `nodes_positions`, `N_CellsSide`, `cell_df`,
`track_seg`, `full_cycle` into `Main`.

- `full_cycle=true`: cells start distributed across the full cell cycle.
  Set to `false` to initialise all cells in G1 (quiescent-like start).

# Example
```julia
setup_cell_lattice!("square", 900.0, 15.0, 12, 60; ParIrr="true")
setup_cell_lattice!("circle", 600.0, 15.0, 12, 40; full_cycle=false)
```
"""
function setup_cell_lattice!(target_geom::String,
                                X_box::Float64,
                                R_cell::Float64,
                                N_sideVox::Int,
                                N_CellsSide::Int;
                                ParIrr::String     = "false",
                                track_seg::Bool    = true,
                                full_cycle::Bool   = true)

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
        full_cycle      = $full_cycle
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

    rel_center_x_local, rel_center_y_local = calculate_centers(0.0, 0.0, gsm2.rd, gsm2.Rn)
    num_cols_local = length(rel_center_x_local)

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
Out-of-range values are clamped: returns `1` if below the first edge,
`length(bins)-1` if above the last edge (never returns `nothing`).

# Example
```julia
set_energy_bins(0.5, [0.0, 0.1, 1.0, 10.0])  # → 2
set_energy_bins(-1.0, [0.0, 0.1, 1.0, 10.0]) # → 1   (clamped)
set_energy_bins(99.0, [0.0, 0.1, 1.0, 10.0]) # → 3   (clamped)
```
"""
function set_energy_bins(value::Float64, bins::Vector{Float64})
    for i in 1:(length(bins) - 1)
        lower = bins[i]
        upper = bins[i + 1]
        (i == 1 && value == lower) && return i
        (value > lower && value ≤ upper) && return i
    end
    # Out-of-range fallback: clamp to first or last valid bin
    return value ≤ bins[1] ? 1 : length(bins) - 1
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
    setup(
        E::Float64,
        particle::String,
        dose::Float64,
        tumor_radius::Float64;
        X_box::Float64      = 600.0,
        X_voxel::Float64    = 300.0,
        R_cell::Float64     = 15.0,
        target_geom::String = "circle",
        calc_type::String   = "full",
        type_AT::String     = "KC",
        track_seg::Bool     = true
    ) -> NamedTuple

High-level wrapper that orchestrates the full irradiation preparation pipeline
(starting from beam/ion setup through cell lattice, population, irradiation
conditions, oxygenation, and quick fluence/time summaries), while making legacy
APIs usable in notebook/async contexts.

This function accepts **four required inputs** (`E`, `particle`, `dose`,
`tumor_radius`) and exposes **keyword arguments** for geometry and model
settings. Internally, it adapts to legacy "bang" functions that set global
variables (e.g. `ion`, `irrad`, `cell_df`) by:

1. Injecting required parameters into `Main` (so functions that read globals
    can find them), and
2. Using `Base.invokelatest` to avoid "world age" errors in IJulia/Pluto or
    async execution.

# Arguments

**Positional**
- `E::Float64`: Particle energy per nucleon (units per your domain convention).
- `particle::String`: Ion species identifier (e.g., `"1H"`, `"4He"`, `"12C"`).
- `dose::Float64`: Prescribed dose (Gy).
- `tumor_radius::Float64`: Target/tumor radius (µm or your chosen unit).

**Keywords (with defaults)**
- `X_box::Float64=600.0`: Size of the simulation box (same units as `R_cell`).
- `X_voxel::Float64=300.0`: Voxel side length used in beam geometry.
- `R_cell::Float64=15.0`: Cell radius used for lattice/population geometry.
- `target_geom::String="circle"`: Target geometry label consumed by the setup functions.
- `calc_type::String="full"`: Beam/geometry calculation mode.
- `type_AT::String="KC"`: AT (radiobiological) model identifier passed to `ATRadius` and `setup_irrad_conditions!`.
- `track_seg::Bool=true`: Whether to enable track segmentation in downstream setup.

# Behavior / Pipeline

1. **Derived geometry**  
    Computes:
    - `N_sideVox = floor(Int, 2*X_box/X_voxel)`
    - `N_CellsSide = 2*floor(Int, X_box/(2*R_cell))`

2. **Global injection for legacy APIs**  
    Injects the following bindings into `Main` (shadowing any existing values):
    `tumor_radius`, `X_voxel`, `X_box`, `R_cell`, `type_AT`, `target_geom`,
    `track_seg`, `N_sideVox`, `N_CellsSide`.

3. **Ion & irradiation setup**  
    Calls `setup_IonIrrad!(dose, E, particle)` (via `Base.invokelatest`) and
    retrieves `ion`, `irrad`, and other globals such as `A`, `Z`, `LET` from `Main`.

4. **Beam geometry**  
    Calls `calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)`
    to get `(R_beam, x_beam, y_beam)`.

5. **AT initialization**  
    Computes `(Rc, Rp, Kp) = ATRadius(ion, irrad, type_AT)` and builds
    `at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)`.

6. **Cell lattice & population**  
    - `setup_cell_lattice!(...)` populates `Main.cell_df`.
    - `setup_cell_population!(...)` updates `Main.cell_df`, using `Main.gsm2` if present.

7. **Irradiation conditions & oxygenation**  
    - `setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)` updates `cell_df`.
    - `set_oxygen!(cell_df; plot_oxygen=false)` updates oxygenation fields.

8. **Summary metrics**  
    Using `irrad.dose`, `irrad.doserate`, and `LET`, computes:
   - `F    = irrad.dose / (1.602e-9 * LET)`
   - `Npar = round(Int, F * π * R_beam^2 * 1e-8)`
    - `zF   = irrad.dose / Npar`
    - `D    = irrad.doserate / zF`
   - `T    = irrad.dose / (zF * D) * 3600`
    and reports a concise textual summary.

# Returns

A `NamedTuple` with:
- `ion`: Ion descriptor (as defined by your domain code).
- `irrad`: Irradiation descriptor (dose, dose rate, LET, etc.).
- `cell_df`: The final cell table/structure after population, conditions, and oxygenation.
- `at_start`: Initial AT state constructed from `(particle, E, A, Z, LET, Rc, Rp, Kp)`.
- `R_beam`, `x_beam`, `y_beam`: Beam radius and centroid coordinates.
- `O2_mean`: Mean oxygen level over `cell_df` where `is_cell == 1`.
- `Npar`, `zF`, `D`, `T`: Particle count estimate, fluence-per-particle, dose-rate-per-particle, and estimated irradiation time (s).

# Side Effects

- **Global state**: Overwrites/injects several bindings in `Main` and relies on
    legacy functions that set global variables (`ion`, `irrad`, `cell_df`, `gsm2`,
    etc.). Avoid calling concurrently in multi-task/threaded workflows unless
    appropriately synchronized.
- **World age handling**: Uses `Base.invokelatest` to ensure the newest method
    definitions are called in notebook/async environments.

# Requirements & Assumptions

- The following functions exist and either return or mutate globals as shown:
    `setup_IonIrrad!`, `calculate_beam_properties`, `ATRadius`, `AT`,
    `setup_cell_lattice!`, `setup_cell_population!`, `setup_irrad_conditions!`,
    `set_oxygen!`.
- The following globals are created by the legacy functions: `ion`, `irrad`,
    (optionally) `A`, `Z`, `LET`, `cell_df`, `gsm2`.
- `cell_df` exposes columns/fields `.O` and `.is_cell`, and supports
    boolean indexing as used in `mean(cell_df.O[cell_df.is_cell .== 1])`.

# Units Notes

- The constant `1.602e-9` and the `1e-8` geometric factor must match your LET and
    length units (e.g., LET in keV/µm, lengths in µm vs cm). Verify consistency
    throughout the codebase to avoid silent unit errors.

# Examples

```julia
# Minimal call with defaults
out = setup(2.0, "1H", 1.5, 300.0)

# Customized geometry and AT model
out = setup(2.0, "12C", 2.0, 250.0;
            X_box=800.0, X_voxel=200.0, R_cell=12.5,
            target_geom="circle", calc_type="full",
            type_AT="KC", track_seg=true)
```
"""
function setup(
    E::Float64,
    particle::String,
    dose::Float64,
    tumor_radius::Float64;
    X_box::Float64      = 600.0,
    X_voxel::Float64    = 300.0,
    R_cell::Float64     = 15.0,
    target_geom::String = "circle",
    calc_type::String   = "full",
    type_AT::String     = "KC",
    track_seg::Bool     = true
)
    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    # Inject all variables that setup functions read directly from Main
    @eval Main begin
        tumor_radius = $tumor_radius
        X_voxel      = $X_voxel
        X_box        = $X_box
        R_cell       = $R_cell
        type_AT      = $type_AT
        target_geom  = $target_geom
        calc_type    = $calc_type
        track_seg    = $track_seg
        N_sideVox    = $N_sideVox
        N_CellsSide  = $N_CellsSide
    end

    Base.invokelatest(setup_IonIrrad!, dose, E, particle)
    ion   = Base.invokelatest(getfield, Main, :ion)
    irrad = Base.invokelatest(getfield, Main, :irrad)
    A     = Base.invokelatest(getfield, Main, :A)
    Z     = Base.invokelatest(getfield, Main, :Z)
    LET   = Base.invokelatest(getfield, Main, :LET)

    R_beam, x_beam, y_beam = Base.invokelatest(
        calculate_beam_properties, calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = Base.invokelatest(ATRadius, ion, irrad, type_AT)
    at_start   = Base.invokelatest(AT, particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    Base.invokelatest(setup_cell_lattice!, target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    cell_df = Base.invokelatest(getfield, Main, :cell_df)

    gsm2 = Base.invokelatest(getfield, Main, :gsm2)
    Base.invokelatest(setup_cell_population!, target_geom, X_box, R_cell,
                        N_sideVox, N_CellsSide, gsm2)
    cell_df = Base.invokelatest(getfield, Main, :cell_df)

    ion   = Base.invokelatest(getfield, Main, :ion)
    irrad = Base.invokelatest(getfield, Main, :irrad)
    Base.invokelatest(setup_irrad_conditions!, ion, irrad, type_AT, cell_df, track_seg)
    cell_df = Base.invokelatest(getfield, Main, :cell_df)

    Base.invokelatest(set_oxygen!, cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    irrad = Base.invokelatest(getfield, Main, :irrad)
    LET   = Base.invokelatest(getfield, Main, :LET)
    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    Npar == 0 && error("Npar computed as zero — beam radius or dose too small (R_beam=$R_beam, dose=$(irrad.dose), LET=$LET)")
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    @eval Main begin
        R_beam = $R_beam
        x_beam = $x_beam
        y_beam = $y_beam
        F      = $F
        Npar   = $Npar
        zF     = $zF
        D      = $D
        T      = $T
    end

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")

    return (
        ion=ion, irrad=irrad, cell_df=cell_df, at_start=at_start,
        R_beam=R_beam, x_beam=x_beam, y_beam=y_beam,
        O2_mean=O2_mean, Npar=Npar, zF=zF, D=D, T=T
    )
end