#! ============================================================================
#! utilities_biology.jl
#!
#! FUNCTIONS
#! ---------
#~ Cell Position Generation
#?   generate_cells_positions_selector(target_geom, ParIrr, X_box, R_cell, N_sideVox, N_CellsSide_default)
#       Dispatcher: picks the right lattice generator based on geometry and irradiation mode.
#?   generate_cells_positions_squaredlattice_3D_new_1layervoxel(X_box, R_cell, N_sideVox)
#       3D square lattice restricted to 1 voxel layer in z (partial irradiation).
#?   generate_cells_positions_squaredlattice_3D_new(X_box, R_cell)
#       Full 3D square lattice, boundary-exclusive.
#?   generate_cells_positions_squaredlattice_3D(X_box, R_cell)
#       Full 3D square lattice, boundary-inclusive (N_CellsSide+1 points per axis).
#
#~ Cell Population & Initialization
#?   populate_cells_wrapper(ParIrr, N, nodes_positions, R_cell, gsm2, cell_df, domain,
#                          tumor_radius, full_cycle, target_geom, type, N_sideVox,
#                          N_CellsSide, X_box, X_voxel)
#       Dispatcher: routes to partial or full irradiation cell population.
#?   create_cells_3D_voxel_1layervoxel_df!(N, nodes_positions, R_cell, SP_, gsm2, cell_df,
#                                         domain, tumor_radius, full_cycle, geometry, type,
#                                         N_sideVox, N_CellsSide, X_box, X_voxel)
#       Mutates cell_df in-place for partial irradiation (1 voxel layer in z).
#?   create_cells_3D_voxel_df!(N, nodes_positions, R_cell, SP_, gsm2, cell_df, domain,
#                              tumor_radius, full_cycle, geometry, type, X_box, X_voxel)
#       Mutates cell_df in-place for full irradiation (entire 3D volume).
#   initialize_G0_phase!(cell_df)
#       Sets contact-inhibited cells (number_nei == 0) to G0. Call before compute_times_domain!
#
#~ Neighbors
#?   compute_neighbors_3d(N, M, L)
#       Returns 26-connected Moore neighborhood adjacency list for an N×M×L lattice.
#
#~ Oxygen
#?   set_oxygen!(cell_df; rim_ox, core_ox, max_dist_ref, plot_oxygen)
#       Assigns radial oxygen gradient (rim→core) to all cells; optionally plots distribution.
#
#~ Constants
#?   PHASE_DURATIONS   — Gamma (shape, scale) parameters per cell cycle phase
#?   PHASE_TRANSITION  — Dict mapping each phase to its successor (G1→S→G2→M→G1)

#! ============================================================================
#! Constants for cell cycle phases
#! ============================================================================

const PHASE_DURATIONS = Dict(
    "G1" => (shape=0.5*11, scale=2.0),
    "S"  => (shape=0.5*7,  scale=2.0),
    "G2" => (shape=0.5*5,  scale=2.0),
    "M"  => (shape=0.5*1,  scale=2.0),
    "G0" => (shape=Inf,    scale=Inf)  # G0 has no intrinsic duration
)

const PHASE_TRANSITION = Dict(
    "G1" => "S",
    "S"  => "G2",
    "G2" => "M",
    "M"  => "G1"
)

"""
generate_cells_positions_selector(target_geom::String,
                                        ParIrr::String,
                                        X_box::Float64,
                                        R_cell::Float64,
                                        N_sideVox::Int64,
                                        N_CellsSide_default::Int64) -> (N::Int, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, N_CellsSide::Int)

Select and execute the appropriate cell‑position generation routine based on the **target geometry** and **irradiation mode**, returning:
- `N`               : total number of generated cells
- `nodes_positions` : vector of `(x, y, z)` coordinates for each cell center (in µm)
- `N_CellsSide`     : number of cells per side (if applicable), otherwise a provided default

# Purpose
This wrapper centralizes the choice of lattice generation:
- **Square geometry**
  - **Partial irradiation** (`ParIrr == "true"`): Generate a 3D square lattice constrained to the irradiated voxel layer, returning `(N, nodes_positions, N_CellsSide)`.
  - **Full irradiation** (`ParIrr != "true"`): Generate a 3D square lattice for the whole target, returning `(N, nodes_positions)` and using `N_CellsSide_default` as placeholder.
- **Circle geometry**: Generate a 3D lattice within a circular target region, returning `(N, nodes_positions)` and `N_CellsSide_default`.

# Arguments
- `target_geom::String`  
    Geometry of the target: `"square"` or `"circle"`.
- `ParIrr::String`  
    Partial irradiation flag as string: `"true"` or `"false"` (string form kept for API compatibility).
- `X_box::Float64`  
    Half‑side of the simulation box (µm). Full box width is `2*X_box`.
- `R_cell::Float64`  
    Cell radius (µm), used to define packing / spacing in the lattice generation.
- `N_sideVox::Int64`  
    Number of voxels per side (relevant for partial irradiation and voxel‑aware layouts).
- `N_CellsSide_default::Int64`  
    Default value returned when the chosen generator does not produce `N_CellsSide`.

# Returns
- `(N, nodes_positions, N_CellsSide)`:
    - `N::Int` — number of cells generated
    - `nodes_positions::Vector{Tuple{Float64,Float64,Float64}}` — coordinates of cell centers
    - `N_CellsSide::Int` — cells per side (meaningful for square/partial mode), otherwise `N_CellsSide_default`

# Behavior
- If `target_geom == "square"` and `ParIrr == "true"`:
    calls `generate_cells_positions_squaredlattice_3D_new_1layervoxel(X_box, R_cell, N_sideVox)`
  which **returns** `(N, nodes_positions, N_CellsSide)`.
- If `target_geom == "square"` and `ParIrr != "true"`:
    calls `generate_cells_positions_squaredlattice_3D_new(X_box, R_cell)`
  which **returns** `(N, nodes_positions)`, and this function forwards `N_CellsSide_default`.
- If `target_geom == "circle"`:
    calls `generate_cells_positions_squaredlattice_3D(X_box, R_cell)`
  which **returns** `(N, nodes_positions)`, and this function forwards `N_CellsSide_default`.

# Notes
- This function prints a concise description of the selected configuration.
- Returns an **empty** position vector with `N = 0` if `target_geom` is unknown.

# Example
```julia
N, nodes_positions, Nside = generate_cells_positions_selector("square", "true", 900.0, 15.0, 12, 0)
```
"""

function generate_cells_positions_selector(target_geom::String, ParIrr::String, X_box::Float64, R_cell::Float64, N_sideVox::Int64, N_CellsSide_default::Int64)
    println("... Generating cells ...")
    if target_geom == "square"
        if ParIrr == "true"
            println("Selected configuration: Square Target, Partial Irradiation")
            # This function returns N_CellsSide as the 3rd argument
            N, nodes_positions, N_CellsSide = generate_cells_positions_squaredlattice_3D_new_1layervoxel(X_box, R_cell, N_sideVox)
            return N, nodes_positions, N_CellsSide
        else
            println("Selected configuration: Square Target, Full Irradiation")
            # This function returns only N and nodes_positions
            N, nodes_positions = generate_cells_positions_squaredlattice_3D_new(X_box, R_cell)
            return N, nodes_positions, N_CellsSide_default
        end
    elseif target_geom == "circle"
        println("Selected configuration: Circle Target")
        # This function returns only N and nodes_positions
        N, nodes_positions = generate_cells_positions_squaredlattice_3D(X_box, R_cell)
        return N, nodes_positions, N_CellsSide_default
    else
        # Fallback/Error case
        println("Warning: Unknown target_geom '", target_geom, "'. Returning empty positions.")
        return 0, Vector{Tuple{Float64,Float64,Float64}}(), N_CellsSide_default
    end
end

"""
generate_cells_positions_squaredlattice_3D_new_1layervoxel(
        X_box::Float64,
        R_cell::Float64,
        N_sideVox::Int64
    ) -> (N::Int, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, N_CellsSide::Int)

Generate a 3D square-lattice arrangement of cell centers, restricted to a **single voxel layer**
in the z-direction. This layout is useful for *partial irradiation* simulations where only a
thin slab or a specific depth range of cells should be populated.

# Purpose
This function produces:
- a uniformly spaced 3D grid of cell centers inside the cubic simulation box,
- but **only for the z-slices belonging to one voxel layer**, determined by `N_sideVox`.

The x–y grid covers the full transverse cross-section, while the z-grid is reduced in depth.

# Geometry and Method
- The full simulation box extends from `-X_box` to `+X_box` in each dimension.
- Cells are placed on a regular square lattice with spacing equal to the cell diameter:

      spacing = 2 * R_cell

- The number of cells along one side is:

        N_CellsSide = floor( (2*X_box) / spacing )

- In the z-direction, only:

        z = 0 : (N_CellsSide / N_sideVox - 1)

  layers are generated, corresponding to **one voxel's thickness**.

# Arguments
- `X_box::Float64`  
  Half-side of the cubic simulation box (µm). Full width is `2 * X_box`.

- `R_cell::Float64`  
    Cell radius (µm), used to define cell spacing.

- `N_sideVox::Int64`  
  Number of voxels along one side of the cube. Determines the **thickness of one voxel layer**.

# Returns
- `N::Int`  
    Total number of generated cells.

- `nodes_positions::Vector{Tuple{Float64,Float64,Float64}}`  
    A vector of `(x, y, z)` coordinates for each cell.

- `N_CellsSide::Int`  
    Number of cells along one side in x and y directions.

# Example
```julia
N, positions, Nside = generate_cells_positions_squaredlattice_3D_new_1layervoxel(900.0, 15.0, 12)
```
"""

function generate_cells_positions_squaredlattice_3D_new_1layervoxel(X_box::Float64, R_cell::Float64, N_sideVox::Int64)
    # Number of cells per side
    N_CellsSide = floor(Int, (2 * X_box) / (2 * R_cell))

    # Constant spacing between centers
    spacing = 2 * R_cell

    # Initialize positions array
    nodes_positions = Vector{Tuple{Float64,Float64,Float64}}()

    # cicle to generate centers
    for z in 0:(Int64(N_CellsSide / N_sideVox)-1)
        for y in 0:(N_CellsSide-1)
            for x in 0:(N_CellsSide-1)
                xc = -X_box + R_cell + x * spacing
                yc = -X_box + R_cell + y * spacing
                zc = -X_box + R_cell + z * spacing
                push!(nodes_positions, (xc, yc, zc))
            end
        end
    end

    N = length(nodes_positions)
    return N, nodes_positions, N_CellsSide
end

"""
generate_cells_positions_squaredlattice_3D_new(
        X_box::Float64,
        R_cell::Float64
    ) -> (N::Int, nodes_positions::Vector{Tuple{Float64,Float64,Float64}})

Generate a full 3D square‑lattice distribution of cell centers inside the
simulation box. This creates a **uniform cubic grid of cells** filling
the entire domain from `-X_box` to `+X_box` in all three directions.

# Purpose
This function is used for **full irradiation** simulations, where the entire
volume of the target is populated with cells. It generates a regular 3D lattice
of coordinates spaced by `2 * R_cell`, the cell diameter.

# Geometry and Method
- The simulation box is cubic with half‑side `X_box` (so full side is `2 * X_box`).
- Cells are placed on a regular grid with spacing:

      spacing = 2 * R_cell

- The number of cells per coordinate axis is:

      N_CellsSide = floor( (2 * X_box) / spacing )

for integer indices `x, y, z`.

The result is a **deterministic, perfectly aligned 3D cell lattice**.

# Arguments
- `X_box::Float64`  
    Half‑size of the simulation cube (µm). Range: `[-X_box, +X_box]`.

- `R_cell::Float64`  
    Cell radius (µm). Determines spacing between cells.

# Returns
- `N::Int`  
    Total number of generated cell centers.

- `nodes_positions::Vector{Tuple{Float64,Float64,Float64}}`  
    A vector of `(x, y, z)` coordinates for every cell center.

# Example
```julia
N, cells = generate_cells_positions_squaredlattice_3D_new(900.0, 15.0)
```
"""

function generate_cells_positions_squaredlattice_3D_new(X_box::Float64, R_cell::Float64)
    # Number of cells per side
    N_CellsSide = floor(Int, (2 * X_box) / (2 * R_cell))

    # Constant spacing between centers
    spacing = 2 * R_cell

    # Initialize positions array
    nodes_positions = Vector{Tuple{Float64,Float64,Float64}}()

    # cicle to generate centers
    for z in 0:(N_CellsSide-1)
        for y in 0:(N_CellsSide-1)
            for x in 0:(N_CellsSide-1)
                xc = -X_box + R_cell + x * spacing
                yc = -X_box + R_cell + y * spacing
                zc = -X_box + R_cell + z * spacing
                push!(nodes_positions, (xc, yc, zc))
            end
        end
    end

    N = length(nodes_positions)
    return N, nodes_positions
end

"""
generate_cells_positions_squaredlattice_3D(
        X_box::Float64,
        R_cell::Float64
    ) -> (N::Int, nodes_positions::Vector{Tuple{Float64,Float64,Float64}})

Generate a **full 3D square lattice** of cell centers that uniformly fills the
simulation domain from `-X_box` to `+X_box` in all three spatial directions.
This generator includes **(N_CellsSide + 1)** points along each axis so that
both boundary planes are included.

# Purpose
This function is used for simulations in which:
- the entire cubic region must be populated with cells  
- the user wants **boundary‑inclusive** grids  
- spacing is uniform and equal to the cell diameter  

This is similar to `generate_cells_positions_squaredlattice_3D_new` but uses a
slightly different grid definition by explicitly including **endpoints**.

# Geometry and Method
Let the simulation box range be:

    x, y, z ∈ [-X_box, X_box]

Cells are placed on a regular lattice with spacing equal to the cell diameter:

    spacing = 2 * R_cell

The number of intervals per side is:

    N_CellsSide = floor( X_box / (2 * R_cell) ) * 2

This yields **N_CellsSide + 1** positions along each axis, ensuring that the
grid includes both extremes.
# Arguments
- `X_box::Float64`  
    Half-width of the cubic simulation domain (µm).

- `R_cell::Float64`  
  Cell radius (µm). The lattice spacing is `2 * R_cell`.

# Returns
- `N::Int`  
    Total number of generated cell coordinates.

- `nodes_positions::Vector{Tuple{Float64,Float64,Float64}}`  
    List of `(x, y, z)` positions for each cell in the grid.

# Example
```julia
N, pos = generate_cells_positions_squaredlattice_3D(900.0, 15.0)
println("Total cells: ", N)
```
"""

function generate_cells_positions_squaredlattice_3D(X_box::Float64, R_cell::Float64)
    # Calculate the number of cells on each side of the lattice
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))
    nodes_positions = Vector{Tuple{Float64,Float64,Float64}}()

    # Loop over the 3D lattice to calculate the positions of the cells
    for z in 1:(N_CellsSide+1)
        for i in 1:(N_CellsSide+1)
            for j in 1:(N_CellsSide+1)
                # Calculate the position of the cell in 3D space
                push!(nodes_positions,
                    (-R_cell * N_CellsSide + (i - 1) * 2 * R_cell, -R_cell * N_CellsSide + (j - 1) * 2 * R_cell, -R_cell * N_CellsSide + (z - 1) * 2 * R_cell))
            end
        end
    end

    # Calculate the total number of cells
    N = (N_CellsSide + 1)^3
    return N, nodes_positions
end

"""
populate_cells_wrapper(
        ParIrr::String,
        N::Int64,
        nodes_positions::Vector{Tuple{Float64,Float64,Float64}},
        R_cell::Float64,
        gsm2::GSM2,
        cell_df::DataFrame,
        domain::Int64,
        tumor_radius::Float64,
        full_cycle::Bool,
        target_geom::String,
        type::String,
        N_sideVox::Int64,
        N_CellsSide::Int64,
        X_box::Float64,
        X_voxel::Float64
    ) -> Nothing

Populate (mutate in-place) the `cell_df` table and related simulation structures with
cell/domain attributes according to the irradiation mode.

# Purpose
This wrapper centralizes the choice between:
- **Partial irradiation (single voxel layer)** → uses `create_cells_3D_voxel_1layervoxel_df!`
- **Full irradiation (standard volume)**      → uses `create_cells_3D_voxel_df!`

It routes to the appropriate lower-level routine based on `ParIrr`, ensuring a single,
consistent entry point for cell population across simulation modes.

# Behavior
- If `ParIrr == "true"`:
    - Prints `"Configuration: Partial Irradiation (1 layer voxel)"`
    - Calls:
    ```julia
    create_cells_3D_voxel_1layervoxel_df!(
        N, nodes_positions, R_cell, SP_, gsm2, cell_df, domain, tumor_radius,
        full_cycle, target_geom, type, N_sideVox, N_CellsSide, X_box, X_voxel
    )
    ```
- Else (any other value):
    - Prints `"Configuration: Full Irradiation (Standard)"`
    - Calls:
    ```julia
    create_cells_3D_voxel_df!(
        N, nodes_positions, R_cell, SP_, gsm2, cell_df, domain, tumor_radius,
        full_cycle, target_geom, type, X_box, X_voxel
    )
    ```

> **Note**: This function **mutates** `cell_df` in place (the `!`-suffixed callees follow Julia
> convention for mutating functions). No value is returned.

# Arguments
- `ParIrr::String`  
    Partial-irradiation flag as string for API compatibility. Use `"true"` for partial irradiation
    (one voxel layer), otherwise full irradiation is assumed.
- `N::Int64`  
    Total number of cells expected/placed.
- `nodes_positions::Vector{Tuple{Float64,Float64,Float64}}`  
    Cell center coordinates `(x,y,z)` in µm.
- `R_cell::Float64`  
    Cell radius in µm.
- `gsm2::GSM2`  
    GSM2 model parameter object (e.g., `GSM2(r, a, b, rd, Rn)`).
- `cell_df::DataFrame`  
  DataFrame containing at least positional columns; will be **augmented in place** with cell/domain fields.
- `domain::Int64`  
    Domain configuration selector for subcellular structures.
- `tumor_radius::Float64`  
    Target/tumor radius in µm (used by geometry-aware population).
- `full_cycle::Bool`  
    If `true`, perform full population cycle (e.g., domains, materials, bookkeeping).
- `target_geom::String`  
    Target geometry label (e.g., `"square"`, `"circle"`).
- `type::String`  
    Track-structure / model type string (e.g., `"KC"`, `"LEM"`).
- `N_sideVox::Int64`  
    Number of voxels per side (relevant in partial-irradiation mode).
- `N_CellsSide::Int64`  
    Cells per side (used by the 1-layer voxel routine).
- `X_box::Float64`  
    Half-side of the simulation box (µm).
- `X_voxel::Float64`  
    Voxel side (µm).

# Side Effects
- Mutates `cell_df` by adding/changing columns that describe per-cell state, domains, and metadata.
- May allocate additional structures referenced indirectly by `cell_df`.

# Returns
- `Nothing` (mutation-based workflow).

# Errors
- Will throw if required downstream functions are unavailable or inputs are inconsistent
    (e.g., negative radii, empty `nodes_positions`, or incompatible lattice/voxel settings).

# Example
```julia
populate_cells_wrapper(
    "false", N, nodes_positions, R_cell, SP, gsm2, cell_df, domain,
    tumor_radius, true, "square", "KC", N_sideVox, N_CellsSide, X_box, X_voxel
)
```
"""
function populate_cells_wrapper(ParIrr::String, N::Int64, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, R_cell::Float64, gsm2::GSM2, cell_df::DataFrame, domain::Int64, tumor_radius::Float64, full_cycle::Bool, target_geom::String, type::String, N_sideVox::Int64, N_CellsSide::Int64, X_box::Float64, X_voxel::Float64)
    println("... Populating cells ...")
    SP_ = 1.
    if ParIrr == "true"
        println("Configuration: Partial Irradiation (1 layer voxel)")
        create_cells_3D_voxel_1layervoxel_df!(N, nodes_positions, R_cell, SP_, gsm2, cell_df, domain, tumor_radius, full_cycle, target_geom, type, N_sideVox, N_CellsSide, X_box, X_voxel)
    else
        println("Configuration: Full Irradiation (Standard)")
        create_cells_3D_voxel_df!(N, nodes_positions, R_cell, SP_, gsm2, cell_df, domain, tumor_radius, full_cycle, target_geom, type, X_box, X_voxel)
    end
end


"""
create_cells_3D_voxel_1layervoxel_df!(
        N::Int64,
        nodes_positions::Vector{Tuple{Float64,Float64,Float64}},
        R_cell::Float64,
        SP_::Float64,
        gsm2::GSM2,
        cell_df::DataFrame,
        domain::Int64,
        tumor_radius::Float64,
        full_cycle::Bool,
        geometry::String,
        type::String,
        N_sideVox::Int64,
        N_CellsSide::Int64,
        X_box::Float64,
        X_voxel::Float64
    ) -> Nothing

Populate `cell_df` with all per‑cell biological, geometric, and voxel‑level attributes
for the **partial‑irradiation** configuration (1‑voxel‑layer mode).  
This function mutates `cell_df` in‑place and does not return a value.

# Purpose
This routine corresponds to the *partial irradiation* scenario, where only a **single voxel
layer** in the z‑direction is populated. It assigns:
- basic cell type attributes (normal/stem)
- O₂ / hypoxia values
- voxel indices `(i_voxel_x, i_voxel_y, i_voxel_z)`
- cell cycle state (if `full_cycle == true`)
- per‑domain damage matrices (`dam_X`, `dam_Y`)
- per‑cell dose containers
- neighbor lists (3D structured grid)

The goal is to fully initialize the simulation state for each cell in a compact,
DataFrame‑based representation.

# How It Works
1. **Neighbor Computation**  
    The function precomputes structured 3D neighbor indices using  
    `compute_neighbors_3d(Ncell, Ncell, Ncell/N_sideVox)`  
    which matches the reduced depth of the 1‑layer voxel configuration.

2. **Multithreaded Cell Initialization (`Threads.@threads`)**  
    For each cell (indexed by `i`):
    - Extract `(x, y, z)` coordinates.
    - Randomly decide whether it is stem or normal (80% normal).
    - Assign cell cycle state:
        - `"M"`, `"S"`, `"G2"` if `full_cycle == true`
        - `"M"` or `"I"` otherwise.
   - Compute its **voxel indices** based on `(x, y, z)` and clip them via `clamp`.
    - Compute tumor inclusion (`sqrt(x² + y² + z²) < tumor_radius`) if geometry = `"circle"`.

3. **Geometry‑Based Cell Inclusion**
    - `"square"` → all cells are included (`is_cell = 1`)
    - `"circle"` → only cells inside tumor radius are included (`is_cell = 1`)
    - otherwise     → excluded (`is_cell = 0`)

4. **Domain Damage Initialization**  
    Computes:
    - `n_dom_sub = floor(Rn / rd)`  (number of subdomains per radial direction)
   - `len_dom_vec = domain * n_dom_sub`  
        and prepares zero‑initialized vectors for domain damage tracking.

5. **Assign Results to `cell_df`**  
    Adds or overwrites columns such as:
    - `.is_cell`, `.is_stem`, `.proliferation`
    - `.i_voxel_x`, `.i_voxel_y`, `.i_voxel_z`
    - `.dam_X`, `.dam_Y`, `.dose`, `.dose_cell`
    - `.O` (oxygen)
    - `.sp` (stopping power)
    - `.cell_cycle`, `.apo_time`, `.death_time`, `.recover_time`, etc.

# Arguments
- `N` — total number of cells
- `nodes_positions` — vector of `(x,y,z)` coordinates
- `R_cell` — cell radius
- `SP_` — stopping power value
- `gsm2` — GSM2 parameter object (`r, a, b, rd, Rn`)
- `cell_df` — DataFrame to be mutated
- `domain` — number of domains for damage tracking
- `tumor_radius` — radius for circular geometry
- `full_cycle` — whether to assign full cell‑cycle distribution
- `geometry` — `"square"` or `"circle"`
- `type` — irradiation/track model label
- `N_sideVox` — number of voxels per dimension
- `N_CellsSide` — number of cells per side in x and y
- `X_box` — half‑side of simulation box
- `X_voxel` — voxel side length

# Returns
- `Nothing` — the DataFrame is mutated in place.

# Notes
- Designed specifically for **partial irradiation simulations** using a
  **1‑voxel‑thickness layer**.
- Multithreading accelerates initialization of large cell populations.
- All allocations are amortized and performed outside of the main DataFrame.

# Example
```julia
create_cells_3D_voxel_1layervoxel_df!(
    N, nodes_positions, R_cell, SP_, gsm2,
    cell_df, domain, tumor_radius, true,
    "square", "KC", N_sideVox, N_CellsSide,
    X_box, X_voxel
)
```
"""
function create_cells_3D_voxel_1layervoxel_df!(N::Int64, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, R_cell::Float64, SP_::Float64, gsm2::GSM2, cell_df::DataFrame, domain::Int64, tumor_radius::Float64, full_cycle::Bool, geometry::String, type::String, N_sideVox::Int64, N_CellsSide::Int64, X_box::Float64, X_voxel::Float64)
    # Compute neighbors for each cell
    Nd = 1
    Ncell = N_CellsSide
    neighbors = compute_neighbors_3d(Ncell, Ncell, Int64(Ncell / N_sideVox))

    is_cell_vec = Vector{Int64}(undef, N)
    is_stem_vec = Vector{Int64}(undef, N)
    proliferation_vec = Vector{Int64}(undef, N)
    cell_cycle_vec = Vector{String}(undef, N)
    O_vec = Vector{Float64}(undef, N)
    i_voxel_x_vec = Vector{Int64}(undef, N)
    i_voxel_y_vec = Vector{Int64}(undef, N)
    i_voxel_z_vec = Vector{Int64}(undef, N)

    Threads.@threads for i in 1:N
        x, y, z = nodes_positions[i]

        cell_line = rand() < 0.8 ? 1 : 0
        is_stem_vec[i] = cell_line

        if full_cycle
            ru = rand() * 24
            if ru <= 1
                cell_cycle = "M"
            elseif ru <= 20
                cell_cycle = "S"
            elseif ru <= 24
                cell_cycle = "G2"
            else
                println("Error")
                cell_cycle = "I"
            end
        else
            cell_cycle = (rand() < 0.25) ? "M" : "I"
        end
        cell_cycle_vec[i] = cell_cycle

        proliferation_vec[i] = 15

        # Voxel calculation
        i_vx = floor(Int64, (x + X_box) / X_voxel) + 1
        i_vy = floor(Int64, (y + X_box) / X_voxel) + 1
        i_vz = floor(Int64, (z + X_box) / X_voxel) + 1

        i_vx = clamp(i_vx, 1, N_sideVox)
        i_vy = clamp(i_vy, 1, N_sideVox)
        i_vz = clamp(i_vz, 1, N_sideVox)

        i_voxel_x_vec[i] = i_vx
        i_voxel_y_vec[i] = i_vy
        i_voxel_z_vec[i] = i_vz

        is_inside_tumor = sqrt(x^2 + y^2 + z^2) < tumor_radius
        O2_TNT = 5.0
        O_vec[i] = O2_TNT

        if geometry == "square"
            is_cell_vec[i] = 1
        elseif geometry == "circle"
            is_cell_vec[i] = is_inside_tumor ? 1 : 0
        else
            is_cell_vec[i] = 0
        end
    end

    cell_df.nei = neighbors
    cell_df.is_cell = is_cell_vec
    cell_df.is_stem = is_stem_vec
    cell_df.proliferation = proliferation_vec
    cell_df.cell_cycle = cell_cycle_vec
    cell_df.O = O_vec
    cell_df.i_voxel_x = i_voxel_x_vec
    cell_df.i_voxel_y = i_voxel_y_vec
    cell_df.i_voxel_z = i_voxel_z_vec

    cell_df.dam_X = [Matrix{Float64}(undef, 0, Nd) for _ in 1:N]
    cell_df.dam_Y = [Matrix{Float64}(undef, 0, Nd) for _ in 1:N]

    n_dom_sub = floor(Int64, gsm2.Rn / gsm2.rd)
    len_dom_vec = domain * n_dom_sub
    cell_df.dam_X_dom = [zeros(Int64, len_dom_vec) for _ in 1:N]
    cell_df.dam_Y_dom = [zeros(Int64, len_dom_vec) for _ in 1:N]

    cell_df.sp = fill(SP_, N)
    cell_df.dose = [zeros(Float64, domain) for _ in 1:N]
    cell_df.dose_cell = zeros(Float64, N)

    cell_df.can_divide = zeros(Int64, N)
    cell_df.number_nei = zeros(Int64, N)

    cell_df.apo_time = zeros(Float64, N)
    cell_df.death_time = zeros(Float64, N)
    cell_df.recover_time = zeros(Float64, N)
    cell_df.cycle_time = zeros(Float64, N)
    cell_df.is_death_rad = zeros(Int64, N)
end

"""
create_cells_3D_voxel_df!(
        N::Int64,
        nodes_positions::Vector{Tuple{Float64,Float64,Float64}},
        R_cell::Float64,
        SP_::Float64,
        gsm2::GSM2,
        cell_df::DataFrame,
        domain::Int64,
        tumor_radius::Float64,
        full_cycle::Bool,
        geometry::String,
        type::String,
        X_box::Float64,
        X_voxel::Float64
    ) -> Nothing

Populate `cell_df` with all geometric, biological, voxel-level, and domain-level
attributes for **full‑irradiation** simulations.  
This is the standard mode that populates the *entire 3D volume* of the target.

This function mutates `cell_df` in-place and returns `nothing`.

# Purpose
This function initializes every cell in the full 3D lattice (no voxel‑layer restriction),
assigning:

- stem/normal identity  
- cell‑cycle state (`M`, `S`, `G2`, or `I`)  
- proliferation parameters  
- voxel indices `(i_voxel_x, i_voxel_y, i_voxel_z)`  
- oxygenation level  
- domain-damage structures (`dam_X`, `dam_Y`, `dam_X_dom`, `dam_Y_dom`)  
- dose containers (`dose`, `dose_cell`)  
- structural/metadata fields (neighbors, can_divide, number_nei, timers, death flags)  

It mirrors the behavior of the partial‑irradiation version, but applies to the **entire cube**.

# How It Works

## 1. Neighbor Computation
## 2. Multithreaded Per‑Cell Initialization
Each cell is initialized in a `Threads.@threads` loop:

- **Random stem assignment**: 80% normal, 20% stem  
- **Cell cycle**:
    - If `full_cycle == true`, uses a realistic distribution over 24h  
    - Otherwise assigns `"M"` or `"I"`  
- **Voxel index computation** across the entire simulation domain  
- **Tumor geometry filtering** (square vs. circle)  
- **Oxygenation** set to constant `5.0` (modifiable later)  

## 3. Domain Initialization
For each cell:
- Empty 2D matrices `dam_X` and `dam_Y` are allocated
- Vectors `dam_X_dom` and `dam_Y_dom` are initialized accordingly

## 4. DataFrame Population
The function writes (or overwrites) the following columns:

- `.nei`  
- `.is_cell`, `.is_stem`, `.proliferation`, `.cell_cycle`  
- `.O`  
- `.i_voxel_x`, `.i_voxel_y`, `.i_voxel_z`  
- `.dam_X`, `.dam_Y`, `.dam_X_dom`, `.dam_Y_dom`  
- `.dose`, `.dose_cell`  
- `.can_divide`, `.number_nei`  
- `.apo_time`, `.death_time`, `.recover_time`, `.cycle_time`, `.is_death_rad`  

# Arguments
- `N`: total number of cells
- `nodes_positions`: vector of `(x, y, z)` coordinates
- `R_cell`: cell radius (µm)
- `SP_`: stopping‑power value assigned to the cell
- `gsm2`: GSM2 parameters (`r, a, b, rd, Rn`)
- `cell_df`: DataFrame to mutate in-place
- `domain`: number of domains per cell
- `tumor_radius`: radius of circular target region
- `full_cycle`: whether to assign realistic cell‑cycle states
- `geometry`: `"square"` or `"circle"`
- `type`: irradiation/track model type
- `X_box`: half-side of simulation cube
- `X_voxel`: voxel side length

# Returns
- `Nothing` (mutation-only function; updates `cell_df` in-place)

# Notes
- This is the **full‑volume** version, used when **all** cells in the 3D grid must be simulated.  
- Shares structural logic with `create_cells_3D_voxel_1layervoxel_df!`, but applies uniformly in z.  
- Threaded for high performance.  
- Uses deterministic grid placement but random biological assignment.

# Example
```julia
create_cells_3D_voxel_df!(
    N, nodes_positions, R_cell, SP_, gsm2,
    cell_df, domain, tumor_radius, true,
    "circle", "KC", X_box, X_voxel
)
```
"""
function create_cells_3D_voxel_df!(N::Int64, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, R_cell::Float64, SP_::Float64, gsm2::GSM2, cell_df::DataFrame, domain::Int64, tumor_radius::Float64, full_cycle::Bool, geometry::String, type::String, X_box::Float64, X_voxel::Float64)
    # Compute neighbors for each cell
    Nd = 1
    Ncell = round(Int64, N^(1 / 3))
    neighbors = compute_neighbors_3d(Ncell, Ncell, Ncell)

    is_cell_vec = Vector{Int64}(undef, N)
    is_stem_vec = Vector{Int64}(undef, N)
    proliferation_vec = Vector{Int64}(undef, N)
    cell_cycle_vec = Vector{String}(undef, N)
    O_vec = Vector{Float64}(undef, N)
    i_voxel_x_vec = Vector{Int64}(undef, N)
    i_voxel_y_vec = Vector{Int64}(undef, N)
    i_voxel_z_vec = Vector{Int64}(undef, N)

    N_sideVox = convert(Int64, floor(2 * X_box / (X_voxel)))

    Threads.@threads for i in 1:N
        x, y, z = nodes_positions[i]

        cell_line = rand() < 0.8 ? 1 : 0
        is_stem_vec[i] = cell_line

        if full_cycle
            ru = rand() * 24
            if ru <= 1
                cell_cycle = "M"
            elseif ru <= 6
                cell_cycle = "G2"
            elseif ru <= 13
                cell_cycle = "S"
            elseif ru <= 24
                cell_cycle = "G1"
            else
                println("Error")
                cell_cycle = "I" # Fallback
            end
        else
            cell_cycle = (rand() < 0.25) ? "M" : "I"
        end
        cell_cycle_vec[i] = cell_cycle

        proliferation_vec[i] = 15

        i_vx = 0
        i_vy = 0
        i_vz = 0

        i_vx = floor(Int64, (x + X_box) / X_voxel) + 1
        i_vy = floor(Int64, (y + X_box) / X_voxel) + 1
        i_vz = floor(Int64, (z + X_box) / X_voxel) + 1

        # Clamp to be safe
        i_vx = clamp(i_vx, 1, N_sideVox)
        i_vy = clamp(i_vy, 1, N_sideVox)
        i_vz = clamp(i_vz, 1, N_sideVox)

        i_voxel_x_vec[i] = i_vx
        i_voxel_y_vec[i] = i_vy
        i_voxel_z_vec[i] = i_vz

        # Check if the cell is inside the tumor
        is_inside_tumor = sqrt(x^2 + y^2 + z^2) < tumor_radius
        O2_TNT = 5.0 # Fixed value from original code
        O_vec[i] = O2_TNT

        if geometry == "square"
            is_cell_vec[i] = 1
        elseif geometry == "circle"
            is_cell_vec[i] = is_inside_tumor ? 1 : 0
        else
            is_cell_vec[i] = 0
        end
    end

    cell_df.nei = neighbors
    cell_df.is_cell = is_cell_vec
    cell_df.is_stem = is_stem_vec
    cell_df.proliferation = proliferation_vec
    cell_df.cell_cycle = cell_cycle_vec
    cell_df.O = O_vec 
    cell_df.i_voxel_x = i_voxel_x_vec
    cell_df.i_voxel_y = i_voxel_y_vec
    cell_df.i_voxel_z = i_voxel_z_vec

    cell_df.dam_X = [Matrix{Float64}(undef, 0, Nd) for _ in 1:N]
    cell_df.dam_Y = [Matrix{Float64}(undef, 0, Nd) for _ in 1:N]

    # dam_X_dom::Vector{Int64} initialized to fill(0, domain * floor(...))
    n_dom_sub = floor(Int64, gsm2.Rn / gsm2.rd)
    len_dom_vec = domain * n_dom_sub
    cell_df.dam_X_dom = [zeros(Int64, len_dom_vec) for _ in 1:N]
    cell_df.dam_Y_dom = [zeros(Int64, len_dom_vec) for _ in 1:N]

    cell_df.sp = fill(SP_, N)

    cell_df.dose = [zeros(Float64, domain) for _ in 1:N]
    cell_df.dose_cell = zeros(Float64, N)

    cell_df.can_divide = zeros(Int64, N)
    cell_df.number_nei = zeros(Int64, N)

    cell_df.apo_time = zeros(Float64, N)
    cell_df.death_time = zeros(Float64, N)
    cell_df.recover_time = zeros(Float64, N)
    cell_df.cycle_time = zeros(Float64, N)
    cell_df.is_death_rad = zeros(Int64, N)
    
    # ============================================================================
    # NEW: Count empty neighbors and initialize G0 for contact-inhibited cells
    # ============================================================================
    for i in 1:N
        if cell_df.is_cell[i] == 1
            # Count empty neighbors
            empty_count = 0
            for nei_idx in cell_df.nei[i]
                if 1 <= nei_idx <= N && cell_df.is_cell[nei_idx] == 0
                    empty_count += 1
                end
            end
            cell_df.number_nei[i] = empty_count
            
            # If no empty neighbors → contact inhibited → G0
            if empty_count == 0
                cell_df.cell_cycle[i] = "G0"
                cell_df.cycle_time[i] = Inf
                cell_df.can_divide[i] = 0
            else
                # Has space to divide
                cell_df.can_divide[i] = 1
                # Initialize cycle_time based on assigned phase
                cell_df.cycle_time[i] = generate_cycle_time(cell_df.cell_cycle[i])
            end
        end
    end
    
    g0_count = sum((cell_df.is_cell .== 1) .& (cell_df.cell_cycle .== "G0"))
    println("Initialized $g0_count cells to G0 (contact-inhibited)")
end

"""
compute_neighbors_3d(N::Int, M::Int, L::Int) 
        -> Vector{Vector{Int}}

Compute the **3D Moore-neighborhood** (26‑connected neighbors) for each vertex in a
regular `N × M × L` lattice.

# Purpose
This function builds the *full adjacency list* of a 3D grid where each cell (voxel)
may have up to **26 neighbors**:
- 6 face‑adjacent
- 12 edge‑adjacent
- 8 corner‑adjacent

The result is a vector where index `idx` contains a list of all valid neighbors of
the cell corresponding to coordinates `(i, j, k)`.

This structure is useful for:
- interaction-based cell simulations  
- diffusive processes  
- mechanical neighborhood queries  
- agent‑based models  
- spatial statistical kernels  

# Lattice Indexing
The 3D grid is mapped into a 1D vector using the standard row-major mapping:
idx = (k − 1) * (N*M) + (j − 1) * N + i
with:
- `i = 1:N`
- `j = 1:M`
- `k = 1:L`

# Neighborhood Definition
For each point `(i, j, k)`, the function checks all combinations:
di, dj, dk ∈ {-1, 0, 1}
excluding `(0, 0, 0)` (the point itself).  
A neighbor is included only if its coordinates remain inside bounds:
1 ≤ ni ≤ N
1 ≤ nj ≤ M
1 ≤ nk ≤ L
# Output
- A `Vector{Vector{Int}}` of length `N*M*L`
- `neighbors[idx]` contains all valid neighbors of the vertex corresponding to that index

The number of neighbors varies depending on whether a vertex lies:
- inside the grid (26 neighbors),
- on a face (17 neighbors),
- on an edge (11 neighbors),
- or on a corner (7 neighbors).

# Example
```julia
nei = compute_neighbors_3d(10, 10, 10)
println("Neighbors of cell 1: ", nei[1])
```
"""
function compute_neighbors_3d(N, M, L)
    num_vertices = N * M * L  # Total number of vertices
    neighbors = Vector{Vector{Int}}(undef, num_vertices)  # Preallocate vector of vectors

    # Loop through each vertex in the lattice
    for k in 1:L
        for j in 1:M
            for i in 1:N
                idx = (k - 1) * N * M + (j - 1) * N + i  # 1D index of (i, j, k)
                neighbor_list = Int[]  # Initialize an empty vector for neighbors

                # Check all 26 possible neighbor positions (-1, 0, 1 shifts in x, y, z)
                for dk in -1:1
                    for dj in -1:1
                        for di in -1:1
                            # Skip the case where all di, dj, dk are zero (the vertex itself)
                            if di == 0 && dj == 0 && dk == 0
                                continue
                            end

                            # Calculate the neighboring (i, j, k) coordinates
                            ni = i + di
                            nj = j + dj
                            nk = k + dk

                            # Ensure the neighbor is within the bounds of the lattice
                            if 1 <= ni <= N && 1 <= nj <= M && 1 <= nk <= L
                                neighbor_idx = (nk - 1) * N * M + (nj - 1) * N + ni
                                push!(neighbor_list, neighbor_idx)
                            end
                        end
                    end
                end

                neighbors[idx] = neighbor_list  # Assign neighbors to the current vertex
            end
        end
    end
    return neighbors
end

"""
set_oxygen!(cell_df::DataFrame;
                rim_ox::Float64 = 7.0,
                core_ox::Float64 = 0.5,
                max_dist_ref::Float64 = 400.0,
                plot_oxygen::Bool = false) -> Nothing

Assign an oxygen concentration to each cell in the spheroid based on the
distance from the spheroid rim. This function **modifies `cell_df` in place** by
adding/overwriting:

- `distance` — radial distance from spheroid center
- `O`   — assigned oxygen concentration

Optionally (`plot_oxygen=true`), it generates a **two‑panel visualization** consisting of:
1. A 3D half‑sphere cut colored by oxygen values  
2. A density plot of the oxygen distribution  

# Biological Model (Oxygen Law)
This function implements a **radial diffusion‑limited oxygen decay** typical of
multicellular tumor spheroids:

1. Let `max_dist` be the spheroid radius (maximum cell distance).
2. Compute distance from spheroid rim:
dist_from_rim = max_dist - distance
3. Oxygen decreases moving inward, following:

- **At the rim** (`dist_from_rim ≤ 0`):  
    ```
    oxygen = rim_ox
    ```

- **Within diffusion-limited zone** (`0 < dist_from_rim < max_dist_ref`):  
    Linear decay from `rim_ox → core_ox`:
    ```
    oxygen = rim_ox - (rim_ox - core_ox) * (dist_from_rim / max_dist_ref)
    ```

- **Beyond diffusion depth** (`dist_from_rim ≥ max_dist_ref`):  
    ```
    oxygen = core_ox
    ```

This approximates the classical experimental observation that spheroids have:
- Highly oxygenated outer proliferative rim  
- Gradually hypoxic intermediate shell  
- A severely hypoxic or anoxic core  

# Arguments
- `cell_df::DataFrame`  
    Must contain columns `x`, `y`, `z`, `is_cell`.

- `rim_ox`  
    Oxygen at the spheroid rim.

- `core_ox`  
    Minimum oxygen at the hypoxic core.

- `max_dist_ref`  
    Effective oxygen diffusion range (µm).  
    Beyond this distance from the rim, oxygen saturates at `core_ox`.

- `plot_oxygen`  
    If `true`, generates a 3D hemispherical scatter plot colored by oxygen and a
    histogram/density plot of the distribution.

# Returns
- `Nothing`.  
  The DataFrame is **mutated in-place**.

# Example
```julia
set_oxygen!(cell_df; rim_ox=7.0, core_ox=0.1, max_dist_ref=450.0, plot_oxygen=true)
```
"""
function set_oxygen!(
    cell_df::DataFrame;
    rim_ox=7.0,
    core_ox=0.1,
    max_dist_ref=400.0,
    plot_oxygen::Bool=false
)
    # ------------------------------------------------------------
    # 1) Compute distance from spheroid center (0,0,0)
    # ------------------------------------------------------------
    cell_df.distance = sqrt.((cell_df.x).^2 .+ (cell_df.y).^2 .+ (cell_df.z).^2)

    # Maximum spheroid radius using only real cells
    max_dist = maximum(cell_df.distance[cell_df.is_cell .== 1])

    # Allocate oxygen column
    cell_df.O = similar(cell_df.distance)

    # ------------------------------------------------------------
    # 2) Oxygen assignment (linear scaling from rim to core)
    # ------------------------------------------------------------
    for i in 1:nrow(cell_df)
        if cell_df.is_cell[i] == 0
            cell_df.O[i] = 0.0
            continue
        end

        # Distance from outer rim
        dist_from_rim = max_dist - cell_df.distance[i]

        if dist_from_rim <= 0
            # At the outermost rim
            cell_df.O[i] = rim_ox
        elseif dist_from_rim < max_dist_ref
            # Linear interpolation from rim_ox → core_ox over 0 → max_dist_ref
            frac = dist_from_rim / max_dist_ref
            cell_df.O[i] = rim_ox - (rim_ox - core_ox) * frac
        else
            # Deeper than 450 µm from rim → core oxygen
            cell_df.O[i] = core_ox
        end
    end

    # ------------------------------------------------------------
    # 3) Optional 2-panel visualization
    # ------------------------------------------------------------
    if plot_oxygen
        # Only real cells
        df_cells_den = cell_df[cell_df.is_cell .== 1, :]
        df_cells_3d  = df_cells_den[df_cells_den.x .>= 0, :]  # half-sphere view

        # 3D scatter
        p1 = scatter(
            df_cells_3d.x, df_cells_3d.y, df_cells_3d.z;
            markersize = 4,
            markerstrokewidth = 0.1,
            marker_z = df_cells_3d.O,
            colorbar = true,
            xlabel = "x (µm)",
            ylabel = "y (µm)",
            zlabel = "z (µm)",
            title = "3D Oxygen Distribution",
            legend = false,
            aspect_ratio = :equal,
            seriescolor = :viridis,
            size = (900, 700),
            camera = (320, 30)  # adjust view if needed
        )

        # Oxygen density histogram
        p2 = density(
            df_cells_den.O;
            xlabel = "Oxygen concentration",
            ylabel = "Density",
            title = "Oxygen Density Distribution",
            legend = false,
            lw = 2,
            c = :blue
        )

        # Combine plots
        display(plot(p1, p2; layout = (1, 2), size = (1400, 600)))
    end

    return nothing
end

"""
Initialize cells to G0 if they have no space to divide (number_nei == 0)
Call this after setting up initial population, before compute_times_domain!
"""
function initialize_G0_phase!(cell_df::DataFrame)
    blocked_count = 0
    
    for i in 1:nrow(cell_df)
        if cell_df.is_cell[i] == 1 && cell_df.number_nei[i] == 0
            # Cell is alive but has no empty neighbors → G0
            cell_df.cell_cycle[i] = "G0"
            cell_df.cycle_time[i] = Inf
            cell_df.can_divide[i] = 0
            blocked_count += 1
        end
    end
    
    println("Initialized $blocked_count cells to G0 (contact-inhibited)")
    
    return nothing
end