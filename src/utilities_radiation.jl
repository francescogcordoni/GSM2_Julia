"""
# ATRadius — Amorphous Track Structure Radii and Dose Amplitude

Compute the amorphous track structure parameters for a given ion using either the
Local Effect Model (LEM) or the Kiefer–Chatterjee (KC) parameterization.

## Purpose
This function implements the **amorphous track structure approximation**, widely
used in radiobiological modelling to describe how energy is distributed radially
around an ion track. It returns three characteristic parameters:

- `Rc` — Core radius of the ion track  
- `Rp` — Penumbra radius  
- `Kp` — Radial dose normalization amplitude  

These parameters are used in track-structure-based models of biological damage.

## Reproducibility
This function is **fully deterministic**. For reproducible workflows:
- provide identical ion properties (`ion.E`, `ion.A`, `ion.Z`, `ion.LET`)
- ensure consistent units across the entire pipeline
- no random components are used, ensuring stable outputs across runs

## Inputs
- `ion::Ion`
    - `ion.E`   — kinetic energy per nucleon (MeV/u)
    - `ion.A`   — mass number
    - `ion.Z`   — atomic number
    - `ion.LET` — unrestricted LET (typically keV/µm)

- `irrad::Irrad` — irradiation conditions (kept for extensibility)

- `type::String`
    - `"LEM"` — Local Effect Model parameterization  
    - `"KC"`  — Kiefer–Chatterjee style parameterization  

## Outputs
Returns `(Rc, Rp, Kp)`:
- `Rc` — core track radius  
- `Rp` — penumbra radius  
- `Kp` — radial dose normalization coefficient  

If the method type is unknown, returns `(-1, -1)`.

## Example
```julia
Rc, Rp, Kp = ATRadius(ion, irrad, "LEM")
```
"""
function ATRadius(ion::Ion, irrad::Irrad, type::String)

    if type == "LEM"
        Rc = 0.01
        Rp = 0.05 * ((ion.E / ion.A)^(1.7))
        LETk = ion.LET * 0.1602

        Kp = (1 / (pi)) * (LETk / (1 * (1 + 2 * log(Rp / Rc))))

    elseif type == "KC"
        AMU2MEV = 931.494027

        β = sqrt(1 - 1 / ((ion.E / AMU2MEV + 1)^2))
        Rc = 0.01116 * β
        Rp = 0.0616 * ((ion.E / ion.A)^(1.7))

        particleEnergy = ion.E
        A = ion.A
        Z = ion.Z
        AMU2MEV = 931.494027

        β = sqrt(1 - 1 / ((particleEnergy / AMU2MEV + 1)^2))
        z_eff = Z * (1 - exp(-125 * β / Z^(2.0 / 3.0)))

        Kp = 1.25 * 0.0001 * (z_eff / β)^2

    else
        println("Error - unknown method")
        return -1, -1
    end

    return Rc, Rp, Kp
end

"""
# Atomic Number Utilities and Stopping Power Loader

This module provides:
1. A fast dictionary mapping element symbols to atomic numbers (`Z_MAP`)
2. A parser function to extract Z from ion notation (e.g., `"12C"`)
3. A loader for stopping power tables from text files

## Reproducibility
These functions are fully deterministic:
- The dictionary is constant and hard‑coded.
- `getZ` performs a pure lookup without side effects.
- `load_stopping_power` always loads the same files and returns identical data 
    given the same directory structure and files.

## Included Functions

### `getZ(ion::AbstractString) -> Int`
Extracts the element symbol from an ion string (e.g., `"12C"`) and returns its
atomic number using a fast dictionary lookup.

### `load_stopping_power() -> Dict{String, Matrix{Float64}}`
Loads stopping‑power data from TXT tables into a dictionary mapping ions to
their corresponding data arrays.

Both functions are designed for speed and clarity in radiobiology simulations.

"""
const Z_MAP = Dict(
    "H"  => 1,  "He" => 2,  "Li" => 3,  "Be" => 4,  "B"  => 5,  "C"  => 6,  "N"  => 7,  "O"  => 8,
    "F"  => 9,  "Ne" => 10, "Na" => 11, "Mg" => 12, "Al" => 13, "Si" => 14, "P"  => 15,
    "S"  => 16, "Cl" => 17, "Ar" => 18, "K"  => 19, "Ca" => 20, "Sc" => 21, "Ti" => 22,
    "V"  => 23, "Cr" => 24, "Mn" => 25, "Fe" => 26, "Co" => 27, "Ni" => 28, "Cu" => 29,
    "Zn" => 30, "Ga" => 31, "Ge" => 32, "As" => 33, "Se" => 34, "Br" => 35, "Kr" => 36,
    "Rb" => 37, "U"  => 92
)

# -------------------------------------------------------------------
# EXTRACT ATOMIC NUMBER
# -------------------------------------------------------------------

"""
getZ(ion::AbstractString) -> Int

Extract the atomic number (Z) from an ion string such as `"12C"` or `"4He"`.

# Method
- Extracts the alphabetical part of the string (e.g., `"C"` from `"12C"`).
- Looks up the symbol in `Z_MAP`.

# Returns
- The atomic number `Z` if recognized.
- `-1` if the symbol is not found.
"""
function getZ(ion::AbstractString)
    particle = filter(isletter, ion)      # Keep only letters, e.g. "C" from "12C"
    return get(Z_MAP, particle, -1)       # Return Z or -1 if missing
end

# -------------------------------------------------------------------
# LOAD STOPPING POWER TABLES
# -------------------------------------------------------------------

"""
load_stopping_power() -> Dict{String, Matrix{Float64}}

Load stopping power tables from disk into a dictionary.

# Directory structure
The function expects files at: stoppingpower/H_water.txt
stoppingpower/He_water.txt...

# Returns
A dictionary mapping ion labels (e.g., `"12C"`) to their numerical stopping
power tables as matrices.

This function is deterministic and suitable for reproducible pipelines.
"""
function load_stopping_power()
    sp = Dict{String,Matrix{Float64}}()

    # Ion-to-element mapping for lookup
    ion_list = [
        ("1H",  "H"),
        ("4He", "He"),
        ("6Li", "Li"),
        ("8Be", "Be"),
        ("10B", "B"),
        ("12C", "C"),
        ("14N", "N"),
        ("16O", "O"),
    ]

    for (ion, element) in ion_list
        sp[ion] = readdlm("data/stoppingpower/$(element)_water.txt", Float64)
    end

    return sp
end

"""
linear_interpolation(ion::String, input_value::Float64, sp::Dict) -> Float64

Perform a linear interpolation of stopping‑power data for a given ion.

# Purpose
Given a stopping‑power table `sp[ion]` (two‑column matrix), this function
computes an interpolated LET or stopping‑power value at the specified
`input_value` (e.g., energy).

It is a simple and deterministic interpolation method used to obtain
physics parameters between tabulated points.

# Inputs
- `ion::String`  
    Key in the dictionary `sp`, e.g. `"1H"`, `"12C"`.

- `input_value::Float64`  
    The x‑value (energy in MeV/u) at which the stopping power
    should be interpolated.

- `sp::Dict{String, Matrix}`  
    Dictionary mapping ion strings to 2‑column matrices:  
    - Column 1: energy values  
    - Column 2: stopping power values  

# Output
- Returns the interpolated stopping power as `Float64`.

# Method
1. Extract the x and y columns from the stopping‑power table.
2. Check that the requested value lies within the tabulated range.
3. Find the two closest points surrounding the input value.
4. Use the standard linear interpolation formula:  
   y = y₁ + (y₂ − y₁) * (x − x₁) / (x₂ − x₁)

# Example
```julia
LET = linear_interpolation("12C", 150.0, stopping_power_dict)
```
"""

function linear_interpolation(ion::String, input_value::Float64, sp::Dict)

    # Read the data from the file (assuming space/tab delimited columns)
    x = sp[ion][:, 1]  # First column (x values)
    y = sp[ion][:, 2] / 10  # Second column (y values)

    # Ensure the input value is within the range of the first column
    if input_value < minimum(x) || input_value > maximum(x)
        error("Input value $input_value is out of range.")
    end

    # Find the indices of the closest values
    idx_lower = findlast(x .<= input_value)  # Index of the largest value ≤ input_value
    idx_upper = findfirst(x .>= input_value) # Index of the smallest value ≥ input_value

    # Get the values for interpolation
    x1, y1 = x[idx_lower], y[idx_lower]
    x2, y2 = x[idx_upper], y[idx_upper]

    # Linear interpolation formula
    if x1 == x2
        interpolated_value = y1
    else
        interpolated_value = y1 + (y2 - y1) * (input_value - x1) / (x2 - x1)
    end

    return interpolated_value
end

"""
calculate_beam_properties(calc_type::String,
                                target_geom::String,
                                X_box::Float64,
                                X_voxel::Float64,
                                tumor_radius::Float64) 
        -> (R_beam, x_beam, y_beam)
Compute the beam radius and beam-center coordinates for the simulation, based on
the selected calculation mode and the target geometry.
# Purpose
This function determines:
- the **effective beam radius** used in the dose deposition simulation
- the **beam center coordinates** in the simulation domain
It supports two simulation strategies:
- `"fast"` : coarse, voxel-based approximation (computationally cheaper)
- `"full"` : geometry-based full-resolution model (more accurate)
# Inputs
- `calc_type::String`  
    `"fast"` → approximate beam based on voxel size  
    `"full"` → compute beam radius based on actual target geometry  
- `target_geom::String`  
    `"circle"` → circular target region  
    `"square"` → square target region  
- `X_box::Float64`  
    Half‑side length of the simulation box (µm).  
    Full box width is `2 * X_box`.
- `X_voxel::Float64`  
    Side length of each voxel (µm).
- `tumor_radius::Float64`  
    Radius of the circular target region (µm), used if `target_geom == "circle"`.
# Outputs
- `R_beam::Float64`  
    Effective beam radius in micrometers.
- `x_beam::Float64`, `y_beam::Float64`  
    Coordinates of the beam center in the simulation plane.
# Notes
- In `"fast"` mode, the beam radius is defined as the radius of the smallest
  circle that fully contains a voxel (`(voxel_side / 2) * sqrt(2)`).
- In `"full"` mode:
    - For a circular target, the beam radius matches the target diagonal.
    - For a square target, it matches the diagonal of the simulation box.
# Example
```julia
R_beam, x_beam, y_beam = calculate_beam_properties("full", "circle", 800, 200, 300)
```
"""
function calculate_beam_properties(calc_type::String,
                                    target_geom::String,
                                    X_box::Float64,
                                    X_voxel::Float64,
                                    tumor_radius::Float64)
    if calc_type == "fast"
        # Beam approximated by the diagonal of a voxel
        R_beam = (X_voxel / 2) * sqrt(2)

        # Beam centered near lower-left corner of the box
        x_beam = -(X_box - X_voxel / 2)
        y_beam = -(X_box - X_voxel / 2)

    elseif calc_type == "full"

        if target_geom == "circle"
            # Beam radius covers the circular target's diagonal
            R_beam = tumor_radius * sqrt(2)

        elseif target_geom == "square"
            # Beam radius covers the entire box diagonal
            R_beam = X_box * sqrt(2)

        else
            error("Unknown target geometry: $target_geom. Use \"circle\" or \"square\".")
        end

        # Beam centered in the simulation domain
        x_beam = 0.0
        y_beam = 0.0

    else
        error("Unknown calc_type: $calc_type. Use \"fast\" or \"full\".")
    end

    return R_beam, x_beam, y_beam
end


"""
compute_energy_box!(irrad_cond::Array{AT},
                        ion::Ion,
                        irrad::Irrad,
                        type_AT::String,
                        cell_df::DataFrame,
                        track_seg::Bool) -> Nothing

Fill the per-layer irradiation condition array `irrad_cond` (one `AT` entry per distinct z-layer)
by propagating the ion’s energy and LET across the spheroid depth.

# What this does
1. Detects the distinct z-levels present in `cell_df` (sorted).
2. Initializes local copies of the ion state (energy, LET, Z, A, label).
3. Computes the amorphous track parameters (Rc, Rp, Kp) for the current ion and irradiation setup.
4. For each z-layer:
    - Stores an `AT` object in `irrad_cond[i]` for that layer using the current ion state and AT parameters.
    - If `track_seg` is false and the current energy is nonzero, updates the ion energy and LET by calling
        `residual_energy_after_distance` over the layer spacing (assumed uniform), then recomputes AT parameters.
5. Restores the original `ion` at the end.

# Inputs
- `irrad_cond::Array{AT}`  
    Pre-allocated array whose length must match the number of distinct z-layers in `cell_df`.
    This array will be filled in place, one `AT` per layer.

- `ion::Ion`  
    Initial ion state (fields used: ion label string, E, LET, Z). This object is not modified by the caller
    because the original is restored at the end, but a local copy is updated during the layer loop.

- `irrad::Irrad`  
    Irradiation settings passed to the AT radius calculator.

- `type_AT::String`  
    Track-structure model selector, e.g., "KC" or "LEM".

- `cell_df::DataFrame`  
    Must contain a column `:z` with the z-coordinates of all cells (real or empty sites). Distinct z-values
    define the layer positions and spacing.

- `track_seg::Bool`  
    If true, the ion state is kept fixed across layers (no energy loss calculation).  
    If false, the ion’s energy and LET are updated after each layer by applying
    `residual_energy_after_distance` using the first z-spacing as the step.

# Assumptions and notes
- The z-grid is assumed uniform. The step size used for residual-energy calculation is
    the first difference `diff(unique_z)[1]` after sorting unique z-values.
- The stopping-power lookup `sp` is expected to be available in the scope of this function
    for `residual_energy_after_distance`.
- The mass number A is parsed from the ion label string (e.g., "12C" → 12) and used consistently.
- The convention `Rk = Rp` is applied when storing AT parameters.
- The function does not return a value; it mutates `irrad_cond` in place and leaves the caller’s `ion`
    unchanged.

# Side effects
- Modifies `irrad_cond` in place by filling each entry with an `AT` object for the corresponding layer.

# Example
```julia
unique_layers = length(unique(cell_df.z))
irrad_cond = Array{AT}(undef, unique_layers)

compute_energy_box!(irrad_cond, ion, irrad, type_AT, cell_df, false)

lets = getfield.(irrad_cond, :LET)
energies = getfield.(irrad_cond, :E)
```
"""
function compute_energy_box!(irrad_cond::Array{AT}, ion::Ion, irrad::Irrad, type_AT::String, cell_df::DataFrame, track_seg::Bool)
    unique_z = sort(unique(cell_df.z))
    # Preserve original ion for the caller
    ion_original = ion

    # Unpack current ion state (will be updated locally if track_seg == false)
    E        = ion.E
    LET      = ion.LET
    Z        = ion.Z
    particle = ion.ion
    A        = parse(Int, match(r"^(\d+)", particle).captures[1])

    # Initial AT parameters for the current ion state
    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    Rk = Rp

    # Per-layer AT storage and optional energy/LET propagation
    for i in 1:size(unique_z, 1)
        irrad_cond[i] = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp)

        # If not tracking segments, propagate energy/LET across one layer step
        if (!track_seg) && (E != 0.0)
            step = diff(unique_z)[1]               # assumes uniform spacing
            E, LET = residual_energy_after_distance(E, Z, A, step, particle, sp)
        end

        # Refresh ion and AT parameters for the next layer
        ion = Ion(particle, E, A, Z, LET, 1.0)
        Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
        Rk = Rp
    end

    # Restore original ion for the caller
    ion = ion_original

    return nothing
end

"""
residual_energy_after_distance(
        E_u::Float64,
        Z::Int,
        A::Int,
        x_um::Float64,
        ion::String,
        sp::Dict;
        dx_um::Float64 = 0.1
    ) -> Tuple{Float64, Float64}

Compute the residual kinetic energy per nucleon and the corresponding LET of an ion
after traveling a given path length inside water, by numerically integrating an
energy loss model.

This function performs a simple stepwise propagation:
- Converts the requested travel distance (in micrometers) to centimeters.
- Uses a small step size (default 0.1 micrometers) to advance through the medium.
- At each step estimates stopping power based on a Bethe-like expression adapted
    for water and an effective charge model, then subtracts energy accordingly.
- Stops when the requested distance is reached or the ion runs out of energy.
- Converts the final total energy back to energy per nucleon.
- Looks up the corresponding LET from the provided stopping power tables via
    `linear_interpolation(ion, E, sp)`.

Arguments:
- E_u        : Initial kinetic energy per nucleon (MeV/u).
- Z          : Ion atomic number.
- A          : Ion mass number.
- x_um       : Path length to traverse, in micrometers.
- ion        : Ion label string used to index the stopping power dictionary (e.g., "1H", "12C").
- sp         : Dictionary mapping ion labels to stopping power tables consumable by `linear_interpolation`.
- dx_um      : Optional step size for the integration in micrometers (default 0.1). Smaller values are slower but more accurate.

Returns:
- A tuple (E, LET):
    - E   : Residual kinetic energy per nucleon (MeV/u) after traveling x_um in water.
    - LET : Corresponding LET from the tables. If E is zero, LET is set to 0.0.

Model notes and assumptions:
- Water is assumed as the medium with fixed properties (density 1 g/cm^3, effective Z and A).
- Uses a non-relativistic to moderately relativistic Bethe-like stopping expression with an effective charge correction.
- The integration is explicit Euler with fixed step size; choose dx_um to balance accuracy and performance.
- The ion is considered stopped when the running total energy becomes zero.
- The interpolation function `linear_interpolation(ion, E, sp)` must be defined and able to return a consistent LET for the given ion and energy.

Example:
```julia
E_res, LET_res = residual_energy_after_distance(150.0, 6, 12, 200.0, "12C", sp; dx_um=0.05)
```
"""
function residual_energy_after_distance(E_u::Float64, Z::Int, A::Int, x_um::Float64, ion::String, sp::Dict; dx_um=0.1)
    # Constants (water medium and physical constants in convenient units)
    K = 0.307                  # MeV·cm²/mol
    me = 0.511                 # MeV
    c = 3e10                   # cm/s
    I = 75e-6                  # MeV (mean excitation energy of water)
    ρ = 1.0                    # g/cm³ (water)
    Z_med = 7.42               # effective Z of water
    A_med = 18.015             # g/mol

    # Particle rest mass in MeV
    M = A * 931.494

    # Convert distances from micrometers to cm
    total_distance_cm = x_um * 1e-4
    step_cm = dx_um * 1e-4

    # Initialize energy (total kinetic energy in MeV)
    E_total = E_u * A

    # Stopping power model at current total energy
    function stopping_power(E::Float64)
        γ = (E + M) / M
        β2 = 1 - 1 / γ^2
        γ2 = γ^2
        β = sqrt(β2)

        me_over_M = me / M
        Tmax = (2 * me * β2 * γ2) / (1 + 2 * γ * me_over_M + me_over_M^2)

        arg = (2 * me * β2 * γ2 * Tmax) / I^2
        if arg <= 0 || β2 <= 0
            return 0.0
        end

        Zeff = Z * (1 - exp(-125 * β * Z^(-2 / 3)))
        dEdx = K * (Z_med / A_med) * (Zeff^2 / β2) * (log(arg) - 2 * β2) * ρ

        return dEdx
    end

    # Integrate energy loss along the path
    distance_covered = 0.0
    while distance_covered < total_distance_cm && E_total > 0
        dEdx = stopping_power(E_total)
        dE = dEdx * step_cm

        if dE < 0
            E_total = 0.0
            break
        end

        # Update energy (ensure energy does not go below zero)
        E_total = max(E_total - dE, 0.0)
        distance_covered += step_cm
    end

    # Convert back to per-nucleon energy
    E = E_total / A

    # Compute LET via table interpolation (0 if stopped)
    LET = (E == 0.0) ? 0.0 : linear_interpolation(ion, E, sp)

    return E, LET
end