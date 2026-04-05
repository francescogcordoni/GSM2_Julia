#! ============================================================================
#! utilities_radiation.jl
#!
#! FUNCTIONS
#! ---------
#~ Amorphous Track Structure
#?   ATRadius(ion, irrad, type) -> (Rc, Rp, Kp)
#       Computes core radius, penumbra radius, and dose amplitude for a given ion.
#       Supports "LEM" and "KC" parameterizations.
#
#~ Atomic Number & Stopping Power
#?   getZ(ion) -> Int
#       Extracts atomic number Z from ion string (e.g. "12C" → 6).
#?   load_stopping_power() -> Dict{String, Matrix{Float64}}
#       Loads stopping power tables from disk for H, He, Li, Be, B, C, N, O in water.
#?   linear_interpolation(ion, input_value, sp) -> Float64
#       Linear interpolation of LET/stopping power at a given energy from tabulated data.
#
#~ Beam & Irradiation Setup
#?   calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
#           -> (R_beam, x_beam, y_beam)
#       Computes effective beam radius and center for "fast" (voxel-based) or
#       "full" (geometry-based) simulation modes.
#?   compute_energy_box!(irrad_cond, ion, irrad, type_AT, cell_df, track_seg) -> Nothing
#       Fills per-layer AT array by propagating ion energy/LET across spheroid depth.
#       Mutates irrad_cond in-place; one AT entry per z-layer.
#
#~ Energy Propagation
#?   residual_energy_after_distance(E_u, Z, A, x_um, ion, sp; dx_um) -> (E, LET)
#       Numerically integrates Bethe-like stopping power to compute residual energy
#       and LET after traveling x_um µm in water. Explicit Euler integration.
#
#~ Constants
#?   Z_MAP — Dict mapping element symbols to atomic numbers (H=1 … Kr=36, Rb=37, U=92)
#       Note: only elements relevant to particle therapy are included; 38–91 are absent.
#! ============================================================================

"""
    ATRadius(ion::Ion, irrad::Irrad, type::String) -> (Rc, Rp, Kp)

Amorphous track structure radii and dose amplitude for a given ion.
Returns core radius `Rc` (µm), penumbra radius `Rp` (µm), and normalization
coefficient `Kp`.

# Parameterizations

**`"LEM"`** (Local Effect Model):
- `Rc = 0.01` µm (fixed)
- `Rp = 0.05 * (E/A)^1.7` µm
- `Kp = LET_keV_um / (π * (1 + 2*log(Rp/Rc)))` (LET converted: ×0.1602)

**`"KC"`** (Kiefer-Chatterjee):
- `β` from relativistic kinematics: `β = sqrt(1 - 1/(E/AMU + 1)²)`
- `Rc = 0.01116 * β` µm
- `Rp = 0.0616 * (E/A)^1.7` µm
- `z_eff = Z * (1 - exp(-125β/Z^(2/3)))` (effective charge)
- `Kp = 1.25 × 10⁻⁴ * (z_eff/β)²`

Throws an error for unknown `type`.

# Example
```julia
Rc, Rp, Kp = ATRadius(ion, irrad, "KC")
Rc, Rp, Kp = ATRadius(ion, irrad, "LEM")
```
"""
function ATRadius(ion::Ion, _irrad::Irrad, type::String)

    if type == "LEM"
        Rc   = 0.01
        Rp   = 0.05 * ((ion.E / ion.A)^(1.7))
        LETk = ion.LET * 0.1602   # keV/µm → MeV·cm⁻¹ (×0.1602)
        Kp   = (1 / π) * (LETk / (1 + 2 * log(Rp / Rc)))

    elseif type == "KC"
        AMU2MEV = 931.494027
        β    = sqrt(1 - 1 / ((ion.E / AMU2MEV + 1)^2))
        Rc   = 0.01116 * β
        Rp   = 0.0616 * ((ion.E / ion.A)^(1.7))
        z_eff = ion.Z * (1 - exp(-125 * β / ion.Z^(2.0 / 3.0)))
        Kp   = 1.25 * 0.0001 * (z_eff / β)^2

    else
        error("ATRadius: unknown type \"$type\". Use \"LEM\" or \"KC\".")
    end

    return Rc, Rp, Kp
end

#! ============================================================================
#! Atomic Number Lookup
#! ============================================================================

#? Z_MAP — element symbol → atomic number (H=1 ... U=92)
const Z_MAP = Dict(
    "H"  => 1,  "He" => 2,  "Li" => 3,  "Be" => 4,  "B"  => 5,  "C"  => 6,
    "N"  => 7,  "O"  => 8,  "F"  => 9,  "Ne" => 10, "Na" => 11, "Mg" => 12,
    "Al" => 13, "Si" => 14, "P"  => 15, "S"  => 16, "Cl" => 17, "Ar" => 18,
    "K"  => 19, "Ca" => 20, "Sc" => 21, "Ti" => 22, "V"  => 23, "Cr" => 24,
    "Mn" => 25, "Fe" => 26, "Co" => 27, "Ni" => 28, "Cu" => 29, "Zn" => 30,
    "Ga" => 31, "Ge" => 32, "As" => 33, "Se" => 34, "Br" => 35, "Kr" => 36,
    "Rb" => 37, "U"  => 92
)

"""
    getZ(ion::AbstractString) -> Int

Extract atomic number Z from an ion string (e.g. `"12C"` → 6, `"4He"` → 2).
Returns `-1` if the element symbol is not found in `Z_MAP`.

# Example
```julia
getZ("12C")   # → 6
getZ("4He")   # → 2
```
"""
function getZ(ion::AbstractString)
    particle = filter(isletter, ion)
    return get(Z_MAP, particle, -1)
end

"""
    load_stopping_power() -> Dict{String, Matrix{Float64}}

Load stopping power tables from `data/stoppingpower/` for H, He, Li, Be, B, C, N, O in water.
Returns a Dict mapping ion labels (e.g. `"12C"`) to 2-column matrices (energy, stopping power).
Fully deterministic — same files always produce same output.

# Example
```julia
sp = load_stopping_power()
```
"""
function load_stopping_power()
    sp = Dict{String, Matrix{Float64}}()
    spdir = joinpath(@__DIR__, "..", "data", "stoppingpower")

    ion_list = [
        ("1H",  "H"),  ("4He", "He"), ("6Li", "Li"), ("8Be", "Be"),
        ("10B", "B"),  ("12C", "C"),  ("14N", "N"),  ("16O", "O"),
    ]

    for (ion, element) in ion_list
        sp[ion] = readdlm(joinpath(spdir, "$(element)_water.txt"), Float64)
    end

    return sp
end

"""
    linear_interpolation(ion::String, input_value::Float64, sp::Dict) -> Float64

Linear interpolation of stopping power at `input_value` (energy in MeV/u) for a given ion.
`sp[ion]` must be a 2-column matrix: column 1 = energy, column 2 = stopping power.
Errors if `input_value` is out of the tabulated range.

# Example
```julia
LET = linear_interpolation("12C", 150.0, sp)
```
"""
function linear_interpolation(ion::String, input_value::Float64, sp::Dict)
    x = sp[ion][:, 1]
    y = sp[ion][:, 2] / 10   # convert MeV·cm²/g → keV/µm (÷10 for water, ρ=1 g/cm³)

    if input_value < minimum(x) || input_value > maximum(x)
        error("Input value $input_value is out of range.")
    end

    idx_lower = findlast(x .<= input_value)
    idx_upper = findfirst(x .>= input_value)

    x1, y1 = x[idx_lower], y[idx_lower]
    x2, y2 = x[idx_upper], y[idx_upper]

    return x1 == x2 ? y1 : y1 + (y2 - y1) * (input_value - x1) / (x2 - x1)
end

"""
    calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
        -> (R_beam, x_beam, y_beam)

Compute effective beam radius and center coordinates.

- `"fast"` mode: beam ≈ voxel diagonal, center at box corner.
- `"full"` mode: beam covers full target diagonal, center at origin.
  - `"circle"` target → `R_beam = tumor_radius * sqrt(2)`
  - `"square"` target → `R_beam = X_box * sqrt(2)`

# Example
```julia
R_beam, x_beam, y_beam = calculate_beam_properties("full", "circle", 800.0, 200.0, 300.0)
```
"""
function calculate_beam_properties(calc_type::String,
                                    target_geom::String,
                                    X_box::Float64,
                                    X_voxel::Float64,
                                    tumor_radius::Float64)
    if calc_type == "fast"
        R_beam = (X_voxel / 2) * sqrt(2)
        x_beam = -(X_box - X_voxel / 2)
        y_beam = -(X_box - X_voxel / 2)

    elseif calc_type == "full"
        if target_geom == "circle"
            R_beam = tumor_radius * sqrt(2)
        elseif target_geom == "square"
            R_beam = X_box * sqrt(2)
        else
            error("Unknown target geometry: $target_geom. Use \"circle\" or \"square\".")
        end
        x_beam = 0.0
        y_beam = 0.0

    else
        error("Unknown calc_type: $calc_type. Use \"fast\" or \"full\".")
    end

    return R_beam, x_beam, y_beam
end

"""
    compute_energy_box!(irrad_cond, ion, irrad, type_AT, cell_df, track_seg) -> Nothing

Fill per-layer AT array by propagating ion energy/LET across spheroid depth.
Mutates `irrad_cond` in-place — one `AT` entry per distinct z-layer in `cell_df`.

- `track_seg = true`  → ion state fixed across all layers (no energy loss).
- `track_seg = false` → energy/LET updated layer-by-layer via
  `residual_energy_after_distance` using the actual per-layer z-spacing
  (`unique_z[i+1] - unique_z[i]`). Non-uniform z-spacing is handled correctly.

Requires `sp` (stopping power dict) in enclosing scope.

# Example
```julia
irrad_cond = Array{AT}(undef, length(unique(cell_df.z)))
compute_energy_box!(irrad_cond, ion, irrad, "KC", cell_df, false)
```
"""
function compute_energy_box!(irrad_cond::Array{AT}, ion::Ion, irrad::Irrad, type_AT::String, cell_df::DataFrame, track_seg::Bool)
    unique_z     = sort(unique(cell_df.z))
    n_layers     = length(unique_z)
    ion_original = ion

    E        = ion.E
    LET      = ion.LET
    Z        = ion.Z
    particle = ion.ion
    A        = parse(Int, match(r"^(\d+)", particle).captures[1])

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    Rk = Rp

    for i in 1:n_layers
        irrad_cond[i] = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp)

        # Propagate energy loss to next layer (skip on last layer — no layer i+1)
        if (!track_seg) && (E != 0.0) && (i < n_layers)
            step = unique_z[i + 1] - unique_z[i]   # actual per-layer spacing (µm)
            E, LET = residual_energy_after_distance(E, Z, A, step, particle, sp)
        end

        ion = Ion(particle, E, A, Z, LET, 1.0)
        Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
        Rk = Rp
    end

    ion = ion_original
    return nothing
end

"""
    residual_energy_after_distance(E_u, Z, A, x_um, ion, sp; dx_um=0.1) -> (E, LET)

Residual energy per nucleon (MeV/u) and LET after traveling `x_um` µm in water.
Uses explicit Euler integration of a Bethe-like stopping power model with effective charge.
Stops early if the ion runs out of energy.

- `dx_um` controls step size (smaller = more accurate, slower).
- LET is looked up via `linear_interpolation` at the final energy; returns 0.0 if stopped.

# Example
```julia
E_res, LET_res = residual_energy_after_distance(150.0, 6, 12, 200.0, "12C", sp; dx_um=0.05)
```
"""
function residual_energy_after_distance(E_u::Float64, Z::Int, A::Int, x_um::Float64, ion::String, sp::Dict; dx_um=0.1)
    # Water medium constants
    K     = 0.307       # MeV·cm²/mol
    me    = 0.511       # MeV
    I     = 75e-6       # MeV (mean excitation energy of water)
    ρ     = 1.0         # g/cm³
    Z_med = 7.42
    A_med = 18.015

    M = A * 931.494     # Ion rest mass (MeV)

    total_distance_cm = x_um * 1e-4
    step_cm           = dx_um * 1e-4
    E_total           = E_u * A

    function stopping_power(E::Float64)
        γ = (E + M) / M
        β2 = 1 - 1 / γ^2
        β  = sqrt(β2)
        γ2 = γ^2

        me_over_M = me / M
        Tmax = (2 * me * β2 * γ2) / (1 + 2 * γ * me_over_M + me_over_M^2)
        arg  = (2 * me * β2 * γ2 * Tmax) / I^2

        (arg <= 0 || β2 <= 0) && return 0.0

        Zeff = Z * (1 - exp(-125 * β * Z^(-2 / 3)))
        return K * (Z_med / A_med) * (Zeff^2 / β2) * (log(arg) - 2 * β2) * ρ
    end

    distance_covered = 0.0
    while distance_covered < total_distance_cm && E_total > 0
        dEdx = stopping_power(E_total)
        # dEdx == 0 means the ion has effectively stopped (β→0 or unphysical state);
        # dEdx < 0 should not occur but guards against numerical drift.
        dEdx <= 0 && (E_total = 0.0; break)
        E_total = max(E_total - dEdx * step_cm, 0.0)
        distance_covered += step_cm
    end

    E   = E_total / A
    LET = (E == 0.0) ? 0.0 : linear_interpolation(ion, E, sp)

    return E, LET
end
