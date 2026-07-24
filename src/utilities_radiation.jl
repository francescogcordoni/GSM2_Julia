#! ============================================================================
#! utilities_radiation.jl   —   corrected
#!
#! ============================================================================

#? MeV·cm²/g → keV/µm at ρ = 1 g/cm³.  SRIM's MeV/µm would be 1000.0.
const STOPPING_UNIT = 0.1
const AMU2MEV       = 931.494027

#! ============================================================================
#! Element lookup
#! ============================================================================

const Z_MAP = Dict(
    "H"  => 1,  "He" => 2,  "Li" => 3,  "Be" => 4,  "B"  => 5,  "C"  => 6,
    "N"  => 7,  "O"  => 8,  "F"  => 9,  "Ne" => 10, "Na" => 11, "Mg" => 12,
    "Al" => 13, "Si" => 14, "P"  => 15, "S"  => 16, "Cl" => 17, "Ar" => 18,
    "K"  => 19, "Ca" => 20, "Sc" => 21, "Ti" => 22, "V"  => 23, "Cr" => 24,
    "Mn" => 25, "Fe" => 26, "Co" => 27, "Ni" => 28, "Cu" => 29, "Zn" => 30,
    "Ga" => 31, "Ge" => 32, "As" => 33, "Se" => 34, "Br" => 35, "Kr" => 36,
    "Rb" => 37, "U"  => 92
)

"""    element_of(ion) -> String    "12C" → "C",  "C" → "C"."""
@inline element_of(ion::AbstractString) = filter(isletter, ion)

"""
    getZ(ion) -> Int

Atomic number from an ion label, e.g. "12C" → 6, "10B" → 5.

Errors on an unknown symbol rather than returning -1: the old sentinel flowed
straight into `Z^(-2/3)` in the effective-charge formula and surfaced as a
DomainError far from the cause.
"""
function getZ(ion::AbstractString)
    sym = element_of(ion)
    haskey(Z_MAP, sym) || error("getZ: unknown element '$sym' in ion label '$ion'")
    return Z_MAP[sym]
end

"""    getA(ion) -> Int    Mass number from an ion label, "12C" → 12."""
function getA(ion::AbstractString)
    m = match(r"^(\d+)", ion)
    m === nothing && error("getA: no mass number in ion label '$ion' (expected e.g. \"12C\")")
    return parse(Int, m.captures[1])
end

#! ============================================================================
#! Stopping power tables
#! ============================================================================

"""
    RangeTable

CSDA range for one isotope, derived from its element's stopping power.
Built on demand and cached, since R ∝ A while the stopping power is not.
"""
struct RangeTable
    A::Int
    R::Vector{Float64}
    lR::Vector{Float64}
    Rmin::Float64
    Rmax::Float64
end

"""
    StoppingTable

Electronic stopping power for one element.

* `E`   [MeV/u]  ascending
* `LET` [keV/µm] unit-converted at load

Logs are precomputed for log-log interpolation; lookups cost one
`searchsortedlast` and no allocation.  `ranges` caches derived range tables
keyed by mass number.
"""
struct StoppingTable
    element::String
    Z::Int
    E::Vector{Float64}
    LET::Vector{Float64}
    lE::Vector{Float64}
    lLET::Vector{Float64}
    Emin::Float64
    Emax::Float64
    ranges::Dict{Int,RangeTable}
end

"""
    build_range(E, LET, A) -> Vector{Float64}

CSDA range [µm] by cumulative trapezoidal integration of

    dx = 1000·A·dE_u / LET        (LET in keV/µm, E_u in MeV/u)

The first point carries the residual range below the table floor, evaluated
assuming LET ∝ √E (the low-energy LSS regime), giving R(E₁) = 1000·A·2E₁/LET₁.
For ¹¹B that offset is ~1.5 µm — small, but omitting it biases every range.
"""
function build_range(E::Vector{Float64}, LET::Vector{Float64}, A::Int)
    n = length(E)
    R = Vector{Float64}(undef, n)
    R[1] = 1000.0 * A * 2 * E[1] / LET[1]
    @inbounds for i in 2:n
        R[i] = R[i-1] + 1000.0 * A * 0.5 * (1/LET[i] + 1/LET[i-1]) * (E[i] - E[i-1])
    end
    return R
end

"""
    range_table!(tab, A) -> RangeTable

Fetch or build the range table for mass number `A`.

NOT thread-safe on first call for a given `A`.  Warm the cache with
`prepare_range!` before entering any threaded region; `compute_energy_box!`
does this automatically.
"""
function range_table!(tab::StoppingTable, A::Int)
    A > 0 || error("range_table!: mass number must be positive, got $A")
    get!(tab.ranges, A) do
        R = build_range(tab.E, tab.LET, A)
        RangeTable(A, R, log.(R), first(R), last(R))
    end
end

"""
    load_stopping_table(path, element) -> StoppingTable

Parse a two-column `<El>_water.txt`.  Keeps lines whose two whitespace-separated
fields both parse as Float64, so CRLF endings and any stray header are handled
without a fixed skip count.
"""
function load_stopping_table(path::AbstractString, element::AbstractString)
    isfile(path) || error("load_stopping_table: no such file: $path")

    E   = Float64[]
    LET = Float64[]
    for line in eachline(path)
        f = split(line)                        # \r and \t count as whitespace
        length(f) == 2 || continue
        a = tryparse(Float64, f[1]); b = tryparse(Float64, f[2])
        (a === nothing || b === nothing) && continue
        push!(E, a); push!(LET, b * STOPPING_UNIT)
    end

    isempty(E) && error("load_stopping_table: no numeric rows in $path")
    issorted(E)    || error("load_stopping_table: energies not ascending in $path")
    all(>(0), LET) || error("load_stopping_table: non-positive stopping power in $path")

    sym = String(element)
    haskey(Z_MAP, sym) || error("load_stopping_table: unknown element '$sym'")

    StoppingTable(sym, Z_MAP[sym], E, LET, log.(E), log.(LET),
                  first(E), last(E), Dict{Int,RangeTable}())
end

"""
    load_stopping_power(; dir, elements) -> Dict{String,StoppingTable}

Load `<El>_water.txt` for each element, keyed by element symbol.
Missing files are reported together rather than throwing on the first one.
"""
function load_stopping_power(;
        dir::AbstractString = joinpath(@__DIR__, "..", "data", "stoppingpower"),
        elements::Vector{String} = ["H", "He", "Li", "Be", "B", "C", "N", "O"])

    sp, missing_files = Dict{String,StoppingTable}(), String[]
    for el in elements
        path = joinpath(dir, "$(el)_water.txt")
        isfile(path) ? (sp[el] = load_stopping_table(path, el)) : push!(missing_files, path)
    end
    isempty(missing_files) || @warn "load_stopping_power: missing tables" files=missing_files
    isempty(sp) && error("load_stopping_power: no tables loaded from $dir")
    return sp
end

"""    table_for(sp, ion) -> StoppingTable    Resolves "12C" and "C" alike."""
function table_for(sp::Dict{String,StoppingTable}, ion::AbstractString)
    el  = element_of(ion)
    tab = get(sp, el, nothing)
    tab === nothing && error("no stopping table for element '$el' (from '$ion'). " *
                             "Loaded: $(sort(collect(keys(sp))))")
    return tab
end

prepare_range!(sp::Dict{String,StoppingTable}, ion::AbstractString, A::Int) =
    range_table!(table_for(sp, ion), A)

#! ============================================================================
#! Interpolation
#! ============================================================================

@inline function interp_loglog(lx::Vector{Float64}, ly::Vector{Float64}, x::Float64)
    lxv = log(x)
    n   = length(lx)
    i   = searchsortedlast(lx, lxv)
    i < 1     && (i = 1)
    i > n - 1 && (i = n - 1)
    @inbounds begin
        t = (lxv - lx[i]) / (lx[i+1] - lx[i])
        return exp(ly[i] + t * (ly[i+1] - ly[i]))
    end
end

@inline function let_at(tab::StoppingTable, E_u::Float64)
    E_u <= tab.Emin && return 0.0
    E_u > tab.Emax && error("let_at: E=$E_u MeV/u above table max $(tab.Emax) for $(tab.element)")
    interp_loglog(tab.lE, tab.lLET, E_u)
end

@inline function range_at(tab::StoppingTable, rt::RangeTable, E_u::Float64)
    E_u <= tab.Emin && return 0.0
    E_u > tab.Emax && error("range_at: E=$E_u MeV/u above table max $(tab.Emax) for $(tab.element)")
    interp_loglog(tab.lE, rt.lR, E_u)
end

@inline function energy_at_range(tab::StoppingTable, rt::RangeTable, R_um::Float64)
    R_um <= rt.Rmin && return 0.0
    R_um > rt.Rmax && error("energy_at_range: R=$R_um µm beyond max $(rt.Rmax) for $(tab.element)-$(rt.A)")
    interp_loglog(rt.lR, tab.lE, R_um)
end

"""
    linear_interpolation(ion, E_u, sp) -> Float64

LET [keV/µm] at `E_u` [MeV/u].  Same call signature as before; log-log
interpolation on a precomputed table, no allocation.  Returns 0 below the table
floor instead of throwing, since energy legitimately reaches zero in propagation.
"""
linear_interpolation(ion::AbstractString, E_u::Float64, sp::Dict{String,StoppingTable}) =
    let_at(table_for(sp, ion), E_u)

#! ============================================================================
#! Energy propagation
#! ============================================================================

"""
    residual_energy_after_distance(E_u, Z, A, x_um, ion, sp) -> (E_u, LET)

Residual energy [MeV/u] and **path-averaged** LET [keV/µm] over the step,
by range inversion:  R₁ = R(E₀) − x,  E₁ = R⁻¹(R₁).

The path-averaged LET, ΔE·A·1000/x, conserves energy over the step exactly;
the entrance or exit value is only first-order accurate and biases the layer
dose.  The two agree closely far from the Bragg peak and diverge as the ion slows.

`Z` is unused, retained for call-site compatibility.
"""
function residual_energy_after_distance(E_u::Float64, Z::Int, A::Int, x_um::Float64,
                                        ion::AbstractString, sp::Dict{String,StoppingTable})
    tab = table_for(sp, ion)
    return residual_energy_after_distance(E_u, x_um, tab, range_table!(tab, A))
end

function residual_energy_after_distance(E_u::Float64, x_um::Float64,
                                        tab::StoppingTable, rt::RangeTable)
    E_u <= tab.Emin && return 0.0, 0.0
    E_u > tab.Emax && error("residual_energy: E=$E_u MeV/u above table max $(tab.Emax)")
    x_um <= 0 && return E_u, let_at(tab, E_u)

    R1 = range_at(tab, rt, E_u) - x_um
    R1 <= rt.Rmin && return 0.0, E_u * rt.A * 1000.0 / x_um   # stops inside the step

    E1 = energy_at_range(tab, rt, R1)
    return E1, (E_u - E1) * rt.A * 1000.0 / x_um
end

#! ============================================================================
#! Amorphous track radii
#! ============================================================================

"""
    ATRadius(ion::Ion, irrad::Irrad, type) -> (Rc, Rp, Kp)

Core radius, penumbra radius [µm] and penumbra coefficient.

**KC** (Kiefer–Chatterjee)
    β  = sqrt(1 - 1/(E/931.494 + 1)²)
    Rc = 0.0116·β
    Rp = 0.0616·E^1.7                      (E in MeV/u)
    z_eff = Z(1 - exp(-125β/Z^{2/3}))
    Kp = 1.25e-4·(z_eff/β)²

**LEM**
    Rc = 0.01 (fixed),  Rp = 0.05·E^1.7,  Kp = LETk/(π(1 + 2ln(Rp/Rc)))

The returned Rp is the radius the domain integral must truncate at *and* the
radius the core dose must be normalised against.  Keep `Rk = Rp` and let
GetRadialLinearDose read `ion.Rk`; never recompute a penumbra radius locally.
"""
function ATRadius_params(ion::Ion, type::String)
    ion.E > 0 || return nothing

    if type == "LEM"
        Rc   = 0.01
        Rp   = 0.05 * ion.E^1.7
        LETk = ion.LET * 0.1602
        Kp   = (1 / π) * (LETk / (1 + 2 * log(Rp / Rc)))

    elseif type == "KC"
        β  = sqrt(1 - 1 / ((ion.E / AMU2MEV + 1)^2))
        Rc = 0.0116 * β                       # was 0.01116
        Rp = 0.0616 * ion.E^1.7               # was 0616  ← missing decimal point
        z_eff = ion.Z * (1 - exp(-125 * β / ion.Z^(2/3)))
        Kp = 1.25e-4 * (z_eff / β)^2

    else
        error("ATRadius: unknown type \"$type\". Use \"LEM\" or \"KC\".")
    end

    Rp > Rc || return nothing
    LETk  = ion.LET * 0.1602
    Dcore = (LETk - 2π * Kp * log(Rp / Rc)) / (π * Rc^2)
    Dcore > 0 || return nothing

    return (Rc = Rc, Rp = Rp, Kp = Kp, Dcore = Dcore)
end

"""
    ATRadius_valid(ion, type) -> Bool

Whether the AT parameterisation is self-consistent at this ion's energy and LET.

Fails when `Rp ≤ Rc` (below ~0.02 MeV/u for protons the penumbra formula
collapses inside the core) or when the penumbra alone would carry more than the
full LET, leaving a negative core dose.

`ion.LET` must be the **local** LET at `ion.E`. A path-averaged or otherwise
mismatched LET will make a perfectly valid energy look invalid, because Kp, Rc
and Rp are all derived from β(E) and the core is normalised against LET.
"""
ATRadius_valid(ion::Ion, type::String) = ATRadius_params(ion, type) !== nothing

function ATRadius(ion::Ion, _irrad::Irrad, type::String)
    ion.E > 0 || error("ATRadius: non-positive energy $(ion.E) MeV/u for $(ion.ion)")
    p = ATRadius_params(ion, type)
    if p === nothing
        error("""
            ATRadius: parameterisation invalid for $(ion.ion) at E=$(ion.E) MeV/u,
            LET=$(ion.LET) keV/µm. Either Rp ≤ Rc (ion effectively stopped) or the
            penumbra alone exceeds LET, giving a negative core dose.
            Check that ion.LET is the LOCAL LET at this energy — a slab-averaged
            LET is inconsistent with Kp and Rp, which come from β(E).
            Use ATRadius_valid to test before calling.""")
    end
    return p.Rc, p.Rp, p.Kp
end

#! ============================================================================
#! Beam setup
#! ============================================================================

"""
    calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
        -> (R_beam, x_beam, y_beam)

Unchanged in behaviour.  Note the sampling disc should be padded by Rp beyond
the scored region, or cells near the edge are starved of contributions from
tracks that fall outside R_beam but within Rp of the boundary.
"""
function calculate_beam_properties(calc_type::String, target_geom::String,
                                   X_box::Float64, X_voxel::Float64, tumor_radius::Float64)
    if calc_type == "fast"
        return (X_voxel / 2) * sqrt(2), -(X_box - X_voxel / 2), -(X_box - X_voxel / 2)
    elseif calc_type == "full"
        R_beam = if target_geom == "circle"
            tumor_radius * sqrt(2)
        elseif target_geom == "square"
            X_box * sqrt(2)
        else
            error("Unknown target geometry: $target_geom. Use \"circle\" or \"square\".")
        end
        return R_beam, 0.0, 0.0
    else
        error("Unknown calc_type: $calc_type. Use \"fast\" or \"full\".")
    end
end

"""
    required_particles(dose_Gy, LET_keV_um, R_beam) -> Int

Npar = D·πR_beam² / (LET·0.1602), from D = Φ·LETk with Φ = Npar/(πR_beam²).
"""
required_particles(dose_Gy::Float64, LET_keV_um::Float64, R_beam::Float64) =
    round(Int, dose_Gy * π * R_beam^2 / (LET_keV_um * 0.1602))

#! ============================================================================
#! Layer propagation
#! ============================================================================

"""
    occupied_layers(cell_df, unique_z) -> BitVector

Which z-layers actually contain a cell (`is_cell == 1`).

`cell_df` holds every lattice node, occupied or not, so `unique(cell_df.z)`
alone says nothing about where the material is.
"""
function occupied_layers(cell_df::DataFrame, unique_z::Vector{Float64})
    hasproperty(cell_df, :is_cell) || error("occupied_layers: cell_df needs :is_cell")
    zidx = Dict{Float64,Int}(z => i for (i, z) in enumerate(unique_z))
    occ  = falses(length(unique_z))
    @inbounds for r in 1:nrow(cell_df)
        if cell_df.is_cell[r] == 1
            j = get(zidx, Float64(cell_df.z[r]), 0)
            j != 0 && (occ[j] = true)
        end
    end
    return occ
end

"""
    absorbing_mask(occ, mode) -> BitVector

Which slabs remove energy. Slab `i` spans `z[i] → z[i+1]`.

* `:occupied` — only slabs whose leading layer holds cells. A run of occupied
  layers `a:b` then absorbs over `z[b+1] - z[a]`, i.e. one slab thickness per
  occupied layer, which is the right path length through the material.
* `:span`     — everything between the first and last occupied layer, so
  interior gaps (a necrotic core, say) still attenuate.
* `:all`      — every slab. The original behaviour.
"""
function absorbing_mask(occ::BitVector, mode::Symbol)
    n = length(occ)
    mode === :all      && return trues(n)
    mode === :occupied && return copy(occ)
    if mode === :span
        any(occ) || return falses(n)
        m = falses(n)
        m[findfirst(occ):findlast(occ)] .= true
        return m
    end
    error("absorbing_mask: unknown mode :$mode. Use :occupied, :span or :all.")
end

"""
    compute_energy_box!(irrad_cond, ion, irrad, type_AT, cell_df, track_seg, sp;
                        absorb = :occupied) -> Nothing

Fill one `AT` per z-layer, propagating energy across the spheroid.

**Energy is lost only in slabs that contain cells.** Previously every gap in
`unique(cell_df.z)` attenuated the beam, including layers with no cells in
them, so an ion entering a spheroid padded by four empty layers arrived at the
first real layer already degraded. With `absorb = :occupied` those slabs are
skipped and the ion reaches the first occupied layer at its full entrance
energy.

`absorb`:
* `:occupied` (default) — only slabs whose leading layer holds cells
* `:span` — everything between the first and last occupied layer, so interior
  voids still attenuate
* `:all` — the original behaviour

Physically this is right when the empty layers are vacuum, or when you are
defining depth zero at the spheroid surface. If they represent culture medium
or any water-equivalent material, the ion *does* lose energy crossing them and
`:all` is the correct choice — the padding is then part of the beam path.

Empty layers still receive a valid `AT` entry, carrying the current
(unattenuated) energy, so `irrad_cond` stays indexable by layer.

Other changes from the original: `sp` is an explicit argument (it was read from
the enclosing scope); the trailing `ion = ion_original` was a no-op on a local
binding and is gone; the LET stored per layer is the path average over that
layer rather than the entrance value; and propagation stops cleanly once the
ion runs out of range instead of feeding a sub-threshold energy into ATRadius.
"""
function compute_energy_box!(irrad_cond::Array{AT}, ion::Ion, irrad::Irrad,
                             type_AT::String, cell_df::DataFrame, track_seg::Bool,
                             sp::Dict{String,StoppingTable};
                             absorb::Symbol = :occupied)
    unique_z = sort(unique(Float64.(cell_df.z)))
    n_layers = length(unique_z)
    n_layers == 0 && return nothing

    particle = ion.ion
    Z, A     = ion.Z, ion.A
    tab      = table_for(sp, particle)
    rt       = range_table!(tab, A)          # warm cache, single-threaded

    occ      = occupied_layers(cell_df, unique_z)
    absorbs  = absorbing_mask(occ, absorb)
    n_occ    = count(occ)

    if !track_seg
        skipped = count(!, view(absorbs, 1:n_layers-1))
        @info "compute_energy_box!: $n_occ/$n_layers layers occupied, " *
              "$skipped slab(s) non-absorbing (absorb=:$absorb)"
        n_occ == 0 && @warn "no occupied layers — the beam will not be attenuated at all"
    end

    E       = ion.E
    stopped = false

    for i in 1:n_layers
        if stopped
            irrad_cond[i] = AT(particle, 0.0, A, Z, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            continue
        end

        # LOCAL LET at this layer's energy — not a slab average. Kp, Rc and Rp all
        # derive from β(E), and the core dose is normalised against LET, so the
        # three must refer to the same E or the normalisation is inconsistent.
        LET_local = let_at(tab, E)
        layer_ion = Ion(particle, E, A, Z, LET_local, 1.0)

        if !ATRadius_valid(layer_ion, type_AT)
            stopped = true
            @info "compute_energy_box!: $particle below AT validity at layer $i/$n_layers " *
                  "(E=$(round(E, sigdigits=4)) MeV/u, $(round(E*A, sigdigits=4)) MeV " *
                  "residual per ion unaccounted)"
            irrad_cond[i] = AT(particle, 0.0, A, Z, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            continue
        end

        Rc, Rp, Kp = ATRadius(layer_ion, irrad, type_AT)
        irrad_cond[i] = AT(particle, E, A, Z, LET_local, 1.0, Rc, Rp, Rp, Kp)

        # slab i → i+1 contributes path length only if it absorbs
        step = (i < n_layers && absorbs[i]) ? unique_z[i+1] - unique_z[i] : 0.0
        if !track_seg && step > 0
            E, _ = residual_energy_after_distance(E, step, tab, rt)
            if E <= tab.Emin
                stopped = true
                i < n_layers && @info "compute_energy_box!: $particle stops at layer $i of $n_layers"
            end
        end
    end
    return nothing
end

#! ============================================================================
#! Validation
#! ============================================================================

"""
    validate_stopping(sp, ion, A; probes = [1.0, 10.0, 50.0, 100.0, 200.0])

Prints LET, CSDA range and an E → R → E round trip.  The round trip exercises
both interpolants and should close to interpolation precision; a large residual
means the range table is not cleanly monotonic.
"""
function validate_stopping(sp::Dict{String,StoppingTable}, ion::AbstractString, A::Int;
                           probes::Vector{Float64} = [1.0, 10.0, 50.0, 100.0, 200.0])
    tab = table_for(sp, ion)
    rt  = range_table!(tab, A)
    println("── $(tab.element)  (Z=$(tab.Z), A=$A) ", "─"^30)
    println("   E [MeV/u] : $(tab.Emin) … $(tab.Emax)   |   R [µm] : ",
            round(rt.Rmin, sigdigits=4), " … ", round(rt.Rmax, sigdigits=5))
    println("      E        LET [keV/µm]      R [µm]      E→R→E")
    for E in probes
        E > tab.Emax && continue
        R = range_at(tab, rt, E)
        println(lpad(round(E, digits=1), 8),
                lpad(round(let_at(tab, E), digits=3), 16),
                lpad(round(R, sigdigits=6), 14),
                lpad(round(energy_at_range(tab, rt, R)/E - 1, sigdigits=2), 12))
    end
    println()
    return nothing
end
