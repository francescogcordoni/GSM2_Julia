"""
distribute_dose_domain(
        x0::Float64, y0::Float64,
        radius::Float64,
        track::Track,
        irrad_cond::AT,
        type::String;
        verbose::Bool = false
    ) -> (dose::Float64, area::Float64, Gyr::Float64)

Compute the energy deposition (**dose**) from a single ion **track** into a circular
target domain of center `(x0, y0)` and radius `radius`. The track geometry is given
by `track` (with impact point `track.x, track.y` and radial extent `track.Rk`), while
the microdosimetric model is controlled by `irrad_cond` and `type`.

The computation considers three geometric cases based on the **impact parameter**
`b = √((track.x - x0)^2 + (track.y - y0)^2)`:

1. **Full impact** (`b ≤ radius`): the track passes through the target center.
2. **Partial overlap** (`radius < b ≤ radius + track.Rk`): annular overlap.
3. **No overlap** (`b > radius + track.Rk`): zero contribution.

Dose accumulation is performed by integrating the **weighted radial track** using
`integrate_weighted_radial_track(…)` over the appropriate radial ranges. The mean
dose density `Gyr` is computed as `dose / area`, with `area` being the effective
geometric normalization used.

# Arguments
- `x0, y0::Float64` : Center of the circular target domain.
- `radius::Float64` : Target radius (same length units as track geometry).
- `track::Track`    : Track data structure, requires at least fields `x`, `y`, `Rk`.
- `irrad_cond::AT`  : Irradiation/microphysics parameters (kernel coefficients, etc.).
- `type::String`    : Microdosimetric model name (e.g., "Katz", "MKM").

# Keywords
- `verbose::Bool=false` : If `true`, prints detailed per‑case diagnostic information.

# Returns
- `(dose, area, Gyr)`:
    - `dose::Float64` : Integrated dose contribution (numerator).
    - `area::Float64` : Effective area used for normalization.
    - `Gyr::Float64`  : Mean dose density (dose / area). Zero if `area == 0`.

# Notes
- Units are assumed to be **consistent** across geometry and model parameters.
- Integration resolution (500 or 100 steps) follows the original logic.
- Fixed previous scoping bug: `rMax = min(track.Rk, b + radius)` (no free `Rk`).

"""
function distribute_dose_domain(
    x0::Float64, y0::Float64,
    radius::Float64,
    track::Track,
    irrad_cond::AT,
    type::String;
    verbose::Bool = false
)

    vprintln(args...) = (verbose ? println(args...) : nothing)

    # --- Geometry: impact parameter b and radial limits ---
    x_track = track.x - x0
    y_track = track.y - y0
    b = sqrt(x_track^2 + y_track^2)

    rMax = min(track.Rk, b + radius)   # ensure we never integrate beyond the track
    rMin = 0.0

    vprintln("\n--- distribute_dose_domain ---")
    vprintln("center = (", x0, ", ", y0, "), radius = ", radius)
    vprintln("track  = (", track.x, ", ", track.y, "), Rk = ", track.Rk)
    vprintln("b = ", b, "  rMin = ", rMin, "  rMax = ", rMax)

    # Accumulators
    area1 = 0.0
    area2 = 0.0
    area3 = 0.0
    dose  = 0.0
    Gyr   = 0.0

    # ------------------------
    # CASE 1: full impact (b ≤ radius)
    # ------------------------
    if b <= radius
        vprintln("[Case 1] b ≤ radius  → full impact")

        # Inner intersection radius (core within domain)
        r_intersection = (b + track.Rk < radius) ? track.Rk : (radius - b)
        r_intersection = max(0.0, r_intersection) # guard against negatives

        area1 = π * r_intersection^2

        # Integrate 0 → r_intersection
        area, integral, _ = integrate_weighted_radial_track(0.0, r_intersection, b, radius, 500, type, irrad_cond)
        dose += integral

        # Integrate r_intersection → rMax (still within domain)
        if rMax > r_intersection
            area, integral, _ = integrate_weighted_radial_track(r_intersection, rMax, b, radius, 500, type, irrad_cond)
            area2 = area
            dose += integral
        end

        # Edge correction if we reach track.Rk
        if rMax == track.Rk
            if track.Rk > radius - b
                theta1 = acos((b / (2 * rMax) + rMax / (2 * b) - (radius^2) / (2 * b * rMax)))
                theta2 = acos((b / (2 * radius) - (rMax^2) / (2 * b * radius) + radius / (2 * b)))
                area3 = π * radius^2 - (theta1 * rMax^2 + theta2 * radius^2 - rMax * b * sin(theta1))
            else
                area3 = π * (radius^2 - r_intersection^2)
            end
        end

        denom_area = area1 + area2 + area3
        Gyr = (denom_area > 0.0) ? dose / denom_area : 0.0
        vprintln("areas: a1=", area1, "  a2=", area2, "  a3=", area3, "  denom=", denom_area)

    # ------------------------
    # CASE 2: partial overlap (radius < b ≤ radius + Rk)
    # ------------------------
    elseif b <= radius + track.Rk
        vprintln("[Case 2] radius < b ≤ radius + Rk  → partial overlap")

        rMin = max(0.0, b - radius)
        step = b <= radius + 0.2 ? 500 : 100   # original resolution heuristic

        area, integral, _ = integrate_weighted_radial_track(rMin, rMax, b, radius, step, type, irrad_cond)
        dose  = integral
        area2 = area

        if rMax == track.Rk
            theta1 = acos((b / (2 * rMax) + rMax / (2 * b) - (radius^2) / (2 * b * rMax)))
            theta2 = acos((b / (2 * radius) - (rMax^2) / (2 * b * radius) + radius / (2 * b)))
            area3 = π * radius^2 - (theta1 * rMax^2 + theta2 * radius^2 - rMax * b * sin(theta1))
        end

        denom_area = area2 + area3
        Gyr = (denom_area > 0.0) ? dose / denom_area : 0.0
        vprintln("areas: a2=", area2, "  a3=", area3, "  denom=", denom_area)

    # ------------------------
    # CASE 3: no overlap (b > radius + Rk)
    # ------------------------
    else
        vprintln("[Case 3] b > radius + Rk  → no overlap")
        dose = 0.0
        area1 = area2 = area3 = 0.0
        Gyr = 0.0
    end

    vprintln("Result → dose=", dose, "  area=", area1 + area2 + area3, "  Gyr=", Gyr, "\n")
    return dose, area1 + area2 + area3, Gyr
end

"""
integrate_weighted_radial_track(
        rMin::Float64, rMax::Float64,
        b::Float64, r_nucleus::Float64,
        step::Int64,
        type::String,
        irrad_cond::AT;
        verbose::Bool = false
    ) -> (area::Float64, integral::Float64, Gyr::Float64)

Integrate the **weighted radial dose contribution** of an ion track over a radial
interval `[rMin, rMax]` relative to the domain.

The integrand has the general form:

    f(r) = GetRadialLinearDose(r, irrad_cond, type) * r * arc_intersection_angle(r, b, r_nucleus)

and the integration uses **logarithmic spacing** between radii to resolve steep
gradients near the track core. A simple **trapezoidal rule** is applied.

The integration also accumulates an **effective area**, defined as:

    area += ∫ (arc_intersection_angle(r) * r) dr

This allows the returned `Gyr = integral / area` to represent a **mean dose density**
over the angular–radial sector intersecting the nucleus (or domain).

# Arguments
- `rMin::Float64` : Lower radial bound (can be zero).
- `rMax::Float64` : Upper radial bound (must be > rMin).
- `b::Float64`    : Impact parameter (distance between track and domain center).
- `r_nucleus::Float64` : Domain radius.
- `step::Int64`   : Number of integration steps (minimum enforced is 3).
- `type::String`  : Microdosimetric model ("Katz", "MKM", etc.).
- `irrad_cond::AT` : Irradiation/model parameters.

# Keywords
- `verbose::Bool=false` : If true, prints detailed integration info.

# Returns
- `area::Float64`    : Effective angular–radial area from the geometric kernel.
- `integral::Float64`: Integrated dose numerator.
- `Gyr::Float64`     : Mean dose density = `integral / area` (zero if area=0).

# Notes
- Uses **logarithmic sampling** even if `rMin = 0` by assigning `log_rMin = -5.0`.
- Automatically clamps `step` to at least 3.
- Breaks early if `r2 == rMax` to avoid overshoot from logarithmic spacing.
"""
function integrate_weighted_radial_track(
    rMin::Float64, rMax::Float64,
    b::Float64, r_nucleus::Float64,
    step::Int64,
    type::String,
    irrad_cond::AT;
    verbose::Bool = false
)

    vprintln(args...) = (verbose ? println(args...) : nothing)

    # --- Basic input validation ---
    if rMax <= 0
        return 0.0, 0.0, 0.0
    end
    if rMax <= rMin
        return 0.0, 0.0, 0.0
    end

    # --- Prepare log-space parameters ---
    log_rMin = (rMin > 0) ? log10(rMin) : -5.0  # allow rMin = 0
    log_rMax = log10(rMax)

    nSteps = max(3, step)
    log_step = (log_rMax - log_rMin) / nSteps

    vprintln("\n--- integrate_weighted_radial_track ---")
    vprintln("rMin=", rMin, "  rMax=", rMax, "  b=", b, "  r_nucleus=", r_nucleus)
    vprintln("nSteps=", nSteps, "  log_rMin=", log_rMin, "  log_rMax=", log_rMax)

    # --- Initial point ---
    r1 = rMin
    arc_weight1 = arc_intersection_angle(r1, b, r_nucleus)
    f1 = GetRadialLinearDose(r1, irrad_cond, type) * r1 * arc_weight1

    # --- Accumulators ---
    integral = 0.0
    area     = 0.0

    # --- Log-spaced trapezoidal rule ---
    for i in 1:nSteps
        log_r2 = log_rMin + log_step * i
        r2 = 10^log_r2

        # Force exact rMax in last step
        if i == nSteps
            r2 = rMax
        end

        arc_weight2 = arc_intersection_angle(r2, b, r_nucleus)
        f2 = GetRadialLinearDose(r2, irrad_cond, type) * r2 * arc_weight2

        # Trapezoidal integration in linear r
        Δr = r2 - r1
        integral += Δr * (f1 + f2) * 0.5

        area += Δr * (arc_weight1 * r1 + arc_weight2 * r2) * 0.5

        vprintln("step=", i, "  r1=", r1, "  r2=", r2, "  f1=", f1, "  f2=", f2)

        # Prepare next iteration
        r1 = r2
        f1 = f2
        arc_weight1 = arc_weight2

        # Break early (avoids overshoot)
        if r2 == rMax
            break
        end
    end

    Gyr = (area > 0.0) ? (integral / area) : 0.0

    vprintln("Result → area=", area, "  integral=", integral, "  Gyr=", Gyr, "\n")
    return area, integral, Gyr
end

"""
    arc_intersection_angle(
        r::Float64,
        b::Float64,
        r_nucleus::Float64;
        verbose::Bool = false
    ) -> Float64

Compute the **angular intersection** (in radians) between:
- a ring at radius `r` around the ion track axis
- a circular domain (typically a nucleus) of radius `r_nucleus`
- offset by an impact parameter `b` (distance between track and domain center)

This angle represents the portion of the circle at radius `r` that lies **inside**
the target domain. It is used as a geometric weight when integrating the radial
dose distribution.

# Geometry
The function implements the classical intersection conditions:

1. **Full intersection** (`r ≤ |b - r_nucleus|`)
    - If the ring is fully inside the domain:
        → angle = 2π when `b < r_nucleus`
    - If the ring lies fully outside:
        → angle = 0

2. **Partial intersection** (`|b - r_nucleus| < r < b + r_nucleus`)
    - Circular arc intersection:
        arg = (b² + r² - r_nucleus²) / (2 b r)
        angle = 2 * acos(arg_clamped)

3. **No intersection** (`r ≥ b + r_nucleus`)
    - angle = 0

# Arguments
- `r::Float64`          : Radial distance from track center.
- `b::Float64`          : Impact parameter (track–domain center distance).
- `r_nucleus::Float64`  : Radius of the domain.

# Keyword
- `verbose::Bool=false` : If true, prints detailed geometric case info.

# Returns
- `θ::Float64` in radians, always within `[0, 2π]`.

# Notes
- Uses `clamp(arg, -1, 1)` for numerical stability.
- No units assumed; all inputs must be in consistent length units.
"""
function arc_intersection_angle(
    r::Float64,
    b::Float64,
    r_nucleus::Float64;
    verbose::Bool = false
)
    vprintln(args...) = (verbose ? println(args...) : nothing)

    vprintln("\n--- arc_intersection_angle ---")
    vprintln("r = ", r, ",  b = ", b, ",  r_nucleus = ", r_nucleus)

    # -----------------------------
    # CASE 1 — Full containment
    # -----------------------------
    if r <= abs(b - r_nucleus)
        if b < r_nucleus
            vprintln("[Case 1] Ring fully inside domain → angle = 2π")
            return 2π
        else
            vprintln("[Case 1] Ring fully outside domain → angle = 0")
            return 0.0
        end
    end

    # -----------------------------
    # CASE 2 — Partial intersection
    # -----------------------------
    if r < b + r_nucleus
        arg = (b^2 + r^2 - r_nucleus^2) / (2 * b * r)
        arg_c = clamp(arg, -1.0, 1.0)
        θ = 2 * acos(arg_c)
        vprintln("[Case 2] Partial intersection → angle = ", θ)
        return θ
    end

    # -----------------------------
    # CASE 3 — No intersection
    # -----------------------------
    vprintln("[Case 3] No intersection → angle = 0")
    return 0.0
end

"""
GetRadialLinearDose(r::Float64, ion::AT, type::String; verbose::Bool = false)

Compute the **radial linear energy deposition** (dose per unit length) at a
distance `r` from the ion track according to the selected microdosimetric model.

Two models are currently supported:

### 1. **KC Model** (`type == "KC"`)
Implements a Katz–Chatterjee–like radial dose formulation:
- Core region (`r ≤ Rc`):
    High-density delta-ray core dose, normalized by core area.
- Penumbra region (`r > Rc`):
    Long-range dose tail proportional to `1/r²`.

Includes:
- Effective charge calculation  
- Beta from particle kinetic energy  
- Penumbra radius calculation  
- Physics coefficients (`Kp`, LET → keV/µm conversion)

### 2. **LEM Model** (`type == "LEM"`)
Implements a Local Effect Model (LEM)-style radial dose:
- Core (`r ≤ Rc`) and penumbra (`r > Rc`) depend on log(Rp/Rc)
- Normalization scaled by typical LEM factors

### Arguments
- `r::Float64`       : Radial distance from the track center.
- `ion::AT`          : Irradiation/microphysics object containing fields:
                        `E`, `LET`, `A`, `Z`, `Rc`, `Rp`, …
- `type::String`     : `"KC"` or `"LEM"`.

### Keyword
- `verbose::Bool = false`  
    If true, prints intermediate values (charge, beta, coefficients, region info).

### Returns
- `D_arc::Float64` : Radial linear dose contribution at distance `r`.

### Notes
- **Units are assumed consistent** (energy in MeV, radii in µm or keV/µm normalization).
- No range-check is enforced: the caller must ensure `r ≥ 0`.
- The physics formulas are **identical** to your original implementation.
"""
function GetRadialLinearDose(r::Float64, ion::AT, type::String; verbose::Bool = false)
    vprintln(args...) = (verbose ? println(args...) : nothing)

    if type == "KC"
        # --- Extract basic parameters ---
        particleEnergy = ion.E
        LET = ion.LET
        A = ion.A
        Z = ion.Z

        # --- Constants ---
        AMU2MEV = 931.494027        # convert A*amu to MeV
        ETA = 0.0116                # Katz constant (keV/µm)
        CONV = 160.2177             # keV/µm to J/kg conversion factor

        # --- Relativistic beta ---
        β = sqrt(1 - 1 / ((particleEnergy / AMU2MEV + 1)^2))

        # --- Penumbra and core radii ---
        r_penumbra = 0.0616 * (particleEnergy / A)^1.7
        r_core = ion.Rc

        # --- Effective charge ---
        z_eff = Z * (1 - exp(-125 * β / Z^(2.0 / 3.0)))

        # --- Penumbra coefficient ---
        Kp = 1.25 * 0.0001 * (z_eff / β)^2

        # --- LET conversion (MeV/µm → keV/µm) ---
        LETk = LET * 0.1602

        # --- Verbose diagnostics ---
        vprintln("KC Model:")
        vprintln("  β = ", β)
        vprintln("  r_core = ", r_core, "   r_penumbra = ", r_penumbra)
        vprintln("  z_eff = ", z_eff, "   Kp = ", Kp)

        # --- Dose calculation ---
        if r <= r_core
            # Core region
            D_arc = LETk
            D_arc -= 2π * Kp * log(r_penumbra / r_core)
            D_arc /= (π * r_core^2)
            D_arc *= 1.05
            vprintln("  [Core] r=", r, " → D=", D_arc)
        else
            # Penumbra region (1/r²)
            D_arc = Kp / (r * r)
            vprintln("  [Penumbra] r=", r, " → D=", D_arc)
        end

        return D_arc


    elseif type == "LEM"
        # --- LET conversion ---
        LETk = ion.LET * 0.1602

        Rc = ion.Rc
        Rp = ion.Rp

        # Normalization factor used by your original implementation
        norm = LETk / (1 * (1 + 2 * log(Rp / Rc)))

        vprintln("LEM Model:")
        vprintln("  Rc = ", Rc, "   Rp = ", Rp)
        vprintln("  LETk = ", LETk, "   norm = ", norm)

        if r <= Rc
            D_arc = (1 / (π * Rc^2)) * norm
            D_arc *= 1.05
            vprintln("  [Core] r=", r, " → D=", D_arc)
        else
            D_arc = (1 / (π * r * r)) * norm
            vprintln("  [Penumbra] r=", r, " → D=", D_arc)
        end

        return D_arc


    else
        error("GetRadialLinearDose: unknown model type ‘$type’. Expected \"KC\" or \"LEM\".")
    end
end






