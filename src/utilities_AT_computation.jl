#! ============================================================================
#! utilities_AT_computation.jl
#!
#! FUNCTIONS
#! ---------
#~ Domain Dose Integration
#?   distribute_dose_domain(x0, y0, radius, track, irrad_cond, type; verbose)
#           -> (dose, area, Gyr)
#       Dose from a single ion track into a circular domain.
#       Handles three geometric cases: full impact (b≤r), partial overlap
#       (r < b ≤ r+Rk), and no overlap (b > r+Rk).
#       Returns integrated dose numerator, normalization area, and mean dose Gyr.
#?   integrate_weighted_radial_track(rMin, rMax, b, r_nucleus, step, type, irrad_cond; verbose)
#           -> (area, integral, Gyr)
#       Log-spaced trapezoidal integration of the radial dose kernel over [rMin, rMax].
#       Integrand: GetRadialLinearDose(r) * r * arc_intersection_angle(r, b, r_nucleus).
#
#~ Geometric Helpers
#?   arc_intersection_angle(r, b, r_nucleus; verbose) -> Float64
#       Angular intersection (radians) between a ring at radius r and a circular
#       domain of radius r_nucleus offset by impact parameter b.
#       Returns 0, 2π, or the partial arc angle. Used as geometric weight in integration.
#
#~ Radial Dose Model
#?   GetRadialLinearDose(r, ion, type; verbose) -> Float64
#       Radial linear dose at distance r from track center.
#       "KC" model: core (LET-based) + penumbra (Kp/r²) with effective charge.
#       "LEM" model: core + penumbra using log(Rp/Rc) normalization.
#! ============================================================================

"""
    distribute_dose_domain(x0, y0, radius, track, irrad_cond, type; verbose=false)
        -> (dose, area, Gyr)

Dose from a single ion track into a circular domain of center `(x0, y0)` and
radius `radius`. Three cases based on impact parameter `b`:
1. `b ≤ radius`          — full impact
2. `radius < b ≤ r+Rk`   — partial overlap
3. `b > radius + Rk`     — no overlap → (0, 0, 0)

Returns `(dose, area, Gyr)` where `Gyr = dose/area` (0 if area=0).

# Example
```julia
dose, area, Gyr = distribute_dose_domain(0.0, 0.0, 5.0, track, irrad_cond, "KC")
```
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

    x_track = track.x - x0
    y_track = track.y - y0
    b    = sqrt(x_track^2 + y_track^2)
    rMax = min(track.Rk, b + radius)

    vprintln("\n--- distribute_dose_domain ---")
    vprintln("center=(", x0, ",", y0, ") radius=", radius,
                "  track=(", track.x, ",", track.y, ") Rk=", track.Rk)
    vprintln("b=", b, "  rMax=", rMax)

    area1 = 0.0; area2 = 0.0; area3 = 0.0
    dose  = 0.0; Gyr   = 0.0

    # ── CASE 1: full impact ──────────────────────────────────────────────────
    if b <= radius
        vprintln("[Case 1] full impact")

        r_intersection = (b + track.Rk < radius) ? track.Rk : (radius - b)
        r_intersection = max(0.0, r_intersection)
        area1 = π * r_intersection^2

        _, integral, _ = integrate_weighted_radial_track(
            0.0, r_intersection, b, radius, 500, type, irrad_cond)
        dose += integral

        if rMax > r_intersection
            area2, integral, _ = integrate_weighted_radial_track(
                r_intersection, rMax, b, radius, 500, type, irrad_cond)
            dose += integral
        end

        if rMax == track.Rk
            if track.Rk > radius - b
                theta1 = acos((b/(2*rMax) + rMax/(2*b) - radius^2/(2*b*rMax)))
                theta2 = acos((b/(2*radius) - rMax^2/(2*b*radius) + radius/(2*b)))
                area3  = π*radius^2 - (theta1*rMax^2 + theta2*radius^2 - rMax*b*sin(theta1))
            else
                area3 = π * (radius^2 - r_intersection^2)
            end
        end

        denom = area1 + area2 + area3
        Gyr   = denom > 0.0 ? dose / denom : 0.0
        vprintln("a1=", area1, " a2=", area2, " a3=", area3, " Gyr=", Gyr)

    # ── CASE 2: partial overlap ──────────────────────────────────────────────
    elseif b <= radius + track.Rk
        vprintln("[Case 2] partial overlap")

        rMin = max(0.0, b - radius)
        step = b <= radius + 0.2 ? 500 : 100

        area2, integral, _ = integrate_weighted_radial_track(
            rMin, rMax, b, radius, step, type, irrad_cond)
        dose = integral

        if rMax == track.Rk
            theta1 = acos((b/(2*rMax) + rMax/(2*b) - radius^2/(2*b*rMax)))
            theta2 = acos((b/(2*radius) - rMax^2/(2*b*radius) + radius/(2*b)))
            area3  = π*radius^2 - (theta1*rMax^2 + theta2*radius^2 - rMax*b*sin(theta1))
        end

        denom = area2 + area3
        Gyr   = denom > 0.0 ? dose / denom : 0.0
        vprintln("a2=", area2, " a3=", area3, " Gyr=", Gyr)

    # ── CASE 3: no overlap ───────────────────────────────────────────────────
    else
        vprintln("[Case 3] no overlap → zero")
        dose = 0.0; Gyr = 0.0
    end

    vprintln("→ dose=", dose, " area=", area1+area2+area3, " Gyr=", Gyr)
    return dose, area1 + area2 + area3, Gyr
end

"""
    integrate_weighted_radial_track(rMin, rMax, b, r_nucleus, step, type, irrad_cond;
                                    verbose=false)
        -> (area, integral, Gyr)

Log-spaced trapezoidal integration of the radial dose kernel over `[rMin, rMax]`.
Integrand: `GetRadialLinearDose(r) * r * arc_intersection_angle(r, b, r_nucleus)`.
`rMin=0` is handled by using `log10(rMin) = -5`. Step clamped to ≥ 3.

Returns `(area, integral, Gyr)` where `Gyr = integral/area` (0 if area=0).

# Example
```julia
area, integral, Gyr = integrate_weighted_radial_track(0.0, 5.0, 2.0, 4.0, 500, "KC", irrad_cond)
```
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

    (rMax <= 0 || rMax <= rMin) && return 0.0, 0.0, 0.0

    log_rMin = rMin > 0 ? log10(rMin) : -5.0
    log_rMax = log10(rMax)
    nSteps   = max(3, step)
    log_step = (log_rMax - log_rMin) / nSteps

    vprintln("integrate rMin=", rMin, " rMax=", rMax, " b=", b, " steps=", nSteps)

    r1          = rMin
    arc_w1      = arc_intersection_angle(r1, b, r_nucleus)
    f1          = GetRadialLinearDose(r1, irrad_cond, type) * r1 * arc_w1
    integral    = 0.0
    area        = 0.0

    for i in 1:nSteps
        r2 = i == nSteps ? rMax : 10^(log_rMin + log_step * i)

        arc_w2 = arc_intersection_angle(r2, b, r_nucleus)
        f2     = GetRadialLinearDose(r2, irrad_cond, type) * r2 * arc_w2
        Δr     = r2 - r1

        integral += Δr * (f1 + f2) * 0.5
        area     += Δr * (arc_w1 * r1 + arc_w2 * r2) * 0.5

        r1 = r2; f1 = f2; arc_w1 = arc_w2
        r2 == rMax && break
    end

    Gyr = area > 0.0 ? integral / area : 0.0
    return area, integral, Gyr
end

"""
    arc_intersection_angle(r, b, r_nucleus; verbose=false) -> Float64

Angular intersection (radians) between a ring at radius `r` around the track
and a circular domain of radius `r_nucleus` offset by impact parameter `b`.

- `r ≤ |b - r_nucleus|` → 2π (ring fully inside domain) or 0 (fully outside)
- `r < b + r_nucleus`   → partial arc: `2*acos((b²+r²-r_nucleus²)/(2br))`
- otherwise             → 0

`arg` is clamped to `[-1, 1]` for numerical stability.

# Example
```julia
θ = arc_intersection_angle(3.0, 2.0, 4.0)
```
"""
function arc_intersection_angle(
    r::Float64,
    b::Float64,
    r_nucleus::Float64;
    verbose::Bool = false
)
    vprintln(args...) = (verbose ? println(args...) : nothing)
    vprintln("arc r=", r, " b=", b, " r_nucleus=", r_nucleus)

    if r <= abs(b - r_nucleus)
        θ = b < r_nucleus ? 2π : 0.0
        vprintln("[Case 1] fully ", b < r_nucleus ? "inside" : "outside", " → ", θ)
        return θ
    end

    if r < b + r_nucleus
        arg = (b^2 + r^2 - r_nucleus^2) / (2 * b * r)
        θ   = 2 * acos(clamp(arg, -1.0, 1.0))
        vprintln("[Case 2] partial arc → ", θ)
        return θ
    end

    vprintln("[Case 3] no intersection → 0")
    return 0.0
end

"""
    GetRadialLinearDose(r::Float64, ion::AT, type::String; verbose=false) -> Float64

Radial linear dose at distance `r` from track center.

- `"KC"` model: core region (`r ≤ Rc`) uses LET-based constant; penumbra uses `Kp/r²`.
  Includes relativistic β, effective charge z_eff, and LET conversion (MeV/µm → keV/µm).
- `"LEM"` model: core and penumbra normalized by `log(Rp/Rc)`.

# Example
```julia
D = GetRadialLinearDose(0.5, irrad_cond, "KC")
D = GetRadialLinearDose(2.0, irrad_cond, "LEM")
```
"""
function GetRadialLinearDose(r::Float64, ion::AT, type::String; verbose::Bool = false)
    vprintln(args...) = (verbose ? println(args...) : nothing)

    if type == "KC"
        AMU2MEV = 931.494027

        β         = sqrt(1 - 1 / ((ion.E / AMU2MEV + 1)^2))
        r_core    = ion.Rc
        z_eff     = ion.Z * (1 - exp(-125 * β / ion.Z^(2.0/3.0)))
        Kp        = 1.25 * 0.0001 * (z_eff / β)^2
        LETk      = ion.LET * 0.1602
        r_penumbra = 0.0616 * (ion.E / ion.A)^1.7

        vprintln("KC: β=", β, " z_eff=", z_eff, " Kp=", Kp, " r_core=", r_core)

        if r <= r_core
            D_arc  = LETk - 2π * Kp * log(r_penumbra / r_core)
            D_arc /= π * r_core^2
            D_arc *= 1.05
            vprintln("  [Core]     r=", r, " → D=", D_arc)
        else
            D_arc = Kp / (r * r)
            vprintln("  [Penumbra] r=", r, " → D=", D_arc)
        end
        return D_arc

    elseif type == "LEM"
        Rc   = ion.Rc
        Rp   = ion.Rp
        LETk = ion.LET * 0.1602
        norm = LETk / (1 + 2 * log(Rp / Rc))

        vprintln("LEM: Rc=", Rc, " Rp=", Rp, " norm=", norm)

        if r <= Rc
            D_arc  = norm / (π * Rc^2)
            D_arc *= 1.05
            vprintln("  [Core]     r=", r, " → D=", D_arc)
        else
            D_arc = norm / (π * r * r)
            vprintln("  [Penumbra] r=", r, " → D=", D_arc)
        end
        return D_arc

    else
        error("GetRadialLinearDose: unknown model '$type'. Expected \"KC\" or \"LEM\".")
    end
end
