# ============================================================================
#! utilities_oxygen.jl
#
#  Analytic (quasi-steady-state) oxygen diffusion for tumour spheroids and
#  dynamic re-oxygenation driven by the live-cell distribution.
#
#  The oxygen profile C(r) is the steady-state solution of the reaction-
#  diffusion equation
#
#        D ∇²C(r) = ρ_cell(r) · q0
#
#  in spherical symmetry, with zero-order consumption (Grimes et al. 2014).
#
#  UNITS: all oxygen values are in % O2.
#         (1% O2 = 7.6 mmHg at 1 atm; convert for OER calls if needed.)
#
#  CALIBRATION of q0:
#    q0 is the volumetric oxygen consumption rate at unit cell-packing density
#    (ρ = 1). It is derived from Freyer & Sutherland (1985) EMT6/Ro in-situ
#    measurements, converted via the Grimes (2014) parameterisation:
#
#      a  = 1.27e-6 m³ kg⁻¹ s⁻¹   (Grimes OCR, reduced ÷ 2.5 per Freyer &
#                                    Sutherland's observation that large-spheroid
#                                    in-situ consumption is ~2.5× lower than
#                                    single-cell rates)
#      Ω  = 3.0318e7 mmHg kg m⁻³   (Henry's law constant)
#      A  = a·Ω = 38.50/2.5 = 15.40 mmHg/s
#         = 15.40 / 7.6 = 2.03 %/s
#
#    At unit density (ρ=1): q0 = A = 2.03 %/s.
#    Predicted viable rim: ~130–155 µm for spheroids of radius 250–600 µm.
#    Necrosis onset radius: ~204 µm  (spheroids smaller than this are fully
#    oxygenated, which is correct for a 50 µm test spheroid).
#
#  References:
#    Grimes D.R. et al. (2014) J R Soc Interface 11:20131124
#    Freyer J.P. & Sutherland R.M. (1985) J Cell Physiol 124:516–524
#
#! FUNCTION INDEX
#?  oxygen_profile_grimes(R, ρ; D, q0, C_rim, C_floor) -> (C_of_r, r_necrotic)
#?  necrotic_radius(R, ρ; D, q0, C_rim)               -> r_necrotic
#?  radial_cell_density(cell_df; nbins, center, ...)   -> (edges, ρ_shell, center, Rmax)
#?  set_oxygen_diffusion!(cell_df; kwargs...)          -> :O column (% O2, analytic)
#?  set_oxygen_zones!(cell_df; kwargs...)              -> :O column (% O2, 3-zone discrete)
#?  update_oxygen!(cell_df; kwargs...)                 -> recompute :O in place (dynamic)
# ============================================================================

#~ ------------------------------------------------------------------------
#~ Physical constants and default parameters  (all oxygen in % O2)
#~ ------------------------------------------------------------------------

#?  OXY_D_DEFAULT — O2 diffusion coefficient in tissue (µm² s⁻¹)
const OXY_D_DEFAULT   = 2.0e3    # ~2000 µm²/s (Grimes 2014, water-like tissue)

#?  OXY_Q0_DEFAULT — volumetric O2 consumption at unit packing density (% s⁻¹)
#  Derived from Freyer & Sutherland (1985) / Grimes (2014):
#    a = 1.27e-6/2.5 m³ kg⁻¹ s⁻¹, Ω = 3.0318e7 mmHg kg m⁻³
#    A = aΩ/7.6 = 2.03 % s⁻¹
#  Gives necrosis onset at R ≈ 204 µm; viable rim ≈ 130–155 µm.
const OXY_Q0_DEFAULT  = 2.03     # % s⁻¹  (Freyer & Sutherland / Grimes)

#?  OXY_RIM_DEFAULT — surface oxygen level (% O2, physioxic culture medium)
const OXY_RIM_DEFAULT = 7.0      # %  (≈ 53 mmHg)

#?  OXY_CORE_FLOOR — floor oxygen in anoxic core (% O2, chronic hypoxia)
const OXY_CORE_FLOOR  = 0.1     # %  (≈ 0.76 mmHg)


#~ ------------------------------------------------------------------------
#?  necrotic_radius — radius below which C(r) reaches the anoxic floor
#
#  The no-flux / zero-oxygen condition at r_n gives the transcendental eq:
#        C_rim = (A/6D) [ R² − 3 r_n² + 2 r_n³/R ]
#  solved by bisection. Returns 0 if the spheroid is too small to deplete O2.
#~ ------------------------------------------------------------------------
function necrotic_radius(R::Float64, ρ::Float64;
                         D::Float64     = OXY_D_DEFAULT,
                         q0::Float64    = OXY_Q0_DEFAULT,
                         C_rim::Float64 = OXY_RIM_DEFAULT)

    A = ρ * q0
    A <= 0 && return 0.0

    g(rn) = (A / (6D)) * (R^2 - 3rn^2 + 2rn^3 / R) - C_rim

    # centre value: if ≥ 0 no necrosis
    (C_rim - A * R^2 / (6D)) >= 0.0 && return 0.0

    lo, hi = 0.0, R
    glo = g(lo)
    glo * g(hi) > 0 && return glo > 0 ? R : 0.0

    for _ in 1:100
        mid = 0.5 * (lo + hi)
        gm  = g(mid)
        (abs(gm) < 1e-9 || (hi - lo) < 1e-6) && return mid
        if glo * gm <= 0
            hi = mid
        else
            lo, glo = mid, gm
        end
    end
    return 0.5 * (lo + hi)
end


#~ ------------------------------------------------------------------------
#?  oxygen_profile_grimes — closed-form radial O2 profile [% O2]
#
#  Returns (C_of_r::Function, r_necrotic::Float64).
#  C_of_r(r) gives the % O2 at radius r from the spheroid centre.
#~ ------------------------------------------------------------------------
function oxygen_profile_grimes(R::Float64, ρ::Float64;
                               D::Float64       = OXY_D_DEFAULT,
                               q0::Float64      = OXY_Q0_DEFAULT,
                               C_rim::Float64   = OXY_RIM_DEFAULT,
                               C_floor::Float64 = OXY_CORE_FLOOR)

    A   = ρ * q0
    r_n = necrotic_radius(R, ρ; D=D, q0=q0, C_rim=C_rim)

    function C_of_r(r::Float64)
        r >= R && return C_rim
        if r_n <= 0.0
            return max(C_rim - (A / (6D)) * (R^2 - r^2), C_floor)
        end
        r <= r_n && return C_floor
        C = C_rim - (A / (6D)) * (R^2 - r^2) -
            (A * r_n^3 / (3D)) * (1.0 / R - 1.0 / r)
        return max(C, C_floor)
    end

    return C_of_r, r_n
end


#~ ------------------------------------------------------------------------
#?  radial_cell_density — live-cell packing density per radial shell
#
#  Returns (edges, ρ_shell, center, Rmax).
#  pos_cols: tuple of three column symbols for x, y, z positions in cell_df.
#~ ------------------------------------------------------------------------
function radial_cell_density(cell_df;
                             nbins::Int      = 40,
                             center          = nothing,
                             R_cell::Float64 = 15.0,
                             pos_cols        = (:x, :y, :z))

    cols = collect(pos_cols)
    for c in cols
        @assert hasproperty(cell_df, c) "radial_cell_density: missing column $c; pass pos_cols=(...)"
    end

    alive = cell_df.is_cell .== 1
    xs = cell_df[alive, cols[1]]
    ys = cell_df[alive, cols[2]]
    zs = cell_df[alive, cols[3]]

    if center === nothing
        cx = sum(xs)/length(xs); cy = sum(ys)/length(ys); cz = sum(zs)/length(zs)
    else
        cx, cy, cz = center
    end

    radii = sqrt.((xs .- cx).^2 .+ (ys .- cy).^2 .+ (zs .- cz).^2)
    Rmax  = isempty(radii) ? 0.0 : maximum(radii)
    Rmax <= 0 && return (Float64[0.0], Float64[0.0], (cx, cy, cz), 0.0)

    edges  = collect(range(0.0, Rmax; length = nbins + 1))
    ρshell = zeros(Float64, nbins)
    v_cell = (4/3) * π * R_cell^3

    for i in 1:nbins
        r_in, r_out   = edges[i], edges[i+1]
        n_in_shell    = count(r -> (r >= r_in) && (r < r_out), radii)
        v_shell       = (4/3) * π * (r_out^3 - r_in^3)
        ρshell[i]     = v_shell > 0 ? clamp(n_in_shell * v_cell / v_shell, 0.0, 1.0) : 0.0
    end

    return edges, ρshell, (cx, cy, cz), Rmax
end


#~ ------------------------------------------------------------------------
#?  set_oxygen_diffusion! — write analytic Grimes O2 profile to :O column
#
#  Drop-in replacement for set_oxygen!.  Writes % O2 to cell_df.O.
#
#  density = :mean   → uniform ρ across spheroid (one closed-form profile)
#  density = :radial → per-shell ρ, profile evaluated locally (cheap, graded)
#~ ------------------------------------------------------------------------
function set_oxygen_diffusion!(cell_df;
                               D::Float64        = OXY_D_DEFAULT,
                               q0::Float64       = OXY_Q0_DEFAULT,
                               C_rim::Float64    = OXY_RIM_DEFAULT,
                               C_floor::Float64  = OXY_CORE_FLOOR,
                               R_cell::Float64   = 15.0,
                               density::Symbol   = :mean,
                               nbins::Int        = 40,
                               pos_cols          = (:x, :y, :z),
                               plot_oxygen::Bool = false,
                               verbose::Bool     = false)

    edges, ρshell, center, Rmax =
        radial_cell_density(cell_df; nbins=nbins, R_cell=R_cell, pos_cols=pos_cols)
    cx, cy, cz = center

    hasproperty(cell_df, :O) || (cell_df.O = fill(C_rim, nrow(cell_df)))

    alive   = cell_df.is_cell .== 1
    n_alive = count(alive)
    Reff    = max(Rmax, R_cell)
    ρ_mean  = n_alive > 0 ?
        clamp(n_alive * (4/3)*π*R_cell^3 / ((4/3)*π*Reff^3), 0.0, 1.0) : 0.0

    cols = collect(pos_cols)

    if density == :mean
        C_of_r, r_n = oxygen_profile_grimes(Reff, ρ_mean;
                          D=D, q0=q0, C_rim=C_rim, C_floor=C_floor)
        @inbounds for i in 1:nrow(cell_df)
            cell_df.is_cell[i] == 1 || continue
            r = sqrt((cell_df[i, cols[1]] - cx)^2 +
                     (cell_df[i, cols[2]] - cy)^2 +
                     (cell_df[i, cols[3]] - cz)^2)
            cell_df.O[i] = C_of_r(r)
        end
        if verbose
            println("set_oxygen_diffusion! [mean]")
            println("  R        = $(round(Rmax,   digits=1)) µm")
            println("  ρ_mean   = $(round(ρ_mean, digits=3))")
            println("  q0       = $q0 %/s  (Freyer & Sutherland / Grimes)")
            println("  r_n      = $(round(r_n,    digits=1)) µm  (necrotic radius)")
            println("  rim      = $(round(Rmax-r_n,digits=1)) µm  (viable rim)")
            println("  O2 range = $(round(minimum(cell_df.O[alive]),digits=2)) – " *
                    "$(round(maximum(cell_df.O[alive]),digits=2)) %")
        end

    elseif density == :radial
        @inbounds for i in 1:nrow(cell_df)
            cell_df.is_cell[i] == 1 || continue
            r = sqrt((cell_df[i, cols[1]] - cx)^2 +
                     (cell_df[i, cols[2]] - cy)^2 +
                     (cell_df[i, cols[3]] - cz)^2)
            b      = clamp(searchsortedlast(edges, r), 1, length(ρshell))
            C_of_r, _ = oxygen_profile_grimes(Reff, ρshell[b];
                           D=D, q0=q0, C_rim=C_rim, C_floor=C_floor)
            cell_df.O[i] = C_of_r(r)
        end
        verbose && println("set_oxygen_diffusion! [radial]: R=$(round(Rmax,digits=1)) µm, " *
                           "ρ_mean=$(round(ρ_mean,digits=3))")
    else
        error("set_oxygen_diffusion!: unknown density=$density. Use :mean or :radial.")
    end

    plot_oxygen && _plot_oxygen_profile(cell_df, center, cols)
    return cell_df
end


#~ ------------------------------------------------------------------------
#?  set_oxygen_zones! — three-zone discrete O2 assignment [% O2]
#
#  Assigns one of three O2 levels based on fractional distance from centre:
#    r < core_frac·Rmax              → anoxic core  (O2_anoxic)
#    r > (1-rim_frac)·Rmax          → normoxic rim  (O2_normoxic)
#    otherwise                       → hypoxic shell (O2_hypoxic)
#~ ------------------------------------------------------------------------
function set_oxygen_zones!(cell_df;
                           O2_normoxic::Float64 = OXY_RIM_DEFAULT,  # 7.0 %
                           O2_hypoxic::Float64  = 1.3,              # % (≈10 mmHg)
                           O2_anoxic::Float64   = OXY_CORE_FLOOR,   # 0.1 %
                           core_frac::Float64   = 0.35,
                           rim_frac::Float64    = 0.30,
                           pos_cols             = (:x, :y, :z),
                           plot_oxygen::Bool    = false,
                           verbose::Bool        = false)

    O2_anoxic <= O2_hypoxic <= O2_normoxic ||
        error("set_oxygen_zones!: require O2_anoxic ≤ O2_hypoxic ≤ O2_normoxic")
    (0.0 < core_frac && 0.0 < rim_frac && core_frac + rim_frac < 1.0) ||
        error("set_oxygen_zones!: core_frac and rim_frac must be positive and sum < 1")

    cols = collect(pos_cols)
    for c in cols
        @assert hasproperty(cell_df, c) "set_oxygen_zones!: missing column $c"
    end

    alive = cell_df.is_cell .== 1
    xs = cell_df[alive, cols[1]]; ys = cell_df[alive, cols[2]]; zs = cell_df[alive, cols[3]]
    isempty(xs) && error("set_oxygen_zones!: no alive cells found")
    cx = sum(xs)/length(xs); cy = sum(ys)/length(ys); cz = sum(zs)/length(zs)
    Rmax = maximum(sqrt.((xs.-cx).^2 .+ (ys.-cy).^2 .+ (zs.-cz).^2))

    r_core = core_frac * Rmax
    r_rim  = (1.0 - rim_frac) * Rmax

    hasproperty(cell_df, :O) || (cell_df.O = fill(O2_normoxic, nrow(cell_df)))

    @inbounds for i in 1:nrow(cell_df)
        cell_df.is_cell[i] == 1 || continue
        r = sqrt((cell_df[i, cols[1]] - cx)^2 +
                 (cell_df[i, cols[2]] - cy)^2 +
                 (cell_df[i, cols[3]] - cz)^2)
        cell_df.O[i] = r < r_core ? O2_anoxic : r > r_rim ? O2_normoxic : O2_hypoxic
    end

    if verbose
        n_a = count(i -> cell_df.is_cell[i]==1 &&
            sqrt((cell_df[i,cols[1]]-cx)^2+(cell_df[i,cols[2]]-cy)^2+(cell_df[i,cols[3]]-cz)^2) < r_core,
            1:nrow(cell_df))
        n_n = count(i -> cell_df.is_cell[i]==1 &&
            sqrt((cell_df[i,cols[1]]-cx)^2+(cell_df[i,cols[2]]-cy)^2+(cell_df[i,cols[3]]-cz)^2) > r_rim,
            1:nrow(cell_df))
        println("set_oxygen_zones!: Rmax=$(round(Rmax,digits=1)) µm | " *
                "anoxic=$n_a ($(O2_anoxic)%) | hypoxic=$(count(alive)-n_a-n_n) ($(O2_hypoxic)%) | " *
                "normoxic=$n_n ($(O2_normoxic)%)")
    end

    plot_oxygen && _plot_oxygen_profile(cell_df, (cx,cy,cz), cols)
    return cell_df
end


#~ ------------------------------------------------------------------------
#?  update_oxygen! — recompute O2 after population changes (reoxygenation)
#
#  Call before each irradiation fraction or every Δt hours in the ABM.
#  As cells die and ρ_cell drops, the necrotic boundary recedes and
#  previously hypoxic cells reoxygenate — the 4th R of radiobiology.
#~ ------------------------------------------------------------------------
function update_oxygen!(cell_df; kwargs...)
    return set_oxygen_diffusion!(cell_df; kwargs...)
end


#~ ------------------------------------------------------------------------
#?  _plot_oxygen_profile — two-panel diagnostic figure [% O2]
#
#  Left  — normalised histogram + KDE of per-cell O2 values
#  Right — O2 (%) vs radial distance from spheroid centre
#~ ------------------------------------------------------------------------
function _plot_oxygen_profile(cell_df, center, cols)
    cx, cy, cz = center
    alive = cell_df.is_cell .== 1
    r = sqrt.((cell_df[alive, cols[1]] .- cx).^2 .+
              (cell_df[alive, cols[2]] .- cy).^2 .+
              (cell_df[alive, cols[3]] .- cz).^2)
    o = cell_df[alive, :O]

    # left panel — O2 distribution
    p_dist = histogram(o;
                       bins      = 40,
                       normalize = :pdf,
                       xlabel    = "O₂ (%)",
                       ylabel    = "density",
                       title     = "O₂ distribution",
                       label     = "histogram",
                       color     = :seagreen,
                       alpha     = 0.45,
                       linecolor = :seagreen)
    try
        density!(p_dist, o; label = "kde", color = :darkgreen, linewidth = 2)
    catch; end

    # right panel — O2 vs radius
    p_prof = scatter(r, o;
                     xlabel     = "radius (µm)",
                     ylabel     = "O₂ (%)",
                     title      = "Radial O₂ profile",
                     markersize = 2, markerstrokewidth = 0, alpha = 0.4,
                     label      = "")

    p = plot(p_dist, p_prof; layout = (1,2), size = (1000, 400))
    display(p)
    return p
end
