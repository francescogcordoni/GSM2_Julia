# ============================================================================
# spheroid_common.jl — shared setup for spheroid_regrowth.jl and spheroid_lineage.jl
#
# • builds ONE lattice/population (same geometry and initial phases for every run)
# • track-structure dose (MC_dose_CPU!) computed ONCE per (ion, dose)
# • lesions resampled from that dose for every replicate (MC_loop_damage!)
# Oxygen: uniform normoxia (oxygen effects are treated in a separate paper).
# ============================================================================
using Base.Threads
using CSV, DataFrames, Distributions, Random, Statistics, StatsBase, Printf
using JLD2, DelimitedFiles, InlineStrings, ProgressMeter
using Plots, StatsPlots

include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ── GSM2 parameters (phase-specific) ────────────────────────────────────────
const RD, RN = 0.8, 7.2
gsm2_cycle = [GSM2(2.780479661191086,  0.012872261720543399, 0.04029756109753225,  RD, RN),  # G1/G0
              GSM2(5.84009601901114,   0.00589118894714544,  0.05794352736120672,  RD, RN),  # S
              GSM2(1.7720064637774506, 0.024306291709970018, 5.704688326522623e-5, RD, RN),  # G2/M
              GSM2(2.5656972960759896, 0.01481379648786136,  0.012663276476522422, RD, RN)]  # mean
setup_GSM2!(gsm2_cycle[4].r, gsm2_cycle[4].a, gsm2_cycle[4].b, RD, RN)

# ── Geometry defaults ───────────────────────────────────────────────────────
const X_VOXEL     = 800.0
const R_CELL      = 15.0
const TARGET_GEOM = "circle"
const CALC_TYPE   = "full"
const TYPE_AT     = "KC"
const TRACK_SEG   = true
const O2_UNIFORM  = 21.0
const QUIESCENCE_K = 0      # G0 when ≤ K of 26 neighbours are empty (0 = original rule)

G(name::Symbol) = Base.invokelatest(getfield, Main, name)     # globals set by setup_*!

"""Lattice + population built once; returns the pristine cell_df (uniform O2)."""
function build_spheroid(tumor_radius::Float64, X_box::Float64; seed::Integer = 1)
    Random.seed!(seed)
    N_sideVox   = Int(floor(2 * X_box / X_VOXEL))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_CELL)))
    setup_IonIrrad!(1.0, 80.0, "1H")
    calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_box, X_VOXEL, tumor_radius)
    setup_cell_lattice!(TARGET_GEOM, X_box, R_CELL, N_sideVox, N_CellsSide;
                        ParIrr = "false", track_seg = TRACK_SEG)
    setup_cell_population!(TARGET_GEOM, X_box, R_CELL, N_sideVox, N_CellsSide, gsm2_cycle[4])
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, G(:cell_df), TRACK_SEG)
    df = G(:cell_df)
    set_oxygen!(df; plot_oxygen = false)
    df.O .= O2_UNIFORM
    set_quiescence_threshold!(QUIESCENCE_K; cell_df = df)
    pristine = deepcopy(df)
    alive = pristine.is_cell .== 1
    @printf("spheroid: %d cells | G0 %.0f%% | lattice sites %d\n", count(alive),
            100 * count(==("G0"), pristine.cell_cycle[alive]) / count(alive), nrow(pristine))
    return pristine
end

"""
Track-structure dose for one (ion, energy, dose) on a copy of the pristine
spheroid. The ion globals stay set for this ion, so call `sample_damage!` for
all its replicates before moving to the next ion or dose.
"""
function dose_copy(E::Float64, particle::String, dose::Float64, pristine::DataFrame;
                   tumor_radius::Float64, X_box::Float64)
    df = deepcopy(pristine)
    dose <= 0 && return df
    setup_IonIrrad!(dose, E, particle)
    R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_box, X_VOXEL, tumor_radius)
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, G(:cell_df), TRACK_SEG)
    Npar = round(Int, G(:irrad).dose / (1.602e-9 * G(:LET)) * π * R_beam^2 * 1e-8)
    @printf("  [%s %.0f MeV/u, %.1f Gy] Npar=%d\n", particle, E, dose, Npar)
    MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, G(:df_center_x), G(:df_center_y),
                 G(:at), gsm2_cycle, TYPE_AT, TRACK_SEG)
    return df
end

"""Fresh lesions from the stored dose (call right after the matching dose_copy)."""
function sample_damage!(df::DataFrame, dose::Float64)
    dose <= 0 && return df
    MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)
    return df
end

"""Alive and not doomed."""
clonogenic(df::DataFrame) = count((df.is_cell .== 1) .& .!isfinite.(df.death_time))
clonogenic(pop::CellPopulation) = count((pop.is_cell .== 1) .& .!isfinite.(pop.death_time))

"""Wilson 95% interval for k successes in n."""
function wilson(k::Integer, n::Integer; z = 1.96)
    n == 0 && return (NaN, NaN, NaN)
    p = k / n; d = 1 + z^2 / n
    c = (p + z^2 / (2n)) / d; h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))
end
