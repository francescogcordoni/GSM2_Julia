# ============================================================================
# scripts/spheroid_fractionation_oxygen.jl
#
# Fractionated irradiation (3 × 2 Gy, 24 h apart) of a LARGE spheroid, with the
# oxygen profile recomputed on the current spheroid right before every fraction,
# compared with the same spheroid at uniform oxygen. 1H 100 MeV and 12C 80 MeV/u.
#
#   oxygen = :diffusion   quasi-steady reaction–diffusion profile of the current
#                         spheroid (radius, centre and local cell density per shell,
#                         so that the space freed by dying cells reoxygenates the core)
#   oxygen = :uniform     every cell at the rim value O2_UNIFORM: same spheroid,
#                         same irradiation, no hypoxia → isolates the oxygen effect
#
# The spheroid radius at the first fraction (~400 µm) is well above the necrosis-onset
# radius R* ≈ 206 µm, so the core is hypoxic. Each realisation grows the spheroid ONCE
# for T_PRE hours and every condition starts from a copy of it.
#
# Outputs (scripts/output/spheroid_fractionation_oxygen/):
#   hourly.csv                 cell numbers per phase every hour (and right after each fraction)
#   fractions.csv              per fraction: dose, survival, oxygen statistics (hypoxic fractions)
#   cells_<ion>_<oxygen>_f<k>.csv   realisation 1: every cell at fraction k (position, O2, dose,
#                              survival probability, doomed) for oxygen/survival-vs-radius figures
#
# HOW TO RUN
#   julia --threads auto scripts/spheroid_fractionation_oxygen.jl
# then: julia results/plot_fractionation_oxygen.jl
# ============================================================================

using Base.Threads, Distributed, CSV, DataFrames, Distributions, Random, Plots, StatsPlots
using ProgressBars, GLM, JLD2, DelimitedFiles, Statistics, StatsBase, Optim, LsqFit
using ProgressMeter, InlineStrings, CUDA, SparseArrays, Printf
using Statistics: mean

println("Threads available: ", nthreads())
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS
# ============================================================================
const T_PRE   = 24.0        # h of growth before the first fraction
const FX_DOSE = 2.0         # Gy per fraction
const N_FX    = 3
const FX_GAP  = 24.0        # h between fractions
const T_AFTER = 72.0        # h after the last fraction
const N_REP   = 3           # stochastic realisations

const IONS = [(tag = "1H_100MeV",  particle = "1H",  E = 100.0),
              (tag = "12C_80MeVu", particle = "12C", E = 80.0)]
const OXYGEN_MODES = [:diffusion, :uniform]
const O2_UNIFORM   = 7.0         # % O2: the rim value of the profile (physioxic)
const DENSITY      = :radial     # local cell density per shell (dead cells free oxygen)
const CONSUME_DOOMED = true      # doomed cells keep consuming oxygen until removed
const HYPOXIC_PCT  = 1.0         # % O2 below which a cell is counted as hypoxic (~7.6 mmHg)
const ANOXIC_PCT   = 0.2         # % O2 below which a cell is counted as (nearly) anoxic

# Large spheroid (necrotic/hypoxic core) in a box with room for 168 h of growth
TUMOR_RADIUS = 400.0
R_CELL       = 15.0
X_BOX        = 800.0
X_VOXEL      = 1600.0

RD = 0.8; RN = 7.2
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

const TYPE_AT     = "KC"
const TRACK_SEG   = true
const TARGET_GEOM = "circle"
const CALC_TYPE   = "full"
const NAT_APO     = 1e-10
const FATE        = :delayed

const OUTDIR = joinpath(@__DIR__, "output", "spheroid_fractionation_oxygen")
mkpath(OUTDIR)

"""
Oxygen of the CURRENT spheroid. :diffusion recomputes the profile from the alive
cells (centre, radius and, with DENSITY = :radial, the local density per shell);
if CONSUME_DOOMED is false, doomed cells are left out of the consumption.
"""
function oxygenate!(df, mode::Symbol; verbose = false)
    if mode === :uniform
        df.O[df.is_cell .== 1] .= O2_UNIFORM
        return df
    end
    masked = Int[]
    if !CONSUME_DOOMED
        masked = findall((df.is_cell .== 1) .& isfinite.(df.death_time))
        df.is_cell[masked] .= 0
    end
    set_oxygen_diffusion!(df; density = DENSITY, C_rim = O2_UNIFORM, plot_oxygen = false, verbose = verbose)
    if !isempty(masked)
        df.is_cell[masked] .= 1
        df.O[masked] .= minimum(df.O[df.is_cell .== 1])     # irrelevant: already doomed
    end
    return df
end

# ============================================================================
# 2. SETUP — identical to example/spheroid.jl (top level)
# ============================================================================
# ============================================================================

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(R_G1, A_G1, B_G1, RD, RN)
gsm2_cycle[2] = GSM2(R_S,  A_S,  B_S,  RD, RN)
gsm2_cycle[3] = GSM2(R_G2, A_G2, B_G2, RD, RN)
gsm2_cycle[4] = GSM2(R_AVG, A_AVG, B_AVG, RD, RN)
setup_GSM2!(R_AVG, A_AVG, B_AVG, RD, RN)

setup_IonIrrad!(FX_DOSE, 100.0, "1H"; type_AT = TYPE_AT)
N_sideVox   = Int(floor(2 * X_BOX / X_VOXEL))
N_CellsSide = 2 * convert(Int64, floor(X_BOX / (2 * R_CELL)))
setup_cell_lattice!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide;
                    ParIrr = "false", track_seg = TRACK_SEG, full_cycle = true)
@eval Main begin
    tumor_radius = $TUMOR_RADIUS
    X_voxel      = $X_VOXEL
    X_box        = $X_BOX
end
setup_cell_population!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide,
                       gsm2_cycle[4]; ParIrr = "false")
setup_irrad_conditions!(ion, irrad, TYPE_AT, cell_df, TRACK_SEG)
oxygenate!(cell_df, :diffusion; verbose = true)

# Fate model: set_fate_model! exists only in the latest utilities_abm.jl
if isdefined(Main, :set_fate_model!)
    set_fate_model!(FATE)
else
    @warn "utilities_abm.jl has no set_fate_model!: compute_times_domain! is used as it is. " *
          "For the delayed fate model use the latest utilities_abm.jl."
end
hasproperty(cell_df, :death_type)  || (cell_df.death_type  = fill(-1, nrow(cell_df)))
hasproperty(cell_df, :dam_X_total) || (cell_df.dam_X_total = zeros(Int, nrow(cell_df)))
hasproperty(cell_df, :dam_Y_total) || (cell_df.dam_Y_total = zeros(Int, nrow(cell_df)))

const N_INIT = count(cell_df.is_cell .== 1)
println("Spheroid: $N_INIT cells")
print_phase_distribution(cell_df; label = "pre-irradiation")
cell_df_pristine = deepcopy(cell_df)

# domain tables of the whole lattice (built by setup_cell_population!, cover every site)
const DFX = df_center_x
const DFY = df_center_y
const AT0 = at

# ============================================================================

# ============================================================================
# 3. HELPERS
# ============================================================================
G(name::Symbol) = Base.invokelatest(getfield, Main, name)

# Advance running timers by an idle interval (fallback if utilities_abm.jl lacks it)
if !isdefined(Main, :_advance_idle!)
    function _advance_idle!(cell_df::DataFrame, δ::Float64)
        δ <= 0 && return nothing
        for col in (cell_df.death_time, cell_df.cycle_time, cell_df.recover_time)
            @inbounds for i in eachindex(col)
                cell_df.is_cell[i] == 1 && isfinite(col[i]) && (col[i] = max(0.0, col[i] - δ))
            end
        end
        return nothing
    end
end

"""Counts on the cell DataFrame (alive cells only): doomed = finite death_time."""
function counts(df::DataFrame)
    alive  = df.is_cell .== 1
    doomed = alive .& isfinite.(df.death_time)
    clono  = alive .& .!isfinite.(df.death_time)
    ph = df.cell_cycle[clono]
    return (total = count(alive), clonogenic = count(clono), doomed = count(doomed),
            G0 = count(==("G0"), ph), G1 = count(==("G1"), ph), S = count(==("S"), ph),
            G2 = count(==("G2"), ph), M = count(==("M"), ph))
end

"""
ABM for `seg` hours from absolute time t0, one hour at a time, counting the cells
on `df` after every hour (no snapshots involved). Appends one row per hour.
"""
function run_segment!(df, seg::Int, t0::Float64, rows::DataFrame, tag::String, rep::Int;
                      final::Bool)
    for h in 1:seg
        ts, _ = run_simulation_abm!(df; nat_apo = NAT_APO, terminal_time = 1.0,
                                    snapshot_times = Int[], verbose = false,
                                    return_dataframes = false, update_input = true)
        t_last = isempty(ts.time) ? 0.0 : ts.time[end]
        _advance_idle!(df, 1.0 - t_last)          # timers relative to the end of the hour
        push!(rows, merge((tag = tag, rep = rep, t = t0 + h), counts(df)))
    end
    return t0 + seg
end

clonogenic(df) = count((df.is_cell .== 1) .& .!isfinite.(df.death_time))

"""Spheroid radius now (µm), from the alive cells."""
current_radius(df) = (a = df.is_cell .== 1;
    maximum(sqrt.(df.x[a] .^ 2 .+ df.y[a] .^ 2 .+ df.z[a] .^ 2)) + R_CELL)

"""
Box check. The lattice is a CUBE with periodic neighbours, so what matters is
whether cells sit on the outermost layer of sites (|x|, |y| or |z| at the face),
not their distance from the centre (corners are far from the faces).
Returns (cells on the outermost layer, max radius, 95th-percentile radius).
"""
function box_check(df)
    a = findall(==(1), df.is_cell)
    face = X_BOX - R_CELL - 1e-6                      # centre of the outermost site layer
    edge = count(i -> max(abs(df.x[i]), abs(df.y[i]), abs(df.z[i])) >= face, a)
    r = sqrt.(df.x[a] .^ 2 .+ df.y[a] .^ 2 .+ df.z[a] .^ 2)
    return edge, maximum(r), quantile(r, 0.95)
end

"""ABM for Δ hours from t0, one hour at a time, recording the counts every hour."""
function run_for!(df, Δ::Float64, t0::Float64, rows::DataFrame, meta)
    nh = round(Int, Δ)
    for h in 1:nh
        ts, _ = run_simulation_abm!(df; nat_apo = NAT_APO, terminal_time = 1.0, snapshot_times = Int[],
                                    verbose = false, return_dataframes = false, update_input = true)
        _advance_idle!(df, 1.0 - (isempty(ts.time) ? 0.0 : ts.time[end]))
        push!(rows, merge(meta, (t = t0 + h,), counts(df)))
    end
    return t0 + nh
end

"""
One fraction on the current spheroid. The oxygen profile is recomputed on the
current spheroid immediately before the dose (it sets the OER of every cell).
"""
function irradiate_fraction!(df::DataFrame, ion_, mode::Symbol, seed::UInt64)
    oxygenate!(df, mode)
    a0 = findall(==(1), df.is_cell)
    O = df.O[a0]
    setup_IonIrrad!(FX_DOSE, ion_.E, ion_.particle; type_AT = TYPE_AT)
    Rnow = current_radius(df)
    R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, Rnow)
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
    Npar = round(Int, G(:irrad).dose / (1.602e-9 * G(:LET)) * π * R_beam^2 * 1e-8)
    MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, DFX, DFY, deepcopy(AT0),
                 gsm2_cycle, TYPE_AT, TRACK_SEG; seed = seed)
    Random.seed!(seed)
    MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)
    for i in a0                                        # GSM2 survival without resetting timers
        ph = df.cell_cycle[i]
        g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
        df.sp[i] = domain_GSM2(df.dam_X_dom[i], df.dam_Y_dom[i], g)
    end
    live = [i for i in a0 if !isfinite(df.death_time[i])]       # not doomed by an earlier fraction
    hyp  = [i for i in live if df.O[i] < HYPOXIC_PCT]
    oxi  = [i for i in live if df.O[i] >= HYPOXIC_PCT]
    stats = (Npar = Npar, R_spheroid = Rnow, mean_dose = mean(df.dose_cell[a0]),
             mean_sp = isempty(live) ? NaN : mean(df.sp[live]),
             sp_hypoxic = isempty(hyp) ? NaN : mean(df.sp[hyp]),
             sp_oxic = isempty(oxi) ? NaN : mean(df.sp[oxi]),
             mean_O2 = mean(O), frac_hypoxic = mean(O .< HYPOXIC_PCT), frac_anoxic = mean(O .< ANOXIC_PCT))
    compute_times_domain!(df, gsm2_cycle; nat_apo = NAT_APO, terminal_time = Inf,
                          verbose = false, summary = false)
    return stats
end

# ============================================================================
# 4. RUN
# ============================================================================
hourly = DataFrame()
fx_rows = DataFrame(ion = String[], oxygen = String[], rep = Int[], fraction = Int[], t = Float64[],
                    clonogenic_before = Int[], clonogenic_after = Int[], SF = Float64[],
                    mean_sp_GSM2 = Float64[], sp_hypoxic = Float64[], sp_oxic = Float64[],
                    mean_dose = Float64[], mean_O2 = Float64[], frac_hypoxic = Float64[],
                    frac_anoxic = Float64[], R_spheroid = Float64[], Npar = Int[])

for rep in 1:N_REP
    grown = deepcopy(cell_df_pristine)
    Random.seed!(hash((:grow, rep)))
    run_for!(grown, T_PRE, -T_PRE, DataFrame(), (ion = "", oxygen = "", rep = rep))
    N0 = count(==(1), grown.is_cell)
    @printf("\n== realisation %d: %d cells, radius %.0f µm at the first fraction ==\n",
            rep, N0, current_radius(grown))

    conds = vcat([(ion = (tag = "control", particle = "", E = 0.0), mode = :diffusion)],
                 [(ion = i, mode = m) for i in IONS for m in OXYGEN_MODES])
    for cnd in conds
        ion_, mode = cnd.ion, cnd.mode
        df = deepcopy(grown)
        meta = (ion = ion_.tag, oxygen = String(mode), rep = rep)
        rows = DataFrame(); push!(rows, merge(meta, (t = 0.0,), counts(df)))
        t = 0.0
        for f in 1:N_FX
            if ion_.tag != "control"
                before = clonogenic(df)
                st = irradiate_fraction!(df, ion_, mode, hash((ion_.tag, mode, rep, f)))
                after = clonogenic(df)
                push!(rows, merge(meta, (t = t + 1e-3,), counts(df)))
                push!(fx_rows, (ion_.tag, String(mode), rep, f, t, before, after, after / max(before, 1),
                                st.mean_sp, st.sp_hypoxic, st.sp_oxic, st.mean_dose, st.mean_O2,
                                st.frac_hypoxic, st.frac_anoxic, st.R_spheroid, st.Npar))
                @printf("  %-11s %-9s fraction %d at %3.0f h: R %.0f µm, hypoxic %.0f%%, SF %.3f (oxic %.3f, hypoxic %.3f)\n",
                        ion_.tag, mode, f, t, st.R_spheroid, 100 * st.frac_hypoxic, after / max(before, 1),
                        st.sp_oxic, st.sp_hypoxic)
                if rep == 1                              # per-cell state for the radial figures
                    a = findall(==(1), df.is_cell)
                    CSV.write(joinpath(OUTDIR, "cells_$(ion_.tag)_$(mode)_f$(f).csv"),
                              DataFrame(x = df.x[a], y = df.y[a], z = df.z[a], O2 = df.O[a],
                                        dose = df.dose_cell[a], sp = df.sp[a],
                                        doomed = isfinite.(df.death_time[a]), phase = String.(df.cell_cycle[a])))
                end
            else
                oxygenate!(df, :diffusion)               # control: profile only (for the record)
            end
            Random.seed!(hash((ion_.tag, mode, rep, f, :abm)))
            t = run_for!(df, f < N_FX ? FX_GAP : T_AFTER, t, rows, meta)
        end
        edge, rmax, _ = box_check(df)
        @printf("  %-11s %-9s end: %d cells (clonogenic %d), max radius %.0f µm%s\n", ion_.tag, mode,
                count(==(1), df.is_cell), clonogenic(df), rmax, edge > 0 ? "  WARNING: box face reached" : "")
        rows[!, :N0] .= N0
        append!(hourly, rows)
        CSV.write(joinpath(OUTDIR, "hourly.csv"), hourly)
        CSV.write(joinpath(OUTDIR, "fractions.csv"), fx_rows)
    end
end
println("\nSaved in $OUTDIR")
