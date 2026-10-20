# ============================================================================
# example/spheroid_fractionation.jl
#
# Fractionated irradiation of a 3-D spheroid, 1H vs 12C (+ unirradiated control),
# several stochastic realisations. Built on the same pipeline as example/spheroid.jl.
#
# SCHEDULE
#   0 → T_PRE            free growth
#   T_PRE + k·FX_GAP     fraction of FX_DOSE (k = 0 … N_FX-1)
#   … + T_AFTER          follow-up after the last fraction
#
# PIPELINE (per condition, per realisation)
#   1. PARAMETERS
#   2. SETUP (once, top level, exactly as example/spheroid.jl)
#   3. GROWTH            ABM for T_PRE hours
#   4. EACH FRACTION     setup_IonIrrad! → beam on the CURRENT spheroid →
#                        setup_irrad_conditions! → (oxygen) → MC_dose_CPU! →
#                        MC_loop_damage! → compute_times_domain! → ABM until the
#                        next fraction
#   5. OUTPUT            CSV for every DataFrame + plots in example/output/spheroid_fractionation/
#
# Fates: FATE = :incremental (default) keeps the cells doomed by an earlier
# fraction and their delayed deaths, and leaves undamaged cells' clocks running.
# compute_cell_survival_GSM2! is NOT called in that mode because it resets the
# timers of every cell; the GSM2 survival probability is computed without resets.
# FATE = :legacy reproduces example/spheroid.jl (timers of all cells reassigned at
# every fraction, direct-Y cells removed at once, no death delay).
#
# HOW TO RUN
#   julia --threads auto example/spheroid_fractionation.jl
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using StatsPlots
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using CUDA
using Statistics: mean
using SparseArrays
using Printf

println("Threads available: ", nthreads())
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS
# ============================================================================

# Schedule
const T_PRE   = 24          # h of growth before the first fraction
const FX_DOSE = 2.0         # Gy per fraction
const N_FX    = 3
const FX_GAP  = 24          # h between fractions
const T_AFTER = 96          # h after the last fraction
const N_REP   = 5           # stochastic realisations per condition

const CONDITIONS = [
    (tag = "control",    particle = "1H",  E = 100.0, dose = 0.0),
    (tag = "1H_100MeV",  particle = "1H",  E = 100.0, dose = FX_DOSE),
    (tag = "12C_80MeVu", particle = "12C", E = 80.0,  dose = FX_DOSE),
]

# Spheroid and cell geometry (box must contain the growth: the lattice is periodic)
TUMOR_RADIUS  = 250.0
R_CELL        = 15.0
X_BOX         = 450.0
X_VOXEL       = 900.0

# GSM2 domain geometry and phase-specific parameters [G1/G0, S, G2/M, average]
RD = 0.8; RN = 7.2
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

# Options
const TYPE_AT     = "KC"
const TRACK_SEG   = true
const TARGET_GEOM = "circle"
const CALC_TYPE   = "full"
const NAT_APO     = 1e-10
const FATE        = :incremental      # :incremental (delayed death) or :legacy (as example/spheroid.jl)
const OXYGEN      = :diffusion        # :diffusion (recomputed at each fraction) or :uniform
const O2_UNIFORM  = 21.0
const SNAP_DT     = 1                 # h, resolution of the saved time series

const OUTDIR = joinpath(@__DIR__, "output", "spheroid_fractionation")
mkpath(OUTDIR)
const FX_TIMES = [T_PRE + (f - 1) * FX_GAP for f in 1:N_FX]

# ============================================================================
# 2. SETUP — identical to example/spheroid.jl (top level)
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
OXYGEN === :diffusion ? set_oxygen_diffusion!(cell_df; density = :mean, plot_oxygen = false) :
                        (cell_df.O .= O2_UNIFORM)

# Fate model: set_fate_model! exists only in the latest utilities_abm.jl
if isdefined(Main, :set_fate_model!)
    set_fate_model!(FATE)
else
    @warn "utilities_abm.jl has no set_fate_model!: compute_times_domain! is used as it is. " *
          "For the incremental model with delayed death use the latest utilities_abm.jl."
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

"""One fraction on the current spheroid, same steps as example/spheroid.jl."""
function irradiate_fraction!(df::DataFrame, c, seed::UInt64)
    setup_IonIrrad!(c.dose, c.E, c.particle; type_AT = TYPE_AT)
    Rnow = current_radius(df)
    R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, Rnow)
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
    OXYGEN === :diffusion && set_oxygen_diffusion!(df; density = :mean, plot_oxygen = false)

    F    = G(:irrad).dose / (1.602e-9 * G(:LET))
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    at_  = deepcopy(AT0)
    MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, DFX, DFY, at_,
                 gsm2_cycle, TYPE_AT, TRACK_SEG; seed = seed)
    Random.seed!(seed)
    MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)

    alive = findall(==(1), df.is_cell)
    if FATE === :legacy
        compute_cell_survival_GSM2!(df, gsm2_cycle)     # resets timers: fine in legacy mode
    else
        for i in alive                                   # GSM2 survival without resetting timers
            ph = df.cell_cycle[i]
            g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
            df.sp[i] = domain_GSM2(df.dam_X_dom[i], df.dam_Y_dom[i], g)
        end
    end
    notdoomed = [i for i in alive if !isfinite(df.death_time[i])]
    mean_dose = mean(df.dose_cell[alive])
    mean_sp   = isempty(notdoomed) ? NaN : mean(df.sp[notdoomed])
    compute_times_domain!(df, gsm2_cycle; nat_apo = NAT_APO, terminal_time = Inf,
                          verbose = false, summary = false)
    return (Npar = Npar, R_beam = R_beam, R_spheroid = Rnow, mean_dose = mean_dose,
            mean_sp = mean_sp)
end

# ============================================================================
# 4. RUN
# ============================================================================

hourly_all = DataFrame()
fx_rows = DataFrame(tag = String[], rep = Int[], fraction = Int[], t = Float64[],
                    clonogenic_before = Int[], clonogenic_after = Int[], SF = Float64[],
                    mean_sp_GSM2 = Float64[], mean_dose = Float64[], Npar = Int[],
                    R_beam = Float64[], R_spheroid = Float64[])

for c in CONDITIONS, rep in 1:N_REP
    println("\n== $(c.tag)  realisation $rep ==")
    df   = deepcopy(cell_df_pristine)
    rows = DataFrame()
    Random.seed!(hash((:grow, rep)))                 # same growth for every condition
    push!(rows, merge((tag = c.tag, rep = rep, t = 0.0), counts(df)))
    t0 = run_segment!(df, T_PRE, 0.0, rows, c.tag, rep; final = false)
    for f in 1:N_FX
        before = clonogenic(df)
        info = (Npar = 0, R_beam = NaN, R_spheroid = current_radius(df), mean_dose = 0.0, mean_sp = 1.0)
        c.dose > 0 && (info = irradiate_fraction!(df, c, hash((c.tag, rep, f))))
        after = clonogenic(df)
        c.dose > 0 && push!(rows, merge((tag = c.tag, rep = rep, t = t0 + 1e-3), counts(df)))
        if c.dose > 0 && after == before
            @warn "$(c.tag) rep $rep fraction $f: no cell was doomed by $(c.dose) Gy " *
                  "(mean dose $(round(info.mean_dose, digits = 3)) Gy, GSM2 SF $(round(info.mean_sp, digits = 3))). " *
                  "Check the dose and the fate step."
        end
        push!(fx_rows, (c.tag, rep, f, t0, before, after, before > 0 ? after / before : NaN,
                        info.mean_sp, info.mean_dose, info.Npar, info.R_beam, info.R_spheroid))
        @printf("  fraction %d at t=%.0f h: dose %.3f Gy  GSM2 SF %.3f  clonogenic %d → %d (SF %.3f)\n",
                f, t0, info.mean_dose, info.mean_sp, before, after, after / max(before, 1))
        Random.seed!(hash((c.tag, rep, f, :abm)))
        t0 = run_segment!(df, f < N_FX ? FX_GAP : T_AFTER, t0, rows, c.tag, rep; final = f == N_FX)
    end
    CSV.write(joinpath(OUTDIR, "hourly_$(c.tag)_rep$(rep).csv"), rows)
    a = findall(==(1), df.is_cell)
    CSV.write(joinpath(OUTDIR, "final_cells_$(c.tag)_rep$(rep).csv"),
              DataFrame(index = df.index[a], x = df.x[a], y = df.y[a], z = df.z[a],
                        phase = String.(df.cell_cycle[a]), doomed = isfinite.(df.death_time[a]),
                        empty_neighbours = df.number_nei[a], O = df.O[a]))
    edge, rmax, r95 = box_check(df)
    @printf("  final radius: max %.0f µm, 95th percentile %.0f µm (box half-size %.0f µm)\n",
            rmax, r95, X_BOX)
    edge > 0 && @warn "$(c.tag) rep $rep: $edge cells on the outermost lattice layer: they " *
                      "neighbour the opposite face (periodic lattice) — enlarge X_BOX"
    append!(hourly_all, rows)
end

CSV.write(joinpath(OUTDIR, "hourly_all.csv"), hourly_all)
CSV.write(joinpath(OUTDIR, "fractions.csv"), fx_rows)
summ = combine(groupby(hourly_all, [:tag, :t]),
               :clonogenic => mean => :clono_mean, :clonogenic => std => :clono_sd,
               :total => mean => :total_mean, :total => std => :total_sd,
               :G0 => mean => :G0_mean, :G0 => std => :G0_sd)
summ.clono_cv = summ.clono_sd ./ max.(summ.clono_mean, 1e-9)
sort!(summ, [:tag, :t])
CSV.write(joinpath(OUTDIR, "hourly_summary.csv"), summ)

# ============================================================================
# 5. PLOTS
# ============================================================================
default(framestyle = :box, lw = 1.6)
paper_style!()
col  = Dict("control" => :gray40, "1H_100MeV" => :steelblue, "12C_80MeVu" => :firebrick)
tags = [c.tag for c in CONDITIONS]
fxlines!(p) = vline!(p, FX_TIMES; color = :black, ls = :dash, lw = 0.8, label = "")

# A. one realisation: cell numbers + phase composition of clonogenic cells
pA = map(tags) do t
    r = hourly_all[(hourly_all.tag .== t) .& (hourly_all.rep .== 1), :]
    p1 = plot(r.t, r.total; color = :black, label = "all cells", ylabel = "cells",
              title = "$t, realisation 1", legend = :topleft)
    plot!(p1, r.t, r.clonogenic; color = col[t], label = "clonogenic")
    plot!(p1, r.t, r.doomed; color = col[t], ls = :dot, label = "doomed")
    fxlines!(p1)
    comp = hcat(r.G0, r.G1, r.S, r.G2, r.M) ./ max.(r.clonogenic, 1)
    p2 = areaplot(r.t, comp; label = ["G0" "G1" "S" "G2" "M"], xlabel = "Time (h)",
                  ylabel = "fraction of clonogenic", ylims = (0, 1), legend = :outerright,
                  seriescolor = [:gray70 :steelblue :seagreen :orange :firebrick])
    fxlines!(p2)
    plot(p1, p2; layout = (2, 1))
end
savefig(plot(pA...; layout = (1, length(pA)), size = (560 * length(pA), 650),
             left_margin = 6Plots.mm, bottom_margin = 6Plots.mm),
        joinpath(OUTDIR, "single_realisation.png"))

# B. all realisations (thin) + mean ± SD; comparison panel
pB = map(tags) do t
    p = plot(xlabel = "Time (h)", ylabel = tex("clonogenic / \$N_0\$"), title = t, legend = :topleft)
    for rep in 1:N_REP
        r = hourly_all[(hourly_all.tag .== t) .& (hourly_all.rep .== rep), :]
        plot!(p, r.t, r.clonogenic ./ N_INIT; color = col[t], alpha = 0.3, lw = 0.8, label = "")
    end
    s = summ[summ.tag .== t, :]
    plot!(p, s.t, s.clono_mean ./ N_INIT; ribbon = s.clono_sd ./ N_INIT, fillalpha = 0.25,
          color = col[t], lw = 2.4, label = tex("mean \$\\pm\$ SD"))
    fxlines!(p)
end
pB2 = plot(xlabel = "Time (h)", ylabel = tex("clonogenic / \$N_0\$"), title = tex("mean \$\\pm\$ SD"), legend = :topleft)
for t in tags
    s = summ[summ.tag .== t, :]
    plot!(pB2, s.t, s.clono_mean ./ N_INIT; ribbon = s.clono_sd ./ N_INIT, fillalpha = 0.25,
          color = col[t], lw = 2.2, label = t)
end
fxlines!(pB2)
savefig(plot(pB..., pB2; layout = (1, length(pB) + 1), size = (480 * (length(pB) + 1), 430),
             left_margin = 6Plots.mm, bottom_margin = 6Plots.mm), joinpath(OUTDIR, "replicates.png"))

# C. variability
pC1 = plot(xlabel = "Time (h)", ylabel = "SD of clonogenic cells", legend = :topleft)
pC2 = plot(xlabel = "Time (h)", ylabel = "CV = SD / mean", legend = :topleft)
for t in tags
    s = summ[summ.tag .== t, :]
    plot!(pC1, s.t, s.clono_sd; color = col[t], label = t)
    plot!(pC2, s.t, s.clono_cv; color = col[t], label = t)
end
fxlines!(pC1); fxlines!(pC2)
savefig(plot(pC1, pC2; layout = (1, 2), size = (1200, 430), left_margin = 6Plots.mm,
             bottom_margin = 6Plots.mm), joinpath(OUTDIR, "variability.png"))

# D. per-fraction surviving fraction: ABM (dots + mean) vs mean GSM2 probability (open)
pD = plot(xlabel = "Fraction", ylabel = "Surviving fraction", xticks = 1:N_FX, legend = :bottomright)
for (j, t) in enumerate(tags[2:end])
    g = fx_rows[fx_rows.tag .== t, :]
    scatter!(pD, g.fraction .+ 0.08 * (j - 1.5), g.SF; color = col[t], alpha = 0.5, ms = 4, label = "")
    m = combine(groupby(g, :fraction), :SF => mean => :SF, :SF => std => :sd,
                :mean_sp_GSM2 => mean => :sp)
    plot!(pD, m.fraction, m.SF; yerror = m.sd, color = col[t], marker = :diamond, lw = 2, label = "$t ABM")
    scatter!(pD, m.fraction, m.sp; color = col[t], marker = :circle, mc = :white, msc = col[t],
             label = "$t GSM2 mean")
end
savefig(pD, joinpath(OUTDIR, "fraction_sf.png"))

println("\nSaved in $OUTDIR:")
foreach(f -> println("  ", f), readdir(OUTDIR))
