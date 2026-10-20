# ============================================================================
# example/spheroid_regrowth.jl
#
# Regrowth delay and spheroid control probability (SCP) after single acute
# doses of 1H and 12C. Same setup and irradiation pipeline as example/spheroid.jl.
#
#   SF0         clonogenic fraction right after fate assignment (alive, not doomed)
#   controlled  no clonogenic cell left → the spheroid can never regrow
#   T_regrow    first hour at which the clonogenic count reaches GROWTH_FOLD × N0
#   delay       median T_regrow(D) − median T_regrow(0 Gy)
#
# Track-structure dose: N_DOSE_REAL Monte Carlo realisations per (ion, dose) on
# the spheroid at irradiation; replicate k uses realisation mod1(k, N_DOSE_REAL)
# and fresh lesions (MC_loop_damage!) + fresh ABM randomness.
# Each run stops as soon as the spheroid reaches STOP_FOLD × N0 clonogenic cells
# (regrowth established) or at T_FOLLOW.
#
# HOW TO RUN
#   julia --threads auto example/spheroid_regrowth.jl
# Outputs: example/output/spheroid_regrowth/
#   curves.csv  runs.csv  summary.csv  dose_check.csv  + figures
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
const N_REP       = 20
const N_DOSE_REAL = 4              # independent track-structure realisations per (ion, dose)
const T_FOLLOW    = 336            # h (14 days) maximum follow-up
const SEG         = 24             # h, ABM segment length (stop test after each segment)
const SNAP_DT     = 6              # h, resolution of the growth curves
const GROWTH_FOLD = 2.0            # regrowth endpoint: GROWTH_FOLD × N0 clonogenic cells
const STOP_FOLD   = 4.0            # stop the run once this is reached
const IONS = [
    (tag = "1H_100MeV",  particle = "1H",  E = 100.0, doses = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
    (tag = "12C_80MeVu", particle = "12C", E = 80.0,  doses = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]),
]

# Small spheroid (~500 cells): control probability becomes reachable at clinical doses
TUMOR_RADIUS = 150.0
R_CELL       = 15.0
X_BOX        = 400.0               # room for STOP_FOLD × N0 (radius × 1.6) and more
X_VOXEL      = 800.0

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
const FATE        = :incremental      # :incremental (delayed death) or :legacy (as example/spheroid.jl)
const OXYGEN      = :diffusion        # :diffusion or :uniform
const O2_UNIFORM  = 21.0

const OUTDIR = joinpath(@__DIR__, "output", "spheroid_regrowth")
mkpath(OUTDIR)

# ============================================================================
# 2. SETUP — identical to example/spheroid.jl (top level)
# ============================================================================
gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(R_G1, A_G1, B_G1, RD, RN)
gsm2_cycle[2] = GSM2(R_S,  A_S,  B_S,  RD, RN)
gsm2_cycle[3] = GSM2(R_G2, A_G2, B_G2, RD, RN)
gsm2_cycle[4] = GSM2(R_AVG, A_AVG, B_AVG, RD, RN)
setup_GSM2!(R_AVG, A_AVG, B_AVG, RD, RN)

setup_IonIrrad!(2.0, 100.0, "1H"; type_AT = TYPE_AT)
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

if isdefined(Main, :set_fate_model!)
    set_fate_model!(FATE)
else
    @warn "utilities_abm.jl has no set_fate_model!: compute_times_domain! is used as it is."
end
hasproperty(cell_df, :death_type)  || (cell_df.death_type  = fill(-1, nrow(cell_df)))
hasproperty(cell_df, :dam_X_total) || (cell_df.dam_X_total = zeros(Int, nrow(cell_df)))
hasproperty(cell_df, :dam_Y_total) || (cell_df.dam_Y_total = zeros(Int, nrow(cell_df)))

const N0 = count(cell_df.is_cell .== 1)
println("Spheroid: $N0 cells")
print_phase_distribution(cell_df; label = "pre-irradiation")
cell_df_pristine = deepcopy(cell_df)
const DFX = df_center_x
const DFY = df_center_y
const AT0 = at
const ALIVE0 = findall(==(1), cell_df_pristine.is_cell)

# ============================================================================
# 3. HELPERS
# ============================================================================
G(name::Symbol) = Base.invokelatest(getfield, Main, name)

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

clonogenic(df::DataFrame) = count((df.is_cell .== 1) .& .!isfinite.(df.death_time))
clonogenic(pop::CellPopulation) = count(.!isfinite.(pop.death_time))     # snapshots hold alive cells only

function wilson(k::Integer, n::Integer; z = 1.96)
    n == 0 && return (NaN, NaN, NaN)
    p = k / n; d = 1 + z^2 / n
    c = (p + z^2 / (2n)) / d; h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))
end

"""
Dose realisations for one (ion, dose) on the spheroid at irradiation:
returns the per-cell dose vectors of the alive cells, their energy_step, and Npar.
The ion globals (ion, irrad, irrad_cond) stay set for this (ion, dose).
"""
function dose_realisations(ion_, D::Float64)
    setup_IonIrrad!(D, ion_.E, ion_.particle; type_AT = TYPE_AT)
    base = deepcopy(cell_df_pristine)
    R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, TUMOR_RADIUS)
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, base, TRACK_SEG)
    F    = G(:irrad).dose / (1.602e-9 * G(:LET))
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    reals = Vector{Vector{Vector{Float64}}}()
    for k in 1:N_DOSE_REAL
        MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), base, DFX, DFY, deepcopy(AT0),
                     gsm2_cycle, TYPE_AT, TRACK_SEG; seed = hash((ion_.tag, D, k)))
        push!(reals, [copy(base.dose[i]) for i in ALIVE0])
    end
    return reals, copy(base.energy_step), Npar, R_beam
end

"""Irradiated copy of the pristine spheroid with dose realisation `real`."""
function irradiated_copy(real, estep, D::Float64, seed::UInt64)
    df = deepcopy(cell_df_pristine)
    D <= 0 && return df, 1.0, 0.0
    df.energy_step .= estep
    for (k, i) in enumerate(ALIVE0)
        df.dose[i] = copy(real[k]); df.dose_cell[i] = mean(real[k])
    end
    Random.seed!(seed)
    MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)
    sp_mean = mean(begin
        ph = df.cell_cycle[i]
        g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
        domain_GSM2(df.dam_X_dom[i], df.dam_Y_dom[i], g)
    end for i in ALIVE0)
    FATE === :legacy && compute_cell_survival_GSM2!(df, gsm2_cycle)
    return df, sp_mean, mean(df.dose_cell[ALIVE0])
end

# ============================================================================
# 4. RUN
# ============================================================================
curves = DataFrame(tag = String[], dose = Float64[], rep = Int[], t = Float64[],
                   clonogenic = Int[], alive = Int[])
runs   = DataFrame(tag = String[], dose = Float64[], rep = Int[], realisation = Int[], N0 = Int[],
                   survivors = Int[], SF0 = Float64[], GSM2_SF = Float64[], mean_dose = Float64[],
                   controlled = Bool[], T_regrow = Float64[], t_end = Float64[])
dose_check = DataFrame(tag = String[], dose = Float64[], Npar = Int[], R_beam = Float64[],
                       mean_dose_realisations = Float64[])

for ion_ in IONS, D in ion_.doses
    println("\n== $(ion_.tag)  $(D) Gy ==")
    reals, estep, Npar, R_beam = D > 0 ? dose_realisations(ion_, D) : (nothing, nothing, 0, NaN)
    D > 0 && push!(dose_check, (ion_.tag, D, Npar, R_beam,
                                mean(mean(mean.(r)) for r in reals)))
    for rep in 1:N_REP
        k = D > 0 ? mod1(rep, N_DOSE_REAL) : 0
        df, spm, dmean = irradiated_copy(D > 0 ? reals[k] : nothing, estep, D,
                                         hash((ion_.tag, D, rep, :damage)))
        if D > 0
            Random.seed!(hash((ion_.tag, D, rep, :fate)))
            compute_times_domain!(df, gsm2_cycle; nat_apo = NAT_APO, terminal_time = Inf,
                                  verbose = false, summary = false)
        end
        surv = clonogenic(df)

        # ABM in segments; stop once regrowth is clearly established
        Random.seed!(hash((ion_.tag, D, rep, :abm)))
        t0 = 0.0; Tr = Inf; nC = surv; nA = count(==(1), df.is_cell)
        push!(curves, (ion_.tag, D, rep, 0.0, nC, nA))
        while t0 < T_FOLLOW && surv > 0
            hours = collect(SNAP_DT:SNAP_DT:SEG)
            ts, snaps = run_simulation_abm!(df; nat_apo = NAT_APO, terminal_time = Float64(SEG),
                                            snapshot_times = hours, verbose = false,
                                            return_dataframes = false, update_input = true)
            _advance_idle!(df, SEG - (isempty(ts.time) ? 0.0 : ts.time[end]))
            for h in hours
                if h == SEG
                    nC = clonogenic(df); nA = count(==(1), df.is_cell)
                elseif haskey(snaps, h)
                    nC = clonogenic(snaps[h]); nA = Int(snaps[h].n_alive)
                end
                push!(curves, (ion_.tag, D, rep, t0 + h, nC, nA))
                isinf(Tr) && nC >= GROWTH_FOLD * N0 && (Tr = t0 + h)
            end
            t0 += SEG
            nC >= STOP_FOLD * N0 && break
            nC == 0 && break
        end
        push!(runs, (ion_.tag, D, rep, k, N0, surv, surv / N0, spm, dmean, surv == 0, Tr, t0))
        @printf("  rep %2d  dose %.2f Gy  SF0 %.4f (GSM2 %.4f)  controlled %s  T_regrow %s\n",
                rep, dmean, surv / N0, spm, surv == 0, isinf(Tr) ? "—" : @sprintf("%.0f h", Tr))
        df = nothing
    end
    reals = nothing; GC.gc()
end
CSV.write(joinpath(OUTDIR, "curves.csv"), curves)
CSV.write(joinpath(OUTDIR, "runs.csv"), runs)
CSV.write(joinpath(OUTDIR, "dose_check.csv"), dose_check)

# ============================================================================
# 5. SUMMARY
# ============================================================================
summ = DataFrame(tag = String[], dose = Float64[], mean_dose = Float64[],
                 SF0 = Float64[], SF0_sd = Float64[], GSM2_SF = Float64[],
                 SCP = Float64[], SCP_lo = Float64[], SCP_hi = Float64[], SCP_poisson = Float64[],
                 T_regrow_med = Float64[], delay = Float64[], delay_lo = Float64[],
                 delay_hi = Float64[], n_regrown = Int[])
for g in groupby(runs, [:tag, :dose])
    p, lo, hi = wilson(count(g.controlled), nrow(g))
    T   = g.T_regrow[isfinite.(g.T_regrow)]
    T0  = runs.T_regrow[(runs.tag .== g.tag[1]) .& (runs.dose .== 0.0)]
    T0f = T0[isfinite.(T0)]
    T0m = isempty(T0f) ? NaN : median(T0f)
    med = isempty(T) ? NaN : median(T)
    q   = isempty(T) ? (NaN, NaN) : (quantile(T, 0.25), quantile(T, 0.75))
    push!(summ, (g.tag[1], g.dose[1], mean(g.mean_dose), mean(g.SF0), std(g.SF0), mean(g.GSM2_SF),
                 p, lo, hi, exp(-N0 * mean(g.GSM2_SF)), med, med - T0m, q[1] - T0m, q[2] - T0m,
                 length(T)))
end
sort!(summ, [:tag, :dose])
CSV.write(joinpath(OUTDIR, "summary.csv"), summ)
show(stdout, summ; allcols = true); println()

function d50(s)
    for i in 1:nrow(s) - 1
        if s.SCP[i] < 0.5 <= s.SCP[i + 1]
            return s.dose[i] + (0.5 - s.SCP[i]) / (s.SCP[i + 1] - s.SCP[i]) * (s.dose[i + 1] - s.dose[i])
        end
    end
    return NaN
end
D50 = Dict(t => d50(summ[summ.tag .== t, :]) for t in unique(summ.tag))
for (t, d) in D50; @printf("SCD50 %-12s = %.2f Gy\n", t, d); end
haskey(D50, "1H_100MeV") && haskey(D50, "12C_80MeVu") &&
    @printf("in-situ RBE (12C vs 1H, at 50%% control) = %.2f\n", D50["1H_100MeV"] / D50["12C_80MeVu"])

# ============================================================================
# 6. PLOTS
# ============================================================================
default(framestyle = :box, lw = 1.8)
paper_style!()
ion_col = Dict("1H_100MeV" => :steelblue, "12C_80MeVu" => :firebrick)

# A. growth curves: clonogenic(t)/N0, median + IQR, one panel per ion
panels = map(IONS) do ion_
    p = plot(yscale = :log10, xlabel = "Time after irradiation (h)",
             ylabel = tex("Clonogenic cells / \$N_0\$"), title = ion_.tag, legend = :bottomright,
             ylims = (0.5 / N0, 2 * STOP_FOLD))
    cols = palette(:viridis, length(ion_.doses))
    for (k, D) in enumerate(ion_.doses)
        c = curves[(curves.tag .== ion_.tag) .& (curves.dose .== D), :]
        agg = combine(groupby(c, :t), :clonogenic => (x -> quantile(x, 0.25)) => :lo,
                      :clonogenic => median => :med, :clonogenic => (x -> quantile(x, 0.75)) => :hi,
                      nrow => :n)
        agg = agg[agg.n .>= N_REP ÷ 2, :]          # runs stop early: show times with enough runs
        sort!(agg, :t)
        y = max.(agg.med, 0.5) ./ N0
        plot!(p, agg.t, y; ribbon = (y .- max.(agg.lo, 0.5) ./ N0, max.(agg.hi, 0.5) ./ N0 .- y),
              fillalpha = 0.2, color = cols[k], label = tex("$(D) Gy"))
    end
    hline!(p, [GROWTH_FOLD]; color = :gray, ls = :dash, lw = 1, label = "regrowth endpoint")
end
savefig(plot(panels...; layout = (1, length(panels)), size = (1300, 460),
             left_margin = 6Plots.mm, bottom_margin = 6Plots.mm),
        joinpath(OUTDIR, "growth_curves.png"))

# B. regrowth delay, C. SCP, D. clonogenic fraction after irradiation (ABM vs GSM2)
pB = plot(xlabel = "Dose (Gy)", ylabel = "Regrowth delay (h)", legend = :topleft)
pC = plot(xlabel = "Dose (Gy)", ylabel = "Spheroid control probability", legend = :topleft,
          ylims = (-0.02, 1.02))
pD = plot(xlabel = "Dose (Gy)", ylabel = "Clonogenic fraction after irradiation",
          yscale = :log10, legend = :bottomleft)
for t in unique(summ.tag)
    s = summ[summ.tag .== t, :]; c = ion_col[t]
    m = .!isnan.(s.delay)
    any(m) && plot!(pB, s.dose[m], s.delay[m];
                    yerror = (s.delay[m] .- s.delay_lo[m], s.delay_hi[m] .- s.delay[m]),
                    marker = :circle, color = c, label = t)
    plot!(pC, s.dose, s.SCP; yerror = (s.SCP .- s.SCP_lo, s.SCP_hi .- s.SCP),
          marker = :circle, color = c, label = "$t (ABM)")
    plot!(pC, s.dose, s.SCP_poisson; ls = :dash, color = c, label = tex("$t (Poisson, exp(-\$N_0\$\$\\cdot\$SF_GSM2))"))
    m = s.SF0 .> 0
    plot!(pD, s.dose[m], s.SF0[m]; marker = :circle, color = c, label = "$t ABM")
    plot!(pD, s.dose, s.GSM2_SF; ls = :dash, color = c, label = "$t GSM2 mean")
end
savefig(plot(pB, pC, pD; layout = (1, 3), size = (1500, 450), left_margin = 6Plots.mm,
             bottom_margin = 6Plots.mm), joinpath(OUTDIR, "delay_scp_sf.png"))

println("\nSaved in $OUTDIR:")
foreach(f -> println("  ", f), readdir(OUTDIR))
