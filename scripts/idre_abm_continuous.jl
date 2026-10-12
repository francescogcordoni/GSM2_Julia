# ============================================================================
# idre_abm_continuous.jl
#
# Inverse dose-rate effect in the full ABM (lattice, contact inhibition,
# delayed death) under continuous irradiation with high-energy protons,
# comparing the current checkpoint rule (:all) with a G2/M checkpoint (:g2).
#
# Needs utilities_abm_fast.jl and utilities_checkpoint.jl included (last) in
# load_utilities.jl.
#
# Run: julia -t auto scripts/idre_abm_continuous.jl
# ============================================================================
using Base.Threads
using CSV, DataFrames, Distributions, Random, Statistics, Printf
using GLM, JLD2, DelimitedFiles, StatsBase, Optim, LsqFit, ProgressMeter
using InlineStrings, SparseArrays, ProgressBars, Distributed
using Plots, StatsPlots

include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ── Settings ────────────────────────────────────────────────────────────────
const E_PROTON     = 200.0                     # MeV
const DOSE         = 4.0                       # Gy
const RATES_GYH    = [0.1, 0.25, 0.37, 0.5, 0.75, 1.0, 1.54, 3.0, 10.0, 60.0]
const MODES        = [:all, :g2]
const N_REP        = 3                         # independent runs per point
const TUMOR_RADIUS = 150.0                     # µm (cells inside are quiescent G0)
const X_BOX        = 400.0                     # µm (free space to grow into)
const OUTDIR       = joinpath(@__DIR__, "..", "data", "idre_abm_continuous")
mkpath(OUTDIR)

const RD, RN = 0.8, 7.2
gsm2_cycle = [GSM2(2.780479661191086, 0.012872261720543399, 0.04029756109753225, RD, RN),   # G1/G0
              GSM2(5.84009601901114,  0.00589118894714544,  0.05794352736120672, RD, RN),   # S
              GSM2(1.7720064637774506, 0.024306291709970018, 5.704688326522623e-5, RD, RN), # G2/M
              GSM2(2.780479661191086, 0.012872261720543399, 0.04029756109753225, RD, RN)]   # fallback
setup_GSM2!(gsm2_cycle[1].r, gsm2_cycle[1].a, gsm2_cycle[1].b, RD, RN)

env  = setup(E_PROTON, "1H", DOSE, TUMOR_RADIUS; X_box = X_BOX)
cell_df0 = deepcopy(env.cell_df)
LET  = linear_interpolation("1H", E_PROTON, sp)
K_TOT = 9.0 * calculate_kappa("1H", LET, 21.0)        # X lesions / cell / Gy
L_TOT = K_TOT * 1e-3
N0 = count(==(1), cell_df0.is_cell)
@printf("cells=%d  LET=%.2f keV/µm  X lesions/cell/Gy=%.1f  G0 at start=%.2f\n",
        N0, LET, K_TOT, count(==("G0"), cell_df0.cell_cycle[cell_df0.is_cell .== 1]) / N0)

# ── Runs ────────────────────────────────────────────────────────────────────
rows = DataFrame(mode = String[], rate_Gyh = Float64[], rep = Int[], T_irr_h = Float64[],
                 SF_plated = Float64[], N_rel = Float64[],
                 fG1 = Float64[], fS = Float64[], fG2 = Float64[], fM = Float64[], fG0 = Float64[])
for mode in MODES, Ḋ in RATES_GYH, rep in 1:N_REP
    df = deepcopy(cell_df0)
    Random.seed!(hash((rep, :abm)))                   # same ABM randomness for every rate/mode
    t = @elapsed _, res = irradiate_continuous_abm!(df, gsm2_cycle; dose = DOSE,
                              dose_rate_gyh = Ḋ, K_tot = K_TOT, L_tot = L_TOT,
                              checkpoint = mode, seed = rep)
    f = res.phase_frac
    push!(rows, (String(mode), Ḋ, rep, res.T_irr, res.SF_plated, res.N_rel,
                 f.G1, f.S, f.G2, f.M, f.G0))
    @printf("%-4s Ḋ=%-6.3g Gy/h rep %d | SF_plated=%.4f  N/N0=%.3f  G2+M=%.2f  G0=%.2f  (%.0f s)\n",
            mode, Ḋ, rep, res.SF_plated, res.N_rel, f.G2 + f.M, f.G0, t)
end
CSV.write(joinpath(OUTDIR, "idre_abm.csv"), rows)

# ── Summary + plots ─────────────────────────────────────────────────────────
summ = combine(groupby(rows, [:mode, :rate_Gyh]),
               :SF_plated => mean => :SF, :SF_plated => (x -> std(x) / sqrt(length(x))) => :SE,
               :N_rel => mean => :N_rel, [:fG2, :fM] => ((a, b) -> mean(a .+ b)) => :fG2M,
               :fG0 => mean => :fG0)
sort!(summ, [:mode, :rate_Gyh])
CSV.write(joinpath(OUTDIR, "idre_abm_summary.csv"), summ)

p1 = plot(xscale = :log10, yscale = :log10, xlabel = "Dose rate (Gy/h)",
          ylabel = "SF (plated) after $(DOSE) Gy", legend = :bottomright)
p2 = plot(xscale = :log10, xlabel = "Dose rate (Gy/h)",
          ylabel = "G2+M fraction at end of exposure", legend = :topright)
for (i, g) in enumerate(groupby(summ, :mode))
    lo = min.(g.SE, 0.9 .* g.SF)
    plot!(p1, g.rate_Gyh, g.SF; yerror = (lo, g.SE), marker = :circle, color = i,
          label = "checkpoint $(g.mode[1])")
    plot!(p2, g.rate_Gyh, g.fG2M; marker = :circle, color = i, label = "checkpoint $(g.mode[1])")
end
savefig(plot(p1, p2; layout = (1, 2), size = (1300, 450), left_margin = 8Plots.mm,
             bottom_margin = 6Plots.mm), joinpath(OUTDIR, "idre_abm.png"))
println("\nResults in ", OUTDIR)
