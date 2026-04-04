using CSV, DataFrames, Plots, LaTeXStrings, Printf, Distributions, Statistics, Random, Base.Threads

# ── Paths ─────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

outdir = joinpath(@__DIR__, "..", "data", "fit_LQ")
mkpath(outdir)

# ── GSM2 parameters ────────────────────────────────────────────────────────────
a_G1 = 0.012872261720543399
b_G1 = 0.04029756109753225
r_G1 = 2.780479661191086

a_S  = 0.00589118894714544
b_S  = 0.05794352736120672
r_S  = 5.84009601901114

a_G2 = 0.024306291709970018
b_G2 = 5.704688326522623e-5
r_G2 = 1.7720064637774506

rd = 0.8
Rn = 7.2

# ── Shared geometry constants ─────────────────────────────────────────────────
(center_x, _) = calculate_centers(0.0, 0.0, rd, Rn)
domain     = length(center_x)
n_repeat   = floor(Int, Rn / rd)
kappa_base  = 56.0 / (n_repeat * domain)
lambda_base = kappa_base * 1e-3
println("domain=$domain  n_repeat=$n_repeat  n_domains=$(n_repeat*domain)")

# ── Pre-sample damage arrays ──────────────────────────────────────────────────
sim_ = 10000
dose = [0.5, 1, 2, 3, 4, 5]
X_dam = zeros(Int64, size(dose, 1), sim_, n_repeat * domain)
Y_dam = zeros(Int64, size(dose, 1), sim_, n_repeat * domain)

for dj in 1:size(dose, 1)
    for s in 1:sim_
        d = dose[dj]
        lambda_X = max(0.0, kappa_base * d)
        lambda_Y = max(0.0, lambda_base * d)
        X_dam[dj, s, :] = rand(Poisson(lambda_X), n_repeat * domain)
        Y_dam[dj, s, :] = rand(Poisson(lambda_Y), n_repeat * domain)
    end
end

const DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

# ── run_phase: typed arg avoids data race with global gsm2 ────────────────────
function run_phase(gsm2_ph::GSM2, X_dam, Y_dam, dose, sim_)
    SP_cell = zeros(Float64, size(dose, 1), sim_)
    for dj in 1:size(dose, 1)
        Threads.@threads for s in 1:sim_
            SP_cell[dj, s] = domain_GSM2(X_dam[dj, s, :], Y_dam[dj, s, :], gsm2_ph)
        end
    end
    return vec(mean(SP_cell, dims=2))
end

# ── G1 ────────────────────────────────────────────────────────────────────────
println("--- Calculating Survival Probabilities G1 ---")
SP_opt_G1 = run_phase(GSM2(r_G1, a_G1, b_G1, rd, Rn), X_dam, Y_dam, dose, sim_)
alpha_G1 = 0.351;  beta_G1 = 0.04
S_LQ_G1  = exp.(-alpha_G1 .* dose .- beta_G1 .* dose .* dose)
println("--- Done G1 ---")

# ── S ─────────────────────────────────────────────────────────────────────────
println("--- Calculating Survival Probabilities S ---")
SP_opt_S = run_phase(GSM2(r_S, a_S, b_S, rd, Rn), X_dam, Y_dam, dose, sim_)
alpha_S = 0.1235;  beta_S = 0.0285
S_LQ_S  = exp.(-alpha_S .* dose .- beta_S .* dose .* dose)
println("--- Done S ---")

# ── G2/M ──────────────────────────────────────────────────────────────────────
println("--- Calculating Survival Probabilities G2 ---")
SP_opt_G2 = run_phase(GSM2(r_G2, a_G2, b_G2, rd, Rn), X_dam, Y_dam, dose, sim_)
alpha_G2 = 0.793;  beta_G2 = 0.0
S_LQ_G2  = exp.(-alpha_G2 .* dose .- beta_G2 .* dose .* dose)
println("--- Done G2 ---")

# ── Plots ─────────────────────────────────────────────────────────────────────
D_fine = collect(range(0, maximum(dose), length=300))

function phase_panel(dose, SP_opt, alpha, beta, D_fine, title_str, col)
    SF_lq_fine = exp.(-alpha .* D_fine .- beta .* D_fine .^ 2)
    p = plot(;
        xlabel        = "Dose (Gy)",
        ylabel        = "Survival fraction",
        yscale        = :log10,
        legend        = :topright,
        size          = (600, 480),
        left_margin   = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        DEFAULTS...,
    )
    plot!(p, D_fine, SF_lq_fine;
        label="LQ", color=:black, linewidth=2.0, linestyle=:dash)
    scatter!(p, dose, SP_opt;
        label="GSM2", color=col, markersize=6, markerstrokewidth=0.5)
    return p
end

p_G1 = phase_panel(dose, SP_opt_G1, alpha_G1, beta_G1, D_fine, "G1",   :royalblue)
p_S  = phase_panel(dose, SP_opt_S,  alpha_S,  beta_S,  D_fine, "S",    :seagreen)
p_G2 = phase_panel(dose, SP_opt_G2, alpha_G2, beta_G2, D_fine, "G2/M", :firebrick)

savefig(p_G1, joinpath(outdir, "LQ_fit_G1.png"));  savefig(p_G1, joinpath(outdir, "LQ_fit_G1.pdf"))
savefig(p_S,  joinpath(outdir, "LQ_fit_S.png"));   savefig(p_S,  joinpath(outdir, "LQ_fit_S.pdf"))
savefig(p_G2, joinpath(outdir, "LQ_fit_G2.png"));  savefig(p_G2, joinpath(outdir, "LQ_fit_G2.pdf"))
println("Saved individual panels")

# Combined 3-panel plot
p_all = plot(p_G1, p_S, p_G2;
    layout = (1, 3),
    size   = (1800, 500),
    DEFAULTS...,
)
display(p_all)
savefig(p_all, joinpath(outdir, "LQ_fit_all.png"))
savefig(p_all, joinpath(outdir, "LQ_fit_all.pdf"))
println("Saved: LQ_fit_all")

# Save numerical results
results_df = DataFrame(
    phase   = vcat(fill("G1", 6), fill("S", 6), fill("G2", 6)),
    dose    = vcat(dose, dose, dose),
    SF_gsm2 = vcat(SP_opt_G1, SP_opt_S, SP_opt_G2),
    SF_lq   = vcat(S_LQ_G1,  S_LQ_S,  S_LQ_G2),
)
CSV.write(joinpath(outdir, "LQ_fit_results.csv"), results_df)
println("Saved: LQ_fit_results.csv")
println("\nAll saved to $outdir")
