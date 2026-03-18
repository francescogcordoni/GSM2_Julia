using CSV, DataFrames
using Plots
using Printf

#~ ============================================================
#~ Configuration — must match run_survival_curve.jl
#~ ============================================================
indir  = joinpath(@__DIR__, "..", "data", "continuous_dr")
outdir = @__DIR__

surv_file = joinpath(indir, "survival_results_12C_10MeV.csv")
meta_file = joinpath(indir, "survival_meta_12C_10MeV.csv")

#~ ============================================================
#~ Load data
#~ ============================================================
surv_df = CSV.read(surv_file, DataFrame)
meta_df = CSV.read(meta_file, DataFrame)

doses_to_run         = surv_df.dose_Gy
doserates_to_run_Gys = meta_df.dose_rate_Gys

println("Loaded survival results:")
println("  Doses    : $doses_to_run")
println("  Dose rates (Gy/s): $doserates_to_run_Gys")

#~ ============================================================
#~ PLOT: Survival fraction vs Dose (log scale)
#~ ============================================================
p = plot(;
    xlabel  = "Dose (Gy)",
    ylabel  = "Survival fraction",
    yscale  = :log10,
    title   = "Survival vs Dose — 12C 10 MeV/u",
    legend  = :topright,
    size    = (800, 550),
    dpi     = 150)

for (j, dr_gys) in enumerate(doserates_to_run_Gys)
    col_name = @sprintf("dr_%.0eGys", dr_gys)
    sf_vec   = surv_df[!, col_name]
    plot!(p, doses_to_run, sf_vec;
          label  = "$(dr_gys) Gy/s",
          marker = :circle,
          lw     = 2)
end

display(p)
savefig(p, joinpath(outdir, "survival_curve_12C_10MeV.png"))
println("Saved: $(joinpath(outdir, "survival_curve_12C_10MeV.png"))")

#~ ============================================================
#~ FINAL PRINT
#~ ============================================================
println("\n", "="^60)
println("ALL PLOTS SAVED TO $outdir/")
println("="^60)
for f in filter(f -> endswith(f, ".png"), sort(readdir(outdir)))
    println("  $outdir/$f")
end
