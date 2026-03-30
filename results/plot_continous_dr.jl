using CSV, DataFrames, Plots, LsqFit, Statistics, Printf, Glob

# ── Paths ─────────────────────────────────────────────────────────────────────
data_dir    = joinpath(@__DIR__, "..", "data", "continuous_dr")
results_dir = joinpath(@__DIR__, "..", "results")
mkpath(results_dir)

# ── LQ model: S = exp(-α·D - β·D²) ──────────────────────────────────────────
lq_model(D, p) = exp.(-p[1] .* D .- p[2] .* D .^ 2)

# ── Collect all survival CSV files ────────────────────────────────────────────
surv_files = filter(f -> occursin("survival_results", f) && endswith(f, ".csv"),
                    readdir(data_dir, join=true))

println("Found $(length(surv_files)) survival files:")
for f in surv_files; println("  ", basename(f)); end

# ── Storage for all fits ──────────────────────────────────────────────────────
all_fits = DataFrame(
    file       = String[],
    dose_rate  = String[],
    alpha      = Float64[],
    beta       = Float64[],
    alpha_err  = Float64[],
    beta_err   = Float64[],
    alpha_beta = Float64[],
    r2         = Float64[],
)

# ── Process each file ─────────────────────────────────────────────────────────
for fpath in surv_files
    tag  = replace(basename(fpath), "survival_results_" => "", ".csv" => "")
    df   = CSV.read(fpath, DataFrame)

    doses     = df.dose_Gy
    dr_cols   = setdiff(names(df), ["dose_Gy"])

    println("\n", "="^60)
    println("File: ", basename(fpath), "  →  tag: $tag")
    println("Dose rates: ", dr_cols)

    # one plot per file — all dose rates overlaid
    p_surv  = plot(title  = "Survival curves — $tag",
                   xlabel = "Dose (Gy)",
                   ylabel = "Survival fraction",
                   yscale = :log10,
                   legend = :topright,
                   framestyle = :box,
                   grid    = true,
                   gridalpha = 0.3,
                   size    = (700, 500),
                   dpi     = 150)

    p_alpha = plot(title  = "LQ α — $tag",
                   xlabel = "Dose rate",
                   ylabel = "α (Gy⁻¹)",
                   legend = false,
                   framestyle = :box,
                   size    = (500, 400),
                   dpi     = 150)

    p_beta  = plot(title  = "LQ β — $tag",
                   xlabel = "Dose rate",
                   ylabel = "β (Gy⁻²)",
                   legend = false,
                   framestyle = :box,
                   size    = (500, 400),
                   dpi     = 150)

    alphas     = Float64[]
    betas      = Float64[]
    alpha_errs = Float64[]
    beta_errs  = Float64[]
    dr_labels  = String[]

    for col in dr_cols
        surv = df[!, col]

        # guard: clamp to avoid log(0) issues
        surv_safe = clamp.(surv, 1e-10, 1.0)

        # initial guess: α=0.1, β=0.05
        p0 = [0.1, 0.05]
        lb = [0.0,  0.0]
        ub = [10.0, 10.0]

        fit_result = nothing
        try
            fit_result = curve_fit(lq_model, doses, surv_safe, p0;
                                   lower = lb, upper = ub)
        catch e
            @warn "Fit failed for $col in $tag" exception = e
            continue
        end

        α, β    = coef(fit_result)
        se      = try stderror(fit_result) catch; [NaN, NaN] end
        α_err, β_err = se

        # R²
        surv_pred = lq_model(doses, [α, β])
        ss_res    = sum((surv_safe .- surv_pred) .^ 2)
        ss_tot    = sum((surv_safe .- mean(surv_safe)) .^ 2)
        r2        = 1.0 - ss_res / ss_tot

        println(@sprintf("  %-20s  α=%.4f±%.4f  β=%.4f±%.4f  α/β=%.2f  R²=%.4f",
                         col, α, α_err, β, β_err, α/β, r2))

        push!(alphas,     α)
        push!(betas,      β)
        push!(alpha_errs, α_err)
        push!(beta_errs,  β_err)
        push!(dr_labels,  col)

        push!(all_fits, (tag, col, α, β, α_err, β_err, α/β, r2))

        # smooth fit curve
        D_fine    = range(0, maximum(doses), length=200)
        surv_fit  = lq_model(collect(D_fine), [α, β])

        label_str = replace(col, "dr_" => "", "Gys" => " Gy/s")
        scatter!(p_surv, doses, surv_safe;
                 label = label_str, markersize = 5, markerstrokewidth = 0.5)
        plot!(p_surv, D_fine, surv_fit;
              label = "", linestyle = :dash, linewidth = 1.5)
    end

    # α and β vs dose rate bar plots
    if !isempty(alphas)
        bar!(p_alpha, dr_labels, alphas;
             yerror = alpha_errs, color = :steelblue, alpha = 0.7,
             xrotation = 30)
        bar!(p_beta,  dr_labels, betas;
             yerror = beta_errs,  color = :firebrick, alpha = 0.7,
             xrotation = 30)
    end

    # save per-file plots
    savefig(p_surv,  joinpath(results_dir, "survival_curves_$(tag).png"))
    savefig(p_alpha, joinpath(results_dir, "alpha_$(tag).png"))
    savefig(p_beta,  joinpath(results_dir, "beta_$(tag).png"))
    println("  Saved plots for $tag")
end

# ── Save fit table ────────────────────────────────────────────────────────────
fits_path = joinpath(results_dir, "lq_fits_all.csv")
CSV.write(fits_path, all_fits)
println("\nSaved fit table: $fits_path")

# ── Summary plot: α/β ratio across all conditions ────────────────────────────
if nrow(all_fits) > 0
    p_ab = plot(title     = "α/β ratio across conditions",
                xlabel    = "Condition",
                ylabel    = "α/β (Gy)",
                legend    = false,
                framestyle = :box,
                size      = (max(600, 80*nrow(all_fits)), 450),
                dpi       = 150,
                bottom_margin = 10Plots.mm)

    labels = all_fits.file .* "\n" .* all_fits.dose_rate
    bar!(p_ab, labels, all_fits.alpha_beta;
         color = :mediumpurple, alpha = 0.8, xrotation = 45)

    savefig(p_ab, joinpath(results_dir, "alpha_beta_summary.png"))
    println("Saved summary plot: alpha_beta_summary.png")
end

println("\nDone. All results in: $results_dir")
println(all_fits)
