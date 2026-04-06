using CSV, DataFrames, Plots, Printf, LsqFit, Statistics, LaTeXStrings

# ── Paths ─────────────────────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data", "continuous_dr")

const DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

# ── Conditions to load ────────────────────────────────────────────────────────
# fit_max_Gy: upper dose limit used for LQ fitting (per condition)
CONDITIONS = [
    (tag="12C_10MeV",  label=L"$^{12}$C 10 MeV/u",  color=:firebrick,  fit_max_Gy=1.3, xlim=2.0, skip_dose_idx=Int[]),
    (tag="12C_100MeV", label=L"$^{12}$C 100 MeV/u", color=:darkorange, fit_max_Gy=Inf, xlim=6.0, skip_dose_idx=[2]),
    (tag="1H_100MeV",  label=L"$^{1}$H 100 MeV/u",  color=:royalblue,  fit_max_Gy=Inf, xlim=6.0, skip_dose_idx=[2]),
]

# Dose-rate columns to exclude from all plots
const EXCLUDE_DR = String[]

# ── LQ fitting helper ─────────────────────────────────────────────────────────
# Parse dose-rate value (Gy/s) from column name, e.g. "dr_1e-05Gys" → 1e-5
function parse_dr(col::AbstractString)
    s = replace(col, "dr_" => "", "Gys" => "")
    return parse(Float64, s)
end

function fit_lq_survival(surv_df::DataFrame, tag::String;
                         fit_max_Gy::Float64 = Inf)
    lq_model(D, p) = exp.(-p[1] .* D .- p[2] .* D .^ 2)
    dr_cols = setdiff(names(surv_df), vcat(["dose_Gy"], EXCLUDE_DR))
    doses   = surv_df.dose_Gy

    params_df = DataFrame(
        tag          = String[],
        dose_rate_Gys = Float64[],
        alpha        = Float64[],
        beta         = Float64[],
        alpha_err    = Float64[],
        beta_err     = Float64[],
        alpha_beta   = Float64[],
        r2           = Float64[],
    )

    for col in dr_cols
        dr_val = parse_dr(col)
        mask   = doses .<= fit_max_Gy
        D_fit  = Float64.(doses[mask])
        S_fit  = Float64.(surv_df[mask, col])

        # Remove zero or negative survival values (can't fit log scale)
        valid  = S_fit .> 0
        D_fit  = D_fit[valid]
        S_fit  = S_fit[valid]

        if length(D_fit) < 3
            @warn "Too few points to fit LQ for $tag / $col"
            push!(params_df, (tag, dr_val, NaN, NaN, NaN, NaN, NaN, NaN))
            continue
        end

        try
            # Fit in log space: log(S) = -α·D - β·D²  (linear in parameters)
            lq_log(D, p) = -p[1] .* D .- p[2] .* D .^ 2
            logS_fit = log.(S_fit)
            fit  = curve_fit(lq_log, D_fit, logS_fit, [0.5, 0.05]; lower=[0.0, 0.0])
            α, β = fit.param
            se   = stderror(fit)
            # R² also in log space
            ss_res = sum((logS_fit .- lq_log(D_fit, fit.param)) .^ 2)
            ss_tot = sum((logS_fit .- mean(logS_fit)) .^ 2)
            r2     = 1.0 - ss_res / ss_tot
            push!(params_df, (tag, dr_val, α, β, se[1], se[2], α / β, r2))
        catch e
            @warn "LQ fit failed for $tag / $col" exception=e
            push!(params_df, (tag, dr_val, NaN, NaN, NaN, NaN, NaN, NaN))
        end
    end
    return params_df
end

# ── Refit and save LQ params for all conditions ───────────────────────────────
for cond in CONDITIONS
    surv_path = joinpath(datadir, "survival_results_$(cond.tag).csv")
    isfile(surv_path) || continue
    surv_df  = CSV.read(surv_path, DataFrame)
    lq_df    = fit_lq_survival(surv_df, cond.tag; fit_max_Gy=cond.fit_max_Gy)
    lq_path  = joinpath(datadir, "lq_params_$(cond.tag).csv")
    CSV.write(lq_path, lq_df)
    println("Fitted & saved: lq_params_$(cond.tag).csv  (fit up to $(cond.fit_max_Gy) Gy)")
end

# Dose-rate palette (one color per dose rate, shared across conditions)
dr_palette = [:navy, :steelblue, :seagreen, :darkorange, :purple]

lq(D, α, β) = exp.(-α .* D .- β .* D .^ 2)

# ── PLOT 1: Dose-survival curves + LQ fit (one panel per condition) ───────────
surv_panels    = Plots.Plot[]
surv_conds_out = []   # CONDITIONS entries for which a panel was built

for cond in CONDITIONS
    surv_path = joinpath(datadir, "survival_results_$(cond.tag).csv")
    lq_path   = joinpath(datadir, "lq_params_$(cond.tag).csv")
    isfile(surv_path) && isfile(lq_path) || (@warn "Missing files for $(cond.tag)"; continue)

    surv_df = CSV.read(surv_path, DataFrame)
    lq_df   = CSV.read(lq_path,   DataFrame)

    doses   = surv_df.dose_Gy
    dr_cols = setdiff(names(surv_df), vcat(["dose_Gy"], EXCLUDE_DR))
    D_fine  = range(0, maximum(doses), length=300)

    p = plot(;
        xlabel        = "Dose (Gy)",
        ylabel        = "Survival fraction",
        yscale        = :log10,
        legend        = :bottomleft,
        size          = (650, 500),
        bottom_margin = 10Plots.mm,
        left_margin   = 10Plots.mm,
        DEFAULTS...,
    )

    for (k, col) in enumerate(dr_cols)
        col_c  = dr_palette[min(k, length(dr_palette))]
        row    = lq_df[lq_df.dose_rate_Gys .== lq_df.dose_rate_Gys[k], :]
        lbl    = latexstring(@sprintf("%.0e~\\mathrm{Gy/s}", lq_df.dose_rate_Gys[k]))

        # LQ fit curve
        if !isempty(row) && !isnan(row.alpha[1])
            α, β = row.alpha[1], row.beta[1]
            plot!(p, collect(D_fine), lq(collect(D_fine), α, β);
                label     = lbl,
                color     = col_c,
                linewidth = 1.8,
            )
        end
    end

    push!(surv_panels, p)
    push!(surv_conds_out, cond)
end


if !isempty(surv_panels)
    ncols  = length(surv_panels)
    p_surv = plot(surv_panels...;
                  layout     = (1, ncols),
                  size       = (650 * ncols, 500),
                  DEFAULTS...)
    display(p_surv)
    savefig(p_surv, joinpath(datadir, "survival_curves_all.png"))
    savefig(p_surv, joinpath(datadir, "survival_curves_all.pdf"))
    println("Saved: survival_curves_all")

    # Also save individual panels
    for (cond, p) in zip(surv_conds_out, surv_panels)
        savefig(p, joinpath(datadir, "survival_curves_$(cond.tag).png"))
        savefig(p, joinpath(datadir, "survival_curves_$(cond.tag).pdf"))
        println("Saved: survival_curves_$(cond.tag)")
    end
end

# ── Load all LQ params ────────────────────────────────────────────────────────
all_lq = DataFrame()
for cond in CONDITIONS
    path = joinpath(datadir, "lq_params_$(cond.tag).csv")
    isfile(path) || continue
    df = CSV.read(path, DataFrame)
    append!(all_lq, df; cols=:union)
end

if isempty(all_lq)
    println("No LQ parameter files found — skipping parameter plots.")
else
    # ── PLOT 2: Dose rate vs α ────────────────────────────────────────────────
    p_alpha = plot(;
        xlabel = "Dose rate (Gy/s)",
        ylabel = L"\alpha\ \mathrm{(Gy^{-1})}",
        xscale = :log10,
        legend = :topright,
        size   = (750, 500),
        DEFAULTS...,
    )

    for cond in CONDITIONS
        sub = all_lq[all_lq.tag .== cond.tag, :]
        isempty(sub) && continue
        sort!(sub, :dose_rate_Gys)
        plot!(p_alpha, sub.dose_rate_Gys, sub.alpha;
            label             = cond.label,
            color             = cond.color,
            linewidth         = 2,
            marker            = :circle,
            markersize        = 6,
            markerstrokewidth = 0.5,
        )
        if !all(isnan.(sub.alpha_err))
            plot!(p_alpha, sub.dose_rate_Gys,
                  sub.alpha .+ sub.alpha_err;
                fillrange         = sub.alpha .- sub.alpha_err,
                fillalpha         = 0.15,
                linealpha         = 0,
                color             = cond.color,
                label             = "",
            )
        end
    end

    display(p_alpha)
    savefig(p_alpha, joinpath(datadir, "alpha_vs_doserate.png"))
    savefig(p_alpha, joinpath(datadir, "alpha_vs_doserate.pdf"))
    println("Saved: alpha_vs_doserate")

    # ── PLOT 3: Dose rate vs β ────────────────────────────────────────────────
    p_beta = plot(;
        xlabel = "Dose rate (Gy/s)",
        ylabel = L"\beta\ \mathrm{(Gy^{-2})}",
        xscale = :log10,
        legend = :topleft,
        size   = (750, 500),
        DEFAULTS...,
    )

    for cond in CONDITIONS
        sub = all_lq[all_lq.tag .== cond.tag, :]
        isempty(sub) && continue
        sort!(sub, :dose_rate_Gys)
        plot!(p_beta, sub.dose_rate_Gys, sub.beta;
            label             = cond.label,
            color             = cond.color,
            linewidth         = 2,
            marker            = :circle,
            markersize        = 6,
            markerstrokewidth = 0.5,
        )
        if !all(isnan.(sub.beta_err))
            plot!(p_beta, sub.dose_rate_Gys,
                  sub.beta .+ sub.beta_err;
                fillrange         = sub.beta .- sub.beta_err,
                fillalpha         = 0.15,
                linealpha         = 0,
                color             = cond.color,
                label             = "",
            )
        end
    end

    display(p_beta)
    savefig(p_beta, joinpath(datadir, "beta_vs_doserate.png"))
    savefig(p_beta, joinpath(datadir, "beta_vs_doserate.pdf"))
    println("Saved: beta_vs_doserate")

    # ── PLOT 4: α/β ratio vs dose rate ───────────────────────────────────────
    p_ab = plot(;
        xlabel = "Dose rate (Gy/s)",
        ylabel = L"\alpha/\beta\ \mathrm{(Gy)}",
        xscale = :log10,
        legend = :topright,
        size   = (750, 500),
        DEFAULTS...,
    )

    for cond in CONDITIONS
        sub = all_lq[all_lq.tag .== cond.tag, :]
        isempty(sub) && continue
        sort!(sub, :dose_rate_Gys)
        plot!(p_ab, sub.dose_rate_Gys, sub.alpha_beta;
            label             = cond.label,
            color             = cond.color,
            linewidth         = 2,
            marker            = :circle,
            markersize        = 6,
            markerstrokewidth = 0.5,
        )
    end

    display(p_ab)
    savefig(p_ab, joinpath(datadir, "alpha_beta_vs_doserate.png"))
    savefig(p_ab, joinpath(datadir, "alpha_beta_vs_doserate.pdf"))
    println("Saved: alpha_beta_vs_doserate")

    # Save combined LQ table
    CSV.write(joinpath(datadir, "lq_params_all.csv"), all_lq)
    println("Saved: lq_params_all.csv")
end

# ── PLOT 5: β vs dose rate with α fixed at 10⁻² Gy/s ────────────────────────
# For each condition:
#   1. Take the free-fit α at the highest dose rate (10⁻² Gy/s).
#   2. Re-fit the LQ model with α fixed → only β is optimised.
#   3. Plot β(dose_rate) alongside a horizontal dashed line at the reference β.
#
# Rationale: at 10⁻² Gy/s the irradiation is quasi-instantaneous, so sublethal
# damage repair is negligible → the fit gives the "true" α for direct lethal
# events. Fixing α isolates the dose-rate effect entirely in β.

function fit_lq_beta_only(surv_df::DataFrame, alpha_fixed::Float64, tag::String;
                           fit_max_Gy::Float64 = Inf)
    lq_beta(D, p) = -alpha_fixed .* D .- p[1] .* D .^ 2   # log-space model
    dr_cols = setdiff(names(surv_df), vcat(["dose_Gy"], EXCLUDE_DR))
    doses   = surv_df.dose_Gy

    params_df = DataFrame(
        tag           = String[],
        dose_rate_Gys = Float64[],
        alpha         = Float64[],   # fixed — same for every row
        beta          = Float64[],
        beta_err      = Float64[],
        alpha_beta    = Float64[],
        r2            = Float64[],
    )

    for col in dr_cols
        dr_val = parse_dr(col)
        mask   = doses .<= fit_max_Gy
        D_fit  = Float64.(doses[mask])
        S_fit  = Float64.(surv_df[mask, col])
        valid  = S_fit .> 0
        D_fit  = D_fit[valid]
        S_fit  = S_fit[valid]
        length(D_fit) < 2 && (push!(params_df, (tag, dr_val, alpha_fixed, NaN, NaN, NaN, NaN)); continue)

        try
            logS_fit = log.(S_fit)
            fit      = curve_fit(lq_beta, D_fit, logS_fit, [0.05]; lower=[0.0])
            β        = fit.param[1]
            se       = try stderror(fit)[1] catch; NaN end
            ss_res   = sum((logS_fit .- lq_beta(D_fit, fit.param)) .^ 2)
            ss_tot   = sum((logS_fit .- mean(logS_fit)) .^ 2)
            r2       = 1.0 - ss_res / ss_tot
            ab       = β > 0 ? alpha_fixed / β : Inf
            push!(params_df, (tag, dr_val, alpha_fixed, β, se, ab, r2))
        catch e
            @warn "Beta-only fit failed for $tag / $col" exception=e
            push!(params_df, (tag, dr_val, alpha_fixed, NaN, NaN, NaN, NaN))
        end
    end
    return params_df
end

let
    ref_dr = 1e-5   # Gy/s — dose rate used to calibrate α

    p_beta_fixed = plot(;
        xlabel = "Dose rate (Gy/s)",
        ylabel = L"\beta\ \mathrm{(Gy^{-2})}",
        title  = latexstring(@sprintf("\\beta\\,(\\alpha\\,\\mathrm{fixed\\ at\\ %.0e\\ Gy/s})", ref_dr)),
        xscale = :log10,
        legend = :topright,
        size   = (750, 500),
        DEFAULTS...,
    )

    for cond in CONDITIONS
        surv_path = joinpath(datadir, "survival_results_$(cond.tag).csv")
        lq_path   = joinpath(datadir, "lq_params_$(cond.tag).csv")
        isfile(surv_path) && isfile(lq_path) || (@warn "Missing files for $(cond.tag)"; continue)

        surv_df = CSV.read(surv_path, DataFrame)
        lq_free = CSV.read(lq_path,   DataFrame)
        sort!(lq_free, :dose_rate_Gys)

        # Extract α from the free fit at the reference dose rate
        ref_row = findfirst(x -> isapprox(x, ref_dr; rtol=0.05), lq_free.dose_rate_Gys)
        if isnothing(ref_row) || isnan(lq_free.alpha[ref_row])
            @warn "Reference dose rate $(ref_dr) Gy/s not found or NaN for $(cond.tag)"
            continue
        end
        alpha_fixed = lq_free.alpha[ref_row]
        beta_ref    = lq_free.beta[ref_row]   # free-fit β at reference — shown as anchor

        println(@sprintf("  %s: α fixed = %.4f (from %.0e Gy/s free fit)",
                         cond.tag, alpha_fixed, ref_dr))

        bo_df = fit_lq_beta_only(surv_df, alpha_fixed, cond.tag;
                                  fit_max_Gy = cond.fit_max_Gy)
        sort!(bo_df, :dose_rate_Gys)
        valid = .!isnan.(bo_df.beta)

        # Save beta-only params
        CSV.write(joinpath(datadir, "lq_params_beta_only_$(cond.tag).csv"), bo_df)

        plot!(p_beta_fixed, bo_df.dose_rate_Gys[valid], bo_df.beta[valid];
              label             = cond.label,
              color             = cond.color,
              linewidth         = 2,
              marker            = :circle,
              markersize        = 6,
              markerstrokewidth = 0.5,
        )

        # Error band
        if !all(isnan.(bo_df.beta_err[valid]))
            errs = bo_df.beta_err[valid]
            plot!(p_beta_fixed, bo_df.dose_rate_Gys[valid],
                  bo_df.beta[valid] .+ errs;
                  fillrange = bo_df.beta[valid] .- errs,
                  fillalpha = 0.15,
                  linealpha = 0,
                  color     = cond.color,
                  label     = "",
            )
        end

        # Star marker at the reference point
        ref_bo = findfirst(x -> isapprox(x, ref_dr; rtol=0.05), bo_df.dose_rate_Gys)
        if !isnothing(ref_bo) && !isnan(bo_df.beta[ref_bo])
            scatter!(p_beta_fixed, [bo_df.dose_rate_Gys[ref_bo]], [bo_df.beta[ref_bo]];
                     color      = cond.color,
                     marker     = :star5,
                     markersize = 11,
                     label      = "",
            )
        end
    end

    vline!(p_beta_fixed, [ref_dr];
           linestyle = :dash, color = :black, alpha = 0.5,
           label = latexstring(@sprintf("\\mathrm{ref\\ %.0e\\ Gy/s}", ref_dr)))

    display(p_beta_fixed)
    savefig(p_beta_fixed, joinpath(datadir, "beta_vs_doserate_fixed_alpha.png"))
    savefig(p_beta_fixed, joinpath(datadir, "beta_vs_doserate_fixed_alpha.pdf"))
    println("Saved: beta_vs_doserate_fixed_alpha")
end

# ── PLOT 6: Survival curves with fixed α (β free per dose rate) ──────────────
# Same layout as PLOT 1, but curves use the beta-only calibration:
#   S(D) = exp(-α_ref · D  -  β(dose_rate) · D²)
# where α_ref is from the free fit at ref_dr = 1e-5 Gy/s.
let
    ref_dr       = 1e-5
    surv_panels6 = Plots.Plot[]
    conds_out6   = []

    for cond in CONDITIONS
        surv_path = joinpath(datadir, "survival_results_$(cond.tag).csv")
        bo_path   = joinpath(datadir, "lq_params_beta_only_$(cond.tag).csv")
        isfile(surv_path) && isfile(bo_path) || (@warn "Missing files for $(cond.tag) (PLOT 6)"; continue)

        surv_df = CSV.read(surv_path, DataFrame)
        bo_df   = CSV.read(bo_path,   DataFrame)
        sort!(bo_df, :dose_rate_Gys)

        doses   = surv_df.dose_Gy
        dr_cols = setdiff(names(surv_df), vcat(["dose_Gy"], EXCLUDE_DR))
        D_fine  = range(0, maximum(doses), length=300)

        alpha_fixed = bo_df.alpha[1]   # same for all rows

        p = plot(;
            xlabel        = "Dose (Gy)",
            ylabel        = "Survival fraction",
            title         = latexstring(
                                string(cond.label) *
                                @sprintf(",\\ \\alpha=%.3f\\ (fixed\\ %.0e\\ \\mathrm{Gy/s})",
                                        alpha_fixed, ref_dr)),
            yscale        = :log10,
            legend        = :bottomleft,
            size          = (650, 500),
            bottom_margin = 10Plots.mm,
            left_margin   = 10Plots.mm,
            DEFAULTS...,
        )

        for (k, col) in enumerate(dr_cols)
            col_c  = dr_palette[min(k, length(dr_palette))]
            dr_val = parse_dr(col)
            lbl    = latexstring(@sprintf("%.0e~\\mathrm{Gy/s}", dr_val))

            # LQ curve with fixed α, β from beta-only fit for this dose rate
            bo_row = findfirst(x -> isapprox(x, dr_val; rtol=0.05), bo_df.dose_rate_Gys)
            if !isnothing(bo_row) && !isnan(bo_df.beta[bo_row])
                β = bo_df.beta[bo_row]
                plot!(p, collect(D_fine), lq.(collect(D_fine), alpha_fixed, β);
                    label     = lbl,
                    color     = col_c,
                    linewidth = 1.8,
                )
            end
        end

        push!(surv_panels6, p)
        push!(conds_out6, cond)
    end

    if !isempty(surv_panels6)
        ncols   = length(surv_panels6)
        p_surv6 = plot(surv_panels6...;
                       layout = (1, ncols),
                       size   = (650 * ncols, 500),
                       DEFAULTS...)
        display(p_surv6)
        savefig(p_surv6, joinpath(datadir, "survival_curves_fixed_alpha_all.png"))
        savefig(p_surv6, joinpath(datadir, "survival_curves_fixed_alpha_all.pdf"))
        println("Saved: survival_curves_fixed_alpha_all")

        for (cond, p) in zip(conds_out6, surv_panels6)
            savefig(p, joinpath(datadir, "survival_curves_fixed_alpha_$(cond.tag).png"))
            savefig(p, joinpath(datadir, "survival_curves_fixed_alpha_$(cond.tag).pdf"))
            println("Saved: survival_curves_fixed_alpha_$(cond.tag)")
        end
    end
end

println("\nAll plots saved to $datadir")
