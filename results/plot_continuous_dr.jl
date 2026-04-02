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
    (tag="12C_10MeV",  label=L"$^{12}$C 10 MeV/u",  color=:firebrick,  fit_max_Gy=1.3),
    (tag="12C_100MeV", label=L"$^{12}$C 100 MeV/u", color=:darkorange, fit_max_Gy=3.0),
    (tag="1H_100MeV",  label=L"$^{1}$H 100 MeV/u",  color=:royalblue,  fit_max_Gy=3.0),
]

# Dose-rate columns to exclude from all plots
const EXCLUDE_DR = String[]

# ── LQ fitting helper ─────────────────────────────────────────────────────────
# Parse dose-rate value (Gy/s) from column name, e.g. "dr_1e-05Gys" → 1e-5
function parse_dr(col::AbstractString)
    s = replace(col, "dr_" => "", "Gys" => "")
    return parse(Float64, s)
end

function fit_lq_survival(surv_df::DataFrame, tag::String; fit_max_Gy::Float64 = Inf)
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
global_ymin    = Inf  # track minimum survival across all conditions

for cond in CONDITIONS
    surv_path = joinpath(datadir, "survival_results_$(cond.tag).csv")
    lq_path   = joinpath(datadir, "lq_params_$(cond.tag).csv")
    isfile(surv_path) && isfile(lq_path) || (@warn "Missing files for $(cond.tag)"; continue)

    surv_df = CSV.read(surv_path, DataFrame)
    lq_df   = CSV.read(lq_path,   DataFrame)

    doses   = surv_df.dose_Gy
    dr_cols = setdiff(names(surv_df), vcat(["dose_Gy"], EXCLUDE_DR))
    D_fine  = range(0, min(3.0, maximum(doses)), length=300)

    # Track global y minimum (skip zeros/negatives which break log scale)
    for col in dr_cols
        vals = filter(>(0), surv_df[!, col])
        isempty(vals) || (global_ymin = min(global_ymin, minimum(vals)))
    end

    p = plot(;
        xlabel        = "Dose (Gy)",
        ylabel        = "Survival fraction",
        yscale        = :log10,
        xlims         = (0, 3.0),
        legend        = :topright,
        size          = (650, 500),
        bottom_margin = 10Plots.mm,
        DEFAULTS...,
    )

    for (k, col) in enumerate(dr_cols)
        col_c  = dr_palette[min(k, length(dr_palette))]
        row    = lq_df[lq_df.dose_rate_Gys .== lq_df.dose_rate_Gys[k], :]
        lbl    = latexstring(@sprintf("%.0e~\\mathrm{Gy/s}", lq_df.dose_rate_Gys[k]))

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

# Apply shared y-axis limits to all panels so every plot has the same scale
if isfinite(global_ymin) && !isempty(surv_panels)
    ylo = 10 ^ floor(log10(global_ymin))   # round down to nearest decade
    for p in surv_panels
        ylims!(p, (ylo, 1.0))
    end
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
        legend = :topright,
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

println("\nAll plots saved to $datadir")
