using CSV, DataFrames, Plots, Printf

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
CONDITIONS = [
    (tag="12C_10MeV",  label="¹²C  10 MeV/u",  color=:firebrick),
    (tag="12C_100MeV", label="¹²C  100 MeV/u", color=:tomato),
    (tag="1H_100MeV",  label="¹H  100 MeV/u",  color=:royalblue),
]

# Dose-rate palette (one color per dose rate, shared across conditions)
dr_palette = [:navy, :steelblue, :seagreen, :darkorange]

lq(D, α, β) = exp.(-α .* D .- β .* D .^ 2)

# ── PLOT 1: Dose-survival curves + LQ fit (one panel per condition) ───────────
surv_panels = Plots.Plot[]

for cond in CONDITIONS
    surv_path = joinpath(datadir, "survival_results_$(cond.tag).csv")
    lq_path   = joinpath(datadir, "lq_params_$(cond.tag).csv")
    isfile(surv_path) && isfile(lq_path) || (@warn "Missing files for $(cond.tag)"; continue)

    surv_df = CSV.read(surv_path, DataFrame)
    lq_df   = CSV.read(lq_path,   DataFrame)

    doses   = surv_df.dose_Gy
    dr_cols = setdiff(names(surv_df), ["dose_Gy"])
    D_fine  = range(0, maximum(doses), length=300)

    p = plot(;
        xlabel = "Dose (Gy)",
        ylabel = "Survival fraction",
        yscale = :log10,
        legend = :topright,
        size   = (650, 500),
        DEFAULTS...,
    )

    for (k, col) in enumerate(dr_cols)
        row = lq_df[lq_df.dose_rate_Gys .== lq_df.dose_rate_Gys[k], :]
        isempty(row) && continue
        α, β = row.alpha[1], row.beta[1]

        dr_val = lq_df.dose_rate_Gys[k]
        lbl    = @sprintf("%.0e Gy/s", dr_val)
        col_c  = dr_palette[min(k, length(dr_palette))]

        scatter!(p, doses, surv_df[!, col];
            label             = lbl,
            color             = col_c,
            markersize        = 5,
            markerstrokewidth = 0.5,
        )
        plot!(p, collect(D_fine), lq(collect(D_fine), α, β);
            label     = "",
            color     = col_c,
            linewidth = 1.8,
            linestyle = :dash,
        )
    end

    push!(surv_panels, p)
end

if !isempty(surv_panels)
    ncols  = length(surv_panels)
    p_surv = plot(surv_panels...;
                  layout = (1, ncols),
                  size   = (650 * ncols, 500),
                  dpi    = 600)
    display(p_surv)
    savefig(p_surv, joinpath(datadir, "survival_curves_all.png"))
    savefig(p_surv, joinpath(datadir, "survival_curves_all.pdf"))
    println("Saved: survival_curves_all")

    # Also save individual panels
    for (cond, p) in zip(CONDITIONS, surv_panels)
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
        ylabel = "α (Gy⁻¹)",
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
        ylabel = "β (Gy⁻²)",
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
        ylabel = "α/β (Gy)",
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
