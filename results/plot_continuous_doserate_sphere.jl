using CSV, DataFrames
using Plots
using Statistics
using Printf
using LaTeXStrings

# ============================================================
# Configuration — must match continuous_doserate_sphere.jl
# ============================================================
indir  = joinpath(@__DIR__, "..", "data", "continuous_doserate_sphere")
outdir = indir
au     = 4.0

# Dose-rate conditions: (Gy/s value, display color)
# Gy/h = Gy/s × 3600 × au
DOSERATES_GYS = [1e-6, 1e-5, 1e-2]

DR_COLORS = Dict(
    1e-6 => :green,
    1e-5 => :steelblue,
    1e-2 => :darkorange)

const DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

dose_label = "1p5Gy"
dr_label(dr_gys::Float64) = @sprintf("%.0e", dr_gys)
dr_plotlabel(dr_gys::Float64) = latexstring(@sprintf("%.0e~\\mathrm{Gy/s}", dr_gys))

# ============================================================
# Load summary
# ============================================================
summary_df = CSV.read(joinpath(indir, "summary_$(dose_label).csv"), DataFrame)
println("Summary loaded:")
println(summary_df)

# Build a lookup: dose_rate_gys → survival_fraction and abm_wall_time
sf_lookup        = Dict(zip(summary_df.dose_rate_gys, summary_df.survival_fraction))
wall_time_lookup = Dict(zip(summary_df.dose_rate_gys, summary_df.irrad_time_h))

# ============================================================
# Load time series for each dose rate
# ============================================================
# Returns (ts_irrad, ts_post) DataFrames for a given Gy/s rate
function load_ts(label::String, dr_gys::Float64)
    lbl       = "$(label)_dr_$(dr_label(dr_gys))Gys"
    path_irr  = joinpath(indir, "$(lbl)_ts_irrad.csv")
    path_post = joinpath(indir, "$(lbl)_ts_post.csv")
    ts_irrad  = isfile(path_irr)  ? CSV.read(path_irr,  DataFrame) : nothing
    ts_post   = isfile(path_post) ? CSV.read(path_post, DataFrame) : nothing
    isnothing(ts_irrad)  && @warn "Not found: $path_irr"
    isnothing(ts_post)   && @warn "Not found: $path_post"
    return ts_irrad, ts_post
end

ts_data = Dict{Float64, NamedTuple}()
for dr_gys in DOSERATES_GYS
    ts_irrad, ts_post = load_ts(dose_label, dr_gys)
    (isnothing(ts_irrad) || isnothing(ts_post)) && continue
    dr_gyh = dr_gys * 3600.0 * au
    ts_data[dr_gys] = (
        ts_irrad      = ts_irrad,
        ts_post       = ts_post,
        dr_gyh        = dr_gyh,
        sf            = get(sf_lookup,        dr_gys, NaN),
        abm_wall_time = get(wall_time_lookup, dr_gys, NaN))
    println("Loaded: $(dr_label(dr_gys)) Gy/s")
end

# ============================================================
# PLOT 1: Dose-rate comparison — total cells over time
# ============================================================
p_total = plot(;
    xlabel = "Time (h)",
    ylabel = "Total cells",
    legend = :topright,
    size   = (900, 550),
    DEFAULTS...,
)

for dr_gys in sort(DOSERATES_GYS)
    haskey(ts_data, dr_gys) || continue
    res   = ts_data[dr_gys]
    col   = get(DR_COLORS, dr_gys, :black)
    t_all = vcat(res.ts_irrad.time,        res.ts_post.time)
    c_all = vcat(res.ts_irrad.total_cells,  res.ts_post.total_cells)
    plot!(p_total, t_all, c_all;
          label = dr_plotlabel(dr_gys),
          lw    = 2,
          color = col)
    vline!(p_total, [res.abm_wall_time];
           label = "", color = col, lw = 1.2, linestyle = :dot)
end

display(p_total)
savefig(p_total, joinpath(outdir, "survival_$(dose_label).png"))
savefig(p_total, joinpath(outdir, "survival_$(dose_label).pdf"))
println("Saved: survival_$(dose_label)")

# ============================================================
# PLOT 2: Phase breakdown — one panel per dose rate
# ============================================================
phase_colors = [:gray, :steelblue, :green, :orange, :red]
phase_labels = ["G0", "G1", "S", "G2", "M"]
phase_fields = [:g0_cells, :g1_cells, :s_cells, :g2_cells, :m_cells]

sorted_rates = sort(DOSERATES_GYS)
n_rates      = length(sorted_rates)

phase_panels = Plots.Plot[]
for dr_gys in sorted_rates
    haskey(ts_data, dr_gys) || continue
    res   = ts_data[dr_gys]
    t_all = vcat(res.ts_irrad.time, res.ts_post.time)

    p = plot(;
        xlabel = "Time (h)",
        ylabel = "Cells",
        legend = :topright,
        size   = (650, 480),
        DEFAULTS...,
    )
    for (fi, fld) in enumerate(phase_fields)
        y_all = vcat(res.ts_irrad[!, fld], res.ts_post[!, fld])
        plot!(p, t_all, y_all;
              label = phase_labels[fi],
              color = phase_colors[fi],
              lw    = 1.8)
    end
    vline!(p, [res.abm_wall_time];
           label = "", color = :black, lw = 1.2, linestyle = :dot)

    push!(phase_panels, p)
end

if !isempty(phase_panels)
    p_phases = plot(phase_panels...;
                    layout = (1, length(phase_panels)),
                    size   = (650 * length(phase_panels), 480),
                    DEFAULTS...)
    display(p_phases)
    savefig(p_phases, joinpath(outdir, "phases_$(dose_label).png"))
    savefig(p_phases, joinpath(outdir, "phases_$(dose_label).pdf"))
    println("Saved: phases_$(dose_label)")

    for (i, dr_gys) in enumerate(sorted_rates)
        savefig(phase_panels[i], joinpath(outdir, "phases_$(dose_label)_$(dr_label(dr_gys))Gys.png"))
        savefig(phase_panels[i], joinpath(outdir, "phases_$(dose_label)_$(dr_label(dr_gys))Gys.pdf"))
    end
end

# ============================================================
# PLOT 3: Normalised survival (N/N₀) over time
# ============================================================
N0 = summary_df.Ntot[1]   # same geometry for all rates — use first row

p_norm = plot(;
    xlabel = "Time (h)",
    ylabel = L"N/N_0",
    legend = :topright,
    size   = (900, 500),
    DEFAULTS...,
)

for dr_gys in sort(DOSERATES_GYS)
    haskey(ts_data, dr_gys) || continue
    res   = ts_data[dr_gys]
    col   = get(DR_COLORS, dr_gys, :black)
    t_all = vcat(res.ts_irrad.time,        res.ts_post.time)
    c_all = vcat(res.ts_irrad.total_cells,  res.ts_post.total_cells)
    plot!(p_norm, t_all, c_all ./ N0;
          label = dr_plotlabel(dr_gys), lw = 2, color = col)
    vline!(p_norm, [res.abm_wall_time];
           label = "", color = col, lw = 1.2, linestyle = :dot)
end
hline!(p_norm, [1.0]; color = :black, ls = :dash, lw = 1, label = L"N_0")
display(p_norm)
savefig(p_norm, joinpath(outdir, "normalised_survival_$(dose_label).png"))
savefig(p_norm, joinpath(outdir, "normalised_survival_$(dose_label).pdf"))
println("Saved: normalised_survival_$(dose_label)")

# ============================================================
# FINAL PRINT
# ============================================================
println("\n", "="^60)
println("ALL PLOTS SAVED TO $outdir/")
println("="^60)
println("Files written:")
for f in filter(f -> endswith(f, ".png") || endswith(f, ".pdf"), sort(readdir(outdir)))
    println("  $outdir/$f")
end
