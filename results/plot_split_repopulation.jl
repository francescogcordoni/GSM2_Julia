using CSV, DataFrames, Plots, JLD2

# ── Load data ──────────────────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data", "split_repopulation")

surv_df  = CSV.read(joinpath(datadir, "surv_curves.csv"),       DataFrame)
pre_full = CSV.read(joinpath(datadir, "phase_times_pre_full.csv"), DataFrame)
pre_G1   = CSV.read(joinpath(datadir, "phase_times_pre_G1.csv"),   DataFrame)
meta     = CSV.read(joinpath(datadir, "metadata.csv"),          DataFrame)

surv_df = surv_df[surv_df.time .<= 30., :]
Nsplit         = meta.Nsplit[1]
Nsplit_2_1H_50 = meta.Nsplit_2_1H_50[1]
particle       = meta.particle[1]
E              = meta.E[1]
dose           = meta.dose[1]

times_split = surv_df.time

# ── Plot 1: Survival vs inter-fraction time ────────────────────────────────────
p1 = plot(
    xlabel      = "Time between fractions (h)",
    ylabel      = "Survival probability",
    framestyle  = :box,
    grid        = true,
    gridalpha   = 0.3,
    size        = (800, 500),
    dpi         = 600,
    fontfamily  = "Computer Modern",
    margin      = 5Plots.mm,
    legend      = :topleft,
)

plot!(p1, times_split, surv_df.surv_abm_full;
    label     = "ABM — full cycle",
    color     = :royalblue,
    linewidth = 2,
    marker    = :circle,
    markersize = 4,
    markerstrokewidth = 0.5,
)
plot!(p1, times_split, surv_df.surv_abm_G1;
    label     = "ABM — G1 only",
    color     = :darkorange,
    linewidth = 2,
    marker    = :diamond,
    markersize = 4,
    markerstrokewidth = 0.5,
)
plot!(p1, times_split, surv_df.surv_noabm;
    label     = "No repopulation",
    color     = :seagreen,
    linewidth = 2,
    marker    = :square,
    markersize = 4,
    markerstrokewidth = 0.5,
)

display(p1)
savefig(p1, joinpath(datadir, "surv_vs_time.png"))
savefig(p1, joinpath(datadir, "surv_vs_time.pdf"))

# ── Plot 2: Alive cells before shot 2 (full cycle) ────────────────────────────
p2 = plot(
    xlabel     = "Time between fractions (h)",
    ylabel     = "Mean alive cells before shot 2",
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    size       = (800, 500),
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
    legend     = :topright,
)

plot!(p2, pre_full.time, pre_full.Nalive;
    label     = "Full cycle",
    color     = :royalblue,
    linewidth = 2,
    marker    = :circle,
    markersize = 4,
    markerstrokewidth = 0.5,
)
plot!(p2, pre_G1.time, pre_G1.Nalive;
    label     = "G1 only",
    color     = :darkorange,
    linewidth = 2,
    marker    = :diamond,
    markersize = 4,
    markerstrokewidth = 0.5,
)

display(p2)
savefig(p2, joinpath(datadir, "alive_vs_time.png"))
savefig(p2, joinpath(datadir, "alive_vs_time.pdf"))

# ── Plot 3: Phase distribution before shot 2 (full cycle) ─────────────────────
phase_colors = Dict("G0" => :gray60, "G1" => :royalblue,
                    "S"  => :seagreen, "G2" => :darkorange, "M" => :firebrick)

p3 = plot(
    xlabel     = "Time between fractions (h)",
    ylabel     = "Mean cells per phase",
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    size       = (800, 500),
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
    legend     = :topright,
)

pre_full_p3 = pre_full[pre_full.time .!= 48.0, :]

for ph in ("G0", "G1", "S", "G2", "M")
    plot!(p3, pre_full_p3.time, pre_full_p3[!, ph];
        label     = ph,
        color     = phase_colors[ph],
        linewidth = 2,
        marker    = :circle,
        markersize = 3,
        markerstrokewidth = 0.5,
    )
end

display(p3)
savefig(p3, joinpath(datadir, "phases_vs_time.png"))
savefig(p3, joinpath(datadir, "phases_vs_time.pdf"))

println("Plots saved to $datadir")
