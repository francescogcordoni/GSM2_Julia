using CSV, DataFrames, Plots

# ── Paths ─────────────────────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data", "inverse_doserate")

df       = CSV.read(joinpath(datadir, "phase_times.csv"),      DataFrame)
nsplit   = CSV.read(joinpath(datadir, "nsplit_reference.csv"), DataFrame)

# ── Color palette ─────────────────────────────────────────────────────────────
type_order = ["1H_2", "1H_10", "1H_100", "12C_15", "12C_20", "12C_80"]
types      = filter(t -> t in unique(df.type), type_order)

palette_colors = [:royalblue, :deepskyblue, :cyan3,
                  :firebrick, :tomato, :lightsalmon]
color_map  = Dict(t => c for (t, c) in zip(type_order, palette_colors))
nsplit_map = Dict(row.type => row.Nsplit for row in eachrow(nsplit))

function label_str(t)
    s = replace(t, "_" => "  ")
    s = replace(s, "1H"  => "¹H")
    replace(s, "12C" => "¹²C")
end

const DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

# ── PLOT 1: Survival vs inter-fraction time ───────────────────────────────────
p1 = plot(;
    xlabel      = "Time between fractions (h)",
    ylabel      = "Survival probability",
    legend      = :topleft,
    legendtitle = "Particle / E (MeV/u)",
    size        = (900, 550),
    DEFAULTS...,
)

for t in types
    sub = sort(df[df.type .== t, :], :time)
    plot!(p1, sub.time, sub.surv_prob;
        label             = label_str(t),
        color             = color_map[t],
        linewidth         = 2,
        marker            = :circle,
        markersize        = 4,
        markerstrokewidth = 0.5,
    )
    if haskey(nsplit_map, t)
        hline!(p1, [nsplit_map[t]];
            color     = color_map[t],
            linestyle = :dash,
            linewidth = 1.2,
            label     = "",
        )
    end
end

display(p1)
savefig(p1, joinpath(datadir, "surv_prob_vs_time.png"))
savefig(p1, joinpath(datadir, "surv_prob_vs_time.pdf"))
println("Saved: surv_prob_vs_time")

# ── PLOT 2: Alive cells before shot 2 ────────────────────────────────────────
p2 = plot(;
    xlabel      = "Time between fractions (h)",
    ylabel      = "Mean alive cells before shot 2",
    legend      = :topleft,
    legendtitle = "Particle / E (MeV/u)",
    size        = (900, 550),
    DEFAULTS...,
)

for t in types
    sub = sort(df[df.type .== t, :], :time)
    plot!(p2, sub.time, sub.Nalive;
        label             = label_str(t),
        color             = color_map[t],
        linewidth         = 2,
        marker            = :circle,
        markersize        = 4,
        markerstrokewidth = 0.5,
    )
end

display(p2)
savefig(p2, joinpath(datadir, "alive_vs_time.png"))
savefig(p2, joinpath(datadir, "alive_vs_time.pdf"))
println("Saved: alive_vs_time")

println("\nAll plots saved to $datadir")
