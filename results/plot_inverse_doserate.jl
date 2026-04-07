using CSV, DataFrames, Plots, LaTeXStrings

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
    m = match(r"^(1H|12C)_(\d+)$", t)
    m === nothing && return t
    particle, energy = m.captures
    sup = particle == "12C" ? "12" : "1"
    sym = particle == "12C" ? "C" : "H"
    return LaTeXString("\$^{$sup}\$$sym $energy MeV/u")
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
    sub = sort(df[df.type .== t .&& df.time .!= 48.0, :], :time)
    plot!(p1, sub.time, sub.surv_prob;
        label             = label_str(t),
        color             = color_map[t],
        linewidth         = 2,
        marker            = :circle,
        markersize        = 4,
        markerstrokewidth = 0.5,
    )
    if haskey(nsplit_map, t)

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

# ── PLOT 3: Cell-cycle distribution before shot 2 ────────────────────────────
const PHASE_COLS   = ["G0", "G1", "S", "G2", "M"]
const PHASE_COLORS = Dict(
    "G0" => :gray,
    "G1" => :steelblue,
    "S"  => :seagreen,
    "G2" => :darkorange,
    "M"  => :firebrick,
)
const PHASE_STYLES = Dict(
    "G0" => :dash,
    "G1" => :solid,
    "S"  => :solid,
    "G2" => :solid,
    "M"  => :solid,
)

# One panel per condition type
phase_panels = Plots.Plot[]

for t in types
    sub = sort(df[df.type .== t, :], :time)
    Nalive = max.(sub.Nalive, 1)

    p = plot(;
        xlabel    = "Time between fractions (h)",
        ylabel    = "Phase fraction",
        legend    = :topright,
        ylims     = (0, 1),
        size      = (650, 480),
        DEFAULTS...,
    )

    for phase in PHASE_COLS
        frac = sub[!, phase] ./ Nalive
        plot!(p, sub.time, frac;
            label     = phase,
            color     = PHASE_COLORS[phase],
            linestyle = PHASE_STYLES[phase],
            linewidth = 1.8,
        )
    end

    push!(phase_panels, p)
end

if !isempty(phase_panels)
    ncols   = length(phase_panels)
    p_phase = plot(phase_panels...;
                   layout = (1, ncols),
                   size   = (650 * ncols, 480),
                   DEFAULTS...)
    display(p_phase)
    savefig(p_phase, joinpath(datadir, "phase_dist_vs_time.png"))
    savefig(p_phase, joinpath(datadir, "phase_dist_vs_time.pdf"))
    println("Saved: phase_dist_vs_time")

    for (t, p) in zip(types, phase_panels)
        savefig(p, joinpath(datadir, "phase_dist_$(t).png"))
        savefig(p, joinpath(datadir, "phase_dist_$(t).pdf"))
    end
end

println("\nAll plots saved to $datadir")
