using CSV, DataFrames
using Plots
using Colors
using Statistics

# ============================================================
# Configuration
# ============================================================
snapshot_hours = [0, 4, 12, 20, 24, 48, 72]
indir  = joinpath(@__DIR__, "..", "data", "spheroid_temporal_evolution")
outdir = indir

CONDITIONS = [
    (label="1H_80MeV",   particle="1H",  energy=80.0, color=:steelblue),
    (label="12C_80MeVu", particle="12C", energy=80.0, color=:red),
]

const PLOT_DEFAULTS = (
    framestyle = :box,
    grid       = true,
    gridalpha  = 0.3,
    dpi        = 600,
    fontfamily = "Computer Modern",
    margin     = 5Plots.mm,
)

# ============================================================
# Load data
# ============================================================
summary_df = CSV.read(joinpath(indir, "summary.csv"), DataFrame)

ts_data = Dict{String, DataFrame}()
for cond in CONDITIONS
    path = joinpath(indir, "$(cond.label)_ts.csv")
    isfile(path) ? ts_data[cond.label] = CSV.read(path, DataFrame) :
                    @warn "Not found: $path"
end

cell_df_pristine = CSV.read(joinpath(indir, "cell_df_pristine.csv"), DataFrame)

# ============================================================
# Helpers
# ============================================================
function load_snapshots(label::String, hours::Vector{Int})
    snaps = Dict{Int, DataFrame}()
    for hr in hours
        path = joinpath(indir, "$(label)_snap_t$(hr)h.csv")
        isfile(path) && (snaps[hr] = CSV.read(path, DataFrame))
    end
    return snaps
end

function attach_z(snap_df::DataFrame, ref::DataFrame)::DataFrame
    z_lookup = Dict{Int32,Int32}(zip(ref.index, ref.z))
    df = copy(snap_df)
    df[!, :z] = Int32[get(z_lookup, idx, Int32(0)) for idx in df.index]
    return df
end

const PHASE_ORDER  = ["G0", "G1", "S", "G2", "M"]
const PHASE_COLORS = Dict(
    "G0" => RGB(0.55, 0.55, 0.55),
    "G1" => RGB(0.27, 0.51, 0.71),
    "S"  => RGB(0.18, 0.63, 0.34),
    "G2" => RGB(0.93, 0.60, 0.13),
    "M"  => RGB(0.80, 0.15, 0.15),
)

function plot_phases(ts_df::DataFrame)
    t = ts_df.time
    p = plot(; xlabel="Time (h)", ylabel="Cell count",
               legend=:topright, size=(700, 420), PLOT_DEFAULTS...)
    plot!(p, t, ts_df.total_cells; label="Alive", lw=2,   color=:black)
    plot!(p, t, ts_df.g1_cells;   label="G1",    lw=1.5, color=:steelblue)
    plot!(p, t, ts_df.s_cells;    label="S",     lw=1.5, color=:green)
    plot!(p, t, ts_df.g2_cells;   label="G2",    lw=1.5, color=:orange)
    plot!(p, t, ts_df.m_cells;    label="M",     lw=1.5, color=:red)
    plot!( p, t, ts_df.g0_cells;   label="G0",    lw=1.5, color=:gray,
            linestyle=:dash)
    return p
end

function plot_phase_fractions(ts_df::DataFrame)
    t   = ts_df.time
    tot = max.(ts_df.total_cells, 1)
    p = plot(; xlabel="Time (h)", ylabel="Phase fraction",
                legend=:topright, ylims=(0, 1),
                size=(700, 420), PLOT_DEFAULTS...)
    plot!(p, t, ts_df.g1_cells ./ tot; label="G1", lw=2,
            color=:steelblue, fill=(0, :steelblue, 0.3))
    plot!(p, t, ts_df.s_cells  ./ tot; label="S",  lw=2,
            color=:green,    fill=(0, :green,    0.3))
    plot!(p, t, ts_df.g2_cells ./ tot; label="G2", lw=2,
            color=:orange,   fill=(0, :orange,   0.3))
    plot!(p, t, ts_df.m_cells  ./ tot; label="M",  lw=2,
            color=:red,      fill=(0, :red,      0.3))
    plot!(p, t, ts_df.g0_cells ./ tot; label="G0", lw=2,
            color=:gray,     fill=(0, :gray,     0.3))
    return p
end

# 3-D half-sphere scatter colored by cell-cycle phase
function plot_spheroid_3d(df::DataFrame)
    alive  = df[df.is_cell .== 1, :]
    half   = alive[alive.x .>= 0, :]

    # Equal axis limits so the sphere is not distorted
    all_c = vcat(half.x, half.y, half.z)
    cmin, cmax = minimum(all_c), maximum(all_c)
    pad  = (cmax - cmin) * 0.04
    lims = (cmin - pad, cmax + pad)

    p = nothing
    for phase in PHASE_ORDER
        sub = half[half.cell_cycle .== phase, :]
        nrow(sub) == 0 && continue
        kw = (
            markersize        = 3,
            markerstrokewidth = 0.0,
            markeralpha       = 0.80,
            color             = PHASE_COLORS[phase],
            label             = phase,
            xlabel            = "x (µm)",
            ylabel            = "y (µm)",
            zlabel            = "z (µm)",
            legend            = :topright,
            colorbar          = false,
            size              = (700, 600),
            camera            = (320, 30),
            xlims             = lims,
            ylims             = lims,
            zlims             = lims,
            framestyle        = :box,
            dpi               = 600,
            fontfamily        = "Computer Modern",
            margin            = 5Plots.mm,
        )
        if p === nothing
            p = scatter(sub.x, sub.y, sub.z; kw...)
        else
            scatter!(p, sub.x, sub.y, sub.z; kw...)
        end
    end
    return p
end

# 2-D equatorial cross-section coloured by cell-cycle phase
function plot_spheroid_slice(df::DataFrame; z_tol::Int=30)
    alive = df[df.is_cell .== 1, :]
    if hasproperty(alive, :z)
        slice = alive[abs.(alive.z) .<= z_tol, :]
    else
        slice = alive
    end

    p = plot(; xlabel="x (µm)", ylabel="y (µm)",
                aspect_ratio = :equal,
                legend       = :topright,
                size         = (600, 600),
                grid         = false,
                framestyle   = :box,
                dpi          = 600,
                fontfamily   = "Computer Modern",
                margin       = 5Plots.mm,
    )

    for phase in PHASE_ORDER
        sub = slice[slice.cell_cycle .== phase, :]
        nrow(sub) == 0 && continue
        scatter!(p, sub.x, sub.y;
            label             = phase,
            color             = PHASE_COLORS[phase],
            markersize        = 4,
            markerstrokewidth = 0.0,
            markeralpha       = 0.85,
        )
    end
    return p
end

# ============================================================
# PLOT 1: Total alive cells over time
# ============================================================
p_total = plot(; xlabel="Time (h)", ylabel="Alive cells",
                    legend=:topleft, size=(900, 500), PLOT_DEFAULTS...)

for cond in CONDITIONS
    haskey(ts_data, cond.label) || continue
    ts_df  = ts_data[cond.label]
    sf_row = filter(r -> r.condition == cond.label, summary_df)
    sf_val = nrow(sf_row) > 0 ? sf_row.survival_fraction[1] : NaN
    label  = replace(cond.label, "_" => "  ") * "  (SF=$(round(sf_val, digits=3)))"
    plot!(p_total, ts_df.time, ts_df.total_cells;
            label=label, lw=2, color=cond.color)
end

display(p_total)
savefig(p_total, joinpath(outdir, "total_cells.png"))
savefig(p_total, joinpath(outdir, "total_cells.pdf"))
println("Saved: total_cells")

# ============================================================
# PLOT 2: Phase breakdown — one panel per condition
# ============================================================
phase_panels = [plot_phases(ts_data[c.label])
                for c in CONDITIONS if haskey(ts_data, c.label)]

ymax_phases = maximum(ylims(p)[2] for p in phase_panels)
for p in phase_panels
    ylims!(p, 0, ymax_phases)
end

p_phases = plot(phase_panels...; layout=(1, length(phase_panels)), legend=:topleft,
                size=(700 * length(phase_panels), 420), dpi=600)
display(p_phases)
savefig(p_phases, joinpath(outdir, "phase_breakdown.png"))
savefig(p_phases, joinpath(outdir, "phase_breakdown.pdf"))
println("Saved: phase_breakdown")

# ============================================================
# PLOT 3: Normalised survival N/N₀
# ============================================================
p_sf = plot(; xlabel="Time (h)", ylabel="Relative cell number (N/N₀)",
                legend=:topright, size=(900, 500), PLOT_DEFAULTS...)

for cond in CONDITIONS
    haskey(ts_data, cond.label) || continue
    ts_df  = ts_data[cond.label]
    sf_row = filter(r -> r.condition == cond.label, summary_df)
    Ntot   = nrow(sf_row) > 0 ? sf_row.Ntot[1] : ts_df.total_cells[1]
    plot!(p_sf, ts_df.time, ts_df.total_cells ./ Ntot;
            label=replace(cond.label, "_" => "  "), lw=2, color=cond.color)
end

display(p_sf)
savefig(p_sf, joinpath(outdir, "normalised_survival.png"))
savefig(p_sf, joinpath(outdir, "normalised_survival.pdf"))
println("Saved: normalised_survival")

# ============================================================
# PLOT 4: Equatorial cross-section snapshots
# ============================================================
for cond in CONDITIONS
    snaps = load_snapshots(cond.label, snapshot_hours)
    isempty(snaps) && continue

    valid_hours = filter(h -> haskey(snaps, h), snapshot_hours)
    panels      = Plots.Plot[]

    for hr in valid_hours
        plot_df = attach_z(snaps[hr], cell_df_pristine)
        push!(panels, plot_spheroid_slice(plot_df))
    end

    isempty(panels) && continue

    # Individual files
    for (i, hr) in enumerate(valid_hours)
        fname = joinpath(outdir, "spheroid_$(cond.label)_t$(hr)h.png")
        savefig(panels[i], fname)
        println("Saved: spheroid_$(cond.label)_t$(hr)h.png")
    end

    # Combined grid — 3 columns × 3 rows
    ncols = 3
    nrows = 3
    p_grid = plot(panels...;
                    layout = (nrows, ncols),
                    size   = (600 * ncols, 600 * nrows),
                    dpi    = 300)
    display(p_grid)
    savefig(p_grid, joinpath(outdir, "spheroid_$(cond.label)_grid.png"))
    savefig(p_grid, joinpath(outdir, "spheroid_$(cond.label)_grid.pdf"))
    println("Saved: spheroid_$(cond.label)_grid")
end

# ============================================================
# PLOT 5: 3-D half-sphere snapshots
# ============================================================
for cond in CONDITIONS
    snaps = load_snapshots(cond.label, snapshot_hours)
    isempty(snaps) && continue

    valid_hours = filter(h -> haskey(snaps, h), snapshot_hours)
    panels3d    = Plots.Plot[]

    for hr in valid_hours
        plot_df = attach_z(snaps[hr], cell_df_pristine)
        push!(panels3d, plot_spheroid_3d(plot_df))
    end

    isempty(panels3d) && continue

    # Individual files
    for (i, hr) in enumerate(valid_hours)
        fname = joinpath(outdir, "spheroid_3d_$(cond.label)_t$(hr)h.png")
        savefig(panels3d[i], fname)
        println("Saved: spheroid_3d_$(cond.label)_t$(hr)h.png")
    end

    # Combined grid — 3 columns × 3 rows
    ncols = 3
    nrows = 3
    p_grid3d = plot(panels3d...;
                    layout = (nrows, ncols),
                    size   = (700 * ncols, 600 * nrows),
                    dpi    = 150)
    display(p_grid3d)
    savefig(p_grid3d, joinpath(outdir, "spheroid_3d_$(cond.label)_grid.png"))
    savefig(p_grid3d, joinpath(outdir, "spheroid_3d_$(cond.label)_grid.pdf"))
    println("Saved: spheroid_3d_$(cond.label)_grid")
end

# ============================================================
# PLOT 6: Phase fractions over time
# ============================================================
frac_panels = [plot_phase_fractions(ts_data[c.label])
                for c in CONDITIONS if haskey(ts_data, c.label)]

p_frac = plot(frac_panels...; layout=(1, length(frac_panels)),
              size=(700 * length(frac_panels), 420), dpi=600)
display(p_frac)
savefig(p_frac, joinpath(outdir, "phase_fractions.png"))
savefig(p_frac, joinpath(outdir, "phase_fractions.pdf"))
println("Saved: phase_fractions")

println("\n", "="^60)
println("ALL PLOTS SAVED TO $outdir")
println("="^60)
