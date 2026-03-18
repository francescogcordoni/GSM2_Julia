using CSV, DataFrames
using Plots
using Colors
using Statistics

# ============================================================
# Configuration — must match run_simulation.jl
# ============================================================
snapshot_hours = [0, 12, 24, 48, 72]
indir  = joinpath(@__DIR__, "..", "data", "spheroid_temporal_evolution")
outdir = @__DIR__

# Condition metadata: (label, particle, energy, line_color)
CONDITIONS = [
    (label="1H_80MeV",   particle="1H",  energy=80.0, color=:steelblue),
    (label="1H_30MeV",   particle="1H",  energy=30.0, color=:darkorange),
    (label="12C_80MeVu", particle="12C", energy=80.0, color=:red),
]

# ============================================================
# Load summary
# ============================================================
summary_df = CSV.read(joinpath(indir, "summary.csv"), DataFrame)
println("Summary loaded:")
println(summary_df)

# ============================================================
# Load time series for each condition
# ============================================================
ts_data = Dict{String, DataFrame}()
for cond in CONDITIONS
    path = joinpath(indir, "$(cond.label)_ts.csv")
    if isfile(path)
        ts_data[cond.label] = CSV.read(path, DataFrame)
        println("Loaded: $path")
    else
        @warn "Time series not found: $path"
    end
end

# ============================================================
# Load pristine cell_df (used for z-coordinate lookup in 3-D plots)
# ============================================================
cell_df_pristine = CSV.read(joinpath(indir, "cell_df_pristine.csv"), DataFrame)

# ============================================================
# Helper: load snapshots for one condition
#   Returns Dict{Int, DataFrame}  hour → DataFrame
# ============================================================
function load_snapshots(label::String, hours::Vector{Int})
    snaps = Dict{Int, DataFrame}()
    for hr in hours
        path = joinpath(indir, "$(label)_snap_t$(hr)h.csv")
        if isfile(path)
            snaps[hr] = CSV.read(path, DataFrame)
        else
            @warn "Snapshot not found: $path"
        end
    end
    return snaps
end

# ============================================================
# Helper: phase-proportion line plot
# ============================================================
function plot_phases(ts_df::DataFrame, label::String)
    t = ts_df.time
    p = plot(t, ts_df.total_cells; label="Alive", lw=2, color=:black,
             xlabel="Time (h)", ylabel="Cell count", title=label)
    plot!(p, t, ts_df.g1_cells; label="G1", lw=1.5, color=:steelblue)
    plot!(p, t, ts_df.s_cells;  label="S",  lw=1.5, color=:green)
    plot!(p, t, ts_df.g2_cells; label="G2", lw=1.5, color=:orange)
    plot!(p, t, ts_df.m_cells;  label="M",  lw=1.5, color=:red)
    plot!(p, t, ts_df.g0_cells; label="G0", lw=1.5, color=:gray,
          linestyle=:dash)
    return p
end

# ============================================================
# Helper: phase fractions over time (filled area)
# ============================================================
function plot_phase_fractions(ts_df::DataFrame, label::String)
    t   = ts_df.time
    tot = max.(ts_df.total_cells, 1)
    p = plot(;
        xlabel="Time (h)", ylabel="Phase fraction",
        title=label, legend=:topright, ylims=(0,1))
    plot!(p, t, ts_df.g1_cells ./ tot; label="G1", lw=2, color=:steelblue,
          fill=(0,:steelblue,0.3))
    plot!(p, t, ts_df.s_cells  ./ tot; label="S",  lw=2, color=:green,
          fill=(0,:green,    0.3))
    plot!(p, t, ts_df.g2_cells ./ tot; label="G2", lw=2, color=:orange,
          fill=(0,:orange,   0.3))
    plot!(p, t, ts_df.m_cells  ./ tot; label="M",  lw=2, color=:red,
          fill=(0,:red,      0.3))
    plot!(p, t, ts_df.g0_cells ./ tot; label="G0", lw=2, color=:gray,
          fill=(0,:gray,     0.3))
    return p
end

# ============================================================
# Helper: 3-D half-sphere scatter coloured by cell cycle
# ============================================================
const PHASE_ORDER  = ["G0", "G1", "S", "G2", "M"]
const PHASE_COLORS = Dict(
    "G0" => RGB(0.55, 0.55, 0.55),   # mid-gray   — quiescent
    "G1" => RGB(0.27, 0.51, 0.71),   # steel blue — gap 1
    "S"  => RGB(0.18, 0.63, 0.34),   # green      — synthesis
    "G2" => RGB(0.93, 0.60, 0.13),   # amber      — gap 2
    "M"  => RGB(0.80, 0.15, 0.15))   # red        — mitosis

function plot_spheroid_halfcut(df::DataFrame, title_str::String)
    # half-sphere cut: x ≥ 0, alive cells only
    alive = df[(df.is_cell .== 1) .& (df.x .>= 0), :]

    p = nothing
    for phase in PHASE_ORDER
        sub = alive[alive.cell_cycle .== phase, :]
        nrow(sub) == 0 && continue
        col = PHASE_COLORS[phase]
        kwargs = (
            markersize        = 4,
            markerstrokewidth = 0.1,
            markeralpha       = 0.85,
            color             = col,
            label             = phase,
            xlabel            = "x (µm)",
            ylabel            = "y (µm)",
            zlabel            = "z (µm)",
            title             = title_str,
            legend            = :topright,
            colorbar          = false,
            size              = (900, 700),
            camera            = (320, 30),
            grid              = true,
            framestyle        = :box,
        )
        if p === nothing
            p = scatter(sub.x, sub.y, sub.z; kwargs...)
        else
            scatter!(p, sub.x, sub.y, sub.z; kwargs...)
        end
    end

    if p === nothing
        p = scatter(Float64[], Float64[], Float64[];
                    title=title_str, xlabel="x (µm)",
                    ylabel="y (µm)", zlabel="z (µm)",
                    size=(900,700), camera=(320,30))
    end
    return p
end

# ============================================================
# Helper: attach z coordinates from pristine cell_df to a snapshot
# ============================================================
function attach_z(snap_df::DataFrame, cell_df_pristine::DataFrame)::DataFrame
    z_lookup = Dict{Int32, Int32}(zip(cell_df_pristine.index,
                                       cell_df_pristine.z))
    df = copy(snap_df)
    df[!, :z] = Int32[get(z_lookup, idx, Int32(0)) for idx in df.index]
    return df
end

# ============================================================
# PLOT 1: Temporal evolution of total cell count
# ============================================================
p_total = plot(;
    xlabel="Time (h)", ylabel="Alive cells",
    title="Spheroid response — 2 Gy",
    legend=:topright, size=(900, 500), dpi=150)

for cond in CONDITIONS
    haskey(ts_data, cond.label) || continue
    ts_df = ts_data[cond.label]
    sf_row = filter(r -> r.condition == cond.label, summary_df)
    sf_val = nrow(sf_row) > 0 ? sf_row.survival_fraction[1] : NaN
    plot!(p_total, ts_df.time, ts_df.total_cells;
          label="$(cond.label)  (SF=$(round(sf_val, digits=3)))",
          lw=2, color=cond.color, linestyle=:solid)
end
display(p_total)
savefig(p_total, joinpath(outdir, "total_cells.png"))
println("Saved: total_cells.png")

# ============================================================
# PLOT 2: Phase breakdown — one panel per condition
# ============================================================
phase_panels = [plot_phases(ts_data[c.label],
                             "$(c.particle) $(c.energy) MeV — 2 Gy")
                for c in CONDITIONS if haskey(ts_data, c.label)]

p_phases = plot(phase_panels...; layout=(1, length(phase_panels)),
                size=(1400, 450), dpi=150)
display(p_phases)
savefig(p_phases, joinpath(outdir, "phase_breakdown.png"))
println("Saved: phase_breakdown.png")

# ============================================================
# PLOT 3: Survival fraction over time (normalised)
# ============================================================
p_sf = plot(;
    xlabel="Time (h)", ylabel="Relative cell number (N/N₀)",
    title="Normalised survival — 2 Gy",
    legend=:topright, size=(900, 500), dpi=150)

for cond in CONDITIONS
    haskey(ts_data, cond.label) || continue
    ts_df  = ts_data[cond.label]
    sf_row = filter(r -> r.condition == cond.label, summary_df)
    Ntot   = nrow(sf_row) > 0 ? sf_row.Ntot[1] : ts_df.total_cells[1]
    norm   = ts_df.total_cells ./ Ntot
    plot!(p_sf, ts_df.time, norm; label=cond.label, lw=2, color=cond.color)
end
hline!(p_sf, [1.0]; color=:black, ls=:dash, lw=1, label="N₀")
display(p_sf)
savefig(p_sf, joinpath(outdir, "normalised_survival.png"))
println("Saved: normalised_survival.png")

# ============================================================
# PLOT 4: 3-D half-sphere snapshots
#   One figure per condition, one panel per snapshot time.
#   z coords joined from cell_df_pristine (not stored in snapshots).
# ============================================================
for cond in CONDITIONS
    snaps = load_snapshots(cond.label, snapshot_hours)
    isempty(snaps) && continue

    part_label  = "$(cond.particle) $(cond.energy) MeV"
    valid_hours = filter(h -> haskey(snaps, h), snapshot_hours)
    panels      = Plots.Plot[]

    for hr in valid_hours
        plot_df = attach_z(snaps[hr], cell_df_pristine)
        n_alive = count(plot_df.is_cell .== 1)
        push!(panels,
              plot_spheroid_halfcut(plot_df,
                  "$part_label  t=$(hr)h  (N=$n_alive)"))
    end

    isempty(panels) && continue

    # Save each time-point panel individually
    for (i, hr) in enumerate(valid_hours)
        fname = joinpath(outdir, "spheroid_$(cond.label)_t$(hr)h.png")
        savefig(panels[i], fname)
        println("Saved: $fname")
    end

    # Combined grid for at-a-glance comparison
    n     = length(panels)
    ncols = min(n, 3)
    nrows = ceil(Int, n / ncols)
    p_grid = plot(panels...;
                  layout     = (nrows, ncols),
                  size       = (900 * ncols, 750 * nrows),
                  dpi        = 100,
                  plot_title = part_label)
    display(p_grid)
    fname_grid = joinpath(outdir, "spheroid_$(cond.label)_grid.png")
    savefig(p_grid, fname_grid)
    println("Saved: $fname_grid")
end

# ============================================================
# PLOT 5: Phase proportions over time (stacked filled area)
# ============================================================
frac_panels = [plot_phase_fractions(ts_data[c.label],
                                     "$(c.particle) $(c.energy) MeV")
               for c in CONDITIONS if haskey(ts_data, c.label)]

p_frac = plot(frac_panels...; layout=(1, length(frac_panels)),
              size=(1400, 450), dpi=150)
display(p_frac)
savefig(p_frac, joinpath(outdir, "phase_fractions.png"))
println("Saved: phase_fractions.png")

# ============================================================
# FINAL PRINT
# ============================================================
println("\n", "="^60)
println("ALL PLOTS SAVED TO $outdir/")
println("="^60)
println("Files written:")
for f in filter(f -> endswith(f, ".png"), sort(readdir(outdir)))
    println("  $outdir/$f")
end
