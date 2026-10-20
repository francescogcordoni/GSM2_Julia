# ============================================================================
# plot_spheroid_3d.jl
#
# 3-D half spheroids from the snapshot CSVs saved by spheroid_temporal_evolution.jl
# (files <label>_snap_t<h>h.csv). Nothing is rerun. Vector PDF, no titles.
#
# The spheroid is cut by the plane y = 0 and the half y ≥ 0 is kept; the camera
# looks from the −y side, so the CUT FACE is in front and the cell-cycle
# structure of the interior (quiescent core, cycling rim) is visible.
#
# Cell positions: the snapshots store lattice indices; x, y, z are taken from
# cell_df_pristine.csv or any *_cell_df_final.csv in the same folder (lattice
# sites never move), reading only the index/x/y/z columns.
#
# Outputs (<data dir>/figures_3d/):
#   spheroid_<label>_t<h>h.pdf    one condition, one time
#   spheroid_<label>_all.pdf      all times of one condition in a row
#   spheroid_all.pdf              conditions as rows, times as columns
#
# Run: julia scripts/plot_spheroid_3d.jl
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using CSV, DataFrames, Printf, Plots
gr()

const DIR  = joinpath(@__DIR__, "..", "data", "spheroid_temporal_evolution")
const FDIR = joinpath(DIR, "figures_3d"); mkpath(FDIR)

# ── view settings ──────────────────────────────────────────────────────────
const CAMERA       = (-30, 20)    # (azimuth, elevation) in degrees; azimuth 0 = view from −y
const SHOW_DOOMED  = true         # doomed cells (scheduled death) drawn in their own colour
const MARKER_SIZE  = 2.0
const EVERY        = 1            # plot every n-th cell (2–3 if the PDFs are too heavy)

# ── positions of every lattice site ────────────────────────────────────────
function load_positions(dir)
    cand = vcat(filter(f -> f == "cell_df_pristine.csv", readdir(dir)),
                filter(f -> endswith(f, "_cell_df_final.csv"), readdir(dir)))
    isempty(cand) && error("no cell_df_pristine.csv or *_cell_df_final.csv in $dir")
    pos = Dict{Int,NTuple{3,Float64}}()
    for f in cand
        d = CSV.read(joinpath(dir, f), DataFrame; select = [:index, :x, :y, :z])
        for r in eachrow(d)
            pos[Int(r.index)] = (Float64(r.x), Float64(r.y), Float64(r.z))
        end
    end
    println("positions of $(length(pos)) lattice sites from $(join(cand, ", "))")
    return pos
end
POS = load_positions(DIR)

# ── snapshots on disk ──────────────────────────────────────────────────────
snapfiles = Dict{String,Vector{Tuple{Int,String}}}()
for f in readdir(DIR)
    m = match(r"^(.*)_snap_t(\d+)h\.csv$", f)
    m === nothing && continue
    push!(get!(snapfiles, m.captures[1], Tuple{Int,String}[]), (parse(Int, m.captures[2]), f))
end
isempty(snapfiles) && error("no *_snap_t<h>h.csv files in $DIR")
labels = sort(collect(keys(snapfiles)))
hours  = sort(unique(h for v in values(snapfiles) for (h, _) in v))
println("conditions: $labels | times: $hours h")

# ── style ──────────────────────────────────────────────────────────────────
default(guidefontsize = 11, tickfontsize = 9, legendfontsize = 9, legendtitlefontsize = 10,
        dpi = 300, left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
        right_margin = 6Plots.mm, top_margin = 2Plots.mm)
paper_style!()
const PHASES = ["G0", "G1", "S", "G2", "M"]
const PCOL   = Dict("G0" => :gray65, "G1" => :steelblue, "S" => :seagreen,
                    "G2" => :darkorange, "M" => :firebrick, "doomed" => :black)
nice(t) = ion_label(t)

"""Half-width of the plotted cube, from the largest snapshot radius (+ one cell)."""
function plot_limit()
    rmax = 0.0
    for (_, v) in snapfiles, (_, f) in v
        d = CSV.read(joinpath(DIR, f), DataFrame; select = [:index])
        for i in d.index
            p = POS[Int(i)]; rmax = max(rmax, sqrt(p[1]^2 + p[2]^2 + p[3]^2))
        end
    end
    return ceil((rmax + 30) / 50) * 50
end
const LIM = plot_limit()

function half_spheroid(file::String, legtitle::String)
    d = CSV.read(joinpath(DIR, file), DataFrame; select = [:index, :cell_cycle, :death_time])
    n = nrow(d)
    x = Vector{Float64}(undef, n); y = similar(x); z = similar(x)
    for k in 1:n
        x[k], y[k], z[k] = POS[Int(d.index[k])]
    end
    phase  = String.(d.cell_cycle)
    doomed = SHOW_DOOMED ? isfinite.(Float64.(d.death_time)) : falses(n)
    keep = findall(y .>= 0)                 # back half; the cut face y = 0 faces the camera
    keep = keep[sortperm(y[keep]; rev = true)][1:EVERY:end]   # far cells first, cut face drawn last

    p = plot(xlabel = tex("x (\$\\mu\$m)"), ylabel = tex("y (\$\\mu\$m)"), zlabel = tex("z (\$\\mu\$m)"),
             xlims = (-LIM, LIM), ylims = (0, LIM), zlims = (-LIM, LIM),
             camera = CAMERA, legend = :outerright, legendtitle = tex(legtitle),
             size = (720, 560), framestyle = :box)
    cats = SHOW_DOOMED ? vcat(PHASES, "doomed") : PHASES
    for cat in cats
        sel = cat == "doomed" ? keep[doomed[keep]] : keep[(phase[keep] .== cat) .& .!doomed[keep]]
        isempty(sel) && continue
        scatter3d!(p, x[sel], y[sel], z[sel]; color = PCOL[cat], ms = MARKER_SIZE, msw = 0,
                   label = cat)
    end
    return p
end

# ── figures ────────────────────────────────────────────────────────────────
save(p, name) = (savefig(p, joinpath(FDIR, name)); println("  ", name))
panels_all = []
for lab in labels
    files = Dict(h => f for (h, f) in snapfiles[lab])
    row = []
    for h in hours
        if haskey(files, h)
            p = half_spheroid(files[h], "$(nice(lab)), t = $h h")
            save(p, "spheroid_$(lab)_t$(h)h.pdf")
        else
            p = plot(framestyle = :none)
        end
        push!(row, p)
    end
    save(plot(row...; layout = (1, length(row)), size = (660 * length(row), 540)),
         "spheroid_$(lab)_all.pdf")
    append!(panels_all, row)
end
save(plot(panels_all...; layout = (length(labels), length(hours)),
          size = (620 * length(hours), 520 * length(labels))), "spheroid_all.pdf")
println("Figures in $FDIR")
