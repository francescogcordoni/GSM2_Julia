# ============================================================================
# example/spheroid_ions.jl
#
# Same pipeline as example/spheroid.jl, run for two radiation qualities on the
# SAME pristine spheroid: 1H 100 MeV and 12C 80 MeV/u (the high-energy ions of
# the dose-rate study). Figures: vector PDF, no titles.
#
#   total_cells.pdf                 total cells vs time, both ions (+ control)
#   phases_<tag>.pdf                total and G0/G1/S/G2/M vs time, one per ion
#   phases_all.pdf                  the phase panels side by side
#   spheroid_<tag>_t<h>h.pdf        3-D half spheroid (y ≤ 0) coloured by phase
#   spheroid_<tag>_all.pdf          all snapshot times of one ion in a row
#   spheroid_all.pdf                ions as rows, times as columns
#   CSV: timeseries_<tag>.csv, summary.csv
#
# HOW TO RUN
#   julia --threads auto example/spheroid_ions.jl
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using StatsPlots
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using CUDA
using Statistics: mean
using SparseArrays
using Printf

println("Threads available: ", nthreads())
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS
# ============================================================================
const DOSE_GY = 2.0
const CONDITIONS = [
    (tag = "control",    particle = "1H",  E = 100.0, dose = 0.0,     label = "control"),
    (tag = "1H_100MeV",  particle = "1H",  E = 100.0, dose = DOSE_GY, label = tex("\$^{1}\$H 100 MeV")),
    (tag = "12C_80MeVu", particle = "12C", E = 80.0,  dose = DOSE_GY, label = tex("\$^{12}\$C 80 MeV/u")),
]

TUMOR_RADIUS  = 350.0
R_CELL        = 15.0
X_BOX         = 550.0
X_VOXEL       = 700.0
RD            = 0.8
RN            = 7.2
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

const TYPE_AT        = "KC"
const TRACK_SEG      = true
const TARGET_GEOM    = "circle"
const CALC_TYPE      = "full"
const TERMINAL_TIME  = 72.0
const SNAPSHOT_HOURS = [0, 6, 12, 24, 48, 72]
const NAT_APO        = 1e-10
const FATE           = :incremental      # :incremental (delayed death) or :legacy
const SEED           = 42

const OUTDIR = joinpath(@__DIR__, "output", "spheroid_ions")
mkpath(OUTDIR)

# ============================================================================
# 2. SETUP — as example/spheroid.jl (top level, once)
# ============================================================================
gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(R_G1, A_G1, B_G1, RD, RN)
gsm2_cycle[2] = GSM2(R_S,  A_S,  B_S,  RD, RN)
gsm2_cycle[3] = GSM2(R_G2, A_G2, B_G2, RD, RN)
gsm2_cycle[4] = GSM2(R_AVG, A_AVG, B_AVG, RD, RN)
setup_GSM2!(R_AVG, A_AVG, B_AVG, RD, RN)

Random.seed!(SEED)
setup_IonIrrad!(DOSE_GY, 100.0, "1H"; type_AT = TYPE_AT)
N_sideVox   = Int(floor(2 * X_BOX / X_VOXEL))
N_CellsSide = 2 * convert(Int64, floor(X_BOX / (2 * R_CELL)))
setup_cell_lattice!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide;
                    ParIrr = "false", track_seg = TRACK_SEG, full_cycle = true)
@eval Main begin
    tumor_radius = $TUMOR_RADIUS
    X_voxel      = $X_VOXEL
    X_box        = $X_BOX
end
setup_cell_population!(TARGET_GEOM, X_BOX, R_CELL, N_sideVox, N_CellsSide,
                       gsm2_cycle[4]; ParIrr = "false")
setup_irrad_conditions!(ion, irrad, TYPE_AT, cell_df, TRACK_SEG)
set_oxygen_diffusion!(cell_df; density = :mean, plot_oxygen = false, verbose = true)

if isdefined(Main, :set_fate_model!)
    set_fate_model!(FATE)
else
    @warn "utilities_abm.jl has no set_fate_model!: compute_times_domain! is used as it is."
end
hasproperty(cell_df, :death_type)  || (cell_df.death_type  = fill(-1, nrow(cell_df)))
hasproperty(cell_df, :dam_X_total) || (cell_df.dam_X_total = zeros(Int, nrow(cell_df)))
hasproperty(cell_df, :dam_Y_total) || (cell_df.dam_Y_total = zeros(Int, nrow(cell_df)))

const N_INIT = count(cell_df.is_cell .== 1)
println("Spheroid: $N_INIT cells")
print_phase_distribution(cell_df; label = "pre-irradiation")
cell_df_pristine = deepcopy(cell_df)
const DFX = df_center_x
const DFY = df_center_y
const AT0 = at
# lattice positions (fixed) by cell index, for the 3-D plots of the snapshots
const POS = Dict(cell_df_pristine.index[i] => (cell_df_pristine.x[i], cell_df_pristine.y[i],
                 cell_df_pristine.z[i]) for i in 1:nrow(cell_df_pristine))

G(name::Symbol) = Base.invokelatest(getfield, Main, name)

# ============================================================================
# 3–6. DOSE, DAMAGE, SURVIVAL, TIMERS, ABM — per condition
# ============================================================================
results = Dict{String,Any}()
summary_df = DataFrame(condition = String[], particle = String[], energy_MeV_u = Float64[],
                       dose_Gy = Float64[], Npar = Int[], mean_cell_dose = Float64[],
                       SF_GSM2 = Float64[], SF_clonogenic = Float64[],
                       N_init = Int[], N_final = Int[])

for c in CONDITIONS
    println("\n", "="^60, "\n  $(c.tag)\n", "="^60)
    df = deepcopy(cell_df_pristine)
    alive = findall(==(1), df.is_cell)
    Npar = 0; dmean = 0.0; sfg = 1.0
    if c.dose > 0
        setup_IonIrrad!(c.dose, c.E, c.particle; type_AT = TYPE_AT)
        R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_BOX, X_VOXEL, TUMOR_RADIUS)
        setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
        F    = G(:irrad).dose / (1.602e-9 * G(:LET))
        Npar = round(Int, F * π * R_beam^2 * 1e-8)
        println("Beam: R_beam = $(round(R_beam, digits = 1)) µm | LET = $(round(G(:LET), digits = 2)) keV/µm | Npar = $Npar")
        @time MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, DFX, DFY, deepcopy(AT0),
                           gsm2_cycle, TYPE_AT, TRACK_SEG; seed = hash((SEED, c.tag)))
        Random.seed!(hash((SEED, c.tag, :damage)))
        MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)
        if FATE === :legacy
            compute_cell_survival_GSM2!(df, gsm2_cycle)      # resets timers: fine for legacy
        else
            for i in alive                                    # GSM2 survival without resets
                ph = df.cell_cycle[i]
                g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
                df.sp[i] = domain_GSM2(df.dam_X_dom[i], df.dam_Y_dom[i], g)
            end
        end
        dmean = mean(df.dose_cell[alive]); sfg = mean(df.sp[alive])
        Random.seed!(hash((SEED, c.tag, :fate)))
        compute_times_domain!(df, gsm2_cycle; nat_apo = NAT_APO, terminal_time = TERMINAL_TIME,
                              verbose = false, summary = true)
    end
    sfc = count((df.is_cell .== 1) .& .!isfinite.(df.death_time)) / N_INIT
    @printf("mean cell dose %.3f Gy | GSM2 SF %.4f | clonogenic SF %.4f\n", dmean, sfg, sfc)

    Random.seed!(hash((SEED, :abm)))                          # same ABM stream for every condition
    ts, snaps = run_simulation_abm!(df; nat_apo = NAT_APO, terminal_time = TERMINAL_TIME,
                                    snapshot_times = SNAPSHOT_HOURS, print_interval = 12.0,
                                    verbose = false, return_dataframes = true, update_input = true)
    # the ABM snapshots a time only when an event falls in that hour: add the final state
    tend = Int(TERMINAL_TIME)
    haskey(snaps, tend) || (snaps[tend] = to_dataframe(CellPopulation(df); alive_only = true))
    tsdf = DataFrame(time = ts.time, total = ts.total_cells, G0 = ts.g0_cells, G1 = ts.g1_cells,
                     S = ts.s_cells, G2 = ts.g2_cells, M = ts.m_cells)
    CSV.write(joinpath(OUTDIR, "timeseries_$(c.tag).csv"), tsdf)
    results[c.tag] = (ts = tsdf, snaps = snaps)
    push!(summary_df, (c.tag, c.particle, c.E, c.dose, Npar, dmean, sfg, sfc, N_INIT,
                       count(==(1), df.is_cell)))
end
CSV.write(joinpath(OUTDIR, "summary.csv"), summary_df)
show(stdout, summary_df; allcols = true); println()

# ============================================================================
# 7. FIGURES (vector PDF, no titles)
# ============================================================================
default(framestyle = :box, lw = 2.0, guidefontsize = 12, tickfontsize = 10,
        legendfontsize = 9, legendtitlefontsize = 10, size = (600, 450), dpi = 300,
        left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, right_margin = 3Plots.mm)
paper_style!()
const COND_COL  = Dict("control" => :gray40, "1H_100MeV" => :steelblue, "12C_80MeVu" => :firebrick)
const PHASES    = ["G0", "G1", "S", "G2", "M"]
const PHASE_COL = Dict("G0" => :gray60, "G1" => :steelblue, "S" => :seagreen,
                       "G2" => :darkorange, "M" => :firebrick)
save(p, name) = (savefig(p, joinpath(OUTDIR, name)); println("  ", name))
lab(tag) = first(c.label for c in CONDITIONS if c.tag == tag)

# Total cells vs time
p_tot = plot(xlabel = "Time after irradiation (h)", ylabel = "Number of cells", legend = :topleft)
for c in CONDITIONS
    r = results[c.tag].ts
    plot!(p_tot, r.time, r.total; color = COND_COL[c.tag], label = c.label)
end
save(p_tot, "total_cells.pdf")

# Total + phases vs time, one per condition
phase_plots = map(CONDITIONS) do c
    r = results[c.tag].ts
    p = plot(xlabel = "Time after irradiation (h)", ylabel = "Number of cells",
             legend = :topleft, legendtitle = c.label)
    plot!(p, r.time, r.total; color = :black, label = "total")
    for ph in PHASES
        plot!(p, r.time, r[!, ph]; color = PHASE_COL[ph], lw = 1.6, label = ph)
    end
    save(p, "phases_$(c.tag).pdf")
    p
end
save(plot(phase_plots...; layout = (1, length(phase_plots)), size = (600 * length(phase_plots), 450),
          link = :y), "phases_all.pdf")

# 3-D half spheroid (cells with y ≤ 0), coloured by phase
lim = (-TUMOR_RADIUS * 1.6, TUMOR_RADIUS * 1.6)
function half_spheroid(snap::DataFrame, tlabel::String)
    pos = [POS[i] for i in snap.index]
    x = first.(pos); y = getindex.(pos, 2); z = last.(pos)
    keep = y .<= 0
    p = plot(xlabel = tex("x (\$\\mu\$m)"), ylabel = tex("y (\$\\mu\$m)"), zlabel = tex("z (\$\\mu\$m)"), xlims = lim, ylims = lim,
             zlims = lim, camera = (30, 25), legend = :topright, legendtitle = tex(tlabel),
             size = (600, 550), framestyle = :box)
    for ph in PHASES
        m = keep .& (String.(snap.cell_cycle) .== ph)
        any(m) || continue
        scatter3d!(p, x[m], y[m], z[m]; color = PHASE_COL[ph], ms = 2.2, msw = 0, label = ph)
    end
    return p
end

hours = sort(collect(SNAPSHOT_HOURS))
grid_rows = []
for c in CONDITIONS
    snaps = results[c.tag].snaps
    row = []
    for h in hours
        if !haskey(snaps, h)
            println("  no snapshot at $h h for $(c.tag)")
            push!(row, plot(framestyle = :none)); continue
        end
        p = half_spheroid(snaps[h], "$(c.label), t = $h h")
        save(p, "spheroid_$(c.tag)_t$(h)h.pdf")
        push!(row, p)
    end
    save(plot(row...; layout = (1, length(row)), size = (550 * length(row), 520)),
         "spheroid_$(c.tag)_all.pdf")
    append!(grid_rows, row)
end
ncol = length(hours)
save(plot(grid_rows...; layout = (length(CONDITIONS), ncol),
          size = (500 * ncol, 480 * length(CONDITIONS))), "spheroid_all.pdf")

println("\nSaved in $OUTDIR")
