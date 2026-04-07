using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using InlineStrings
using CUDA
using Statistics: mean
using Printf

nthreads()

# ============================================================
# Load functions
# ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

sp = load_stopping_power()

# ============================================================
# GSM2 parameters
# ============================================================
a_G1 = 0.012872261720543399;  b_G1 = 0.04029756109753225;  r_G1 = 2.780479661191086
a_S  = 0.00589118894714544;   b_S  = 0.05794352736120672;  r_S  = 5.84009601901114
a_G2 = 0.024306291709970018;  b_G2 = 5.704688326522623e-5; r_G2 = 1.7720064637774506
a    = 0.01481379648786136;   b    = 0.012663276476522422; r    = 2.5656972960759896
rd   = 0.8;  Rn = 7.2

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)

setup_GSM2!(r, a, b, rd, Rn)

# ============================================================
# Geometry and irradiation parameters
# ============================================================
dose_frac1    = 1.0       # Gy — first fraction
dose_frac2    = 1.0       # Gy — second fraction (each replicate)
E_proton      = 80.0      # MeV
particle      = "1H"
tumor_radius  = 300.0
X_box         = 300.0
X_voxel       = 800.0
R_cell        = 15.0
target_geom   = "circle"
calc_type     = "full"
type_AT       = "KC"
track_seg     = true

t_repair      = 48.0      # hours between fractions (post-frac1 repair window)
t_post_frac2  = 26.0      # hours to simulate after second fraction
n_replicates  = 100       # number of stochastic second-fraction runs

N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

datadir = joinpath(@__DIR__, "..", "data", "stochastic_time_ev")
mkpath(datadir)

# ============================================================
# Helpers
# ============================================================
function ts_to_df(ts::SimulationTimeSeries)
    DataFrame(
        time           = ts.time,
        total_cells    = ts.total_cells,
        g0_cells       = ts.g0_cells,
        g1_cells       = ts.g1_cells,
        s_cells        = ts.s_cells,
        g2_cells       = ts.g2_cells,
        m_cells        = ts.m_cells,
        stem_cells     = ts.stem_cells,
        non_stem_cells = ts.non_stem_cells)
end

function reset_cell!(dst::DataFrame, src::DataFrame)
    for col in names(src)
        S = src[!, col]; D_ = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S); copyto!(D_[i], S[i]); end
        else
            copyto!(D_, S)
        end
    end
end

# ============================================================
# PHASE 1: Build geometry and irradiate with first fraction
# ============================================================
println("\n", "="^60)
println("  PHASE 1 — geometry setup + first fraction ($(dose_frac1) Gy)")
println("="^60)

setup_IonIrrad!(dose_frac1, E_proton, particle)
R_beam, _, _ = calculate_beam_properties(calc_type, target_geom,
                                          X_box, X_voxel, tumor_radius)
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                    ParIrr="false", track_seg=track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox,
                       N_CellsSide, gsm2_cycle[4])
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

# Spheroid mask
cell_df.is_cell = ifelse.(
    (cell_df.x .^ 2 .+ cell_df.y .^ 2 .+ cell_df.z .^ 2 .<= tumor_radius^2) .&
    ((cell_df.x .÷ Int(2R_cell) .+ cell_df.y .÷ Int(2R_cell) .+
      cell_df.z .÷ Int(2R_cell)) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df)
    cell_df.number_nei[i] = length(cell_df.nei[i]) -
                             sum(cell_df.is_cell[cell_df.nei[i]])
end
set_oxygen!(cell_df; plot_oxygen=false)
cell_df.O .= 21.0

Ntot = count(cell_df.is_cell .== 1)
println("  Spheroid cells: $Ntot")

# First fraction: dose + damage
cell_df_frac1 = deepcopy(cell_df)
F    = irrad.dose / (1.602e-9 * LET)
Npar = round(Int, F * π * R_beam^2 * 1e-8)
println("  Npar = $Npar   R_beam = $(round(R_beam, digits=2))")

@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_frac1,
                    df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_frac1, irrad_cond, gsm2_cycle)

println("  Mean X damage (frac1): $(round(mean(sum.(cell_df_frac1.dam_X_dom[cell_df_frac1.is_cell .== 1])), digits=2))")

# ============================================================
# PHASE 2: ABM repair window — 48 hours post first fraction
# ============================================================
println("\n", "="^60)
println("  PHASE 2 — ABM repair window ($(t_repair) h)")
println("="^60)

compute_times_domain!(cell_df_frac1, gsm2_cycle;
                      nat_apo = 1e-10, terminal_time = t_repair,
                      verbose = false, summary = true)

ts_repair, _ = run_simulation_abm!(cell_df_frac1;
                    nat_apo           = 1e-10,
                    terminal_time     = t_repair,
                    snapshot_times    = Int[],
                    print_interval    = 6.0,
                    verbose           = true,
                    return_dataframes = false,
                    update_input      = true)

ts_repair_df = ts_to_df(ts_repair)
# time axis starts at 0 relative to frac1; keep as-is for the combined plot
CSV.write(joinpath(datadir, "ts_repair_frac1.csv"), ts_repair_df)

n_alive_post_repair = count(cell_df_frac1.is_cell .== 1)
println("  Alive after repair: $n_alive_post_repair / $Ntot")
println("  SF after repair:    $(round(n_alive_post_repair / Ntot, digits=4))")

# Save the post-repair state — this is the SHARED starting point for all replicates
cell_df_post_repair = deepcopy(cell_df_frac1)
CSV.write(joinpath(datadir, "cell_df_post_repair.csv"), cell_df_post_repair)
println("  Saved: cell_df_post_repair.csv")

# ============================================================
# PHASE 3: Second fraction irradiation parameters
#   setup_IonIrrad! with the same ion re-derives Npar/R_beam/zF
#   from dose_frac2 without touching the cell geometry.
# ============================================================
setup_IonIrrad!(dose_frac2, E_proton, particle)
R_beam2, _, _ = calculate_beam_properties(calc_type, target_geom,
                                           X_box, X_voxel, tumor_radius)
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

F2    = irrad.dose / (1.602e-9 * LET)
Npar2 = round(Int, F2 * π * R_beam2^2 * 1e-8)
println("\n  Second fraction: Npar=$Npar2   R_beam=$(round(R_beam2, digits=2))")

# Precompute second-fraction damage once on a clean copy,
# then reuse dam_X_dom / dam_Y_dom across replicates.
cell_df_frac2_template = deepcopy(cell_df_post_repair)
@time MC_dose_fast!(ion, Npar2, R_beam2, irrad_cond, cell_df_frac2_template,
                    df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_frac2_template, irrad_cond, gsm2_cycle)
println("  Mean X damage (frac2 template): $(round(mean(sum.(cell_df_frac2_template.dam_X_dom[cell_df_frac2_template.is_cell .== 1])), digits=2))")

# ============================================================
# PHASE 4: 100 stochastic replicates of the second fraction
# ============================================================
println("\n", "="^60)
println("  PHASE 4 — $(n_replicates) stochastic replicates  ($(t_post_frac2) h post frac2)")
println("="^60)

# Storage: one row per replicate per time point
# We collect total_cells columns: rep_001, rep_002, ...
all_ts = Vector{DataFrame}(undef, n_replicates)
cell_work = deepcopy(cell_df_frac2_template)

for rep in 1:n_replicates
    println("  Replicate $rep / $n_replicates")

    # Reset to post-repair + second-fraction damage state
    reset_cell!(cell_work, cell_df_frac2_template)

    compute_times_domain!(cell_work, gsm2_cycle;
                          nat_apo = 1e-10, terminal_time = t_post_frac2,
                          verbose = false, summary = false)

    ts_rep, _ = run_simulation_abm!(cell_work;
                        nat_apo           = 1e-10,
                        terminal_time     = t_post_frac2,
                        snapshot_times    = Int[],
                        print_interval    = t_post_frac2 + 1.0,
                        verbose           = false,
                        return_dataframes = false,
                        update_input      = true)

    # Shift time so t=0 is the moment of the second fraction
    ts_df_rep = ts_to_df(ts_rep)
    # time in ts_rep starts at 0 relative to the ABM call; keep as-is
    all_ts[rep] = ts_df_rep

    # Save individual replicate
    CSV.write(joinpath(datadir, @sprintf("ts_rep_%03d.csv", rep)), ts_df_rep)

    GC.gc()
end

# ============================================================
# SAVE combined results
# ============================================================

# Master table: columns = time (union grid), rows = replicates
# Use the time vector from rep 1 as reference grid
t_grid = all_ts[1].time

combined_df = DataFrame(time = t_grid)
for rep in 1:n_replicates
    col = Symbol(@sprintf("rep_%03d", rep))
    # Align to t_grid (interpolate if lengths differ)
    if length(all_ts[rep].time) == length(t_grid)
        combined_df[!, col] = all_ts[rep].total_cells
    else
        # Fallback: nearest-neighbour alignment on t_grid
        rep_t = all_ts[rep].time
        rep_c = all_ts[rep].total_cells
        aligned = [rep_c[argmin(abs.(rep_t .- t))] for t in t_grid]
        combined_df[!, col] = aligned
    end
end
CSV.write(joinpath(datadir, "ts_all_replicates.csv"), combined_df)
println("Saved: ts_all_replicates.csv")

# Summary: mean and std of total_cells at each time point
rep_cols   = [Symbol(@sprintf("rep_%03d", r)) for r in 1:n_replicates]
cell_mat   = Matrix{Float64}(combined_df[!, rep_cols])
mean_cells = vec(mean(cell_mat, dims=2))
std_cells  = vec(std(cell_mat,  dims=2))

summary_ts = DataFrame(
    time       = t_grid,
    mean_cells = mean_cells,
    std_cells  = std_cells,
    lo_cells   = max.(mean_cells .- std_cells, 0.0),
    hi_cells   = mean_cells .+ std_cells)
CSV.write(joinpath(datadir, "ts_summary.csv"), summary_ts)
println("Saved: ts_summary.csv")

# Repair phase time series (already saved above, confirm)
println("Saved: ts_repair_frac1.csv")

# ============================================================
# FINAL PRINT
# ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO $datadir/")
println("="^60)
for f in sort(readdir(datadir))
    println("  $datadir/$f")
end
