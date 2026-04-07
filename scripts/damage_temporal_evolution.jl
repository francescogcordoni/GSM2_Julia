using Base.Threads
using CSV, DataFrames
using Distributions
using Random
using Statistics
using Printf
using ProgressMeter

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
# Geometry — smaller sphere than spheroid_temporal_evolution
# ============================================================
dose         = 2.0
tumor_radius = 200.0   # smaller sphere
X_box        = 200.0
X_voxel      = 800.0
R_cell       = 15.0
target_geom  = "circle"
calc_type    = "full"
type_AT      = "KC"
track_seg    = true

N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

datadir = joinpath(@__DIR__, "..", "data", "damage_temporal_evolution")
mkpath(datadir)

# ============================================================
# Build shared pristine geometry once
# ============================================================
E_ref = 80.0;  particle_ref = "1H"
setup_IonIrrad!(dose, E_ref, particle_ref)
R_beam, _, _ = calculate_beam_properties(calc_type, target_geom,
                                          X_box, X_voxel, tumor_radius)
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                    ParIrr="false", track_seg=track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox,
                       N_CellsSide, gsm2_cycle[4])
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

set_oxygen!(cell_df; plot_oxygen=false)

cell_df_pristine = deepcopy(cell_df)
Ntot_ref = count(cell_df_pristine.is_cell .== 1)
println("Spheroid cells (tumor_radius=$tumor_radius): $Ntot_ref")
println("Initial phase distribution:")
for phase in ["G0", "G1", "S", "G2", "M"]
    n = count(cell_df_pristine.cell_cycle[cell_df_pristine.is_cell .== 1] .== phase)
    println("  $phase : $n  ($(round(100n/Ntot_ref, digits=1))%)")
end
CSV.write(joinpath(datadir, "cell_df_pristine.csv"), cell_df_pristine)

# ============================================================
# Helper: irradiate a copy (no ABM, damage only)
# ============================================================
function irradiate_copy(E::Float64, particle::String,
                         cell_df_pristine::DataFrame,
                         gsm2_cycle::Vector{GSM2};
                         dose::Float64         = 2.0,
                         tumor_radius::Float64 = 200.0,
                         X_box::Float64        = 200.0,
                         X_voxel::Float64      = 800.0,
                         target_geom::String   = "circle",
                         calc_type::String     = "full",
                         type_AT::String       = "KC",
                         track_seg::Bool       = true)

    setup_IonIrrad!(dose, E, particle)
    R_beam_, _, _ = calculate_beam_properties(calc_type, target_geom,
                                               X_box, X_voxel, tumor_radius)
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    cell_copy = deepcopy(cell_df_pristine)

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam_^2 * 1e-8)
    println("  [$particle $(E) MeV] Npar=$Npar  R_beam=$(round(R_beam_,digits=2))")

    @time MC_dose_CPU!(ion, Npar, R_beam_, irrad_cond, cell_copy,
                        df_center_x, df_center_y, at,
                        gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_copy, irrad_cond, gsm2_cycle)

    println("  [$particle $(E) MeV] Mean X damage: " *
            "$(round(mean(sum.(cell_copy.dam_X_dom[cell_copy.is_cell .== 1])), digits=2))")

    return cell_copy
end

# ============================================================
# Helper: compute per-cell damage trajectories and save
# ============================================================
function compute_and_save_trajectories(label::String,
                                        cell_copy::DataFrame,
                                        gsm2_cycle::Vector{GSM2};
                                        outdir::String = datadir)

    println("\n", "="^60)
    println("  DAMAGE TRAJECTORIES: $label")
    println("="^60)

    active_idx = findall(cell_copy.is_cell .== 1)
    Ntot = length(active_idx)
    println("  Active cells: $Ntot")

    # We store one row per (cell_id, event_index) with columns:
    # cell_id, phase, event_idx, time, X_total, Y_total, death_code
    rows = DataFrame(
        cell_id    = Int[],
        phase      = String[],
        event_idx  = Int[],
        time       = Float64[],
        X_total    = Int[],
        Y_total    = Int[],
        death_code = Int[],
    )

    @showprogress "  Computing trajectories..." for ci in active_idx
        X_dom = cell_copy.dam_X_dom[ci]
        Y_dom = cell_copy.dam_Y_dom[ci]
        phase = String(cell_copy.cell_cycle[ci])

        # Determine which gsm2 to use based on phase
        gsm2_idx = phase == "G1" ? 1 :
                   phase == "S"  ? 2 :
                   phase == "G2" ? 3 : 4
        gsm2 = gsm2_cycle[gsm2_idx]

        times_traj, X_traj, Y_traj, death_code =
            compute_repair_domain_trajectory(X_dom, Y_dom, gsm2)

        for (k, (t, Xv, Yv)) in enumerate(zip(times_traj, X_traj, Y_traj))
            push!(rows, (ci, phase, k, t, sum(Xv), sum(Yv), death_code))
        end
    end

    CSV.write(joinpath(outdir, "$(label)_trajectories.csv"), rows)
    println("  Saved $(nrow(rows)) trajectory rows → $(label)_trajectories.csv")

    # Summary per cell: just initial damage, death_code, final time
    summary_rows = DataFrame(
        cell_id       = Int[],
        phase         = String[],
        X_init        = Int[],
        Y_init        = Int[],
        death_code    = Int[],
        final_time    = Float64[],
    )
    for ci in active_idx
        sub = rows[rows.cell_id .== ci, :]
        isempty(sub) && continue
        push!(summary_rows, (
            ci,
            sub.phase[1],
            sub.X_total[1],
            sub.Y_total[1],
            sub.death_code[1],
            sub.time[end],
        ))
    end
    CSV.write(joinpath(outdir, "$(label)_summary.csv"), summary_rows)

    sf = count(summary_rows.death_code .== 0) / Ntot
    println("  Survival fraction (no ABM): $(round(sf, digits=4))")
    println("  Saved summary → $(label)_summary.csv")

    return (rows = rows, summary = summary_rows, sf = sf, Ntot = Ntot)
end

# ============================================================
# IRRADIATE: 3 conditions
# ============================================================
println("\n--- Irradiating: 1H 80 MeV ---")
cell_1H_80 = irradiate_copy(80.0, "1H", cell_df_pristine, gsm2_cycle)

println("\n--- Irradiating: 1H 30 MeV ---")
cell_1H_30 = irradiate_copy(30.0, "1H", cell_df_pristine, gsm2_cycle)

println("\n--- Irradiating: 12C 80 MeV/u ---")
cell_12C_80 = irradiate_copy(80.0, "12C", cell_df_pristine, gsm2_cycle)

# Save irradiated cell states
CSV.write(joinpath(datadir, "cell_irrad_1H_80.csv"),  cell_1H_80)
CSV.write(joinpath(datadir, "cell_irrad_1H_30.csv"),  cell_1H_30)
CSV.write(joinpath(datadir, "cell_irrad_12C_80.csv"), cell_12C_80)

# ============================================================
# COMPUTE DAMAGE TRAJECTORIES (no ABM)
# ============================================================
res_1H_80  = compute_and_save_trajectories("1H_80MeV",   cell_1H_80,  gsm2_cycle)
res_1H_30  = compute_and_save_trajectories("1H_30MeV",   cell_1H_30,  gsm2_cycle)
res_12C_80 = compute_and_save_trajectories("12C_80MeVu", cell_12C_80, gsm2_cycle)

# ============================================================
# FINAL SUMMARY
# ============================================================
summary_df = DataFrame(
    condition         = ["1H_80MeV",   "1H_30MeV",   "12C_80MeVu"],
    particle          = ["1H",          "1H",          "12C"],
    energy_MeV        = [80.0,          30.0,          80.0],
    Ntot              = [res_1H_80.Ntot,  res_1H_30.Ntot,  res_12C_80.Ntot],
    survival_fraction = [res_1H_80.sf,    res_1H_30.sf,    res_12C_80.sf],
)
CSV.write(joinpath(datadir, "summary.csv"), summary_df)
println("\nSummary:")
println(summary_df)

println("\nALL RESULTS SAVED TO $datadir/")
println("="^60)
println("Files written:")
for f in sort(readdir(datadir))
    println("  $datadir/$f")
end
