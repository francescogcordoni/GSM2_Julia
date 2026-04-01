using CSV, DataFrames, Printf
using Statistics

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
# Fixed irradiation parameters — 1 Gy protons 100 MeV
# ============================================================
dose        = 1.0
E           = 100.0
particle    = "1H"
target_geom = "circle"
calc_type   = "full"
type_AT     = "KC"
track_seg   = true
R_cell      = 15.0
X_voxel     = 800.0
terminal_time = 72.0   # h

# Spheroid sizes to benchmark (tumor radius in µm)
RADII = [100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.]

datadir = joinpath(@__DIR__, "..", "data", "benchmark_spheroid_scaling")
mkpath(datadir)

# ============================================================
# Result table
# ============================================================
results = DataFrame(
    tumor_radius_um = Float64[],
    N_cells         = Int[],
    t_irrad_s       = Float64[],
    t_abm_s         = Float64[],
    t_total_s       = Float64[],
)

# ============================================================
# Benchmark loop
# ============================================================
setup_IonIrrad!(dose, E, particle)   # sets global ion/irrad/LET

for tumor_radius in RADII
    X_box = tumor_radius

    println("\n", "="^60)
    @printf("  RADIUS = %.0f µm\n", tumor_radius)
    println("="^60)

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    # ── Build geometry ──────────────────────────────────────────
    R_beam, _, _ = calculate_beam_properties(calc_type, target_geom,
                                              X_box, X_voxel, tumor_radius)
    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox,
                           N_CellsSide, gsm2_cycle[4])
    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
    set_oxygen!(cell_df; plot_oxygen=false)

    Ncells = count(cell_df.is_cell .== 1)
    @printf("  Cells: %d\n", Ncells)

    # ── Irradiation timing ──────────────────────────────────────
    cell_irrad = deepcopy(cell_df)

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    @printf("  Npar = %d\n", Npar)

    t_irrad = @elapsed begin
        MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_irrad,
                     df_center_x, df_center_y, at,
                     gsm2_cycle, type_AT, track_seg)
        MC_loop_damage!(ion, cell_irrad, irrad_cond, gsm2_cycle)
    end
    @printf("  Irradiation time : %.2f s\n", t_irrad)

    # ── ABM timing ──────────────────────────────────────────────
    t_abm = @elapsed begin
        compute_times_domain!(cell_irrad, gsm2_cycle;
                              nat_apo       = 1e-10,
                              terminal_time = terminal_time,
                              verbose       = false,
                              summary       = false)
        run_simulation_abm!(cell_irrad;
                            nat_apo           = 1e-10,
                            terminal_time     = terminal_time,
                            print_interval    = 24.0,
                            verbose           = false,
                            return_dataframes = false,
                            update_input      = false)
    end
    @printf("  ABM time         : %.2f s\n", t_abm)
    @printf("  Total time       : %.2f s\n", t_irrad + t_abm)

    push!(results, (tumor_radius, Ncells, t_irrad, t_abm, t_irrad + t_abm))
end

# ============================================================
# Save
# ============================================================
CSV.write(joinpath(datadir, "scaling_1H_100MeV_1Gy.csv"), results)

println("\n", "="^60)
println("RESULTS:")
println(results)
println("\nSaved → $(joinpath(datadir, "scaling_1H_100MeV_1Gy.csv"))")
