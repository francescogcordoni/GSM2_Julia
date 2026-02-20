using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using StatsPlots
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings

nthreads()

#*Nd is the dimension fo the space, the nucleus of the cell is assumed to be a cylinder. 
#*The cell a sphere around the center.
#*The geometry can be more complicated but for now it is fine

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "utilities_structures.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_general.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_radiation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_GSM2.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_biology.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_env.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_dose_computation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_AT_computation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_plot.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_abm.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_plot_abm.jl"))

#&Stopping power
sp = load_stopping_power()

#~ ==========================================================================================
#~ ============================== Generate GSM2 =============================================
#~ ==========================================================================================

#& Select cell line for GSM2 parameters (to be externalized in the future)
#& cell_line = "HSG"

#& GSM2 model parameters (typical ranges noted in comments)
a  = 0.01    # lethal rate [1/h] 
b  = 0.30    # pairwise lethal rate [1/h]
r  = 4.30    # repair rate [1/h]
rd = 0.80    # domain radius [µm]
Rn = 7.2     # nucleus radius [µm]

#G1 -> 12h
a_G1 = 0.012872261720543399
b_G1 = 0.04029756109753225
r_G1 = 2.780479661191086

#S -> 8h
a_S = 0.00589118894714544
b_S = 0.05794352736120672
r_S = 5.84009601901114

#G2 - M -> 3h + 1h
a_G2 = 0.024306291709970018
b_G2 = 5.704688326522623e-5
r_G2 = 1.7720064637774506

#mixed
a = 0.01481379648786136
b = 0.012663276476522422
r = 2.5656972960759896
rd = 0.8;
Rn = 7.2;    

gsm2_cycle = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn); #! G1
gsm2_cycle[2] = GSM2(r_S, a_S, b_S, rd, Rn); #! S
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn); #! G2 - M
gsm2_cycle[4] = GSM2(r, a, b, rd, Rn); #! mixed


#& Construct the GSM2 object
setup_GSM2!(r, a, b, rd, Rn)

#~ ============================================================
#~ =================== Simulation Parameters ==================
#~ ============================================================

#& Box size (µm)
#& A square box with half‑side X_box → full side = 2*X_box
X_box    = 900.0      # corresponds to a full 1.8 mm box
println("X_box        :", X_box)

#& Voxel size (µm)
X_voxel  = 300.0      # voxel side = 300 µm
println("X_voxel      :", X_voxel)

#& Number of voxels per side
N_sideVox = Int(floor(2 * X_box / X_voxel))
println("N_sideVox    :", N_sideVox)

#& Cell nucleus radius (µm)
r_nucleus = Rn       # typical mammalian nucleus ~7–10 µm 
#! too many reduce to one
#! prev. R = r_nucleus = r_nucl = 7.2      check if non defined and set unique r_nucleus
println("R_nucleus    :", r_nucleus)

#& Cell radius (µm)
R_cell    = 15.0      # corresponds to ~30 µm diameter
println("R_cell       :", R_cell)

N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))


#~ ==========================================================================================
#~ ========================== Ion and Irradiation Settings ==================================
#~ ==========================================================================================

E        = 50.0         # Ion kinetic energy (MeV/u)
particle = "1H"    
dose = 1.
setup_IonIrrad!(dose, E, particle)

#~ ==========================================================================================
#~ ============================= Simulation Configuration ===================================
#~ ==========================================================================================

#& Geometry of the target region
target_geom = "circle"        # Options: "square", "circle"

#& Calculation mode
calc_type   = "full"          # Options: "full" -> all layers, "fast" -> only first layer

#& Tumor or target radius (µm)
tumor_radius = 250.0

#& Compute beam parameters based on geometry and calculation mode
#! check description on inputs
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);


#~ ==========================================================================================
#~ =========================== Amorphous Track Structure ====================================
#~ ==========================================================================================

#& Compute the amorphous track structure radii and dose normalization
type_AT = "KC"
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);

#& In this model, Rk is taken equal to the penumbra radius Rp
Rk = Rp  #! remove Rk

#& Construct the initial AT (track structure) object
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);


#~ ==========================================================================================
#~ ===================== Partial Irradiation & Cell Lattice Initialization ==================
#~ ==========================================================================================

#& Partial irradiation flag ("true" / "false" as strings to match existing APIs)
ParIrr = "false"  # use "true" to enable partial irradiation
track_seg = true  # use "true" to enable track segmentation
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="false", track_seg = track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
println("Number of cells = ", sum(cell_df.is_cell .== 1))

setup_irrad_conditions!(
    ion, irrad, type_AT,
    cell_df,
    track_seg
)

#~ ==========================================================================================
#~ ===================================== set O2 =============================================
#~ ==========================================================================================

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])

#~ ==========================================================================================
#~ ================================== compute dose ==========================================
#~ ==========================================================================================

cell_df_copy = deepcopy(cell_df)

F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600

@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_dose_cell(cell_df_copy, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)

#~ ==========================================================================================
#~ ================================= compute survival =======================================
#~ ==========================================================================================
# Compute cell survival
compute_cell_survival_GSM2!(cell_df_copy, gsm2_cycle)
#plot_survival_probability_cell(cell_df_copy, layer_plot = true)

mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])

#~ ==========================================================================================
#~ =================================== compute ABM ==========================================
#~ ==========================================================================================

nat_apo = 10^-10
compute_times_domain!(cell_df_copy, gsm2_cycle, nat_apo)
plot_times(cell_df_copy)
cell_df_ = deepcopy(cell_df_copy) 

plot_initial_distributions(cell_df_)
print_phase_distribution(cell_df_, label="After Initialization")
plot_phase_proportions_alive(cell_df_)

# Basic plot with bar chart and 3D hemisphere
plot_cell_cycle_distribution(cell_df_)

ts, snapshots = run_simulation_abm!(cell_df_, nat_apo)
plot_simulation_results(ts)

print_phase_distribution(cell_df, label="After Irradiation")

# After simulation
print_phase_distribution(cell_df, label="After 48h Simulation")
plot_phase_comparison_before_after(cell_df_copy, cell_df_)

# Advanced analysis
display(plot_analysis_dashboard(ts))

# Growth rate analysis
display(plot_growth_rate(ts))

# Phase proportions
display(plot_phase_proportions(ts))

# Snapshot comparison
display(plot_snapshot_comparison(snapshots, times=[0, 6, 12, 24]))

# Spatial visualization (if coordinates available)
display(plot_spatial_distribution(snapshots[12]))

# Statistical summary
print_simulation_summary(ts)

# Export data
#export_timeseries_csv(ts, "simulation_results.csv")

# Create animation (if spatial data)
create_spatial_animation(snapshots, output_file="cell_dynamics.gif")


# Access snapshots
snapshot_6h = snapshots[6]  # Cells at 6 hours
snapshot_24h = snapshots[24]  # Cells at 24 hours

# Access time series data
final_cell_count = ts.total_cells[end]





