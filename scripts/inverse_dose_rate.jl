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
dose = 1.5
setup_IonIrrad!(dose, E, particle)

#~ ==========================================================================================
#~ ============================= Simulation Configuration ===================================
#~ ==========================================================================================

#& Geometry of the target region
target_geom = "circle"        # Options: "square", "circle"

#& Calculation mode
calc_type   = "full"          # Options: "full" -> all layers, "fast" -> only first layer

#& Tumor or target radius (µm)
tumor_radius = 300.0

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

cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

cell_df_original = deepcopy(cell_df_copy)

@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_dose_cell(cell_df_copy, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)

cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0


cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

surv_prob = Array{Float64, 1}()
#~ ==========================================================================================
#~ compute istantenous irradiation 5Gy
#~ ==========================================================================================

cell_df_istant = deepcopy(cell_df_copy)
cell_irrad = deepcopy(cell_df_istant)
MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_irrad, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_irrad, irrad_cond, gsm2_cycle)
    
cell_df_istant.dam_X_dom .+= cell_irrad.dam_X_dom
cell_df_istant.dam_Y_dom .+= cell_irrad.dam_Y_dom
compute_cell_survival_GSM2!(cell_df_istant, gsm2_cycle)
mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
nat_apo = 10^-10
compute_times_domain!(cell_df_istant, gsm2_cycle, nat_apo)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

times_split = [0.05, 0.1, 0.2, 0.5, 6.0, 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0, 72.0, 96.0]
for t in times_split 
    println(t)
    cell_ = deepcopy(cell_df_original)
    X_prev = cell_.dam_X_total
    compute_times_domain!(cell_, gsm2_cycle, nat_apo, terminal_time = t)
    cell_.is_cell[isfinite.(cell_.death_time)] .= 0
    run_simulation_abm!(cell_, nat_apo, terminal_time = t)

    cell_irrad = deepcopy(cell_)
    MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_irrad, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_irrad, irrad_cond, gsm2_cycle)
    X_new = cell_irrad.dam_X_total
    
    cell_.dam_X_dom .+= cell_irrad.dam_X_dom
    cell_.dam_Y_dom .+= cell_irrad.dam_Y_dom
    X_post = cell_.dam_X_total + cell_irrad.dam_X_total
    compute_times_domain!(cell_, gsm2_cycle, nat_apo)
    cell_df_split_ = cell_[cell_.is_cell .== 1, :]
    push!(surv_prob, size(cell_df_split_[.!isfinite.(cell_df_split_.death_time),:], 1)/Ntot)
end
pushfirst!(times_split, 0.0)
Plots.plot(times_split, surv_prob)



Plots.plot(X_prev)
Plots.plot!(X_new)
Plots.plot!(X_post)






df_sub = filter(:is_cell => ==(1), cell_)

# Optionally drop missing labels
df_sub = dropmissing(df_sub, :cell_cycle)

# Count per category and convert to proportions
counts = countmap(df_sub.cell_cycle)          # Dict{String, Int}
cats   = collect(keys(counts))
vals   = collect(values(counts))
props  = vals 

# Bar plot of proportions
default(fontfamily = "sans")
bar(
    cats, props;
    legend = false,
    xlabel = "cell_cycle",
    ylabel = "Proportion (within is_cell == 1)",
    title  = "Cell cycle proportions among is_cell == 1",
    bar_width = 0.8,
    color = "#D55E00",   # Francesco's preferred rust orange 😉
    framestyle = :box,
    yticks = 0:0.1:1.0,
)
