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
using Statistics

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
include(joinpath(@__DIR__, "..", "src", "utilities_TCP_NTCP.jl"))

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

gsm2_cycle = Array{GSM2}(undef, 1)
gsm2_cycle[1] = GSM2(r, a, b, rd, Rn);
gsm2 = gsm2_cycle[1]

#& Construct the GSM2 object
setup_GSM2!(r, a, b, rd, Rn)


#~ ============================================================
#~ =================== Simulation Parameters ==================
#~ ============================================================

#& Box size (µm)
#& A square box with half‑side X_box → full side = 2*X_box
X_box    = 1800.0      # 900.0 um corresponds to a full 1.8 mm box 1800.0
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

#& Number of cells per box side
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

#& Target geometry and tumor radius
tumor_radius = X_box
target_geom = "square"        # Options: "square", "circle"

#& Dose calculation type
calc_type   = "fast"          # Options: "full", "fast"

#& Beam properties
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);

#& Irradiation conditions
ParIrr = "true"  # use "true" to enable partial irradiation
track_seg = true  # use "true" to enable track segment

#& AT type
type_AT = "KC"

#& Setup cell lattice and population
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr = ParIrr, track_seg = track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2; ParIrr = ParIrr)
println("Number of cells = ", sum(cell_df.is_cell .== 1))

if calc_type == "fast"
    cell_df_first_voxel = deepcopy(cell_df[(cell_df.i_voxel_x .== 1) .& (cell_df.i_voxel_y .== 1), :])
    cell_df_first_voxel.is_cell .= 1
end

#N_cell_voxel = zeros(Int64, N_sideVox, N_sideVox, N_sideVox)
#or i in 1:N_sideVox
#    for j in 1:N_sideVox
#        for k in 1:N_sideVox
#            N_cell_voxel[i,j,k] = sum(cell_df[(cell_df.i_voxel_x .== i) .& (cell_df.i_voxel_y .== j) .& (cell_df.i_voxel_z .== k), :].is_cell)
#        end
#    end 
#end
#N_cell_voxel

#vscodedisplay(cell_df)
#vscodedisplay(cell_df_first_voxel)

#& Dose array and fractions
max_total_dose_array = 90.0;
dose_step = 50; #150
dose_array = collect(range(start=max_total_dose_array/dose_step,stop=max_total_dose_array,length=dose_step));
NFraction = [5]; #5, 15, 30

#& Oxygenation array
O2_array = [7.0];
#O2_array = vcat(0.5, collect(range(start = 1.0, stop = 7.0, step = 2.0)))

#& Seriality parameter array
N_PI = 3
m_FSU_array = positive_integer_divisors(N_sideVox*N_sideVox) # serial FSU
m_FSU_array = filter(x -> x % N_PI == 0, m_FSU_array)
n_FSU_array = floor.(Int, (N_sideVox*N_sideVox)./m_FSU_array) # parallel FSU
s = 1.0./n_FSU_array
ParIrr_array = [1, 2, 3]/N_PI #,2,3

#& Initialise Voxel array
(N_sideVox, VoxArray) = CreationArrayVoxels_NTCP(X_box, X_voxel);


#~ ============================================================
#~ ====================== Simulation loop =====================
#~ ============================================================

#& Initialise array for outputs
D_voxel_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(ParIrr_array))
NTCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(O2_array), length(s), length(ParIrr_array))
#cell_df_copy = deepcopy(cell_df)
#cell_df_temp = deepcopy(cell_df)
#cell_df_first_voxel_copy = deepcopy(cell_df_first_voxel)
#vscodedisplay(cell_df_temp)

#& Loop over fractions 
for i_frac in eachindex(NFraction)
    #& Dose per fraction array
    dose_vec = collect(range(start=(max_total_dose_array/NFraction[i_frac])/dose_step,stop=max_total_dose_array/NFraction[i_frac],length=dose_step))
    
    #& Loop over doses
    for i_D in eachindex(dose_vec)
        #& Dose per fraction
        dose = dose_vec[i_D]
        println("Dose: ", dose)
        
        #& Setup irradiation conditions
        E        = 280.0 #100.0         # Ion kinetic energy (MeV/u)
        particle = "12C"    #1H
        setup_IonIrrad!(dose, E, particle)

        Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
        Rk = Rp  #! remove Rk
        at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

        setup_irrad_conditions!(
            ion, irrad, type_AT,
            cell_df,
            track_seg
        )

        #& Set oxygen level
        set_oxygen!(cell_df; rim_ox=7.0, core_ox=0.1, max_dist_ref=250.0, plot_oxygen=false)
        #cell_df.O .= 7.
        #cell_df.O .= ifelse.(cell_df.distance .< 400, 0.1, 6.0)

        #& Copy cell_df
        cell_df_copy = deepcopy(cell_df)

        if calc_type == "fast"
            cell_df_first_voxel_copy = deepcopy(cell_df_first_voxel) # = cell_df_first_voxel
        end

        #& Calculate dose
        F = irrad.dose / (1.602 * 10^(-9) * LET)
        Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
        zF = irrad.dose / Npar
        D = irrad.doserate / zF
        T = irrad.dose / (zF * D) * 3600

        if calc_type == "full"
            @time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
        elseif calc_type == "fast"
            #@time MC_dose_fast_sim1voxel_NTCP!(ion, Npar, x_beam, y_beam, R_beam, irrad_cond, cell_df_first_voxel_copy, df_center_x, df_center_y, at, gsm2, type_AT, track_seg, cell_df_copy, ParIrr);
            @time MC_dose_CPU_sim1voxel!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg, cell_df_first_voxel_copy, ParIrr; x_cb = x_beam, y_cb = y_beam)
        end

        #& Loop over O2 levels
        for i_O2 in eachindex(O2_array)
            #& Copy cell_df
            cell_df_temp = deepcopy(cell_df_copy)

            #& Update oxygen level
            cell_df_temp.O .= O2_array[i_O2]
            
            #& Calculate damage
            MC_loop_damage!(ion, cell_df_temp, irrad_cond, gsm2_cycle)

            #& Compute survival probability for each cell
            gsm2 = gsm2_cycle[1]
            compute_cell_survival_GSM2_NTCP!(cell_df_temp, gsm2; NFrac=NFraction[i_frac])

            #& Compute survival probability for each voxel
            Survival_Voxels_NTCP!(VoxArray, X_voxel, cell_df_temp);

            #& Loop over seriality parameters and partial irradiation conditions
            for i_s in eachindex(s)
                for i_PI in eachindex(ParIrr_array)
                    #& Compute NTCP and dose
                    NTCP_all[i_D, i_frac, i_O2, i_s, i_PI], D_voxel_all[i_D, i_frac, i_PI] = compute_NTCP_nxm_PI(VoxArray, m_FSU_array[i_s], n_FSU_array[i_s], ParIrr_array[i_PI])
                    #NTCP_all[i_D, i_frac, i_O2, i_s, i_PI], D_voxel_all[i_D, i_frac, i_PI] = compute_NTCP_Box_s(VoxArray, N_sideVox, s[i_s])
                end
            end

        end

    end

end

#& Plot NTCP vs dose per fraction
i_s = 1 #s
i_frac = 1 #NFraction
i_O2 = 1 #O2

plt_NTCP = Plots.plot();
for i_PI in eachindex(ParIrr_array)
plt_NTCP = Plots.plot!(D_voxel_all[:, i_frac, i_PI], NTCP_all[:, i_frac, i_O2, i_s, i_PI], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = string("s =", s[i_s], " - PI =", ParIrr_array[i_PI]), xlims=(-0.99, 8.1))
end 
display(plt_NTCP)

i_s = 10
plt_NTCP = Plots.plot();
for i_PI in eachindex(ParIrr_array)
plt_NTCP = Plots.plot!(D_voxel_all[:, i_frac, i_PI], NTCP_all[:, i_frac, i_O2, i_s, i_PI], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = string("s =", s[i_s], " - PI =", ParIrr_array[i_PI]), xlims=(-0.99, 8.1))
end
display(plt_NTCP)

   
