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
#a = 0.01481379648786136
#b = 0.012663276476522422
#r = 2.5656972960759896
#rd = 0.8;
#Rn = 7.2;    

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
X_box    = 300.0      # 900.0 um corresponds to a full 1.8 mm box
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
target_geom = "circle"        # Options: "square", "circle"

#& Dose calculation type
calc_type   = "full"

#& Beam properties
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);

#& Irradiation conditions
ParIrr = "false"  # use "true" to enable partial irradiation
track_seg = true  # use "true" to enable track segment

#& AT type
type_AT = "KC"

#& Setup cell lattice and population
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="false", track_seg = track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
println("Number of cells = ", sum(cell_df.is_cell .== 1))

#& Assign cell type
rs = zeros(Int64, size(cell_df, 1))
for i in cell_df.index[cell_df.is_cell .== 1]
    u = rand()
    if u <= 0.1
        rs[i] = 1
    else
        rs[i] = 2
    end
end

#& Count cells of each type
N_1 = sum(rs .== 1)
N_2 = sum(rs .== 2)
N_1_perc = N_1 / (N_1 + N_2)
N_2_perc = N_2 / (N_1 + N_2)
println("N_1_perc = ", N_1_perc)
println("N_2_perc = ", N_2_perc)

#& Dose array and fractions
max_total_dose_array = 90.0; #90.0
dose_step = 60; #60
dose_array = collect(range(start=max_total_dose_array/dose_step,stop=max_total_dose_array,length=dose_step));
NFraction = [15]; #15, 30

#& Oxygenation array
O2_array = [7.0];
i_O2 = 1;


#~ ============================================================
#~ ====================== Simulation loop =====================
#~ ============================================================

#& Initialise array for outputs
X_dam_array = Array{Array{Int64}}(undef, length(O2_array), length(NFraction), 3)
Y_dam_array = Array{Array{Int64}}(undef, length(O2_array), length(NFraction), 3)
SP_array = Array{Array{Float64}}(undef, length(dose_array), length(O2_array), length(NFraction), 3)
Dcell_array = Array{Array{Float64}}(undef, length(dose_array), length(NFraction))

TCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(O2_array), 3)
D_voxel_all = Array{Float64}(undef, length(dose_array),length(NFraction))

S = Array{Float64, 1}()
S_hat = Array{Float64, 1}()
S_lower = Array{Float64, 1}()
S_upper = Array{Float64, 1}()

S_S = Array{Float64, 1}()
S_hat_S = Array{Float64, 1}()
S_lower_S = Array{Float64, 1}()
S_upper_S = Array{Float64, 1}()

S_R = Array{Float64, 1}()
S_hat_R = Array{Float64, 1}()
S_lower_R = Array{Float64, 1}()
S_upper_R = Array{Float64, 1}()

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
        E        = 100.0         # Ion kinetic energy (MeV/u)
        particle = "1H"    
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
        set_oxygen!(cell_df; plot_oxygen=false)
        cell_df.O .= 7.
        #cell_df.O .= ifelse.(cell_df.distance .< 400, 0.1, 6.0)

        #& Copy cell_df
        cell_df_copy = deepcopy(cell_df)

        #& Calculate dose
        F = irrad.dose / (1.602 * 10^(-9) * LET)
        Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
        zF = irrad.dose / Npar
        D = irrad.doserate / zF
        T = irrad.dose / (zF * D) * 3600

        @time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)

        #& Calculate damage
        MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

        X_dam_array[i_O2, i_frac, i_line] = map(sum, cell_df_temp.dam_X_dom[cell_df_temp.is_cell .> 0])
        Y_dam_array[i_O2, i_frac, i_line] = map(sum, cell_df_temp.dam_Y_dom[cell_df_temp.is_cell .> 0])

        #& Co-culture
        cell_df_temp = deepcopy(cell_df_copy)
        i_line = 1

        #& Calculate survival probability
        Sp = Array{Float64, 1}()
        for i in cell_df_temp.index[cell_df_temp.is_cell .== 1]
            if rs[i] == 1
                gsm2 = gsm2_cycle[2]
            elseif rs[i] == 2
                gsm2 = gsm2_cycle[1]
            end

            SP_cell = domain_GSM2(cell_df_temp.dam_X_dom[i], cell_df_temp.dam_Y_dom[i], gsm2)
            push!(Sp, SP_cell)
        end
        (p_hat, lower, upper) = survival_ci(Sp)
        push!(S, mean(Sp))
        push!(S_hat, mean(p_hat))
        push!(S_lower, mean(lower))
        push!(S_upper, mean(upper))

        Dcell_array[i_D, i_frac] = cell_df_temp.dose_cell[cell_df_temp.is_cell .> 0]

        #& Calculate survival probability after fraction
        Sp_NFr = Sp.^NFraction[i_frac]
        SP_array[i_D, i_O2, i_frac, i_line] = Sp_NFr

        #& Compute TCP and dose
        D_voxel_all[i_D, i_frac] = mean(cell_df_temp.dose_cell[cell_df_temp.is_cell .== 1])
        TCP_all[i_D, i_frac, i_O2, i_line] = compute_TCP(Sp_NFr)


        #& Sensitive cell line
        cell_df_temp = deepcopy(cell_df_copy)
        i_line = 2

        #& Calculate survival probability
        Sp = Array{Float64, 1}()
        for i in cell_df_temp.index[cell_df_temp.is_cell .== 1]
            gsm2 = gsm2_cycle[1]

            SP_cell = domain_GSM2(cell_df_temp.dam_X_dom[i], cell_df_temp.dam_Y_dom[i], gsm2)
            push!(Sp, SP_cell)
        end
        (p_hat, lower, upper) = survival_ci(Sp)
        push!(S_S, mean(Sp))
        push!(S_hat_S, mean(p_hat))
        push!(S_lower_S, mean(lower))
        push!(S_upper_S, mean(upper))

        X_dam_array[i_O2, i_frac, i_line] = map(sum, cell_df_temp.dam_X_dom[cell_df_temp.is_cell .> 0])
        Y_dam_array[i_O2, i_frac, i_line] = map(sum, cell_df_temp.dam_Y_dom[cell_df_temp.is_cell .> 0])

        #& Calculate survival probability after fraction
        Sp_NFr = Sp.^NFraction[i_frac]
        SP_array[i_D, i_O2, i_frac, i_line] = Sp_NFr
        
        #& Compute TCP
        TCP_all[i_D, i_frac, i_O2, i_line] = compute_TCP(Sp_NFr)


        #& Resitant cell line
        cell_df_temp = deepcopy(cell_df_copy)
        i_line = 3

        #& Calculate survival probability
        Sp = Array{Float64, 1}()
        for i in cell_df_temp.index[cell_df_temp.is_cell .== 1]
            gsm2 = gsm2_cycle[2]

            SP_cell = domain_GSM2(cell_df_temp.dam_X_dom[i], cell_df_temp.dam_Y_dom[i], gsm2)
            push!(Sp, SP_cell)
        end
        (p_hat, lower, upper) = survival_ci(Sp)
        push!(S_R, mean(Sp))
        push!(S_hat_R, mean(p_hat))
        push!(S_lower_R, mean(lower))
        push!(S_upper_R, mean(upper))

        X_dam_array[i_O2, i_frac, i_line] = map(sum, cell_df_temp.dam_X_dom[cell_df_temp.is_cell .> 0])
        Y_dam_array[i_O2, i_frac, i_line] = map(sum, cell_df_temp.dam_Y_dom[cell_df_temp.is_cell .> 0])

        #& Calculate survival probability after fraction
        Sp_NFr = Sp.^NFraction[i_frac]
        SP_array[i_D, i_O2, i_frac, i_line] = Sp_NFr

        #& Compute TCP
        TCP_all[i_D, i_frac, i_O2, i_line] = compute_TCP(Sp_NFr)

    end

end

#& Plot survival probability vs dose
label_legend = ["Co-culture", "Sensitive", "Resitant"]

plt_SP = Plots.plot(D_voxel_all[:, 1], mean.(SP_array[:, 1, 1, 1]).^(1/NFraction[1]), xlabel = "Dose per fraction [Gy]", ylabel = "Mean surviving probability", legend = :best, yscale = :log10, label = label_legend[1])
plt_SP = Plots.plot!(D_voxel_all[:, 1], mean.(SP_array[:, 1, 1, 2]).^(1/NFraction[1]), xlabel = "Dose per fraction [Gy]", ylabel = "Mean surviving probability", legend = :best, yscale = :log10, label = label_legend[2])
plt_SP = Plots.plot!(D_voxel_all[:, 1], mean.(SP_array[:, 1, 1, 3]).^(1/NFraction[1]), xlabel = "Dose per fraction [Gy]", ylabel = "Mean surviving probability", legend = :best, yscale = :log10, label = label_legend[3])

#& Plot TCP vs dose
plt_NTCP = Plots.plot(D_voxel_all[:, 1].*NFraction[1], TCP_all[:, 1, 1, 1], xlabel = "Total dose [Gy]", ylabel = "NTCP", label = label_legend[1])
plt_NTCP = Plots.plot!(D_voxel_all[:, 1].*NFraction[1], TCP_all[:, 1, 1, 2], xlabel = "Total dose [Gy]", ylabel = "NTCP", label = label_legend[2])
plt_NTCP = Plots.plot!(D_voxel_all[:, 1].*NFraction[1], TCP_all[:, 1, 1, 3], xlabel = "Total dose [Gy]", ylabel = "NTCP", label = label_legend[3], xlims=(-0.99, 101))