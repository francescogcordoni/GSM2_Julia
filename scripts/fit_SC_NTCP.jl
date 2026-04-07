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
#a  = 1.21*10^(-2)    # lethal rate [1/h] 
#b  = 2.5*10^(-2)   # pairwise lethal rate [1/h]
#r  = 3.23    # repair rate [1/h]
rd = 0.80    # domain radius [µm]
Rn = 7.2     # nucleus radius [µm]

lb = [3.0, 0.01, 0.01] #[2.5, 0.005, 0.01]
ub = [4.0, 0.09, 0.09] #[4.5, 0.1, 0.5]

samp = 500

rv = rand(Uniform(lb[1], ub[1]), samp)
av = 10 .^(rand(Uniform(log10(lb[2]), log10(ub[2])), samp))
bv = 10 .^(rand(Uniform(log10(lb[3]), log10(ub[3])), samp))

gsm2_cycle = Array{GSM2}(undef, samp)
for i in 1:samp
    gsm2_cycle[i] = GSM2(rv[i], av[i], bv[i], rd, Rn);
end

#& Construct the GSM2 object
setup_GSM2!(rv[1], av[1], bv[1], rd, Rn)


#~ ============================================================
#~ =================== Simulation Parameters ==================
#~ ============================================================

#& Spinal Cord size (µm)
#& A square box with half-side X_box → full side = 2*X_box
X_SC = 3800.0      # 900.0 um corresponds to a full 1.8 mm box
println("X_SC         :", X_SC)

#& FSU size (µm)
X_FSU = [200.0, 400.0, 3800.0, 7600.0] #1900 NO
println("X_FSU        :", X_FSU)

#& Box size (µm)
#& A square box with half‑side X_box → full side = 2*X_box
X_box    = 100.0      # 900.0 um corresponds to a full 1.8 mm box
println("X_box        :", X_box)

#& Voxel size (µm)
X_voxel  = 200.0      # voxel side = 300 µm
println("X_voxel      :", X_voxel)

#& Number of voxels per side
#2 * X_box / X_voxel
N_sideVox = Int(floor(2 * X_box / X_voxel))
println("N_sideVox    :", N_sideVox)

#& Cell nucleus radius (µm)
r_nucleus = Rn       # typical mammalian nucleus ~7–10 µm 
#! too many reduce to one
#! prev. R = r_nucleus = r_nucl = 7.2      check if non defined and set unique r_nucleus
println("R_nucleus    :", r_nucleus)

#& Cell radius (µm)
R_cell    = 10.0      # corresponds to ~30 µm diameter
println("R_cell       :", R_cell)

#& Number of cells per box side
#X_box / (2 * R_cell)
#X_voxel / (2 * R_cell)
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

#& Target geometry and tumor radius
tumor_radius = X_box
target_geom = "square"        # Options: "square", "circle"

#& Dose calculation type
calc_type   = "full"          # Options: "full", "fast"

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
max_total_dose_array = 50.0;
dose_step = 50;
dose_array = collect(range(start=max_total_dose_array/dose_step,stop=max_total_dose_array,length=dose_step));
NFraction = [1, 2, 6, 18]; # 

#& Oxygenation array
O2_array = [7.0];
#O2_array = vcat(0.5, collect(range(start = 1.0, stop = 7.0, step = 2.0)))

#& Seriality parameter array
m_FSU_array = positive_integer_divisors(N_sideVox*N_sideVox*N_sideVox) # serial FSU
n_FSU_array = floor.(Int, (N_sideVox*N_sideVox*N_sideVox)./m_FSU_array) # parallel FSU
s = 1.0./n_FSU_array
#s = [0.08333333333333333, 0.16666666666666666, 0.25, 0.3333333333333333, 0.5, 1.0]

#& Initialise Voxel array
(N_sideVox, VoxArray) = CreationArrayVoxels_NTCP(X_box, X_voxel)

#& Number of FSUs per side (SC)
N_sideFSU_SC = Int(floor(2 * X_SC / X_FSU[4]))
println("N_sideFSU_SC    :", N_sideFSU_SC)

#& Number of Voxels per side (FSU)
N_sideVox_FSU = Int(floor(X_FSU[4] / X_voxel))
println("N_sideVox_FSU    :", N_sideVox_FSU)

#& Initialise Voxel array for the whole volume
VoxArray_SC = Array{Array{Voxel,3}}(undef, N_sideFSU_SC)
VoxArray_FSU = Array{Array{Voxel,3}}(undef, N_sideVox_FSU)


#~ ============================================================
#~ ====================== Simulation loop =====================
#~ ============================================================

#& Initialise array for outputs
D_voxel_all = Array{Float64}(undef, length(dose_array), length(NFraction))
NTCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(gsm2_cycle))

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
        E        = 280.0         # Ion kinetic energy (MeV/u)
        particle = "12C"    
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
        cell_df.O .= O2_array[1]
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
            @time MC_dose_fast_sim1voxel_NTCP!(ion, Npar, x_beam, y_beam, R_beam, irrad_cond, cell_df_first_voxel_copy, df_center_x, df_center_y, at, gsm2, type_AT, track_seg, cell_df_copy, ParIrr);
        end

        #& Loop over GSM2 a,b,r
        for i_abr in eachindex(gsm2_cycle)
            gsm2 = gsm2_cycle[i_abr]

            #& Loop over FSUs
            for i_FSU in 1:size(VoxArray_SC)[1]
                #& Loop over voxels
                for i_vox in 1:size(VoxArray_FSU)[1]
                    #& Copy cell_df
                    cell_df_temp = deepcopy(cell_df_copy)
                    
                    #& Calculate damage
                    MC_loop_damage!(ion, cell_df_temp, irrad_cond, gsm2_cycle)

                    #& Calculate survival probability for each cell
                    #gsm2 = gsm2_cycle[1]
                    compute_cell_survival_GSM2_NTCP!(cell_df_temp, gsm2; NFrac=NFraction[i_frac])
                    println( gsm2_cycle[i_abr])

                    #SP_array[i_D, i_frac, i_O2] = cell_df_temp.sp[cell_df_temp.is_cell .== 1]

                    # Compute survival probability and dose for each voxel
                    VoxArray_temp = deepcopy(VoxArray)
                    Survival_Voxels_NTCP!(VoxArray_temp, X_voxel, cell_df_temp);
                    VoxArray_FSU[i_vox] = VoxArray_temp

                end

                VoxArray_SC[i_FSU] = deepcopy(VoxArray_FSU[1])
                #VoxArray_SC_temp = prod(v[1].ni for v in VoxArray_FSU)
                VoxArray_SC[i_FSU][1].ni = prod(v[1].ni for v in VoxArray_FSU)
                VoxArray_SC[i_FSU][1].Dose = VoxArray_FSU[1][1].Dose


            end

            #& Compute NTCP and dose
            #i_O2 = 1
            NTCP_all[i_D, i_frac, i_abr], D_voxel_all[i_D, i_frac] = compute_NTCP_Box_s_fit(VoxArray_SC, N_sideFSU_SC, s[1])
        end

    end

end


#Save NTCP data
E        = 280.0         # Ion kinetic energy (MeV/u)
particle = "12C"   
O2 = O2_array[1]

rab = DataFrame(r = rv, a = av, b = bv)
par = string(particle, "_", E) #"C280MeV" #p100MeV
geom = string(convert(Int64, X_box*2), "x", convert(Int64, X_box*2), "x", convert(Int64, X_voxel)) #"3600x3600"
if target_geom == "square"
    foldername = string("NTCP_190326", "_", par, "_", geom)
    #O2 = "7"
    cd(mkdir(string("C:/Users/yodat/Documents/Simulazioni_Marco/NTCP/", foldername))) do
        filename = string(par, "_", geom, "_", O2, "_abr.csv");
        CSV.write(filename, rab, writeheader=true)
        for i_frac in eachindex(NFraction)
            dose_array = collect(range(start=(max_total_dose_array/NFraction[i_frac])/dose_step,stop=max_total_dose_array/NFraction[i_frac],length=dose_step))
            filename = string(par, "_", geom, "_", O2, "_AbsDose_Fr_", string(NFraction[i_frac]), ".csv");
            CSV.write(filename, Tables.table(D_voxel_all[:, i_frac]'), writeheader=true, header=string.(dose_array))
            for i_O2 in eachindex(O2_array)
                    filename = string(par, "_", geom, "_O2_", string(round(O2_array[i_O2]; digits=2)), "_NTCP_D_s_Fr_", string(NFraction[i_frac]), "_abr_fit.csv");
                    CSV.write(filename, Tables.table((NTCP_all[:, i_frac, i_O2, :])), writeheader=true, header=string.(rv));
                    #filename = string(par, "_", geom, "_", O2, "_gEUD_D_s_PI", string(ParIrr[j]), ".csv");
                    #CSV.write(filename, Tables.table((GEUDArray[:,:,j])'), writeheader=true, header=string.(DoseArray));
            end
        end
        #filename = string(par, "_", geom, "_", O2, "_abr_", string(NFraction[i_frac]), ".csv");
        #CSV.write(filename, Tables.table(D_voxel_all[:, i_frac]'), writeheader=true, header=string.(["r", "a", "b"]))

    end
end

plt_NTCP = Plots.scatter(D_voxel_all[:, 1], NTCP_all[:, 1, 1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[1])
plt_NTCP = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, 1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[2])
plt_NTCP = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, 1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[3])

plt_MTCP = Plots.scatter(xlabel = "Dose per fraction [Gy]", ylabel = "MTCP")
for i_abr in eachindex(gsm2_cycle)
    plt_NTCP = Plots.scatter!(D_voxel_all[:, 1], NTCP_all[:, 1, i_abr], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = i_abr)
end
display(plt_NTCP)

using Plots, Loess

# Fr 1
x= Array{Float64}(undef, 50, 0)
y= Array{Float64}(undef, 50, 0)
TD50_F1 = Array{Float64}(undef, 500, 1)
plt_NTCP_F1 = Plots.plot(xlabel = "Dose per fraction [Gy]", ylabel = "NTCP")
for i_abr in eachindex(gsm2_cycle)
    # Loess regression
    x1 = vec(D_voxel_all[:, 1])
    x = hcat(x, x1)
    y1 = vec(NTCP_all[:, 1, i_abr])
    y = hcat(y, y1)
    model_loess = loess(x[:,i_abr], y[:,i_abr], span=0.015)
    xs = range(minimum(x[:,i_abr]), maximum(x[:,i_abr]), length=1000)
    ys = predict(model_loess, xs)
    idx_50 = argmin(abs.(ys .- 0.5))
    TCP_50_F1 = ys[idx_50]
    TD50_F1[i_abr] = xs[idx_50]

    plt_NTCP_F1 = Plots.plot!(xs, ys, label = i_abr)

end
display(plt_NTCP)

idx_fit_F1 = argmin(abs.(TD50_F1 .- 18))
TD50_F1[idx_fit_F1]
abr_F1 = gsm2_cycle[idx_fit_F1]

# Fr 2
x= Array{Float64}(undef, 50, 0)
y= Array{Float64}(undef, 50, 0)
TD50_F2 = Array{Float64}(undef, 500, 1)
plt_NTCP_F2 = Plots.plot(xlabel = "Dose per fraction [Gy]", ylabel = "NTCP")
for i_abr in eachindex(gsm2_cycle)
    # Loess regression
    x1 = vec(D_voxel_all[:, 2])
    x = hcat(x, x1)
    y1 = vec(NTCP_all[:, 2, i_abr])
    y = hcat(y, y1)
    model_loess = loess(x[:,i_abr], y[:,i_abr], span=0.015)
    xs = range(minimum(x[:,i_abr]), maximum(x[:,i_abr]), length=1000)
    ys = predict(model_loess, xs)
    idx_50 = argmin(abs.(ys .- 0.5))
    TCP_50_F2 = ys[idx_50]
    TD50_F2[i_abr] = xs[idx_50]

    plt_NTCP_F2 = Plots.plot!(xs, ys, label = i_abr)

end
display(plt_NTCP)

idx_fit_F2 = argmin(abs.(TD50_F2 .- 12.5))
TD50_F2[idx_fit_F6]
abr_F2 = gsm2_cycle[idx_fit_F2]

# Fr 6
x= Array{Float64}(undef, 50, 0)
y= Array{Float64}(undef, 50, 0)
TD50_F6 = Array{Float64}(undef, 500, 1)
plt_NTCP_F6 = Plots.plot(xlabel = "Dose per fraction [Gy]", ylabel = "NTCP")
for i_abr in eachindex(gsm2_cycle)
    # Loess regression
    x1 = vec(D_voxel_all[:, 3])
    x = hcat(x, x1)
    y1 = vec(NTCP_all[:, 3, i_abr])
    y = hcat(y, y1)
    model_loess = loess(x[:,i_abr], y[:,i_abr], span=0.015)
    xs = range(minimum(x[:,i_abr]), maximum(x[:,i_abr]), length=1000)
    ys = predict(model_loess, xs)
    idx_50 = argmin(abs.(ys .- 0.5))
    TCP_50_F6 = ys[idx_50]
    TD50_F6[i_abr] = xs[idx_50]

    plt_NTCP_F6 = Plots.plot!(xs, ys, label = i_abr)

end
display(plt_NTCP)

idx_fit_F6 = argmin(abs.(TD50_F6 .- 6.5))
TD50_F6[idx_fit_F6]
abr_F6 = gsm2_cycle[idx_fit_F6]

# Fr 18
x= Array{Float64}(undef, 50, 0)
y= Array{Float64}(undef, 50, 0)
TD50_F18 = Array{Float64}(undef, 500, 1)
plt_NTCP_F18 = Plots.plot(xlabel = "Dose per fraction [Gy]", ylabel = "NTCP")
for i_abr in eachindex(gsm2_cycle)
    # Loess regression
    x1 = vec(D_voxel_all[:, 4])
    x = hcat(x, x1)
    y1 = vec(NTCP_all[:, 4, i_abr])
    y = hcat(y, y1)
    model_loess = loess(x[:,i_abr], y[:,i_abr], span=0.015)
    xs = range(minimum(x[:,i_abr]), maximum(x[:,i_abr]), length=1000)
    ys = predict(model_loess, xs)
    idx_50 = argmin(abs.(ys .- 0.5))
    TCP_50_F18 = ys[idx_50]
    TD50_F18[i_abr] = xs[idx_50]

    plt_NTCP_F18 = Plots.plot!(xs, ys, label = i_abr)

end
display(plt_NTCP)

idx_fit_F18 = argmin(abs.(TD50_F18 .- 3.0))
TD50_F18[idx_fit_F18]
abr_F18 = gsm2_cycle[idx_fit_F18]

#
plt_NTCP_1 = Plots.scatter(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[1])
plt_NTCP_1 = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[2])
plt_NTCP_1 = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[3])
plt_NTCP_1 = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[4])

plt_NTCP_2 = Plots.scatter(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[1])
plt_NTCP_2 = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[2])
plt_NTCP_2 = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[3])
plt_NTCP_2 = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[4])

plt_NTCP_6 = Plots.scatter(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[1])
plt_NTCP_6 = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[2])
plt_NTCP_6 = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[3])
plt_NTCP_6 = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[4])

plt_NTCP_18 = Plots.scatter(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[1])
plt_NTCP_18 = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[2])
plt_NTCP_18 = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[3])
plt_NTCP_18 = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = NFraction[4])

#
plt_NTCP = Plots.scatter(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F1)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F2)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F6)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 1], NTCP_all[:, 1, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F18)

plt_NTCP = Plots.scatter(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F1)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F2)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F6)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 2], NTCP_all[:, 2, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F18)

plt_NTCP = Plots.scatter(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F1)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F2)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F6)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 3], NTCP_all[:, 3, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F18)

plt_NTCP = Plots.scatter(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F1], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F1)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F2], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F2)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F6], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F6)
plt_NTCP = Plots.scatter!(D_voxel_all[:, 4], NTCP_all[:, 4, idx_fit_F18], xlabel = "Dose per fraction [Gy]", ylabel = "NTCP", label = idx_fit_F18)
