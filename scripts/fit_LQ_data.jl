using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using Distances
using WebIO, PlotlyJS
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using DifferentialEquations
using OrdinaryDiffEq
using StatsPlots
using Sobol 

nthreads() 

#*Nd is the dimension fo the space, the nucleus of the cell is assumed to be a cylinder. 
#*The cell a sphere around the center.
#*The geometry can be more complicated but for now it is fine

type = "domain"
Nd = 1;

#*initialize structures
include("./utilities_structures_dr.jl")
include("./utilities_dr.jl")

#~###############################################################
#~########## creation of a tumor-NT environment #################
#~###############################################################

#* tumor is a sphere of 3 mm dimeter, outer part [radius 3-2 mm] is hypoxic 3.5%
#* inner part [radius 2-1 mm] more hypoxic 2%, necrotic core [radius 1-0 mm] is necrotic 1%
#* the rest is normal tissue, oxygantion 7%
#* I am defining a new function that creates the above specifi environment,we can later try to generalize it to be user defined
#* Oxygenation -it is defined in percentage: [1] = hypoxic for necrotic tumor core - 
#&[3,4] = hypoxic tumor - [7,10] = normal tissue -  [21] = normoxic cells
O2 = 21.;
#*Surival Probability - initially set as 1, it must be change using the code
SP = 1.; 

#&Size Cell nucleus
R = r_nucleus = r_nucl = Rn = 7.2; #8µm
rd = 0.8
#~##################### Choose Ion   #############################################
type_AT = "KC"
E = 240.
particle = "1H"
A = 1
Z = getZ(particle)
LET = linear_interpolation(particle, E, sp)
ion = Ion(particle, E, 1, Z, LET, 1.0)
dose = 2;
irrad = Irrad(dose, 1, 0.18)
#~##################### Choose Irradiation Dose ######################################
Rc = Rp = Rk = 0.;
irrad = Irrad(dose, 1., 0.18);
DoseRate_h = F = D = T = 0.;
(center_x, center_y) = calculate_centers(0., 0., rd, Rn)
domain = size(center_x)[1]

Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp
at_start = AT(particle, E, A, Z, LET, 1., Rc, Rp, Rk, Rp + 1, Kp)

R_beam = Rp + Rn
x_beam, y_beam = 0., 0.; #Coordiante of the center of the beam

#~##################### Convert dose, dose rate in proton Flux ######
DoseRate_h = irrad.doserate;
F = irrad.dose / (1.602 * 10^(-9) * ion.LET);

#~###### Compute Dose for circle beam
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)));
#d = zn/(T*zF)
zF = irrad.dose / Npar;
dr = DoseRate_h / zF;
T = irrad.dose / (zF * D);
track_seg = true

num_cols = domain
df_center_x = DataFrame(Matrix{Float64}(undef, 1, num_cols), :auto)
df_center_x[1, :] = center_x
rename!(df_center_x, Symbol.("center_$i" for i in 1:num_cols))
df_center_x.index .= 1

df_center_y = DataFrame(Matrix{Float64}(undef, 1, num_cols), :auto)
df_center_y[1, :] = center_y
rename!(df_center_y, Symbol.("center_$i" for i in 1:num_cols))
df_center_y.index .= 1

#~###############################################################
#~########## creation of a tumor-NT environment #################
#~###############################################################

#* tumor is a sphere of 3 mm dimeter, outer part [radius 3-2 mm] is hypoxic 3.5%
#* inner part [radius 2-1 mm] more hypoxic 2%, necrotic core [radius 1-0 mm] is necrotic 1%
#* the rest is normal tissue, oxygantion 7%
#* I am defining a new function that creates the above specifi environment,we can later try to generalize it to be user defined
#* Oxygenation -it is defined in percentage: [1] = hypoxic for necrotic tumor core - 
#&[3,4] = hypoxic tumor - [7,10] = normal tissue -  [21] = normoxic cells
O2 = 7.;
#*Surival Probability - initially set as 1, it must be change using the code
SP = 1.; 

#~########################################################################################################
#~#################################### Generate Nodes position ###########################################
#~####### TODO: Functio dose=1;n to input a arbitrery configuration of cell (e.g. from CellSim3D)#########
#~########################################################################################################
#~################ Choose cell_line for GSM2 paramters ###################################################
#&cell_line = "HSG"; #
a = 7.82*10^-3;
b = 1.83*10^-2;
r = 2.5;
rd = 0.8;
Rn = 7.2;    
gsm2 = GSM2(r, a, b, rd, Rn);  

at = DataFrame(zeros(size(df_center_y, 1), (size(df_center_y, 2) - 1)), :auto)
rename!(at, Symbol.("center_$i" for i in 1:(size(df_center_y, 2) - 1)))
at.index = df_center_y.index

irrad_cond = at_start
lets = irrad_cond.LET
energies = irrad_cond.E

MC_loop_ions_singlecell_fast!(Npar, x_beam, y_beam, irrad_cond, gsm2, df_center_x, df_center_y, at, R_beam, type_AT, ion);
mean(at[1, 1:(end - 1)])
n_repeat = floor(Int64, gsm2.Rn / gsm2.rd)

temp_cell = Cell(0.0, 0.0, 0.0, [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "", Matrix{Float64}(undef,0,0), Matrix{Float64}(undef,0,0), [], [], 21., 0.0, [], 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
kappa_base = 55/(n_repeat*domain)
lambda_base = kappa_base * 1e-3 # Factor from original MC_loop_damage_domain!

sim_ = 1000
dose = [0.5, 1, 2, 3, 4, 5]
X_dam = zeros(Int64, size(dose, 1), sim_, n_repeat*domain)
Y_dam = zeros(Int64, size(dose, 1), sim_, n_repeat*domain)
dose = [0.5, 1, 2, 3, 4, 5]
for dj in 1:size(dose, 1)
    for s in 1:sim_
        d = dose[dj]
        lambda_X = max(0.0, kappa_base * d) # Mean for X damage
        lambda_Y = max(0.0, lambda_base * d) # Mean for Y damage
        X_dam[dj, s, :] = rand(Poisson(lambda_X), n_repeat*domain)
        Y_dam[dj, s, :] = rand(Poisson(lambda_Y), n_repeat*domain)
    end
end

sim_grid = 10^3

pa = rand(sim_grid)
amin = log10(10^-5)
amx = log10(1)
a_s = 10 .^ (amin .+ (amx - amin) .* vec(pa))

pb = rand(sim_grid)
bmin = log10(10^-5)
bmx = log10(1)
b_s = 10 .^ (bmin .+ (bmx - bmin) .* vec(pb))

pr = rand(sim_grid)
rmin = 0.5
rmx = 10
r_s = (rmin .+ (rmx - rmin) .* vec(pr))

Plots.plot(a_s, r_s, xaxis = :log10)

# G1 phase
results_df = DataFrame(a = Float64[], b = Float64[], r = Float64[], err = Float64[])

alpha = 0.351
beta = 0.04
SP_grid = zeros(Float64, sim_grid, size(dose, 1))
for gj in 1:sim_grid
    a = a_s[gj]
    b = b_s[gj]
    r = r_s[gj]
    rd = 0.8
    Rn = 7.2

    gsm2 = GSM2(r, a, b, rd, Rn)
    println("--- Calculating Survival Probabilities ---")
    SP_cell = zeros(Float64, size(dose, 1), sim_)
    p = ProgressBar(1:size(dose, 1))
    for dj in p
        @Threads.threads for s in 1:sim_
            current_X_dam = X_dam[dj, s, :]
            current_Y_dam = Y_dam[dj, s, :]
            SP_cell[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
        end
    end

    println("--- Processing Results ---")
    SP_grid[gj, :] = vec(mean(SP_cell, dims = 2))

    S_LQ = exp.(-alpha .* dose .- beta .* dose .* dose)
    err = mean(((S_LQ - SP_grid[gj, :]) .^ 2) ./ S_LQ)

    push!(results_df, (a, b, r, err))
end

sorted_df_G1 = sort(results_df, :err)
sel = 1
a_G1 = sorted_df_G1.a[sel]
b_G1 = sorted_df_G1.b[sel]
r_G1 = sorted_df_G1.r[sel]
println("Best fit parameters:")
println("a: ", a_G1)
println("b: ", b_G1)
println("r: ", r_G1)
#a: 0.012872261720543399
#b: 0.04029756109753225
#r: 2.780479661191086
#Plots.scatter(sorted_df_G1.r, sorted_df_G1.a, marker_z = sorted_df_G1.err, colorbar = true, yaxis = :log10,
#        xlabel = "r", ylabel = "a", label = "", title = "r vs a colored by err")


gsm2 = GSM2(r_G1, a_G1, b_G1, rd, Rn)
println("--- Calculating Survival Probabilities ---")
SP_cell_G1 = zeros(Float64, size(dose, 1), sim_)
p = ProgressBar(1:size(dose, 1))
for dj in p
    @Threads.threads for s in 1:sim_
        current_X_dam = X_dam[dj, s, :]
        current_Y_dam = Y_dam[dj, s, :]
        SP_cell_G1[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
    end
end
println("--- Processing Results ---")
SP_opt_G1 = vec(mean(SP_cell_G1, dims = 2))
S_LQ_G1 = exp.(-alpha .* dose .- beta .* dose .* dose)
err = mean(((S_LQ - SP_opt) .^ 2) ./ S_LQ)

default(palette = palette(:darktest))
Plots.plot(dose, S_LQ, label = "LQ", yscale = :log10, seriestype = :scatter, color = :black)
Plots.plot!(dose, SP_opt, label = "GSM2", yscale = :log10, linewidth = 3)



# S phase
results_df = DataFrame(a = Float64[], b = Float64[], r = Float64[], err = Float64[])

alpha = 0.1235
beta = 0.0285
SP_grid = zeros(Float64, sim_grid, size(dose, 1))
for gj in 1:sim_grid
    a = a_s[gj]
    b = b_s[gj]
    r = r_s[gj]
    rd = 0.8
    Rn = 7.2

    gsm2 = GSM2(r, a, b, rd, Rn)
    println("--- Calculating Survival Probabilities ---")
    SP_cell = zeros(Float64, size(dose, 1), sim_)
    p = ProgressBar(1:size(dose, 1))
    for dj in p
        @Threads.threads for s in 1:sim_
            current_X_dam = X_dam[dj, s, :]
            current_Y_dam = Y_dam[dj, s, :]
            SP_cell[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
        end
    end

    println("--- Processing Results ---")
    SP_grid[gj, :] = vec(mean(SP_cell, dims = 2))

    S_LQ = exp.(-alpha .* dose .- beta .* dose .* dose)
    err = mean(((S_LQ - SP_grid[gj, :]) .^ 2) ./ S_LQ)

    push!(results_df, (a, b, r, err))
end

sorted_df_S = sort(results_df, :err)
sel = 1
a_S = sorted_df_S.a[sel]
b_S = sorted_df_S.b[sel]
r_S = sorted_df_S.r[sel]
println("Best fit parameters:")
println("a: ", a_S)
println("b: ", b_S)
println("r: ", r_S)
#a: 0.00589118894714544
#b: 0.05794352736120672
#r: 5.84009601901114
#Plots.scatter(sorted_df_G1.r, sorted_df_G1.a, marker_z = sorted_df_G1.err, colorbar = true, yaxis = :log10,
#        xlabel = "r", ylabel = "a", label = "", title = "r vs a colored by err")


gsm2 = GSM2(r_S, a_S, b_S, rd, Rn)
println("--- Calculating Survival Probabilities ---")
SP_cell_S = zeros(Float64, size(dose, 1), sim_)
p = ProgressBar(1:size(dose, 1))
for dj in p
    @Threads.threads for s in 1:sim_
        current_X_dam = X_dam[dj, s, :]
        current_Y_dam = Y_dam[dj, s, :]
        SP_cell_S[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
    end
end
println("--- Processing Results ---")
SP_opt_S = vec(mean(SP_cell_S, dims = 2))
S_LQ_S = exp.(-alpha .* dose .- beta .* dose .* dose)
err = mean(((S_LQ - SP_opt) .^ 2) ./ S_LQ)

default(palette = palette(:darktest))
Plots.plot(dose, S_LQ, label = "LQ", yscale = :log10, seriestype = :scatter, color = :black)
Plots.plot!(dose, SP_opt, label = "GSM2", yscale = :log10, linewidth = 3)


# G2 phase
results_df = DataFrame(a = Float64[], b = Float64[], r = Float64[], err = Float64[])

alpha = 0.793
beta = 0.0
SP_grid = zeros(Float64, sim_grid, size(dose, 1))
for gj in 1:sim_grid
    a = a_s[gj]
    b = b_s[gj]
    r = r_s[gj]
    rd = 0.8
    Rn = 7.2

    gsm2 = GSM2(r, a, b, rd, Rn)
    println("--- Calculating Survival Probabilities ---")
    SP_cell = zeros(Float64, size(dose, 1), sim_)
    p = ProgressBar(1:size(dose, 1))
    for dj in p
        @Threads.threads for s in 1:sim_
            current_X_dam = X_dam[dj, s, :]
            current_Y_dam = Y_dam[dj, s, :]
            SP_cell[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
        end
    end

    println("--- Processing Results ---")
    SP_grid[gj, :] = vec(mean(SP_cell, dims = 2))

    S_LQ = exp.(-alpha .* dose .- beta .* dose .* dose)
    err = mean(((S_LQ - SP_grid[gj, :]) .^ 2) ./ S_LQ)

    push!(results_df, (a, b, r, err))
end

sorted_df_G2 = sort(results_df, :err)
sel = 3
a_G2 = sorted_df_G2.a[sel]
b_G2 = sorted_df_G2.b[sel]
r_G2 = sorted_df_G2.r[sel]
println("Best fit parameters:")
println("a: ", a_G2)
println("b: ", b_G2)
println("r: ", r_G2)
#a: 0.024306291709970018
#b: 5.704688326522623e-5
#r: 1.7720064637774506
#Plots.scatter(sorted_df_G1.r, sorted_df_G1.a, marker_z = sorted_df_G1.err, colorbar = true, yaxis = :log10,
#        xlabel = "r", ylabel = "a", label = "", title = "r vs a colored by err")


gsm2 = GSM2(r_G2, a_G2, b_G2, rd, Rn)
println("--- Calculating Survival Probabilities ---")
SP_cell_G2 = zeros(Float64, size(dose, 1), sim_)
p = ProgressBar(1:size(dose, 1))
for dj in p
    @Threads.threads for s in 1:sim_
        current_X_dam = X_dam[dj, s, :]
        current_Y_dam = Y_dam[dj, s, :]
        SP_cell_G2[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
    end
end
println("--- Processing Results ---")
SP_opt_G2 = vec(mean(SP_cell_G2, dims = 2))
S_LQ_G2 = exp.(-alpha .* dose .- beta .* dose .* dose)
err = mean(((S_LQ - SP_opt) .^ 2) ./ S_LQ)

default(palette = palette(:darktest))
Plots.plot(dose, S_LQ_G1, label = "LQ G1", yscale = :log10, seriestype = :scatter)
Plots.plot!(dose, SP_opt_G1, label = "GSM2 G1", yscale = :log10, linewidth = 3)
Plots.plot!(dose, S_LQ_S, label = "LQ S", yscale = :log10, seriestype = :scatter)
Plots.plot!(dose, SP_opt_S, label = "GSM2 S", yscale = :log10, linewidth = 3)
Plots.plot!(dose, S_LQ_G2, label = "LQ G2", yscale = :log10, seriestype = :scatter)
Plots.plot!(dose, SP_opt_G2, label = "GSM2 G2", yscale = :log10, linewidth = 3)


# S phase
alpha_S = 0.1235
beta_S = 0.0285
alpha_S/beta_S
a_S
b_S
r_S
r_S/(a_S + b_S)

# G1 phase
alpha_G1 = 0.351
beta_G1 = 0.04
alpha_G1/beta_G1
a_G1
b_G1
r_G1
r_G1/(a_G1 + b_G1)

# G2 phase
alpha_G2 = 0.793
beta_G2 = 0.0
a_G2
b_G2
r_G2
r_G2/(a_G2 + b_G2)


x = [a_S, a_G1, a_G2, r_S, r_G1, r_G2]
y = [alpha_S, alpha_G1, alpha_G2, alpha_S, alpha_G1, alpha_G2]
Plots.scatter([x[1], x[4]], [y[1], y[4]], label = "S", color = :blue, xaxis = :log10,
        xlabel = "a or r", ylabel = "alpha", legend = :topright, size = (800, 600))
Plots.scatter!([x[2], x[5]], [y[2], y[5]], label = "G1", color = :green)
Plots.scatter!([x[3], x[6]], [y[3], y[6]], label = "G2", color = :red)

Plots.plot([b_S/a_S, b_G1/a_G1, b_G2/a_G2], [beta_S, beta_G1, beta_G2], seriestype = :scatter, label = "")


x = [b_S/a_S, b_G1/a_G1, b_G2/a_G2]
y = [beta_S, beta_G1, beta_G2]
Plots.scatter([x[1]], [y[1]], label = "S", color = :blue, xaxis = :log10,
        xlabel = "b/a", ylabel = "beta", legend = :topright, size = (800, 600))
Plots.scatter!([x[2]], [y[2]], label = "G1", color = :green)
Plots.scatter!([x[3]], [y[3]], label = "G2", color = :red)






# mixed phase -> 1/2 G1 + 1/3 S + 1/6 G2-M
results_df = DataFrame(a = Float64[], b = Float64[], r = Float64[], err = Float64[])

alpha = 0.3869441 
beta = 0.009053933
SP_grid = zeros(Float64, sim_grid, size(dose, 1))
for gj in 1:sim_grid
    a = a_s[gj]
    b = b_s[gj]
    r = r_s[gj]
    rd = 0.8
    Rn = 7.2

    gsm2 = GSM2(r, a, b, rd, Rn)
    println("--- Calculating Survival Probabilities ---")
    SP_cell = zeros(Float64, size(dose, 1), sim_)
    p = ProgressBar(1:size(dose, 1))
    for dj in p
        @Threads.threads for s in 1:sim_
            current_X_dam = X_dam[dj, s, :]
            current_Y_dam = Y_dam[dj, s, :]
            SP_cell[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
        end
    end

    println("--- Processing Results ---")
    SP_grid[gj, :] = vec(mean(SP_cell, dims = 2))

    S_LQ = exp.(-alpha .* dose .- beta .* dose .* dose)
    err = mean(((S_LQ - SP_grid[gj, :]) .^ 2) ./ S_LQ)

    push!(results_df, (a, b, r, err))
end

sorted_df_mixed = sort(results_df, :err)
sel = 1
a_mixed = sorted_df_mixed.a[sel]
b_mixed = sorted_df_mixed.b[sel]
r_mixed = sorted_df_mixed.r[sel]
println("Best fit parameters:")
println("a: ", a_mixed)
println("b: ", b_mixed)
println("r: ", r_mixed)
#a: 0.00589118894714544
#b: 0.05794352736120672
#r: 5.84009601901114
#Plots.scatter(sorted_df_G1.r, sorted_df_G1.a, marker_z = sorted_df_G1.err, colorbar = true, yaxis = :log10,
#        xlabel = "r", ylabel = "a", label = "", title = "r vs a colored by err")


gsm2 = GSM2(r_mixed, a_mixed, b_mixed, rd, Rn)
println("--- Calculating Survival Probabilities ---")
SP_cell_mixed = zeros(Float64, size(dose, 1), sim_)
p = ProgressBar(1:size(dose, 1))
for dj in p
    @Threads.threads for s in 1:sim_
        current_X_dam = X_dam[dj, s, :]
        current_Y_dam = Y_dam[dj, s, :]
        SP_cell_mixed[dj, s] = domain_GSM2(current_X_dam, current_Y_dam, gsm2)
    end
end
println("--- Processing Results ---")
SP_opt_mixed = vec(mean(SP_cell_mixed, dims = 2))
S_LQ_mixed = exp.(-alpha .* dose .- beta .* dose .* dose)
err = mean(((S_LQ_mixed - SP_opt_mixed) .^ 2) ./ S_LQ_mixed)

default(palette = palette(:darktest))
Plots.plot(dose, S_LQ_mixed, label = "LQ", yscale = :log10, seriestype = :scatter, color = :black)
Plots.plot!(dose, SP_opt_mixed, label = "GSM2", yscale = :log10, linewidth = 3)


