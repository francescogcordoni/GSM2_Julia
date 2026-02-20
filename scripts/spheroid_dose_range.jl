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
X_box    = 400.0      # corresponds to a full 1.8 mm box
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

tumor_radius = 250.0
target_geom = "circle"        # Options: "square", "circle"
calc_type   = "full"
R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type,
        target_geom,
        X_box,
        X_voxel,
        tumor_radius
    );
ParIrr = "false"  # use "true" to enable partial irradiation
track_seg = true  # use "true" to enable track segment
type_AT = "KC"
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="false", track_seg = track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
println("Number of cells = ", sum(cell_df.is_cell .== 1))
dose_vec = [0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.]
rs = zeros(Int64, size(cell_df, 1))
for i in cell_df.index[cell_df.is_cell .== 1]
    u = rand()
    if u < 0.2
        rs[i] = 1
    else
        rs[i] = 2
    end
end

S = Array{Float64, 1}()
S_hat = Array{Float64, 1}()
S_lower = Array{Float64, 1}()
S_upper = Array{Float64, 1}()
for dose in dose_vec
    println("Dose: ", dose)
    
    E        = 50.0         # Ion kinetic energy (MeV/u)
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

    set_oxygen!(cell_df; plot_oxygen=false)
    cell_df.O .= 21.

    #cell_df.O .= ifelse.(cell_df.distance .< 400, 0.1, 6.0)
    cell_df_copy = deepcopy(cell_df)

    F = irrad.dose / (1.602 * 10^(-9) * LET)
    Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
    zF = irrad.dose / Npar
    D = irrad.doserate / zF
    T = irrad.dose / (zF * D) * 3600

    @time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    Sp = Array{Float64, 1}()
    for i in cell_df_copy.index[cell_df_copy.is_cell .== 1]
        if rs[i] == 1
            gsm2 = gsm2_cycle[2]
        else
            gsm2 = gsm2_cycle[4]
        end
        
        SP_cell = domain_GSM2(cell_df_copy.dam_X_dom[i], cell_df_copy.dam_Y_dom[i], gsm2)
        push!(Sp, SP_cell)
    end
    (p_hat, lower, upper) = survival_ci(Sp)
    push!(S, mean(Sp))
    push!(S_hat, mean(p_hat))
    push!(S_lower, mean(lower))
    push!(S_upper, mean(upper))
end
Plots.plot(dose_vec, S, label = "SP", legend = :best, yscale = :log10)

# color palette (your preferred rust orange)
primary = "#D55E00"   # line color
bandcol = :lightgray  # band fill color; change if you prefer

# Build symmetrical ribbon around Ŝ using offsets to lower/upper bounds
lower_dev = S_hat .- S_lower
upper_dev = S_upper .- S_hat

plt = Plots.plot(
    dose_vec, S_hat;
    ribbon = (lower_dev, upper_dev),   # shaded band [lower, upper]
    fillalpha = 0.25,
    color = primary,
    label = "Ŝ (mean)",
    xlabel = "Dose",
    ylabel = "Survival probability",
    legend = :topright,
    lw = 2,
    yscale = :log10
)
display(plt)







S_2 = Array{Float64, 1}()
rs_2 = zeros(Int64, size(cell_df, 1))
for i in cell_df_copy.index[cell_df_copy.is_cell .== 1]
    u = rand()
    if u < 0.5
        rs_2[i] = 1
    else
        rs_2[i] = 2
    end
end
S_hat_2 = Array{Float64, 1}()
S_lower_2 = Array{Float64, 1}()
S_upper_2 = Array{Float64, 1}()
for dose in dose_vec
    println("Dose: ", dose)
    
    E        = 50.0         # Ion kinetic energy (MeV/u)
    particle = "1H"    
    setup_IonIrrad!(dose, E, particle)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
    Rk = Rp  #! remove Rk
    at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="false", track_seg = track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
    println("Number of cells = ", sum(cell_df.is_cell .== 1))

    setup_irrad_conditions!(
        ion, irrad, type_AT,
        cell_df,
        track_seg
    )

    set_oxygen!(cell_df; plot_oxygen=false)
    cell_df.O .= 21.

    #cell_df.O .= ifelse.(cell_df.distance .< 400, 0.1, 6.0)
    cell_df_copy = deepcopy(cell_df)

    F = irrad.dose / (1.602 * 10^(-9) * LET)
    Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
    zF = irrad.dose / Npar
    D = irrad.doserate / zF
    T = irrad.dose / (zF * D) * 3600

    @time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    Sp = Array{Float64, 1}()
    for i in cell_df_copy.index[cell_df_copy.is_cell .== 1]
        if rs_2[i] == 1
            gsm2 = gsm2_cycle[2]
        else
            gsm2 = gsm2_cycle[4]
        end
        
        SP_cell = domain_GSM2(cell_df_copy.dam_X_dom[i], cell_df_copy.dam_Y_dom[i], gsm2)
        push!(Sp, SP_cell)
    end

    (p_hat, lower, upper) = survival_ci(Sp)

    push!(S_2, mean(Sp))
    push!(S_hat_2, mean(p_hat))
    push!(S_lower_2, mean(lower))
    push!(S_upper_2, mean(upper))
end
Plots.plot(dose_vec, S_2, label = "SP", legend = :best, yscale = :log10)

# color palette (your preferred rust orange)
primary = "#D55E00"   # line color
bandcol = :lightgray  # band fill color; change if you prefer

# Build symmetrical ribbon around Ŝ using offsets to lower/upper bounds
lower_dev_2 = S_hat_2 .- S_lower_2
upper_dev_2 = S_upper_2 .- S_hat_2

plt = Plots.plot(
    dose_vec, S_hat_2;
    ribbon = (lower_dev, upper_dev),   # shaded band [lower, upper]
    fillalpha = 0.25,
    color = primary,
    label = "Ŝ (mean)",
    xlabel = "Dose",
    ylabel = "Survival probability",
    legend = :topright,
    lw = 2,
    yscale = :log10
)
display(plt)








S_3 = Array{Float64, 1}()
rs_3 = zeros(Int64, size(cell_df, 1))
for i in cell_df_copy.index[cell_df_copy.is_cell .== 1]
    u = rand()
    if u < 0.8
        rs_3[i] = 1
    else
        rs_3[i] = 2
    end
end
S_hat_3 = Array{Float64, 1}()
S_lower_3 = Array{Float64, 1}()
S_upper_3 = Array{Float64, 1}()
for dose in dose_vec
    println("Dose: ", dose)
    
    E        = 50.0         # Ion kinetic energy (MeV/u)
    particle = "1H"    
    setup_IonIrrad!(dose, E, particle)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
    Rk = Rp  #! remove Rk
    at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="false", track_seg = track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
    println("Number of cells = ", sum(cell_df.is_cell .== 1))

    setup_irrad_conditions!(
        ion, irrad, type_AT,
        cell_df,
        track_seg
    )

    set_oxygen!(cell_df; plot_oxygen=false)
    cell_df.O .= 21.

    #cell_df.O .= ifelse.(cell_df.distance .< 400, 0.1, 6.0)
    cell_df_copy = deepcopy(cell_df)

    F = irrad.dose / (1.602 * 10^(-9) * LET)
    Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
    zF = irrad.dose / Npar
    D = irrad.doserate / zF
    T = irrad.dose / (zF * D) * 3600

    @time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    Sp = Array{Float64, 1}()
    for i in cell_df_copy.index[cell_df_copy.is_cell .== 1]
        if rs_3[i] == 1
            gsm2 = gsm2_cycle[2]
        else
            gsm2 = gsm2_cycle[4]
        end
        
        SP_cell = domain_GSM2(cell_df_copy.dam_X_dom[i], cell_df_copy.dam_Y_dom[i], gsm2)
        push!(Sp, SP_cell)
    end

    (p_hat, lower, upper) = survival_ci(Sp)
    
    push!(S_3, mean(Sp))
    push!(S_hat_3, mean(p_hat))
    push!(S_lower_3, mean(lower))
    push!(S_upper_3, mean(upper))
end
Plots.plot(dose_vec, S_3, label = "SP", legend = :best, yscale = :log10)

# color palette (your preferred rust orange)
primary = "#D55E00"   # line color
bandcol = :lightgray  # band fill color; change if you prefer

# Build symmetrical ribbon around Ŝ using offsets to lower/upper bounds
lower_dev_3 = S_hat_3 .- S_lower_3
upper_dev_3 = S_upper_3 .- S_hat_3

plt = Plots.plot(
    dose_vec, S_hat;
    ribbon = (lower_dev, upper_dev),   # shaded band [lower, upper]
    fillalpha = 0.25,
    color = RGB(0., 0., 0.),
    label = "Ŝ (20% resitant - 80% sensitive)",
    xlabel = "Dose",
    ylabel = "Survival probability",
    legend = :topright,
    lw = 2,
    yscale = :log10
)
plt = Plots.plot!(
    dose_vec, S_hat_2;
    ribbon = (lower_dev_2, upper_dev_2),   # shaded band [lower, upper]
    fillalpha = 0.25,
    color = RGB(0.0, 0.8, 0.0),
    label = "Ŝ (50% resitant - 50% sensitive)",
    xlabel = "Dose",
    ylabel = "Survival probability",
    legend = :topright,
    lw = 2,
    yscale = :log10
)
plt = Plots.plot!(
    dose_vec, S_hat_3;
    ribbon = (lower_dev_3, upper_dev_3),   # shaded band [lower, upper]
    fillalpha = 0.25,
    color = RGB(1.0, 0.65, 0.0),
    label = "Ŝ (80% resitant - 20% sensitive)",
    xlabel = "Dose",
    ylabel = "Survival probability",
    legend = :topright,
    lw = 2,
    yscale = :log10
)
display(plt)

cell_df_copy.rs .= rs
cell_df_copy.rs_2 .= rs_2
cell_df_copy.rs_3 .= rs_3
df_active = cell_df_copy[cell_df_copy.is_cell .== 1, :]
df3 = df_active[df_active.x .>= 0, :]

cat1 = ifelse.(df3.rs .== 1, 1, 2)
cat2 = ifelse.(df3.rs_2 .== 1, 1, 2)
cat3 = ifelse.(df3.rs_3 .== 1, 1, 2)

p1 = scatter(
    df3.x, df3.y, df3.z;
    markersize = 4,
    markerstrokewidth = 0.1,
    marker_z = cat1,                    # still drive colors via z
    seriescolor = cgrad([RGB(0,0,0), RGB(1,0,0)], 2, categorical = true),
    clims = (0.5, 2.5),                 # clamp exactly two categories
    colorbar = false,
    xlabel = "x (µm)", ylabel = "y (µm)", zlabel = "z (µm)",
    title = "3D Cell Distribution (20% resitant - 80% sensitive)",
    legend = false, aspect_ratio = :equal,
    size = (900, 700), camera = (320, 30)
)
p2 = scatter(
    df3.x, df3.y, df3.z;
    markersize = 4,
    markerstrokewidth = 0.1,
    marker_z = cat2,                    # still drive colors via z
    seriescolor = cgrad([RGB(0,0,0), RGB(1,0,0)], 2, categorical = true),
    clims = (0.5, 2.5),                 # clamp exactly two categories
    colorbar = false,
    xlabel = "x (µm)", ylabel = "y (µm)", zlabel = "z (µm)",
    title = "3D Cell Distribution (50% resitant - 50% sensitive)",
    legend = false, aspect_ratio = :equal,
    size = (900, 700), camera = (320, 30)
)
p3 = scatter(
    df3.x, df3.y, df3.z;
    markersize = 4,
    markerstrokewidth = 0.1,
    marker_z = cat3,                    # still drive colors via z
    seriescolor = cgrad([RGB(0,0,0), RGB(1,0,0)], 2, categorical = true),
    clims = (0.5, 2.5),                 # clamp exactly two categories
    colorbar = false,
    xlabel = "x (µm)", ylabel = "y (µm)", zlabel = "z (µm)",
    title = "3D Cell Distribution (80% resitant - 20% sensitive)",
    legend = false, aspect_ratio = :equal,
    size = (900, 700), camera = (320, 30)
)


using Plots
using Measures
using Statistics

l = @layout [a b c; d{0.5h}]

fig = plot(p1, p2, p3, plt;
            layout=l,
            size=(1200, 800),
            left_margin=8mm, right_margin=5mm,
            top_margin=5mm,  bottom_margin=5mm)

display(fig)