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
using InlineStrings
using CUDA
using Statistics: mean

nthreads()

#*Nd is the dimension fo the space, the nucleus of the cell is assumed to be a cylinder. 
#*The cell a sphere around the center.
#*The geometry can be more complicated but for now it is fine

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

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
function setup(
    E::Float64,
    particle::String,
    dose::Float64,
    tumor_radius::Float64;
    X_box::Float64      = 600.0,
    X_voxel::Float64    = 300.0,
    R_cell::Float64     = 15.0,
    target_geom::String = "circle",
    calc_type::String   = "full",
    type_AT::String     = "KC",
    track_seg::Bool     = true
)
    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    # Inject all variables that setup functions read directly from Main
    @eval Main begin
        tumor_radius = $tumor_radius
        X_voxel      = $X_voxel
        X_box        = $X_box
        R_cell       = $R_cell
        type_AT      = $type_AT
        target_geom  = $target_geom
        track_seg    = $track_seg
        N_sideVox    = $N_sideVox
        N_CellsSide  = $N_CellsSide
    end

    Base.invokelatest(setup_IonIrrad!, dose, E, particle)
    ion   = Base.invokelatest(getfield, Main, :ion)
    irrad = Base.invokelatest(getfield, Main, :irrad)
    A     = Base.invokelatest(getfield, Main, :A)
    Z     = Base.invokelatest(getfield, Main, :Z)
    LET   = Base.invokelatest(getfield, Main, :LET)

    R_beam, x_beam, y_beam = Base.invokelatest(
        calculate_beam_properties, calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = Base.invokelatest(ATRadius, ion, irrad, type_AT)
    at_start   = Base.invokelatest(AT, particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    Base.invokelatest(setup_cell_lattice!, target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                      ParIrr="false", track_seg=track_seg)
    cell_df = Base.invokelatest(getfield, Main, :cell_df)

    gsm2 = Base.invokelatest(getfield, Main, :gsm2)
    Base.invokelatest(setup_cell_population!, target_geom, X_box, R_cell,
                      N_sideVox, N_CellsSide, gsm2)
    cell_df = Base.invokelatest(getfield, Main, :cell_df)

    ion   = Base.invokelatest(getfield, Main, :ion)
    irrad = Base.invokelatest(getfield, Main, :irrad)
    Base.invokelatest(setup_irrad_conditions!, ion, irrad, type_AT, cell_df, track_seg)
    cell_df = Base.invokelatest(getfield, Main, :cell_df)

    Base.invokelatest(set_oxygen!, cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    irrad = Base.invokelatest(getfield, Main, :irrad)
    LET   = Base.invokelatest(getfield, Main, :LET)
    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")

    return (
        ion=ion, irrad=irrad, cell_df=cell_df, at_start=at_start,
        R_beam=R_beam, x_beam=x_beam, y_beam=y_beam,
        O2_mean=O2_mean, Npar=Npar, zF=zF, D=D, T=T
    )
end

E            = 2.0
    particle     = "1H"
    dose         = 1.5
    tumor_radius = 300.0
    run_ion_irradiation(
    E,
    particle,
    dose,
    tumor_radius)




    E            = 2.0
    particle     = "1H"
    dose         = 1.5
    tumor_radius = 300.0

    # Optional parameters
    X_box       = 600.0
    X_voxel     = 300.0
    R_cell      = 15.0
    target_geom = "circle"
    calc_type   = "full"
    type_AT     = "KC"
    track_seg   = true

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)

    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")


cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

for i in 1:nrow(cell_df_copy)
    if cell_df_copy.is_cell[i] == 1
        ru = rand() * 24
        if ru <= 1
            cell_cycle = "M"
        elseif ru <= 6
            cell_cycle = "G2"
        elseif ru <= 13
            cell_cycle = "S"
        elseif ru <= 24
            cell_cycle = "G1"
        else
            println("Error")
            cell_cycle = "I" # Fallback
        end
        cell_df_copy.cell_cycle[i] = cell_cycle
    end
end
cell_df_original = deepcopy(cell_df_copy)
plot_phase_proportions_alive(cell_df_original)

cols = [1:5; 7:10]
CSV.write("cell_pop_async.csv", cell_df_original[:,cols])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
#plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)

#vscodedisplay(cell_df_copy[cell_df_copy.is_cell .== 1, :])

cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

print_phase_distribution(cell_df_original)
plot_phase_proportions_alive(cell_df_original)

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
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_1H_50 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2
Nsplit_2_1H_50 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])

surv_prob = Array{Float64, 1}()
compute_times_domain!(cell_df_istant, gsm2_cycle)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
plot_times(cell_df_istant)

tumor_radius = 500.0
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
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp  #! remove Rk
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])
cell_df.O .= 21.
F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)

mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])


# ── Setup ─────────────────────────────────────────────────────────────────────
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
nsim        = 10
phase_keys  = ("G0", "G1", "S", "G2", "M")

cell_df_ref = deepcopy(cell_df_copy)

function reset_cell!(dst, src)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

cell_work = deepcopy(cell_df_ref)

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0
    n_valid         = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)

            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            n_valid         += 1
            Nalive_pre_acc  += count(alive_pre)
            Nalive_post_acc += count(alive_post)
            surv_acc        += count(alive_post) / Ntot
            for (i, k) in enumerate(phase_keys)
                phase_pre_acc[i]  += counts_pre[k]
                phase_post_acc[i] += counts_post[k]
            end

        catch e
            @warn "sim=$sim t=$t failed" exception=(e, catch_backtrace())
        end
    end

    inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
    pre   = phase_pre_acc  .* inv_n
    post  = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1], pre[2], pre[3], pre[4], pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)

    GC.gc()
end


pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

phase_times_pre_1H_2 = deepcopy(phase_times_pre)
surv_prob_1H_2 = deepcopy(surv_prob)





######################1H 10

    E            = 10.0
    particle     = "1H"
    dose         = 1.5
    tumor_radius = 300.0

    # Optional parameters
    X_box       = 600.0
    X_voxel     = 300.0
    R_cell      = 15.0
    target_geom = "circle"
    calc_type   = "full"
    type_AT     = "KC"
    track_seg   = true

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)

    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")


cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

cell_df_copy = deepcopy(cell_df_original)
plot_phase_proportions_alive(cell_df_copy)
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
#plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)

#vscodedisplay(cell_df_copy[cell_df_copy.is_cell .== 1, :])

cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

print_phase_distribution(cell_df_original)
plot_phase_proportions_alive(cell_df_original)

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
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_1H_50 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2
Nsplit_2_1H_50 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])


surv_prob = Array{Float64, 1}()
compute_times_domain!(cell_df_istant, gsm2_cycle)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
plot_times(cell_df_istant)

tumor_radius = 500.0
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
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp  #! remove Rk
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])
cell_df.O .= 21.
F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)

mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])


# ── Setup ─────────────────────────────────────────────────────────────────────
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
nsim        = 10
phase_keys  = ("G0", "G1", "S", "G2", "M")

cell_df_ref = deepcopy(cell_df_copy)

function reset_cell!(dst, src)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

cell_work = deepcopy(cell_df_ref)

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0
    n_valid         = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)

            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            n_valid         += 1
            Nalive_pre_acc  += count(alive_pre)
            Nalive_post_acc += count(alive_post)
            surv_acc        += count(alive_post) / Ntot
            for (i, k) in enumerate(phase_keys)
                phase_pre_acc[i]  += counts_pre[k]
                phase_post_acc[i] += counts_post[k]
            end

        catch e
            @warn "sim=$sim t=$t failed" exception=(e, catch_backtrace())
        end
    end

    inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
    pre   = phase_pre_acc  .* inv_n
    post  = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1], pre[2], pre[3], pre[4], pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)

    GC.gc()
end


pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)


phase_times_pre_1H_10 = deepcopy(phase_times_pre)
surv_prob_1H_10 = deepcopy(surv_prob)





###########1H 100


    E            = 100.0
    particle     = "1H"
    dose         = 1.5
    tumor_radius = 300.0

    # Optional parameters
    X_box       = 600.0
    X_voxel     = 300.0
    R_cell      = 15.0
    target_geom = "circle"
    calc_type   = "full"
    type_AT     = "KC"
    track_seg   = true

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)

    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

cell_df_copy = deepcopy(cell_df_original)
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)


#vscodedisplay(cell_df_copy[cell_df_copy.is_cell .== 1, :])

cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

print_phase_distribution(cell_df_original)
plot_phase_proportions_alive(cell_df_original)

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
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_1H_100 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2
Nsplit_2_1H_100 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])


surv_prob = Array{Float64, 1}()
compute_times_domain!(cell_df_istant, gsm2_cycle)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
plot_times(cell_df_istant)

tumor_radius = 500.0
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
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp  #! remove Rk
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])
cell_df.O .= 21.
F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)

mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])


# ── Setup ─────────────────────────────────────────────────────────────────────
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
nsim        = 10
phase_keys  = ("G0", "G1", "S", "G2", "M")

cell_df_ref = deepcopy(cell_df_copy)

function reset_cell!(dst, src)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

cell_work = deepcopy(cell_df_ref)

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0
    n_valid         = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)

            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            n_valid         += 1
            Nalive_pre_acc  += count(alive_pre)
            Nalive_post_acc += count(alive_post)
            surv_acc        += count(alive_post) / Ntot
            for (i, k) in enumerate(phase_keys)
                phase_pre_acc[i]  += counts_pre[k]
                phase_post_acc[i] += counts_post[k]
            end

        catch e
            @warn "sim=$sim t=$t failed" exception=(e, catch_backtrace())
        end
    end

    inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
    pre   = phase_pre_acc  .* inv_n
    post  = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1], pre[2], pre[3], pre[4], pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)

    GC.gc()
end


Plots.plot(times_split, surv_prob_noabm)

pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)


phase_times_pre_1H_100 = deepcopy(phase_times_pre)
surv_prob_1H_100 = deepcopy(surv_prob)





###########12C 80


    E            = 80.0
    particle     = "12C"
    dose         = 1.5
    tumor_radius = 300.0

    # Optional parameters
    X_box       = 600.0
    X_voxel     = 300.0
    R_cell      = 15.0
    target_geom = "circle"
    calc_type   = "full"
    type_AT     = "KC"
    track_seg   = true

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)

    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

cell_df_copy = deepcopy(cell_df_original)
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)


#vscodedisplay(cell_df_copy[cell_df_copy.is_cell .== 1, :])

cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

print_phase_distribution(cell_df_original)
plot_phase_proportions_alive(cell_df_original)

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
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_12C_80 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2
Nsplit_2_12C_80 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])


surv_prob = Array{Float64, 1}()
compute_times_domain!(cell_df_istant, gsm2_cycle)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
plot_times(cell_df_istant)

tumor_radius = 500.0
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
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp  #! remove Rk
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])
cell_df.O .= 21.
F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)

mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])


# ── Setup ─────────────────────────────────────────────────────────────────────
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
nsim        = 10
phase_keys  = ("G0", "G1", "S", "G2", "M")

cell_df_ref = deepcopy(cell_df_copy)

function reset_cell!(dst, src)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

cell_work = deepcopy(cell_df_ref)

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0
    n_valid         = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)

            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            n_valid         += 1
            Nalive_pre_acc  += count(alive_pre)
            Nalive_post_acc += count(alive_post)
            surv_acc        += count(alive_post) / Ntot
            for (i, k) in enumerate(phase_keys)
                phase_pre_acc[i]  += counts_pre[k]
                phase_post_acc[i] += counts_post[k]
            end

        catch e
            @warn "sim=$sim t=$t failed" exception=(e, catch_backtrace())
        end
    end

    inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
    pre   = phase_pre_acc  .* inv_n
    post  = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1], pre[2], pre[3], pre[4], pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)

    GC.gc()
end

Plots.plot(times_split, surv_prob_noabm)

pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

phase_times_pre_12C_80 = deepcopy(phase_times_pre)
surv_prob_12C_80 = deepcopy(surv_prob)




###########12C 20


    E            = 20.0
    particle     = "12C"
    dose         = 1.5
    tumor_radius = 300.0

    # Optional parameters
    X_box       = 600.0
    X_voxel     = 300.0
    R_cell      = 15.0
    target_geom = "circle"
    calc_type   = "full"
    type_AT     = "KC"
    track_seg   = true

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)

    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

cell_df_copy = deepcopy(cell_df_original)
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)


#vscodedisplay(cell_df_copy[cell_df_copy.is_cell .== 1, :])

cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

print_phase_distribution(cell_df_original)
plot_phase_proportions_alive(cell_df_original)

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
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_12C_40 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2
Nsplit_2_12C_40 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])


surv_prob = Array{Float64, 1}()
compute_times_domain!(cell_df_istant, gsm2_cycle)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
plot_times(cell_df_istant)

tumor_radius = 500.0
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
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp  #! remove Rk
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])
cell_df.O .= 21.
F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)

mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])


# ── Setup ─────────────────────────────────────────────────────────────────────
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
nsim        = 10
phase_keys  = ("G0", "G1", "S", "G2", "M")

cell_df_ref = deepcopy(cell_df_copy)

function reset_cell!(dst, src)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

cell_work = deepcopy(cell_df_ref)

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0
    n_valid         = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)

            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            n_valid         += 1
            Nalive_pre_acc  += count(alive_pre)
            Nalive_post_acc += count(alive_post)
            surv_acc        += count(alive_post) / Ntot
            for (i, k) in enumerate(phase_keys)
                phase_pre_acc[i]  += counts_pre[k]
                phase_post_acc[i] += counts_post[k]
            end

        catch e
            @warn "sim=$sim t=$t failed" exception=(e, catch_backtrace())
        end
    end

    inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
    pre   = phase_pre_acc  .* inv_n
    post  = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1], pre[2], pre[3], pre[4], pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)

    GC.gc()
end

Plots.plot(times_split, surv_prob_noabm)

pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

phase_times_pre_12C_20 = deepcopy(phase_times_pre)
surv_prob_12C_20 = deepcopy(surv_prob)




###########12C 20


    E            = 15.0
    particle     = "12C"
    dose         = 1.5
    tumor_radius = 300.0

    # Optional parameters
    X_box       = 600.0
    X_voxel     = 300.0
    R_cell      = 15.0
    target_geom = "circle"
    calc_type   = "full"
    type_AT     = "KC"
    track_seg   = true

    N_sideVox   = Int(floor(2 * X_box / X_voxel))
    N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))

    setup_IonIrrad!(dose, E, particle)

    R_beam, x_beam, y_beam = calculate_beam_properties(
        calc_type, target_geom, X_box, X_voxel, tumor_radius)

    Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
    at_start   = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rp, Kp)

    setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide;
                        ParIrr="false", track_seg=track_seg)
    setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

    setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

    set_oxygen!(cell_df; plot_oxygen=false)
    O2_mean = mean(cell_df.O[cell_df.is_cell .== 1])

    F    = irrad.dose / (1.602e-9 * LET)
    Npar = round(Int, F * π * R_beam^2 * 1e-8)
    zF   = irrad.dose / Npar
    D    = irrad.doserate / zF
    T    = irrad.dose / (zF * D) * 3600

    println("Npar   : $Npar")
    println("R_beam : $(round(R_beam, digits=2))")
    println("O2     : $(round(O2_mean, digits=3))")

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1,
    0
)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) - sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end

cell_df_copy = deepcopy(cell_df_original)
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)


#vscodedisplay(cell_df_copy[cell_df_copy.is_cell .== 1, :])

cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

print_phase_distribution(cell_df_original)
plot_phase_proportions_alive(cell_df_original)

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
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_12C_40 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2
Nsplit_2_12C_40 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])


surv_prob = Array{Float64, 1}()
compute_times_domain!(cell_df_istant, gsm2_cycle)
cell_df_istant_ = cell_df_istant[cell_df_istant.is_cell .== 1, :]
push!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
plot_times(cell_df_istant)

tumor_radius = 500.0
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
R_beam, x_beam, y_beam = calculate_beam_properties(
    calc_type,
    target_geom,
    X_box,
    X_voxel,
    tumor_radius
);
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT);
Rk = Rp  #! remove Rk
at_start = AT(particle, E, A, Z, LET, 1.0, Rc, Rp, Rk, Kp);

set_oxygen!(cell_df; plot_oxygen = false)
O2_mean = mean(cell_df.O[cell_df.is_cell.==1])
cell_df.O .= 21.
F = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF = irrad.dose / Npar
D = irrad.doserate / zF
T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)

mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])


# ── Setup ─────────────────────────────────────────────────────────────────────
times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
nsim        = 10
phase_keys  = ("G0", "G1", "S", "G2", "M")

cell_df_ref = deepcopy(cell_df_copy)

function reset_cell!(dst, src)
    for col in names(src)
        S = src[!, col]
        D = dst[!, col]
        if eltype(S) <: Vector
            @inbounds for i in eachindex(S)
                copyto!(D[i], S[i])
            end
        else
            copyto!(D, S)
        end
    end
end

cell_work = deepcopy(cell_df_ref)

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0
    n_valid         = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)

            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            n_valid         += 1
            Nalive_pre_acc  += count(alive_pre)
            Nalive_post_acc += count(alive_post)
            surv_acc        += count(alive_post) / Ntot
            for (i, k) in enumerate(phase_keys)
                phase_pre_acc[i]  += counts_pre[k]
                phase_post_acc[i] += counts_post[k]
            end

        catch e
            @warn "sim=$sim t=$t failed" exception=(e, catch_backtrace())
        end
    end

    inv_n = n_valid > 0 ? 1.0 / n_valid : 0.0
    pre   = phase_pre_acc  .* inv_n
    post  = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1], pre[2], pre[3], pre[4], pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)

    GC.gc()
end

Plots.plot(times_split, surv_prob_noabm)

pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)


phase_times_pre_12C_15 = deepcopy(phase_times_pre)
surv_prob_12C_15 = deepcopy(surv_prob)



@assert nrow(phase_times_pre_12C_15) == length(surv_prob_12C_15)
@assert nrow(phase_times_pre_12C_20) == length(surv_prob_12C_20)
@assert nrow(phase_times_pre_12C_80) == length(surv_prob_12C_80)
@assert nrow(phase_times_pre_1H_2)  == length(surv_prob_1H_2)
@assert nrow(phase_times_pre_1H_10)  == length(surv_prob_1H_10)
@assert nrow(phase_times_pre_1H_100) == length(surv_prob_1H_100)

phase_times_pre_12C_15[!, :surv_prob] = surv_prob_12C_15
phase_times_pre_12C_20[!, :surv_prob] = surv_prob_12C_20
phase_times_pre_12C_80[!, :surv_prob] = surv_prob_12C_80
phase_times_pre_1H_2[!,  :surv_prob] = surv_prob_1H_2
phase_times_pre_1H_10[!,  :surv_prob] = surv_prob_1H_10
phase_times_pre_1H_100[!, :surv_prob] = surv_prob_1H_100

phase_times_pre_12C_15[!, :type] .= "12C_15"
phase_times_pre_12C_20[!, :type] .= "12C_20"
phase_times_pre_12C_80[!, :type] .= "12C_80"
phase_times_pre_1H_2[!,  :type] .= "1H_2"
phase_times_pre_1H_10[!,  :type] .= "1H_10"
phase_times_pre_1H_100[!, :type] .= "1H_100"

phase_times_pre_all = vcat(phase_times_pre_12C_15,
                            phase_times_pre_12C_20,
                            phase_times_pre_12C_80,
                            phase_times_pre_1H_2,
                            phase_times_pre_1H_10,
                            phase_times_pre_1H_100;
                            cols = :union)

CSV.write("phase_times_async.csv", phase_times_pre_all)


