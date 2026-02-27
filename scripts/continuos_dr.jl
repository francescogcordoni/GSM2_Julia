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
using SparseArrays

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
gsm2_cycle[1] = GSM2(r, a, b, rd, Rn); #! G1
gsm2_cycle[2] = GSM2(r, a, b, rd, Rn); #! S
gsm2_cycle[3] = GSM2(r, a, b, rd, Rn); #! G2 - M
gsm2_cycle[4] = GSM2(r, a, b, rd, Rn); #! mixed


#& Construct the GSM2 object
setup_GSM2!(r, a, b, rd, Rn)

#~ ============================================================
#~ =================== Simulation Parameters ==================
#~ ============================================================

E            = 50.0
particle     = "1H"
dose         = 1.
tumor_radius = 300.0
X_box = 310.
setup(E, particle, dose, tumor_radius, X_box = X_box)
cell_df_copy = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg, single_particle = true)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)


lut = Vector{Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}}(undef, Npar)
@time @threads for p in 1:Npar

    # Each thread gets its OWN copy — this is key
    cell_df_copy = deepcopy(cell_df)

    MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y,
                    at, gsm2_cycle, type_AT, track_seg, single_particle = true)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    particle_damage = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    for row in eachrow(cell_df_copy)
        x = row.dam_X_dom
        y = row.dam_Y_dom

        if all(iszero, x) && all(iszero, y)
            continue
        end

        particle_damage[row.index] = (copy(x), copy(y))
    end

    lut[p] = particle_damage  # safe: each thread writes to a different index p
end

result_df = deepcopy(cell_df)
for p in 1:100
    for (idx, (x, y)) in lut[p]
        result_df[idx, :dam_X_dom] .+= x
        result_df[idx, :dam_Y_dom] .+= y
    end
end
plot_damage(result_df, layer_plot = true)

jldsave("lut_1H_1Gy_50MeV.jld2"; lut)
#lut = load("lut.jld2", "lut")



lut = Vector{Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}}(undef, 10)
@time for p in 1:10

    # Reset cell_df_copy for this particle
    cell_df_copy = deepcopy(cell_df)

    # Run simulation for single particle
    MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y,
                    at, gsm2_cycle, type_AT, track_seg, single_particle = true)
    MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

    particle_damage = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    for row in eachrow(cell_df_copy)
        x = row.dam_X_dom
        y = row.dam_Y_dom

        if all(iszero, x) && all(iszero, y)
            continue
        end

        particle_damage[row.index] = (copy(x), copy(y))
    end

    lut[p] = particle_damage
end

cell_df_copy[idx, :dam_X_dom] .+= x
cell_df_copy[idx, :dam_Y_dom] .+= y

au = 4.
doses_to_run = [0.1, 0.5, 1.0, 1.5, 2., 2.5, 3., 4.] # Doses from 1 to 5 Gy
doserates_to_run_Gys = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-2, 1e-1] # These values are in Gy/s
doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au # Converted to Gy/h for the simulation


    R_beam = at_start.Rp + Rn
    F = irrad.dose / (1.602e-9 * ion.LET)
    Npar = round(Int, F * (pi * (R_beam)^2 * 1e-8))
    if Npar == 0
        Npar = 1 # Ensure at least one particle to avoid division by zero
    end
    zF = irrad.dose / Npar
    dr = irrad.doserate / zF









#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)

cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0
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

p1 = Plots.plot(times_split, surv_prob)

p2 = Plots.plot(phase_times_pre.time, phase_times_pre.Nalive, label = "Alive Cells")
p2 = Plots.plot!(phase_times.time, phase_times.G1, label = "G1")
p2 = Plots.plot!(phase_times.time, phase_times.S, label = "S")
p2 = Plots.plot!(phase_times.time, phase_times.G2, label = "G2")
p2 = Plots.plot!(phase_times.time, phase_times.M, label = "M")
p2 = Plots.plot!(phase_times.time, phase_times.G0, label = "G0")

plot(p1, p2, layout=(2, 1), size=(1000, 800))

phase_times_pre_1H_50 = phase_times_pre
surv_prob_1H_50 = surv_prob





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

cell_df_original = deepcopy(cell_df_copy)

@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)

cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0
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
pushfirst!(surv_prob_noabm, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

p1 = Plots.plot(times_split, surv_prob)
p1 = Plots.plot!(times_split, surv_prob_noabm)

p2 = Plots.plot(phase_times_pre.time, phase_times_pre.Nalive, label = "Alive Cells")
p2 = Plots.plot!(phase_times.time, phase_times.G1, label = "G1")
p2 = Plots.plot!(phase_times.time, phase_times.S, label = "S")
p2 = Plots.plot!(phase_times.time, phase_times.G2, label = "G2")
p2 = Plots.plot!(phase_times.time, phase_times.M, label = "M")
p2 = Plots.plot!(phase_times.time, phase_times.G0, label = "G0")

plot(p1, p2, layout=(2, 1), size=(1000, 800))

phase_times_pre_1H_100 = phase_times_pre
surv_prob_1H_100 = surv_prob





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

cell_df_original = deepcopy(cell_df_copy)

@time MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
plot_scalar_cell(cell_df_copy, :dose_cell, layer_plot = true)

#~ ==========================================================================================
#~ ================================== compute damage ========================================
#~ ==========================================================================================

MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy, layer_plot = true)

cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0
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
pushfirst!(surv_prob_noabm, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

p1 = Plots.plot(times_split, surv_prob)
p1 = Plots.plot!(times_split, surv_prob_noabm)

p2 = Plots.plot(phase_times_pre.time, phase_times_pre.Nalive, label = "Alive Cells")
p2 = Plots.plot!(phase_times_pre.time, phase_times_pre.G1, label = "G1")
p2 = Plots.plot!(phase_times_pre.time, phase_times_pre.S, label = "S")
p2 = Plots.plot!(phase_times_pre.time, phase_times_pre.G2, label = "G2")
p2 = Plots.plot!(phase_times_pre.time, phase_times_pre.M, label = "M")
p2 = Plots.plot!(phase_times_pre.time, phase_times_pre.G0, label = "G0")

plot(p1, p2, layout=(2, 1), size=(1000, 800))

phase_times_pre_12C_80 = phase_times_pre
surv_prob_12C_80 = surv_prob



####### Plots

p1_1H_50 = Plots.plot(times_split, surv_prob_1H_50)

p2_1H_50 = Plots.plot(phase_times_pre_1H_50.time, phase_times_pre_1H_50.Nalive, label = "Alive Cells")
p2_1H_50 = Plots.plot!(phase_times_pre_1H_50.time, phase_times_pre_1H_50.G1, label = "G1")
p2_1H_50 = Plots.plot!(phase_times_pre_1H_50.time, phase_times_pre_1H_50.S, label = "S")
p2_1H_50 = Plots.plot!(phase_times_pre_1H_50.time, phase_times_pre_1H_50.G2, label = "G2")
p2_1H_50 = Plots.plot!(phase_times_pre_1H_50.time, phase_times_pre_1H_50.M, label = "M")
p2_1H_50 = Plots.plot!(phase_times_pre_1H_50.time, phase_times_pre_1H_50.G0, label = "G0")

plot(p1_1H_50, p2_1H_50, layout=(2, 1), size=(1000, 800))


p1_1H_100 = Plots.plot(times_split, surv_prob_1H_100)

p2_1H_100 = Plots.plot(phase_times_pre_1H_100.time, phase_times_pre_1H_100.Nalive, label = "Alive Cells")
p2_1H_100 = Plots.plot!(phase_times_pre_1H_100.time, phase_times_pre_1H_100.G1, label = "G1")
p2_1H_100 = Plots.plot!(phase_times_pre_1H_100.time, phase_times_pre_1H_100.S, label = "S")
p2_1H_100 = Plots.plot!(phase_times_pre_1H_100.time, phase_times_pre_1H_100.G2, label = "G2")
p2_1H_100 = Plots.plot!(phase_times_pre_1H_100.time, phase_times_pre_1H_100.M, label = "M")
p2_1H_100 = Plots.plot!(phase_times_pre_1H_100.time, phase_times_pre_1H_100.G0, label = "G0")

plot(p1_1H_100, p2_1H_100, layout=(2, 1), size=(1000, 800))


p1_12C_80 = Plots.plot(times_split, surv_prob_12C_80)

p2_12C_80 = Plots.plot(phase_times_pre_12C_80.time, phase_times_pre_12C_80.Nalive, label = "Alive Cells")
p2_12C_80 = Plots.plot(phase_times_pre_12C_80.time, phase_times_pre_12C_80.G1, label = "G1")
p2_12C_80 = Plots.plot(phase_times_pre_12C_80.time, phase_times_pre_12C_80.S, label = "S")
p2_12C_80 = Plots.plot(phase_times_pre_12C_80.time, phase_times_pre_12C_80.G2, label = "G2")
p2_12C_80 = Plots.plot(phase_times_pre_12C_80.time, phase_times_pre_12C_80.M, label = "M")
p2_12C_80 = Plots.plot(phase_times_pre_12C_80.time, phase_times_pre_12C_80.G0, label = "G0")

plot(p1_12C_80, p2_12C_80, layout=(2, 1), size=(1000, 800))





