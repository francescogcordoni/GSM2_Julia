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
using Printf

nthreads()

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

sp = load_stopping_power()

#~ ============================================================
#~ GSM2 parameters
#~ ============================================================
a_G1 = 0.012872261720543399
b_G1 = 0.04029756109753225
r_G1 = 2.780479661191086

a_S  = 0.00589118894714544
b_S  = 0.05794352736120672
r_S  = 5.84009601901114

a_G2 = 0.024306291709970018
b_G2 = 5.704688326522623e-5
r_G2 = 1.7720064637774506

a  = 0.01481379648786136
b  = 0.012663276476522422
r  = 2.5656972960759896
rd = 0.8
Rn = 7.2

gsm2_cycle = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)

setup_GSM2!(r, a, b, rd, Rn)

datadir = joinpath(@__DIR__, "..", "data", "inverse_doserate")
mkpath(datadir)

# ++ ADD: helper to kill scheduled deaths consistently
function kill_scheduled_deaths!(cell_df; update_neighbors=true)
    for i in 1:nrow(cell_df)
        if cell_df.is_cell[i] == 1 && isfinite(cell_df.death_time[i])
            cell_df.is_cell[i] = 0
            if update_neighbors
                for nei_idx in cell_df.nei[i]
                    cell_df.number_nei[nei_idx] =
                        length(cell_df.nei[nei_idx]) -
                        sum(cell_df.is_cell[cell_df.nei[nei_idx]])
                end
            end
        end
    end
end

# ++ ADD: correct reference using Gillespie directly
function compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)
    cell_ref_shot1 = deepcopy(cell_df_ref)
    compute_times_domain!(cell_ref_shot1, gsm2_cycle)
    kill_scheduled_deaths!(cell_ref_shot1; update_neighbors=false)
    surv_shot1 = count(i -> cell_ref_shot1.is_cell[i] == 1,
                       1:nrow(cell_ref_shot1)) / Ntot

    cell_ref_shot2 = deepcopy(cell_df_ref)
    for i in findall(cell_ref_shot2.is_cell .== 1)
        cell_ref_shot2.dam_X_dom[i] .= cell_df_second.dam_X_dom[i]
        cell_ref_shot2.dam_Y_dom[i] .= cell_df_second.dam_Y_dom[i]
    end
    compute_times_domain!(cell_ref_shot2, gsm2_cycle)
    kill_scheduled_deaths!(cell_ref_shot2; update_neighbors=false)
    surv_shot2 = count(i -> cell_ref_shot2.is_cell[i] == 1,
                       1:nrow(cell_ref_shot2)) / Ntot

    Nsplit = surv_shot1 * surv_shot2
    println("  Gillespie shot1 survival : $(round(surv_shot1, digits=4))")
    println("  Gillespie shot2 survival : $(round(surv_shot2, digits=4))")
    println("  Correct reference        : $(round(Nsplit,     digits=4))")
    return Nsplit
end

#~ ============================================================
#~ Simulation Parameters
#~ ============================================================
E            = 2.0
particle     = "1H"
dose         = 1.5
tumor_radius = 300.0
setup(E, particle, dose, tumor_radius)

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                 sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end
cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

tumor_radius = 500.0
track_seg    = true
setup_cell_lattice!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide; ParIrr="false", track_seg=track_seg)
setup_cell_population!(target_geom, X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
println("Number of cells = ", sum(cell_df.is_cell .== 1))
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
R_beam, x_beam, y_beam = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT)
Rk = Rp
cell_df.O .= 21.
F    = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF   = irrad.dose / Npar
D    = irrad.doserate / zF
T    = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

times_split = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
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

# ++ ADD: compute correct reference for this condition
Nsplit_1H_2 = compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)

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

            #kill_scheduled_deaths!(cell_work; update_neighbors=true)

            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)

            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)

            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end

            compute_times_domain!(cell_work, gsm2_cycle)

            # ++ ADD: kill scheduled deaths from shot 2
            kill_scheduled_deaths!(cell_work; update_neighbors=false)

            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& isinf.(cell_work.death_time)

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

phase_times_pre_1H_2 = deepcopy(phase_times_pre)
surv_prob_1H_2       = deepcopy(surv_prob)

######################1H 10
E = 10.; particle = "1H"; dose = 1.5; tumor_radius = 300.
setup(E, particle, dose, tumor_radius)

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                 sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end
cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

tumor_radius = 500.0
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
R_beam, x_beam, y_beam = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT); Rk = Rp
cell_df.O .= 21.
F    = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF   = irrad.dose / Npar; D = irrad.doserate / zF; T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

times_split = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
cell_df_ref = deepcopy(cell_df_copy)
cell_work   = deepcopy(cell_df_ref)
phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()

# ++ ADD
Nsplit_1H_10 = compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)

for t in times_split
    println("ABM t = $t")
    Nalive_pre_acc = 0.0; Nalive_post_acc = 0.0
    phase_pre_acc  = zeros(5); phase_post_acc = zeros(5)
    surv_acc = 0.0; n_valid = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)
            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            # ++ ADD
            #kill_scheduled_deaths!(cell_work; update_neighbors=true)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)
            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)
            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end
            compute_times_domain!(cell_work, gsm2_cycle)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=false)
            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& isinf.(cell_work.death_time)
            n_valid += 1
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
    pre = phase_pre_acc .* inv_n; post = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)
    GC.gc()
end

phase_times_pre_1H_10 = deepcopy(phase_times_pre)
surv_prob_1H_10       = deepcopy(surv_prob)

###########1H 100
E = 100.0; particle = "1H"; dose = 1.5; tumor_radius = 300.0
setup(E, particle, dose, tumor_radius)

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                 sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end
cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

tumor_radius = 500.0
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
R_beam, x_beam, y_beam = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT); Rk = Rp
cell_df.O .= 21.
F    = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF   = irrad.dose / Npar; D = irrad.doserate / zF; T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

times_split = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
cell_df_ref = deepcopy(cell_df_copy)
cell_work   = deepcopy(cell_df_ref)
phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()

# ++ ADD
Nsplit_1H_100 = compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)

for t in times_split
    println("ABM t = $t")
    Nalive_pre_acc = 0.0; Nalive_post_acc = 0.0
    phase_pre_acc  = zeros(5); phase_post_acc = zeros(5)
    surv_acc = 0.0; n_valid = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)
            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            # ++ ADD
            #kill_scheduled_deaths!(cell_work; update_neighbors=true)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)
            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)
            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end
            compute_times_domain!(cell_work, gsm2_cycle)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=false)
            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& isinf.(cell_work.death_time)
            n_valid += 1
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
    pre = phase_pre_acc .* inv_n; post = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)
    GC.gc()
end

phase_times_pre_1H_100 = deepcopy(phase_times_pre)
surv_prob_1H_100       = deepcopy(surv_prob)

###########12C 80
E = 80.0; particle = "12C"; dose = 1.5; tumor_radius = 300.0
setup(E, particle, dose, tumor_radius)

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                 sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end
cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

tumor_radius = 500.0
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
R_beam, x_beam, y_beam = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT); Rk = Rp
cell_df.O .= 21.
F    = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF   = irrad.dose / Npar; D = irrad.doserate / zF; T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

times_split = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
cell_df_ref = deepcopy(cell_df_copy)
cell_work   = deepcopy(cell_df_ref)
phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()

# ++ ADD
Nsplit_12C_80 = compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)

for t in times_split
    println("ABM t = $t")
    Nalive_pre_acc = 0.0; Nalive_post_acc = 0.0
    phase_pre_acc  = zeros(5); phase_post_acc = zeros(5)
    surv_acc = 0.0; n_valid = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)
            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=true)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)
            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)
            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end
            compute_times_domain!(cell_work, gsm2_cycle)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=false)
            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& isinf.(cell_work.death_time)
            n_valid += 1
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
    pre = phase_pre_acc .* inv_n; post = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)
    GC.gc()
end

phase_times_pre_12C_80 = deepcopy(phase_times_pre)
surv_prob_12C_80       = deepcopy(surv_prob)

###########12C 20
E = 20.0; particle = "12C"; dose = 1.5; tumor_radius = 300.0
setup(E, particle, dose, tumor_radius)

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                 sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end
cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

tumor_radius = 500.0
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
R_beam, x_beam, y_beam = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT); Rk = Rp
cell_df.O .= 21.
F    = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF   = irrad.dose / Npar; D = irrad.doserate / zF; T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

times_split = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
cell_df_ref = deepcopy(cell_df_copy)
cell_work   = deepcopy(cell_df_ref)
phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()

# ++ ADD
Nsplit_12C_20 = compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)

for t in times_split
    println("ABM t = $t")
    Nalive_pre_acc = 0.0; Nalive_post_acc = 0.0
    phase_pre_acc  = zeros(5); phase_post_acc = zeros(5)
    surv_acc = 0.0; n_valid = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)
            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=true)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)
            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)
            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end
            compute_times_domain!(cell_work, gsm2_cycle)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=false)
            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& isinf.(cell_work.death_time)
            n_valid += 1
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
    pre = phase_pre_acc .* inv_n; post = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)
    GC.gc()
end

phase_times_pre_12C_20 = deepcopy(phase_times_pre)
surv_prob_12C_20       = deepcopy(surv_prob)

###########12C 15
E = 15.0; particle = "12C"; dose = 1.5; tumor_radius = 300.0
setup(E, particle, dose, tumor_radius)

cell_df_copy = deepcopy(cell_df)
cell_df.O .= 21.
cell_df_copy.is_cell = ifelse.(
    (cell_df_copy.x.^2 .+ cell_df_copy.y.^2 .+ cell_df_copy.z.^2 .<= 300^2) .&
    ((cell_df_copy.x .÷ 30 .+ cell_df_copy.y .÷ 30 .+ cell_df_copy.z .÷ 30) .% 2 .== 0),
    1, 0)
for i in 1:nrow(cell_df_copy)
    cell_df_copy.number_nei[i] = length(cell_df_copy.nei[i]) -
                                 sum(cell_df_copy.is_cell[cell_df_copy.nei[i]])
end
cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
cell_df_original = deepcopy(cell_df_copy)
Ntot = size(cell_df_original[cell_df_original.is_cell .== 1, :], 1)

tumor_radius = 500.0
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
R_beam, x_beam, y_beam = calculate_beam_properties(calc_type, target_geom, X_box, X_voxel, tumor_radius)
Rc, Rp, Kp = ATRadius(ion, irrad, type_AT); Rk = Rp
cell_df.O .= 21.
F    = irrad.dose / (1.602 * 10^(-9) * LET)
Npar = round(Int, F * (pi * (R_beam)^2 * 10^(-8)))
zF   = irrad.dose / Npar; D = irrad.doserate / zF; T = irrad.dose / (zF * D) * 3600
cell_df_second = deepcopy(cell_df)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)

times_split = [0.01, 0.1, 0.2, 0.5, 1., 2., 3., 4., 5., 6.0, 8., 10., 12.0, 14., 16., 18., 19., 20., 21., 22., 23., 24.0, 25., 26., 27., 30., 48.0]
cell_df_ref = deepcopy(cell_df_copy)
cell_work   = deepcopy(cell_df_ref)
phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
surv_prob       = Vector{Float64}()

# ++ ADD
Nsplit_12C_15 = compute_correct_reference(cell_df_ref, cell_df_second, gsm2_cycle, Ntot)

for t in times_split
    println("ABM t = $t")
    Nalive_pre_acc = 0.0; Nalive_post_acc = 0.0
    phase_pre_acc  = zeros(5); phase_post_acc = zeros(5)
    surv_acc = 0.0; n_valid = 0

    for sim in 1:nsim
        try
            reset_cell!(cell_work, cell_df_ref)
            compute_times_domain!(cell_work, gsm2_cycle; terminal_time=t)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=true)
            run_simulation_abm!(cell_work; terminal_time=t, verbose=false)
            counts_pre = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_pre  = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)
            for i in findall(cell_work.is_cell .== 1)
                cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
            end
            compute_times_domain!(cell_work, gsm2_cycle)
            # ++ ADD
            kill_scheduled_deaths!(cell_work; update_neighbors=false)
            counts_post = count_phase_alive(cell_work; phase_col=:cell_cycle)
            alive_post  = (cell_work.is_cell .== 1) .& isinf.(cell_work.death_time)
            n_valid += 1
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
    pre = phase_pre_acc .* inv_n; post = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)
    GC.gc()
end

phase_times_pre_12C_15 = deepcopy(phase_times_pre)
surv_prob_12C_15       = deepcopy(surv_prob)

#~ ============================================================
#~ Assemble and save
#~ ============================================================
for (df, sp, label) in [
    (phase_times_pre_1H_2,   surv_prob_1H_2,   "1H_2"),
    (phase_times_pre_1H_10,  surv_prob_1H_10,  "1H_10"),
    (phase_times_pre_1H_100, surv_prob_1H_100, "1H_100"),
    (phase_times_pre_12C_80, surv_prob_12C_80, "12C_80"),
    (phase_times_pre_12C_20, surv_prob_12C_20, "12C_20"),
    (phase_times_pre_12C_15, surv_prob_12C_15, "12C_15"),
]
    @assert nrow(df) == length(sp) "length mismatch for $label"
    df[!, :surv_prob] = sp
    df[!, :type]      .= label
end

phase_times_pre_all = vcat(
    phase_times_pre_1H_2,
    phase_times_pre_1H_10,
    phase_times_pre_1H_100,
    phase_times_pre_12C_80,
    phase_times_pre_12C_20,
    phase_times_pre_12C_15;
    cols=:union)

CSV.write(joinpath(datadir, "phase_times.csv"), phase_times_pre_all)
println("Saved phase_times.csv")

# Save reference (Nsplit) values
nsplit_df = DataFrame(
    type   = ["1H_2",        "1H_10",        "1H_100",        "12C_80",        "12C_20",        "12C_15"],
    Nsplit = [Nsplit_1H_2,   Nsplit_1H_10,   Nsplit_1H_100,   Nsplit_12C_80,   Nsplit_12C_20,   Nsplit_12C_15],
)
CSV.write(joinpath(datadir, "nsplit_reference.csv"), nsplit_df)
println("Saved nsplit_reference.csv")
println("Saved split_dose_survival.png")