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
E            = 100.0
particle     = "1H"
dose         = 1.5
tumor_radius = 300.0
setup(E, particle, dose, tumor_radius)

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

cell_df_copy.cell_cycle .= "G1"
cell_df_copy.can_divide .= 0

cell_df_original = deepcopy(cell_df_copy)
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
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
MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_irrad, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_irrad, irrad_cond, gsm2_cycle)
    
cell_df_istant.dam_X_dom .+= cell_irrad.dam_X_dom
cell_df_istant.dam_Y_dom .+= cell_irrad.dam_Y_dom
compute_cell_survival_GSM2!(cell_df_istant, gsm2_cycle)
compute_cell_survival_GSM2!(cell_irrad, gsm2_cycle)

mean(cell_df_istant[cell_df_istant.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])
mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

Nsplit_1H_50 = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2


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
@time MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_second, df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
MC_loop_damage!(ion, cell_df_second, irrad_cond, gsm2_cycle)
mean(cell_df_second.dose_cell[cell_df_second.is_cell .== 1])
cell_df_second_sp = deepcopy(cell_df_second)
compute_times_domain!(cell_df_second_sp, gsm2_cycle)
Nsplit_2_1H_50 = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])*mean(cell_df_second_sp[cell_df_copy.is_cell .== 1, :sp])

findall(i -> cell_df_second_sp.is_cell[i] == 1 && isfinite(cell_df_second_sp.death_time[i]) && isfinite(cell_df_second_sp.cycle_time[i]), 1:nrow(cell_df_second_sp))

# ── Setup ─────────────────────────────────────────────────────────────────────
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


            # in ABM loop, after compute_times_domain!(cell_work, gsm2_cycle):
#for i in findall(cell_work.is_cell .== 1)
#    if isfinite(cell_work.death_time[i])
#        cell_work.is_cell[i] = 0
#    end
#end

# then count:
alive_post = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)



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

phase_times_pre_1H_100 = deepcopy(phase_times_pre)
surv_prob_1H_100 = deepcopy(surv_prob)

Plots.plot(times_split, surv_prob_1H_100)
Plots.hline!([Nsplit_2_1H_50], color=:red, linestyle=:dash, linewidth=2, label="Reference (no repair)")



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
gsm2_cycle[2] = GSM2(r_G1, a_G1, b_G1, rd, Rn)
gsm2_cycle[3] = GSM2(r_G1, a_G1, b_G1, rd, Rn)
gsm2_cycle[4] =GSM2(r_G1, a_G1, b_G1, rd, Rn)


#& Construct the GSM2 object
setup_GSM2!(r, a, b, rd, Rn)

#~ ============================================================
#~ =================== Simulation Parameters ==================
#~ ============================================================

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


#            # in ABM loop, after compute_times_domain!(cell_work, gsm2_cycle):
#for i in findall(cell_work.is_cell .== 1)
#    if isfinite(cell_work.death_time[i])
#        cell_work.is_cell[i] = 0
#    end
#end

# then count:
alive_post = (cell_work.is_cell .== 1) .& .!isfinite.(cell_work.death_time)



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

phase_times_pre_1H_100_G1 = deepcopy(phase_times_pre)
surv_prob_1H_100_G1 = deepcopy(surv_prob)

Plots.plot(times_split, surv_prob_1H_100, label="Full cycle")
Plots.plot!(times_split, surv_prob_1H_100_G1, label="Only G1")
Plots.hline!([Nsplit_2_1H_50], color=:red, linestyle=:dash, linewidth=2, label="Reference")




surv_prob = Vector{Float64}()

for t in times_split
    println("No-ABM t = $t")

    surv_acc = 0.0
    n_valid  = 0

    for sim in 1:nsim
        reset_cell!(cell_work, cell_df_ref)

        # ── Shot 1: repair for time t ─────────────────────────────────────
        # timeout cells get residual damage written back to dam_X_dom/dam_Y_dom
        # survivors get dam_X_dom/dam_Y_dom cleared
        # scheduled/immediate deaths get is_cell=0
        compute_times_domain!(cell_work, gsm2_cycle; terminal_time = Float64(t))

        # kill scheduled deaths (finite death_time but is_cell still 1)
        for i in 1:nrow(cell_work)
            if cell_work.is_cell[i] == 1 && isfinite(cell_work.death_time[i])
                cell_work.is_cell[i] = 0
                for nei_idx in cell_work.nei[i]
                    cell_work.number_nei[nei_idx] =
                        length(cell_work.nei[nei_idx]) - sum(cell_work.is_cell[cell_work.nei[nei_idx]])
                end
            end
        end

        # ── Shot 2: add damage on top of residual ────────────────────────
        # for survivors: dam_X_dom=0 + shot2 damage
        # for timeout:   dam_X_dom=residual + shot2 damage
        for i in findall(cell_work.is_cell .== 1)
            cell_work.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
            cell_work.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
        end

        # ── Evaluate survival after shot 2 ───────────────────────────────
        # no terminal_time: run Gillespie to completion
        compute_times_domain!(cell_work, gsm2_cycle)

        # kill scheduled deaths from shot 2 as well
        for i in 1:nrow(cell_work)
            if cell_work.is_cell[i] == 1 && isfinite(cell_work.death_time[i])
                cell_work.is_cell[i] = 0
            end
        end

        alive_post = count(i -> cell_work.is_cell[i] == 1 && isinf(cell_work.death_time[i]), 1:nrow(cell_work))
        surv_acc  += alive_post / Ntot
        n_valid   += 1
    end

    push!(surv_prob, surv_acc / n_valid)
    GC.gc()
end

surv_prob_1H_100_noabm = deepcopy(surv_prob)




# shot 1 survival — run Gillespie on shot1 damage alone
cell_ref_shot1 = deepcopy(cell_df_ref)  # has shot1 damage, is the reference
compute_times_domain!(cell_ref_shot1, gsm2_cycle)
for i in 1:nrow(cell_ref_shot1)
    if cell_ref_shot1.is_cell[i] == 1 && isfinite(cell_ref_shot1.death_time[i])
        cell_ref_shot1.is_cell[i] = 0
    end
end
surv_shot1 = count(i -> cell_ref_shot1.is_cell[i] == 1, 1:nrow(cell_ref_shot1)) / Ntot

# shot 2 survival — run Gillespie on shot2 damage alone, on fresh undamaged cells
cell_ref_shot2 = deepcopy(cell_df_ref)
# replace shot1 damage with shot2 damage
for i in findall(cell_ref_shot2.is_cell .== 1)
    cell_ref_shot2.dam_X_dom[i] .= cell_df_second.dam_X_dom[i]
    cell_ref_shot2.dam_Y_dom[i] .= cell_df_second.dam_Y_dom[i]
end
compute_times_domain!(cell_ref_shot2, gsm2_cycle)
for i in 1:nrow(cell_ref_shot2)
    if cell_ref_shot2.is_cell[i] == 1 && isfinite(cell_ref_shot2.death_time[i])
        cell_ref_shot2.is_cell[i] = 0
    end
end
surv_shot2 = count(i -> cell_ref_shot2.is_cell[i] == 1, 1:nrow(cell_ref_shot2)) / Ntot

Nsplit = surv_shot1 * surv_shot2

Plots.plot(times_split, surv_prob_1H_100, label="Full cycle")
Plots.plot!(times_split, surv_prob_1H_100_G1, label="Only G1")
Plots.plot!(times_split, surv_prob_1H_100_noabm, label="No ABM")
Plots.hline!([Nsplit], color=:red, linestyle=:dash, linewidth=2, label="Reference")

# ── Save results ──────────────────────────────────────────────────────────────
datadir = joinpath(@__DIR__, "..", "data", "split_repopulation")
mkpath(datadir)

# Survival curves
surv_df = DataFrame(
    time          = times_split,
    surv_abm_full = surv_prob_1H_100,
    surv_abm_G1   = surv_prob_1H_100_G1,
    surv_noabm    = surv_prob_1H_100_noabm,
)
CSV.write(joinpath(datadir, "surv_curves.csv"), surv_df)

# Phase times
CSV.write(joinpath(datadir, "phase_times_pre_full.csv"), phase_times_pre_1H_100)
CSV.write(joinpath(datadir, "phase_times_pre_G1.csv"),   phase_times_pre_1H_100_G1)
CSV.write(joinpath(datadir, "phase_times_post_G1.csv"),  phase_times)

# Scalar metadata
meta_df = DataFrame(
    Nsplit         = [Nsplit],
    Nsplit_2_1H_50 = [Nsplit_2_1H_50],
    Ntot           = [Ntot],
    particle       = [particle],
    E              = [E],
    dose           = [dose],
)
CSV.write(joinpath(datadir, "metadata.csv"), meta_df)

# JLD2 — all variables in one file
jldsave(joinpath(datadir, "split_repopulation.jld2");
    times_split,
    surv_prob_1H_100,
    surv_prob_1H_100_G1,
    surv_prob_1H_100_noabm,
    phase_times_pre_1H_100,
    phase_times_pre_1H_100_G1,
    phase_times_post_G1 = phase_times,
    Nsplit,
    Nsplit_2_1H_50,
    Ntot,
    particle,
    E,
    dose,
)

println("Results saved to $datadir")

