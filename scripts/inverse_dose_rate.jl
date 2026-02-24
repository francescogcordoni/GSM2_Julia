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
include(joinpath(@__DIR__, "..", "src", "utilities_structures.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_general.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_radiation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_GSM2.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_biology.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_env.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_dose_computation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_dose_computation_GPU.jl"))
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
X_box    = 600.0      # corresponds to a full 1.8 mm box
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
cell_df.O .= 21.
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

Nsplit = mean(cell_irrad[cell_irrad.is_cell .== 1, :sp])^2

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
#times_split = [0.01, 0.1, 0.2, 0.5, 6.0, 8., 10., 12.0]

nsim        = 100
phase_keys  = ("G0", "G1", "S", "G2", "M")

# Freeze reference — never touched again
const cell_df_ref = deepcopy(cell_df_copy)


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

# Rebuild pool with clean deep copies
while isready(pool)
    take!(pool)
end
for _ in 1:n_workers
    put!(pool, deepcopy(cell_df_ref))
end

phase_times     = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
phase_times_pre = DataFrame(time=Float64[], Nalive=Float64[], G0=Float64[], G1=Float64[], S=Float64[], G2=Float64[], M=Float64[])
#surv_prob       = Vector{Float64}()
surv_prob_noabm = Vector{Float64}(undef, length(times_split))

# ── Loop 1: ABM ───────────────────────────────────────────────────────────────
for t in times_split
    println("ABM t = $t")

    results = Channel{NamedTuple}(nsim)

    @sync for _ in 1:nsim
        Threads.@spawn begin
            cell_ = take!(pool)                    # borrow a copy
            try
                reset_cell!(cell_, cell_df_ref)

                compute_times_domain!(cell_, gsm2_cycle; terminal_time=t)
                run_simulation_abm!(cell_; terminal_time=t)

                counts_pre = count_phase_alive(cell_; phase_col=:cell_cycle)
                alive_pre  = (cell_.is_cell .== 1) .& .!isfinite.(cell_.death_time)

                is_cell_mask = cell_.is_cell .== 1
                cell_.dam_X_dom[is_cell_mask] .+= cell_df_second.dam_X_dom[is_cell_mask]
                cell_.dam_Y_dom[is_cell_mask] .+= cell_df_second.dam_Y_dom[is_cell_mask]

                compute_times_domain!(cell_, gsm2_cycle)

                counts_post = count_phase_alive(cell_; phase_col=:cell_cycle)
                alive_post  = (cell_.is_cell .== 1) .& .!isfinite.(cell_.death_time)

                put!(results, (
                    alive_pre  = count(alive_pre),
                    alive_post = count(alive_post),
                    counts_pre = counts_pre,
                    counts_post = counts_post,
                    surv       = count(alive_post) / Ntot
                ))
            finally
                put!(pool, cell_)                  # always return the copy
            end
        end
    end
    close(results)

    # Reduce — single-threaded, no races
    inv_n           = 1.0 / nsim
    Nalive_pre_acc  = 0.0
    Nalive_post_acc = 0.0
    phase_pre_acc   = zeros(5)
    phase_post_acc  = zeros(5)
    surv_acc        = 0.0

    for r in results
        Nalive_pre_acc  += r.alive_pre
        Nalive_post_acc += r.alive_post
        surv_acc        += r.surv
        for (i, k) in enumerate(phase_keys)
            phase_pre_acc[i]  += r.counts_pre[k]
            phase_post_acc[i] += r.counts_post[k]
        end
    end

    pre  = phase_pre_acc  .* inv_n
    post = phase_post_acc .* inv_n
    push!(phase_times_pre, (t, Nalive_pre_acc*inv_n,  pre[1],  pre[2],  pre[3],  pre[4],  pre[5]))
    push!(phase_times,     (t, Nalive_post_acc*inv_n, post[1], post[2], post[3], post[4], post[5]))
    push!(surv_prob, surv_acc * inv_n)
end

# ── Loop 2: no ABM ────────────────────────────────────────────────────────────
for (ti, t) in enumerate(times_split)
    println("noABM t = $t")

    results = Channel{Float64}(nsim)

    @sync for _ in 1:nsim
        Threads.@spawn begin
            cell_ = take!(pool)
            try
                reset_cell!(cell_, cell_df_ref)

                # First shot repair up to time t
                compute_times_domain!(cell_, gsm2_cycle; terminal_time=t)

                # Kill cells with scheduled death, keep survivors and timeouts
                cell_.is_cell[isfinite.(cell_.death_time)] .= 0

                # Add second shot damage only to still-alive cells
                for i in findall(cell_.is_cell .== 1)
                    cell_.dam_X_dom[i] .+= cell_df_second.dam_X_dom[i]
                    cell_.dam_Y_dom[i] .+= cell_df_second.dam_Y_dom[i]
                end

                # Force all alive cells to G1 (no ABM, no division)
                for i in findall(cell_.is_cell .== 1)
                    cell_.cell_cycle[i] = "G1"
                end

                # Second shot repair, no terminal_time limit
                compute_times_domain!(cell_, gsm2_cycle)

                # Kill cells that die from combined damage
                cell_.is_cell[isfinite.(cell_.death_time)] .= 0

                surv = count((cell_.is_cell .== 1) .& .!isfinite.(cell_.death_time)) / Ntot
                put!(results, surv)
            finally
                put!(pool, cell_)
            end
        end
    end
    close(results)

    surv_prob_noabm[ti] = sum(results) / nsim
end

Plots.plot(times_split, surv_prob_noabm)

pushfirst!(times_split, 0.0)
#pushfirst!(surv_prob, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)
pushfirst!(surv_prob_noabm, size(cell_df_istant_[.!isfinite.(cell_df_istant_.death_time),:], 1)/Ntot)

p1 = Plots.plot(times_split, surv_prob[2:end])
p1 = Plots.plot!(times_split, surv_prob_noabm[2:end])

p2 = Plots.plot(phase_times_pre.time, phase_times_pre.Nalive, label = "Alive Cells")
p2 = Plots.plot!(phase_times.time, phase_times.G1, label = "G1")
p2 = Plots.plot!(phase_times.time, phase_times.S, label = "S")
p2 = Plots.plot!(phase_times.time, phase_times.G2, label = "G2")
p2 = Plots.plot!(phase_times.time, phase_times.M, label = "M")
p2 = Plots.plot!(phase_times.time, phase_times.G0, label = "G0")

plot(p1, p2, layout=(2, 1), size=(1000, 800))




