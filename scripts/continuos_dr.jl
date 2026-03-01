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
cell_df.O .= 21.

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0
#cell_df_copy = deepcopy(cell_df)

# Step 1 — fast, parallel dose LUT
@time lut = MC_precompute_lut!(
    ion, Npar, R_beam, irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg;
    chunk_size = 50_000
)

@time damage_lut = precompute_damage_lut!(
    lut, cell_df_copy, irrad_cond, gsm2_cycle, ion;
    chunk_size = 50_000
)

#filter!(p -> !isempty(p), damage_lut)

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end

for p in 1:length(damage_lut)
    for (cell_idx, (x, y)) in damage_lut[p]
        cell_df_copy.dam_X_dom[cell_idx] .+= x
        cell_df_copy.dam_Y_dom[cell_idx] .+= y
    end
end
cell_df_copy[!, :dam_X_total] = sum.(cell_df_copy.dam_X_dom)
cell_df_copy[!, :dam_Y_total] = sum.(cell_df_copy.dam_Y_dom)
plot_damage(cell_df_copy, layer_plot = true)

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0
#jldsave("lut_1H_1Gy_50MeV.jld2"; damage_lut)
#lut = load("lut.jld2", "lut")

au = 4.
doses_to_run          = [0.1, 0.5, 1.0, 1.5, 2., 2.5, 3., 4.]
doserates_to_run_Gys  = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-2, 1e-1]
doserates_to_run_Gyh  = doserates_to_run_Gys .* 3600.0 .* au

survival_results = zeros(length(doses_to_run), length(doserates_to_run_Gyh))
Npar_effect      = length(damage_lut)

# Pre-extract vectors once — avoid repeated DataFrame access
is_cell_base  = copy(cell_df_copy.is_cell)
dam_X_base    = deepcopy(cell_df_copy.dam_X_dom)
dam_Y_base    = deepcopy(cell_df_copy.dam_Y_dom)
death_time_base = copy(cell_df_copy.death_time)
Ntot          = count(==(1), is_cell_base)

# Pre-build index → row map once
index_to_row = Dict(idx => r for (r, idx) in enumerate(cell_df_copy.index))

for (j, dose_rate_gyh) in enumerate(doserates_to_run_Gyh)
    println("Running dose rate: $dose_rate_gyh Gy/h")
    dr = dose_rate_gyh / zF

    for (k, dose) in enumerate(doses_to_run)
        println("  Running dose: $dose Gy")

        N_dose = round(Int, dose * Npar_effect)
        times_ = rand(Exponential(1/dr), N_dose)

        # Reset from base vectors — much faster than deepcopy of full df
        is_cell    = copy(is_cell_base)
        dam_X_dom  = deepcopy(dam_X_base)
        dam_Y_dom  = deepcopy(dam_Y_base)
        death_time = copy(death_time_base)

        # Write back to df once before loop
        cell_df_dr = cell_df_copy
        cell_df_dr[!, :is_cell]    = is_cell
        cell_df_dr[!, :dam_X_dom]  = dam_X_dom
        cell_df_dr[!, :dam_Y_dom]  = dam_Y_dom
        cell_df_dr[!, :death_time] = death_time

        for i in 1:N_dose
            lut_idx = mod1(i, Npar_effect)

            # Direct vector access — no DataFrame overhead
            @inbounds for (idx, (x, y)) in damage_lut[lut_idx]
                row = index_to_row[idx]
                is_cell[row] == 0 && continue  # skip dead cells
                dam_X_dom[row] .+= x
                dam_Y_dom[row] .+= y
            end

            # Sync to df only for compute_times_domain!
            cell_df_dr[!, :dam_X_dom]  = dam_X_dom
            cell_df_dr[!, :dam_Y_dom]  = dam_Y_dom
            cell_df_dr[!, :is_cell]    = is_cell

            compute_times_domain!(cell_df_dr, gsm2_cycle; terminal_time = times_[i])

            # Pull death_time back and update is_cell/damage in vectors
            death_time = cell_df_dr.death_time
            @inbounds for r in 1:length(is_cell)
                if isfinite(death_time[r])
                    is_cell[r] = 0
                    fill!(dam_X_dom[r], 0)
                    fill!(dam_Y_dom[r], 0)
                end
            end
        end

        survival_results[k, j] = count(r -> !isfinite(death_time[r]) && is_cell[r] == 1,
                                        1:length(is_cell)) / Ntot

        println("    Survival: $(survival_results[k,j])")
    end
end

using Plots

p = plot(
    xlabel  = "Dose (Gy)",
    ylabel  = "Survival fraction",
    yscale  = :log10,
    title   = "Survival vs Dose",
    legend  = :topright
)

for (j, dr) in enumerate(doserates_to_run_Gys)
    plot!(p, doses_to_run, survival_results[:, j],
            label  = "$(dr) Gy/s",
            marker = :circle,
            lw     = 2)
end

display(p)

