using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using CUDA
using Statistics: mean
using SparseArrays
using Printf

nthreads()

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

#& Stopping power
sp = load_stopping_power()

#~ ============================================================
#~ GSM2 parameters
#~ ============================================================

#& G1 -> 12h
a_G1 = 0.012872261720543399;  b_G1 = 0.04029756109753225;  r_G1 = 2.780479661191086

#& S -> 8h
a_S = 0.00589118894714544;    b_S = 0.05794352736120672;    r_S = 5.84009601901114

#& G2 - M -> 3h + 1h
a_G2 = 0.024306291709970018;  b_G2 = 5.704688326522623e-5;  r_G2 = 1.7720064637774506

#& mixed
a  = 0.01481379648786136;     b  = 0.012663276476522422;    r  = 2.5656972960759896
rd = 0.8;                     Rn = 7.2

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)   #! G1
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)   #! S
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)   #! G2 - M
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)    #! mixed

setup_GSM2!(r, a, b, rd, Rn)

# ── LQ fitting helper ─────────────────────────────────────────────────────────
function fit_lq_survival(doses, surv_df, doserates_Gys; tag="")
    lq(D, p) = exp.(-p[1] .* D .- p[2] .* D .^ 2)
    dr_cols   = setdiff(names(surv_df), ["dose_Gy"])
    params_df = DataFrame(
        tag           = String[],
        dose_rate_Gys = Float64[],
        alpha         = Float64[],
        beta          = Float64[],
        alpha_err     = Float64[],
        beta_err      = Float64[],
        alpha_beta    = Float64[],
        r2            = Float64[],
    )
    for (j, col) in enumerate(dr_cols)
        surv = clamp.(surv_df[!, col], 1e-10, 1.0)
        local fit_res
        try
            fit_res = curve_fit(lq, doses, surv, [0.1, 0.05];
                                lower=[0.0, 0.0], upper=[10.0, 10.0])
        catch e
            @warn "LQ fit failed for $col ($tag)" exception=e
            continue
        end
        α, β   = coef(fit_res)
        se     = try stderror(fit_res) catch; [NaN, NaN] end
        pred   = lq(doses, [α, β])
        ss_res = sum((surv .- pred) .^ 2)
        ss_tot = sum((surv .- mean(surv)) .^ 2)
        r2     = 1.0 - ss_res / ss_tot
        push!(params_df, (tag, doserates_Gys[j], α, β, se[1], se[2], α / β, r2))
        println(@sprintf("  %-22s  α=%.4f±%.4f  β=%.4f±%.4f  α/β=%.2f  R²=%.4f",
                         col, α, se[1], β, se[2], α / β, r2))
    end
    return params_df
end

#~ ============================================================
#~ Simulation parameters
#~ ============================================================
E            = 10.0
particle     = "12C"
dose         = 4.0
tumor_radius = 450.0
X_box        = 560.0
au           = 4.0

setup(E, particle, dose, tumor_radius; X_box = X_box)

cell_df_copy = deepcopy(cell_df)
cell_df.O   .= 21.0

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0

#~ ============================================================
#~ LUT precomputation
#~ ============================================================
println("Precomputing dose LUT...")
@time lut = MC_precompute_lut!(
    ion, Npar, R_beam, irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg;
    chunk_size = 50_000)

println("Precomputing damage LUT...")
@time damage_lut = precompute_damage_lut!(
    lut, cell_df_copy, irrad_cond, gsm2_cycle, ion;
    chunk_size = 50_000)

jldsave("lut_12C_4Gy_10MeV.jld2"; damage_lut)

Npar_effect = length(damage_lut)
println("Npar_effect = $Npar_effect")

#~ ============================================================
#~ Base active-cell template
#~ ============================================================
active_cells_base = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
for row in eachrow(cell_df_copy)
    row.is_cell != 1 && continue
    active_cells_base[row.index] = (zeros(Int, length(row.dam_X_dom)),
                                    zeros(Int, length(row.dam_Y_dom)))
end
Ntot = length(active_cells_base)
println("Total cells (Ntot) = $Ntot")

outdir = joinpath(@__DIR__, "..", "data", "continuous_dr")
mkpath(outdir)

#~ ============================================================
#~ Survival vs dose loop
#~ ============================================================
doses_to_run         = [0.1, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0]
doserates_to_run_Gys = [1e-5, 1e-4, 1e-3, 1e-2]
doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

survival_results = zeros(length(doses_to_run), length(doserates_to_run_Gyh))
gsm2             = gsm2_cycle[1]

for (j, dose_rate_gyh) in enumerate(doserates_to_run_Gyh)
    println("\nRunning dose rate: $dose_rate_gyh Gy/h")
    dr = dose_rate_gyh / zF

    for (k, dose) in enumerate(doses_to_run)
        N_dose     = round(Int, dose * Npar_effect/4)
        times_full = rand(Exponential(1.0 / dr), N_dose)
        lut_order  = mod1.(randperm(N_dose), Npar_effect)

        times_filtered   = Float64[]
        lut_indices      = Int[]
        accumulated_time = 0.0

        for i in 1:N_dose
            lut_idx          = lut_order[i]
            accumulated_time += times_full[i]
            if !isempty(damage_lut[lut_idx])
                push!(times_filtered,   accumulated_time)
                push!(lut_indices,      lut_idx)
                accumulated_time = 0.0
            end
        end
        println("  [dose=$(dose) Gy] Effective particles: $(length(lut_indices)) / $N_dose")

        # Reset active cells from base template
        active_cells = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}(
            idx => (copy(X), copy(Y))
            for (idx, (X, Y)) in active_cells_base)

        # Main particle loop
        for i in 1:length(lut_indices)
            lut_idx = lut_indices[i]

            # 1. Apply damage
            @inbounds for (idx, (x, y)) in damage_lut[lut_idx]
                !haskey(active_cells, idx) && continue
                active_cells[idx][1] .+= x
                active_cells[idx][2] .+= y
            end

            # 2. Repair window: interval until next hit (Inf for last particle)
            t = i < length(lut_indices) ? times_filtered[i + 1] : Inf

            # 3. SSA repair
            to_delete = Int[]
            for (cell_idx, (X, Y)) in active_cells
                all(iszero, X) && all(iszero, Y) && continue
                death_time, _, _, X_new, Y_new =
                    compute_repair_domain(X, Y, gsm2;
                                          terminal_time = t, au = au)
                if isfinite(death_time)
                    push!(to_delete, cell_idx)
                else
                    active_cells[cell_idx] = (X_new, Y_new)
                end
            end

            for idx in to_delete
                delete!(active_cells, idx)
            end
        end

        survival_results[k, j] = length(active_cells) / Ntot
        println("  [dose=$(dose) Gy] Survival: $(round(survival_results[k,j], digits=4))")
    end
end

#~ ============================================================
#~ Save results
#~ ============================================================
# Save survival matrix with labelled columns (one per dose rate)
col_names = [@sprintf("dr_%.0eGys", dr) for dr in doserates_to_run_Gys]
surv_df   = DataFrame(survival_results, col_names)
insertcols!(surv_df, 1, :dose_Gy => doses_to_run)
CSV.write(joinpath(outdir, "survival_results_12C_10MeV.csv"), surv_df)
println("\nSaved: $(joinpath(outdir, "survival_results_12C_10MeV.csv"))")

# Save metadata so the plot script knows the axes
meta_df = DataFrame(
    dose_rate_Gys = doserates_to_run_Gys,
    dose_rate_Gyh = doserates_to_run_Gyh)
CSV.write(joinpath(outdir, "survival_meta_12C_10MeV.csv"), meta_df)
println("Saved: $(joinpath(outdir, "survival_meta_12C_10MeV.csv"))")

# LQ fit
println("\nFitting LQ model — 12C 10 MeV:")
lq_params = fit_lq_survival(doses_to_run, surv_df, doserates_to_run_Gys; tag="12C_10MeV")
CSV.write(joinpath(outdir, "lq_params_12C_10MeV.csv"), lq_params)
println("Saved: lq_params_12C_10MeV.csv")

#~ ============================================================
#~ FINAL PRINT
#~ ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO $outdir/")
println("="^60)
for f in sort(readdir(outdir))
    println("  $outdir/$f")
end





#~ ============================================================
#~ Simulation parameters
#~ ============================================================
E            = 100.0
particle     = "12C"
dose         = 0.5
tumor_radius = 350.0
X_box        = 460.0
au           = 4.0

setup(E, particle, dose, tumor_radius; X_box = X_box)

cell_df_copy = deepcopy(cell_df)
cell_df.O   .= 21.0

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0

#~ ============================================================
#~ LUT precomputation
#~ ============================================================
println("Precomputing dose LUT...")
@time lut = MC_precompute_lut!(
    ion, Npar, R_beam, irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg;
    chunk_size = 50_000)

println("Precomputing damage LUT...")
@time damage_lut = precompute_damage_lut!(
    lut, cell_df_copy, irrad_cond, gsm2_cycle, ion;
    chunk_size = 50_000)

jldsave("lut_12C_4Gy_100MeV.jld2"; damage_lut)

Npar_effect = length(damage_lut)
println("Npar_effect = $Npar_effect")

#~ ============================================================
#~ Base active-cell template
#~ ============================================================
active_cells_base = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
for row in eachrow(cell_df_copy)
    row.is_cell != 1 && continue
    active_cells_base[row.index] = (zeros(Int, length(row.dam_X_dom)),
                                    zeros(Int, length(row.dam_Y_dom)))
end
Ntot = length(active_cells_base)
println("Total cells (Ntot) = $Ntot")

outdir = joinpath(@__DIR__, "..", "data", "continuous_dr")
mkpath(outdir)

#~ ============================================================
#~ Survival vs dose loop
#~ ============================================================
doses_to_run         = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5]
doserates_to_run_Gys = [1e-5, 1e-4, 1e-3, 1e-2]
doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

survival_results = zeros(length(doses_to_run), length(doserates_to_run_Gyh))
gsm2             = gsm2_cycle[1]

for (j, dose_rate_gyh) in enumerate(doserates_to_run_Gyh)
    println("\nRunning dose rate: $dose_rate_gyh Gy/h")
    dr = dose_rate_gyh / zF

    for (k, dose) in enumerate(doses_to_run)
        N_dose     = round(Int, dose * Npar_effect/0.5)
        times_full = rand(Exponential(1.0 / dr), N_dose)
        lut_order  = mod1.(randperm(N_dose), Npar_effect)

        times_filtered   = Float64[]
        lut_indices      = Int[]
        accumulated_time = 0.0

        for i in 1:N_dose
            lut_idx          = lut_order[i]
            accumulated_time += times_full[i]
            if !isempty(damage_lut[lut_idx])
                push!(times_filtered,   accumulated_time)
                push!(lut_indices,      lut_idx)
                accumulated_time = 0.0
            end
        end
        println("  [dose=$(dose) Gy] Effective particles: $(length(lut_indices)) / $N_dose")

        # Reset active cells from base template
        active_cells = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}(
            idx => (copy(X), copy(Y))
            for (idx, (X, Y)) in active_cells_base)

        # Main particle loop
        for i in 1:length(lut_indices)
            lut_idx = lut_indices[i]

            # 1. Apply damage
            @inbounds for (idx, (x, y)) in damage_lut[lut_idx]
                !haskey(active_cells, idx) && continue
                active_cells[idx][1] .+= x
                active_cells[idx][2] .+= y
            end

            # 2. Repair window: interval until next hit (Inf for last particle)
            t = i < length(lut_indices) ? times_filtered[i + 1] : Inf

            # 3. SSA repair
            to_delete = Int[]
            for (cell_idx, (X, Y)) in active_cells
                all(iszero, X) && all(iszero, Y) && continue
                death_time, _, _, X_new, Y_new =
                    compute_repair_domain(X, Y, gsm2;
                                          terminal_time = t, au = au)
                if isfinite(death_time)
                    push!(to_delete, cell_idx)
                else
                    active_cells[cell_idx] = (X_new, Y_new)
                end
            end

            for idx in to_delete
                delete!(active_cells, idx)
            end
        end

        survival_results[k, j] = length(active_cells) / Ntot
        println("  [dose=$(dose) Gy] Survival: $(round(survival_results[k,j], digits=4))")
    end
end

#~ ============================================================
#~ Save results
#~ ============================================================
# Save survival matrix with labelled columns (one per dose rate)
col_names = [@sprintf("dr_%.0eGys", dr) for dr in doserates_to_run_Gys]
surv_df   = DataFrame(survival_results, col_names)
insertcols!(surv_df, 1, :dose_Gy => doses_to_run)
CSV.write(joinpath(outdir, "survival_results_12C_100MeV.csv"), surv_df)
println("\nSaved: $(joinpath(outdir, "survival_results_12C_100MeV.csv"))")

meta_df = DataFrame(
    dose_rate_Gys = doserates_to_run_Gys,
    dose_rate_Gyh = doserates_to_run_Gyh)
CSV.write(joinpath(outdir, "survival_meta_12C_100MeV.csv"), meta_df)
println("Saved: $(joinpath(outdir, "survival_meta_12C_100MeV.csv"))")

println("\nFitting LQ model — 12C 100 MeV:")
lq_params = fit_lq_survival(doses_to_run, surv_df, doserates_to_run_Gys; tag="12C_100MeV")
CSV.write(joinpath(outdir, "lq_params_12C_100MeV.csv"), lq_params)
println("Saved: lq_params_12C_100MeV.csv")

#~ ============================================================
#~ FINAL PRINT
#~ ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO $outdir/")
println("="^60)
for f in sort(readdir(outdir))
    println("  $outdir/$f")
end






#~ ============================================================
#~ Simulation parameters
#~ ============================================================
E            = 100.0
particle     = "1H"
dose         = 0.5
tumor_radius = 150.0
X_box        = 260.0
au           = 4.0

setup(E, particle, dose, tumor_radius; X_box = X_box)

cell_df_copy = deepcopy(cell_df)
cell_df.O   .= 21.0

for i in 1:nrow(cell_df_copy)
    fill!(cell_df_copy.dam_X_dom[i], 0)
    fill!(cell_df_copy.dam_Y_dom[i], 0)
end
cell_df_copy.dam_X_total .= 0
cell_df_copy.dam_Y_total .= 0

#~ ============================================================
#~ LUT precomputation
#~ ============================================================
println("Precomputing dose LUT...")
@time lut = MC_precompute_lut!(
    ion, Npar, R_beam, irrad_cond, cell_df,
    df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg;
    chunk_size = 50_000)

println("Precomputing damage LUT...")
@time damage_lut = precompute_damage_lut!(
    lut, cell_df_copy, irrad_cond, gsm2_cycle, ion;
    chunk_size = 50_000)

jldsave("lut_1H_4Gy_100MeV.jld2"; damage_lut)

Npar_effect = length(damage_lut)
println("Npar_effect = $Npar_effect")

#~ ============================================================
#~ Base active-cell template
#~ ============================================================
active_cells_base = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
for row in eachrow(cell_df_copy)
    row.is_cell != 1 && continue
    active_cells_base[row.index] = (zeros(Int, length(row.dam_X_dom)),
                                    zeros(Int, length(row.dam_Y_dom)))
end
Ntot = length(active_cells_base)
println("Total cells (Ntot) = $Ntot")

outdir = joinpath(@__DIR__, "..", "data", "continuous_dr")
mkpath(outdir)

#~ ============================================================
#~ Survival vs dose loop
#~ ============================================================
doses_to_run         = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
doserates_to_run_Gys = [1e-5, 1e-4, 1e-3, 1e-2]
doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

survival_results = zeros(length(doses_to_run), length(doserates_to_run_Gyh))
gsm2             = gsm2_cycle[1]

for (j, dose_rate_gyh) in enumerate(doserates_to_run_Gyh)
    println("\nRunning dose rate: $dose_rate_gyh Gy/h")
    dr = dose_rate_gyh / zF

    for (k, dose) in enumerate(doses_to_run)
        N_dose     = round(Int, dose * Npar_effect/0.5)
        times_full = rand(Exponential(1.0 / dr), N_dose)
        lut_order  = mod1.(randperm(N_dose), Npar_effect)

        times_filtered   = Float64[]
        lut_indices      = Int[]
        accumulated_time = 0.0

        for i in 1:N_dose
            lut_idx          = lut_order[i]
            accumulated_time += times_full[i]
            if !isempty(damage_lut[lut_idx])
                push!(times_filtered,   accumulated_time)
                push!(lut_indices,      lut_idx)
                accumulated_time = 0.0
            end
        end
        println("  [dose=$(dose) Gy] Effective particles: $(length(lut_indices)) / $N_dose")

        # Reset active cells from base template
        active_cells = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}(
            idx => (copy(X), copy(Y))
            for (idx, (X, Y)) in active_cells_base)

        # Main particle loop
        for i in 1:length(lut_indices)
            lut_idx = lut_indices[i]

            # 1. Apply damage
            @inbounds for (idx, (x, y)) in damage_lut[lut_idx]
                !haskey(active_cells, idx) && continue
                active_cells[idx][1] .+= x
                active_cells[idx][2] .+= y
            end

            # 2. Repair window: interval until next hit (Inf for last particle)
            t = i < length(lut_indices) ? times_filtered[i + 1] : Inf

            # 3. SSA repair
            to_delete = Int[]
            for (cell_idx, (X, Y)) in active_cells
                all(iszero, X) && all(iszero, Y) && continue
                death_time, _, _, X_new, Y_new =
                    compute_repair_domain(X, Y, gsm2;
                                          terminal_time = t, au = au)
                if isfinite(death_time)
                    push!(to_delete, cell_idx)
                else
                    active_cells[cell_idx] = (X_new, Y_new)
                end
            end

            for idx in to_delete
                delete!(active_cells, idx)
            end
        end

        survival_results[k, j] = length(active_cells) / Ntot
        println("  [dose=$(dose) Gy] Survival: $(round(survival_results[k,j], digits=4))")
    end
end

#~ ============================================================
#~ Save results
#~ ============================================================
# Save survival matrix with labelled columns (one per dose rate)
col_names = [@sprintf("dr_%.0eGys", dr) for dr in doserates_to_run_Gys]
surv_df   = DataFrame(survival_results, col_names)
insertcols!(surv_df, 1, :dose_Gy => doses_to_run)
CSV.write(joinpath(outdir, "survival_results_1H_100MeV.csv"), surv_df)
println("\nSaved: $(joinpath(outdir, "survival_results_1H_100MeV.csv"))")

meta_df = DataFrame(
    dose_rate_Gys = doserates_to_run_Gys,
    dose_rate_Gyh = doserates_to_run_Gyh)
CSV.write(joinpath(outdir, "survival_meta_1H_100MeV.csv"), meta_df)
println("Saved: $(joinpath(outdir, "survival_meta_1H_100MeV.csv"))")

println("\nFitting LQ model — 1H 100 MeV:")
lq_params = fit_lq_survival(doses_to_run, surv_df, doserates_to_run_Gys; tag="1H_100MeV")
CSV.write(joinpath(outdir, "lq_params_1H_100MeV.csv"), lq_params)
println("Saved: lq_params_1H_100MeV.csv")

#~ ============================================================
#~ FINAL PRINT
#~ ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO $outdir/")
println("="^60)
for f in sort(readdir(outdir))
    println("  $outdir/$f")
end


