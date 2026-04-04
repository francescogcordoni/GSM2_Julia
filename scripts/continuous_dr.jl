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

# ============================================================
# Load functions
# ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

sp = load_stopping_power()

# ============================================================
# GSM2 parameters
# ============================================================
a_G1 = 0.012872261720543399;  b_G1 = 0.04029756109753225;  r_G1 = 2.780479661191086
a_S  = 0.00589118894714544;   b_S  = 0.05794352736120672;  r_S  = 5.84009601901114
a_G2 = 0.024306291709970018;  b_G2 = 5.704688326522623e-5; r_G2 = 1.7720064637774506
a    = 0.01481379648786136;   b    = 0.012663276476522422; r    = 2.5656972960759896
rd   = 0.8;                   Rn   = 7.2

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)

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

# ============================================================
# Fast survival loop
# Speedups vs original:
#   1. Flat arrays instead of Dict for active cells  →  no hash overhead
#   2. Threads.@threads over (dose, dose_rate) pairs →  fully independent,
#      each thread has its own RNG and local cell arrays
#   3. Dead-cell bitmask replaces Dict deletion
# ============================================================
function run_survival_fast(
        damage_lut,
        base_X::Vector{Vector{Int}},   # pre-zeroed, length = Ntot
        base_Y::Vector{Vector{Int}},
        cell_id_to_pos::Dict{Int,Int},  # cell_df index → 1:Ntot position
        doses_to_run,
        doserates_to_run_Gys,
        doserates_to_run_Gyh,
        gsm2, zF, Npar_effect, dose_ref, au;
        tag = "")

    Ntot   = length(base_X)
    ndoses = length(doses_to_run)
    ndrs   = length(doserates_to_run_Gyh)

    survival_results = zeros(ndoses, ndrs)

    # Flatten (k, j) into a single index so @threads can distribute evenly
    combos = [(k, j) for j in 1:ndrs for k in 1:ndoses]

    Threads.@threads for ci in eachindex(combos)
        k, j    = combos[ci]
        dose    = doses_to_run[k]
        dr_gyh  = doserates_to_run_Gyh[j]
        dr_gys  = doserates_to_run_Gys[j]
        dr      = dr_gyh / zF           # convert to natural time units

        rng     = Random.default_rng()  # each thread has its own RNG in Julia ≥ 1.7

        N_dose     = round(Int, dose * Npar_effect / dose_ref)
        times_full = rand(rng, Exponential(1.0 / dr), N_dose)
        lut_order  = mod1.(randperm(rng, N_dose), Npar_effect)

        # Filter to particles that actually deposit damage
        times_filtered   = Float64[]
        lut_indices      = Int[]
        acc              = 0.0
        for i in 1:N_dose
            acc += times_full[i]
            if !isempty(damage_lut[lut_order[i]])
                push!(times_filtered, acc)
                push!(lut_indices,    lut_order[i])
                acc = 0.0
            end
        end

        # Thread-local cell state (copy of zeroed base)
        local_X  = [copy(v) for v in base_X]
        local_Y  = [copy(v) for v in base_Y]
        is_alive = fill(true, Ntot)

        for i in eachindex(lut_indices)
            lut_idx = lut_indices[i]

            # 1. Apply damage from this particle to hit cells
            @inbounds for (cell_idx, (x, y)) in damage_lut[lut_idx]
                pos = get(cell_id_to_pos, cell_idx, 0)
                (pos == 0 || !is_alive[pos]) && continue
                local_X[pos] .+= x
                local_Y[pos] .+= y
            end

            # 2. Repair window until next particle (Inf for last)
            t = i < length(lut_indices) ? times_filtered[i + 1] : Inf

            # 3. SSA repair — sequential over cells
            #    (outer @threads already parallelises across (dose, dr) combos;
            #     adding nested @threads here would over-subscribe the thread pool)
            @inbounds for ci2 in 1:Ntot
                !is_alive[ci2] && continue
                all(iszero, local_X[ci2]) && all(iszero, local_Y[ci2]) && continue
                death_time, _, _, X_new, Y_new =
                    compute_repair_domain(local_X[ci2], local_Y[ci2], gsm2;
                                          terminal_time = t, au = au)
                if isfinite(death_time)
                    is_alive[ci2] = false
                else
                    local_X[ci2] = X_new
                    local_Y[ci2] = Y_new
                end
            end
        end

        sf = sum(is_alive) / Ntot
        survival_results[k, j] = sf
        println(@sprintf("  [%s | dose=%.2f Gy | dr=%.0e Gy/s] SF=%.4f  (eff.par=%d/%d)",
                         tag, dose, dr_gys, sf, length(lut_indices), N_dose))
    end

    return survival_results
end

# ============================================================
# Shared output dir
# ============================================================
outdir = joinpath(@__DIR__, "..", "data", "continuous_dr")
mkpath(outdir)

# ============================================================
# ── CONDITION 1: 12C 10 MeV/u ────────────────────────────
# ============================================================
E            = 10.0
particle     = "12C"
dose_ref     = 4.0
tumor_radius = 450.0
X_box        = 560.0
au           = 4.0
tag          = "12C_10MeV"
setup(E, particle, dose_ref, tumor_radius; X_box = X_box)

let
    E            = 10.0
    particle     = "12C"
    dose_ref     = 4.0
    tumor_radius = 450.0
    X_box        = 560.0
    au           = 4.0
    tag          = "12C_10MeV"

    setup(E, particle, dose_ref, tumor_radius; X_box = X_box)

    cell_df_copy = deepcopy(cell_df)
    cell_df.O   .= 21.0
    for i in 1:nrow(cell_df_copy)
        fill!(cell_df_copy.dam_X_dom[i], 0)
        fill!(cell_df_copy.dam_Y_dom[i], 0)
    end
    cell_df_copy.dam_X_total .= 0
    cell_df_copy.dam_Y_total .= 0

    println("\n", "="^60)
    println("CONDITION: $tag")
    println("="^60)

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

    jldsave(joinpath(outdir, "lut_$(tag).jld2"); damage_lut)

    Npar_effect = length(damage_lut)
    println("Npar_effect = $Npar_effect")

    # Build flat-array base template and index map
    cell_ids = [row.index for row in eachrow(cell_df_copy) if row.is_cell == 1]
    Ntot     = length(cell_ids)
    println("Total cells (Ntot) = $Ntot")

    cell_id_to_pos = Dict(idx => ci for (ci, idx) in enumerate(cell_ids))
    base_X = [zeros(Int, length(cell_df_copy.dam_X_dom[ci])) for ci in
              [findfirst(==(id), cell_df_copy.index) for id in cell_ids]]
    base_Y = [zeros(Int, length(cell_df_copy.dam_Y_dom[ci])) for ci in
              [findfirst(==(id), cell_df_copy.index) for id in cell_ids]]

    doses_to_run         = [0.1, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0]
    doserates_to_run_Gys = [1e-5, 5e-5, 1e-4, 1e-3, 1e-2]
    doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

    println("\nRunning survival loop ($(nthreads()) threads, $(length(doses_to_run)*length(doserates_to_run_Gys)) combos)...")
    @time survival_results = run_survival_fast(
        damage_lut, base_X, base_Y, cell_id_to_pos,
        doses_to_run, doserates_to_run_Gys, doserates_to_run_Gyh,
        gsm2_cycle[1], zF, Npar_effect, dose_ref, au; tag = tag)

    col_names = [@sprintf("dr_%.0eGys", dr) for dr in doserates_to_run_Gys]
    surv_df   = DataFrame(survival_results, col_names)
    insertcols!(surv_df, 1, :dose_Gy => doses_to_run)
    CSV.write(joinpath(outdir, "survival_results_$(tag).csv"), surv_df)
    println("Saved: survival_results_$(tag).csv")

    meta_df = DataFrame(dose_rate_Gys = doserates_to_run_Gys,
                        dose_rate_Gyh = doserates_to_run_Gyh)
    CSV.write(joinpath(outdir, "survival_meta_$(tag).csv"), meta_df)
    println("Saved: survival_meta_$(tag).csv")

    println("\nFitting LQ model — $tag:")
    lq_params = fit_lq_survival(doses_to_run, surv_df, doserates_to_run_Gys; tag = tag)
    CSV.write(joinpath(outdir, "lq_params_$(tag).csv"), lq_params)
    println("Saved: lq_params_$(tag).csv")
end

# ============================================================
# ── CONDITION 2: 12C 100 MeV/u ───────────────────────────
# ============================================================
E            = 80.0
particle     = "12C"
dose_ref     = 0.5
tumor_radius = 200.0
X_box        = 250.0
au           = 4.0
tag          = "12C_100MeV"
setup(E, particle, dose_ref, tumor_radius; X_box = X_box)
let
    E            = 80.0
    particle     = "12C"
    dose_ref     = 0.5
    tumor_radius = 200.0
    X_box        = 250.0
    au           = 4.0
    tag          = "12C_100MeV"

    setup(E, particle, dose_ref, tumor_radius; X_box = X_box)

    cell_df_copy = deepcopy(cell_df)
    cell_df.O   .= 21.0
    for i in 1:nrow(cell_df_copy)
        fill!(cell_df_copy.dam_X_dom[i], 0)
        fill!(cell_df_copy.dam_Y_dom[i], 0)
    end
    cell_df_copy.dam_X_total .= 0
    cell_df_copy.dam_Y_total .= 0

    println("\n", "="^60)
    println("CONDITION: $tag")
    println("="^60)

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

    jldsave(joinpath(outdir, "lut_$(tag).jld2"); damage_lut)

    Npar_effect = length(damage_lut)
    println("Npar_effect = $Npar_effect")

    cell_ids = [row.index for row in eachrow(cell_df_copy) if row.is_cell == 1]
    Ntot     = length(cell_ids)
    println("Total cells (Ntot) = $Ntot")

    cell_id_to_pos = Dict(idx => ci for (ci, idx) in enumerate(cell_ids))
    base_X = [zeros(Int, length(cell_df_copy.dam_X_dom[ci])) for ci in
              [findfirst(==(id), cell_df_copy.index) for id in cell_ids]]
    base_Y = [zeros(Int, length(cell_df_copy.dam_Y_dom[ci])) for ci in
              [findfirst(==(id), cell_df_copy.index) for id in cell_ids]]

    doses_to_run         = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    doserates_to_run_Gys = [1e-5, 5e-5, 1e-4, 1e-3, 1e-2]
    doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

    println("\nRunning survival loop ($(nthreads()) threads, $(length(doses_to_run)*length(doserates_to_run_Gys)) combos)...")
    @time survival_results = run_survival_fast(
        damage_lut, base_X, base_Y, cell_id_to_pos,
        doses_to_run, doserates_to_run_Gys, doserates_to_run_Gyh,
        gsm2_cycle[1], zF, Npar_effect, dose_ref, au; tag = tag)

    col_names = [@sprintf("dr_%.0eGys", dr) for dr in doserates_to_run_Gys]
    surv_df   = DataFrame(survival_results, col_names)
    insertcols!(surv_df, 1, :dose_Gy => doses_to_run)
    CSV.write(joinpath(outdir, "survival_results_$(tag).csv"), surv_df)
    println("Saved: survival_results_$(tag).csv")

    meta_df = DataFrame(dose_rate_Gys = doserates_to_run_Gys,
                        dose_rate_Gyh = doserates_to_run_Gyh)
    CSV.write(joinpath(outdir, "survival_meta_$(tag).csv"), meta_df)
    println("Saved: survival_meta_$(tag).csv")

    println("\nFitting LQ model — $tag:")
    lq_params = fit_lq_survival(doses_to_run, surv_df, doserates_to_run_Gys; tag = tag)
    CSV.write(joinpath(outdir, "lq_params_$(tag).csv"), lq_params)
    println("Saved: lq_params_$(tag).csv")
end

# ============================================================
# ── CONDITION 3: 1H 100 MeV ──────────────────────────────
# ============================================================
E            = 100.0
particle     = "1H"
dose_ref     = 0.5
tumor_radius = 200.0
X_box        = 250.0
au           = 4.0
tag          = "1H_100MeV"
setup(E, particle, dose_ref, tumor_radius; X_box = X_box)

let
    E            = 100.0
    particle     = "1H"
    dose_ref     = 0.5
    tumor_radius = 200.0
    X_box        = 250.0
    au           = 4.0
    tag          = "1H_100MeV"

    setup(E, particle, dose_ref, tumor_radius; X_box = X_box)

    cell_df_copy = deepcopy(cell_df)
    cell_df.O   .= 21.0
    for i in 1:nrow(cell_df_copy)
        fill!(cell_df_copy.dam_X_dom[i], 0)
        fill!(cell_df_copy.dam_Y_dom[i], 0)
    end
    cell_df_copy.dam_X_total .= 0
    cell_df_copy.dam_Y_total .= 0

    println("\n", "="^60)
    println("CONDITION: $tag")
    println("="^60)

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

    jldsave(joinpath(outdir, "lut_$(tag).jld2"); damage_lut)

    Npar_effect = length(damage_lut)
    println("Npar_effect = $Npar_effect")

    cell_ids = [row.index for row in eachrow(cell_df_copy) if row.is_cell == 1]
    Ntot     = length(cell_ids)
    println("Total cells (Ntot) = $Ntot")

    cell_id_to_pos = Dict(idx => ci for (ci, idx) in enumerate(cell_ids))
    base_X = [zeros(Int, length(cell_df_copy.dam_X_dom[ci])) for ci in
              [findfirst(==(id), cell_df_copy.index) for id in cell_ids]]
    base_Y = [zeros(Int, length(cell_df_copy.dam_Y_dom[ci])) for ci in
              [findfirst(==(id), cell_df_copy.index) for id in cell_ids]]

    doses_to_run         = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    doserates_to_run_Gys = [1e-5, 5e-5, 1e-4, 1e-3, 1e-2]
    doserates_to_run_Gyh = doserates_to_run_Gys .* 3600.0 .* au

    println("\nRunning survival loop ($(nthreads()) threads, $(length(doses_to_run)*length(doserates_to_run_Gys)) combos)...")
    @time survival_results = run_survival_fast(
        damage_lut, base_X, base_Y, cell_id_to_pos,
        doses_to_run, doserates_to_run_Gys, doserates_to_run_Gyh,
        gsm2_cycle[1], zF, Npar_effect, dose_ref, au; tag = tag)

    col_names = [@sprintf("dr_%.0eGys", dr) for dr in doserates_to_run_Gys]
    surv_df   = DataFrame(survival_results, col_names)
    insertcols!(surv_df, 1, :dose_Gy => doses_to_run)
    CSV.write(joinpath(outdir, "survival_results_$(tag).csv"), surv_df)
    println("Saved: survival_results_$(tag).csv")

    meta_df = DataFrame(dose_rate_Gys = doserates_to_run_Gys,
                        dose_rate_Gyh = doserates_to_run_Gyh)
    CSV.write(joinpath(outdir, "survival_meta_$(tag).csv"), meta_df)
    println("Saved: survival_meta_$(tag).csv")

    println("\nFitting LQ model — $tag:")
    lq_params = fit_lq_survival(doses_to_run, surv_df, doserates_to_run_Gys; tag = tag)
    CSV.write(joinpath(outdir, "lq_params_$(tag).csv"), lq_params)
    println("Saved: lq_params_$(tag).csv")
end

# ============================================================
# Final summary
# ============================================================
println("\n", "="^60)
println("ALL RESULTS SAVED TO $outdir/")
println("="^60)
for f in sort(readdir(outdir))
    println("  $outdir/$f")
end
