#! ============================================================================
#! utilities_MC_gpu.jl  (patched — track_seg passed to MC_loop_ions_domain_tsc_matrix!)
#!
#! FIX: CPU TSC branch of MC_dose_fast! was missing `track_seg` as the 12th
#!      argument to MC_loop_ions_domain_tsc_matrix!.
#!      Closest candidate signature:
#!        MC_loop_ions_domain_tsc_matrix!(::Int64, ::Float64, ::Float64,
#!          ::Vector{AT}, ::GSM2, ::Matrix{Float64}, ::Matrix{Float64},
#!          ::Matrix{Float64}, ::Float64, ::String, ::Ion, ::Bool)
#!      The Bool is track_seg.
#! ============================================================================

const BLOCK_SIZE             = 256
const GPU_PARTICLE_THRESHOLD = 1_000_000

"""
    _log_index(x, log_vmin, log_vmax, n) -> Int32

Log-spaced bin index for `x` in `[10^log_vmin, 10^log_vmax]` divided into `n` bins.
Clamped to `[1, n]`. Callable from GPU device code.
"""
@inline function _log_index(x        :: Float64,
                             log_vmin :: Float64,
                             log_vmax :: Float64,
                             n        :: Int32) :: Int32
    t   = (log10(x) - log_vmin) / (log_vmax - log_vmin)
    idx = Int32(1) + Int32(floor(t * Float64(n - Int32(1))))
    return clamp(idx, Int32(1), n)
end

"""
    _mc_dose_kernel_fast!(out, dom_x, dom_y, px, py, Np, dose_vec, sim_,
                          log_impact_min, log_impact_max,
                          core_radius_sq, mid_radius_sq, penumbra_radius_sq,
                          core_dose, Kp)

CUDA kernel — one thread per domain point. Accumulates dose from Np primaries.
"""
function _mc_dose_kernel_fast!(
    out                :: CuDeviceVector{Float64},
    dom_x              :: CuDeviceVector{Float64},
    dom_y              :: CuDeviceVector{Float64},
    px                 :: CuDeviceVector{Float64},
    py                 :: CuDeviceVector{Float64},
    Np                 :: Int32,
    dose_vec           :: CuDeviceVector{Float64},
    sim_               :: Int32,
    log_impact_min     :: Float64,
    log_impact_max     :: Float64,
    core_radius_sq     :: Float64,
    mid_radius_sq      :: Float64,
    penumbra_radius_sq :: Float64,
    core_dose          :: Float64,
    Kp                 :: Float64
)
    k = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    k > length(dom_x) && return nothing

    kx = dom_x[k]
    ky = dom_y[k]
    local_acc = 0.0

    @inbounds for ip in Int32(1):Np
        dx      = kx - px[ip]
        dy      = ky - py[ip]
        dist_sq = dx*dx + dy*dy

        if dist_sq <= core_radius_sq
            local_acc += core_dose

        elseif dist_sq <= mid_radius_sq
            dist  = sqrt(dist_sq)
            idx_l = _log_index(dist, log_impact_min, log_impact_max, sim_)

            if idx_l >= sim_
                local_acc += dose_vec[sim_]
            else
                step = (log_impact_max - log_impact_min) / Float64(sim_ - Int32(1))
                x1   = exp10(log_impact_min + Float64(idx_l - Int32(1)) * step)
                x2   = exp10(log_impact_min + Float64(idx_l)            * step)
                y1   = dose_vec[idx_l]
                y2   = dose_vec[idx_l + Int32(1)]
                local_acc += y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
            end

        elseif dist_sq < penumbra_radius_sq
            local_acc += Kp / dist_sq
        end
    end

    out[k] = local_acc
    return nothing
end

"""
    _build_lookup(irrad_cond, gsm2, type_AT)
        -> (dose_vec, core_dose, core_radius_sq, mid_radius_sq,
            penumbra_radius_sq, Kp, log_impact_min, log_impact_max)

Builds a CPU-side log-spaced dose lookup table (1000 samples).
"""
function _build_lookup(irrad_cond, gsm2, type_AT)
    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp

    lower_bound_log    = max(1e-9, gsm2.rd - 10Rc)
    core_radius_sq     = (gsm2.rd - 10Rc)^2
    mid_radius_sq      = (gsm2.rd + 150Rc)^2
    penumbra_radius_sq = Rp^2

    sim_     = 1000
    impact_p = 10 .^ range(log10(lower_bound_log),
                            stop   = log10(gsm2.rd + 150Rc),
                            length = sim_)

    lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]
    Threads.@threads for i in 1:sim_
        tid   = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _d, _r, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd,
                                              track, irrad_cond[1], type_AT)
        lookup_threads[tid][i] = Gyr
    end

    dose_vec  = sum(lookup_threads)
    core_dose = dose_vec[1]

    return (dose_vec, core_dose,
            core_radius_sq, mid_radius_sq, penumbra_radius_sq, Kp,
            log10(impact_p[1]), log10(impact_p[end]))
end

"""
    _gpu_run_single!(cu_out, cu_dom_x, cu_dom_y, x_list, y_list, cu_dose,
                     core_radius_sq, mid_radius_sq, penumbra_radius_sq,
                     core_dose, Kp, log_impact_min, log_impact_max) -> Nothing

Uploads particle coordinates to GPU, launches kernel, synchronizes, frees temps.
"""
function _gpu_run_single!(
    cu_out             :: CuVector{Float64},
    cu_dom_x           :: CuVector{Float64},
    cu_dom_y           :: CuVector{Float64},
    x_list             :: Vector{Float64},
    y_list             :: Vector{Float64},
    cu_dose            :: CuVector{Float64},
    core_radius_sq     :: Float64,
    mid_radius_sq      :: Float64,
    penumbra_radius_sq :: Float64,
    core_dose          :: Float64,
    Kp                 :: Float64,
    log_impact_min     :: Float64,
    log_impact_max     :: Float64
)
    Np            = Int32(length(x_list))
    sim_          = Int32(length(cu_dose))
    total_domains = length(cu_dom_x)
    nblocks       = cld(total_domains, BLOCK_SIZE)

    println("   Upload $Np particles | $nblocks blocks × $BLOCK_SIZE threads | $total_domains domains")

    cu_px = CuArray(x_list)
    cu_py = CuArray(y_list)
    fill!(cu_out, 0.0)

    CUDA.@cuda(
        threads = BLOCK_SIZE,
        blocks  = nblocks,
        _mc_dose_kernel_fast!(
            cu_out, cu_dom_x, cu_dom_y,
            cu_px, cu_py, Np,
            cu_dose, sim_,
            log_impact_min, log_impact_max,
            core_radius_sq, mid_radius_sq, penumbra_radius_sq,
            core_dose, Kp
        )
    )

    CUDA.synchronize()
    CUDA.unsafe_free!(cu_px)
    CUDA.unsafe_free!(cu_py)
    println("   Kernel finished.")
end

"""
    MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy,
                  df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg;
                  x_cb, y_cb)

Auto CPU/GPU dispatch wrapper for dose computation.

FIX applied: CPU TSC branch now passes `track_seg` as the required 12th
argument to `MC_loop_ions_domain_tsc_matrix!`.
"""
function MC_dose_fast!(
    ion          :: Ion,
    Npar         :: Int64,
    R_beam       :: Float64,
    irrad_cond   :: Vector{AT},
    cell_df_copy :: DataFrame,
    df_center_x  :: DataFrame,
    df_center_y  :: DataFrame,
    at           :: DataFrame,
    gsm2_cycle   :: Vector{GSM2},
    type_AT      :: String,
    track_seg    :: Bool;
    x_cb         :: Float64 = 0.,
    y_cb         :: Float64 = 0.
)
    println("\n───────────────────────────────────────────────")
    println("🔧  MC_dose_fast!  (auto CPU/GPU dispatch)")
    cuda_ok = CUDA.functional()
    use_gpu = cuda_ok && (Npar >= GPU_PARTICLE_THRESHOLD)
    println("    CUDA available   : $cuda_ok")
    println("    Npar             : $Npar")
    println("    GPU threshold    : $GPU_PARTICLE_THRESHOLD")
    println("    Backend          : $(use_gpu ? "GPU 🚀" : "CPU 🖥")")
    println("    track_seg        : $track_seg")
    println("───────────────────────────────────────────────")

    t_start = time()
    gsm2    = gsm2_cycle[1]

    function _get_representatives(cell_df, energy_step_filter=nothing)
        cell_df_is = if energy_step_filter === nothing
            filter(row -> row.is_cell == 1, cell_df)
        else
            filter(row -> row.is_cell == 1 && row.energy_step == energy_step_filter, cell_df)
        end
        nrow(cell_df_is) == 0 && return nothing, nothing, nothing, nothing

        grouped_df      = combine(groupby(cell_df_is, [:x, :y]),
                                  :index => first => :representative_index)
        rep_indices_set = Set(grouped_df.representative_index)

        sx = filter(row -> row.index in rep_indices_set, df_center_x)
        sy = filter(row -> row.index in rep_indices_set, df_center_y)
        sa = filter(row -> row.index in rep_indices_set, at)
        return rep_indices_set, sx, sy, sa
    end

    function _sample_hits(Np)
        x_list = Vector{Float64}(undef, Np)
        y_list = Vector{Float64}(undef, Np)
        Threads.@threads for ip in 1:Np
            x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
        end
        return x_list, y_list
    end

    # ─────────────────────────────────────────────────────────────────────────
    # TRACK-SEGMENT branch
    # ─────────────────────────────────────────────────────────────────────────
    if track_seg

        if use_gpu
            println("• [GPU] TSC mode")

            # Pass df_center_x / df_center_y / at in full — no index filtering.
            # Filtering by representative indices from cell_df_copy causes
            # empty matrices (total_domains = 0) when cell_df_copy is a
            # deepcopy whose .index values differ from the pre-built at DataFrame.
            # The original working top-level scripts always pass full DataFrames.
            dose_vec, core_dose, cr_sq, mr_sq, pr_sq, Kp, lmin, lmax =
                _build_lookup([irrad_cond[1]], gsm2, type_AT)

            mat_x, mat_y, mat_at = dataframes_to_matrices(df_center_x,
                                                           df_center_y, at)
            num_cells, num_dom   = size(mat_x)
            total_domains        = num_cells * num_dom

            println("   Domains: $num_cells cells × $num_dom domains = $total_domains")
            if total_domains == 0
                @warn "dataframes_to_matrices returned empty matrices — check df_center_x/at."
                return
            end

            cu_dom_x = CuArray(vec(mat_x'))
            cu_dom_y = CuArray(vec(mat_y'))
            cu_dose  = CuArray(dose_vec)
            cu_out   = CUDA.zeros(Float64, total_domains)

            Np = rand(Poisson(Npar))
            println("   Sampling $Np particles (Poisson($Npar))...")
            x_list, y_list = _sample_hits(Np)

            _gpu_run_single!(cu_out, cu_dom_x, cu_dom_y, x_list, y_list, cu_dose,
                             cr_sq, mr_sq, pr_sq, core_dose, Kp, lmin, lmax)

            mat_at .= reshape(Array(cu_out), num_dom, num_cells)'
            matrix_to_dataframe!(at, mat_at)

        else
            println("• [CPU] TSC mode")
            cell_df_is = filter(row -> row.is_cell == 1, cell_df_copy)
            nrow(cell_df_is) == 0 && (@warn "No cells with is_cell = 1 → skipping."; return)

            grouped_df      = combine(groupby(cell_df_is, [:x, :y]),
                                      :index => first => :representative_index)
            rep_indices_set = Set(grouped_df.representative_index)
            println("  → Found $(length(rep_indices_set)) representative cells")

            cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
            cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
            at_single        = filter(row -> row.index in rep_indices_set, at)

            mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x,
                                                           cell_df_single_y,
                                                           at_single)

            # ── FIX: pass track_seg as the required 12th Bool argument ────────
            MC_loop_ions_domain_tsc_matrix!(
                Npar, x_cb, y_cb, [irrad_cond[1]], gsm2,
                mat_x, mat_y, mat_at, R_beam, type_AT, ion, track_seg
            )
            # ─────────────────────────────────────────────────────────────────

            matrix_to_dataframe!(at_single, mat_at)

            # CPU path done — fall through to the unified copy-back below
        end

        # Unified copy-back for both paths:
        # GPU path: `at` was updated in-place by matrix_to_dataframe!
        # CPU path: `at_single` results were written back into `at` rows by
        #           matrix_to_dataframe!(at_single, ...) — at_single is a
        #           filtered view sharing the same underlying row objects,
        #           so `at` already contains the updated values.
        MC_loop_copy_dose_domain_fast!(cell_df_copy, at, at)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYERED (non-TSC) branch
    # ─────────────────────────────────────────────────────────────────────────
    else
        if use_gpu
            Np = rand(Poisson(Npar))
            println("• Sampling $Np particles...")
            x_list, y_list = _sample_hits(Np)

            for id in unique(cell_df_copy.energy_step)
                println("  → Layer $id")

                _, sx, sy, at_single = _get_representatives(cell_df_copy, id)
                sx === nothing && (println("    (empty)"); continue)

                println("• [GPU] Layered mode")

                dose_vec, core_dose, cr_sq, mr_sq, pr_sq, Kp, lmin, lmax =
                    _build_lookup([irrad_cond[id]], gsm2, type_AT)

                mat_x, mat_y, mat_at = dataframes_to_matrices(sx, sy, at_single)
                num_cells, num_dom   = size(mat_x)
                total_domains        = num_cells * num_dom

                cu_dom_x = CuArray(vec(mat_x'))
                cu_dom_y = CuArray(vec(mat_y'))
                cu_dose  = CuArray(dose_vec)
                cu_out   = CUDA.zeros(Float64, total_domains)

                _gpu_run_single!(cu_out, cu_dom_x, cu_dom_y, x_list, y_list, cu_dose,
                                 cr_sq, mr_sq, pr_sq, core_dose, Kp, lmin, lmax)

                mat_at .= reshape(Array(cu_out), num_dom, num_cells)'
                matrix_to_dataframe!(at_single, mat_at)
                MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, at_single, at, id)
            end

        else
            println("• [CPU] non-TSC mode")

            Np = rand(Poisson(Npar))
            println("• Sampling $Np particle hits...")

            x_list = Vector{Float64}(undef, Np)
            y_list = Vector{Float64}(undef, Np)
            Threads.@threads for ip in 1:Np
                x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
            end

            for id in unique(cell_df_copy.energy_step)
                println("  → Layer $id")
                cell_df_is = filter(row -> (row.is_cell == 1) &&
                                           (row.energy_step == id), cell_df_copy)
                if nrow(cell_df_is) == 0
                    println("    (empty → skip)")
                    continue
                end

                grouped_df      = combine(groupby(cell_df_is, [:x, :y]),
                                          :index => first => :representative_index)
                rep_indices_set = Set(grouped_df.representative_index)

                cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
                cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
                at_single        = filter(row -> row.index in rep_indices_set, at)

                mat_x, mat_y, mat_at = dataframes_to_matrices(cell_df_single_x,
                                                               cell_df_single_y,
                                                               at_single)

                MC_loop_ions_domain_matrix!(
                    x_list, y_list, [irrad_cond[id]], gsm2,
                    mat_x, mat_y, mat_at, type_AT, ion
                )

                matrix_to_dataframe!(at_single, mat_at)
                MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, at_single, at, id)
            end
        end
    end

    dt = round(time() - t_start; digits=3)
    println("───────────────────────────────────────────────")
    println("🎉  MC_dose_fast! finished in $(dt)s")
    println("───────────────────────────────────────────────\n")
end
