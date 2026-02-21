"""
mc_gpu.jl — Monte Carlo kernels with automatic CPU/GPU dispatch

Dispatch logic
--------------
    use_gpu = CUDA.functional() && (Npar >= GPU_PARTICLE_THRESHOLD)

    CPU path (Npar < threshold OR no CUDA):
    → Calls your original DataFrame-based functions directly:
        MC_loop_ions_domain_tsc_fast!   (track_seg = true)
        MC_loop_ions_domain_fast!       (track_seg = false)
        Zero DataFrame↔matrix conversion overhead. Identical performance
        to your original code.

    GPU path (Npar >= threshold AND CUDA available):
    → Converts DataFrames to matrices, runs the CUDA kernel, converts back.
    → Single kernel launch, no atomics, O(1) log-spaced lookup index.

Tunable constants
-----------------
    GPU_PARTICLE_THRESHOLD  default 1_000_000
    BLOCK_SIZE              default 256

To override at runtime (without editing this file):
    import Main: GPU_PARTICLE_THRESHOLD
    GPU_PARTICLE_THRESHOLD = 500_000    # lower threshold
    GPU_PARTICLE_THRESHOLD = 0          # always GPU (if available)
    GPU_PARTICLE_THRESHOLD = typemax(Int)  # always CPU
"""

using CUDA
using Distributions
using Statistics: mean

const BLOCK_SIZE             = 256
const GPU_PARTICLE_THRESHOLD = 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# O(1) log-spaced index  (replaces binary searchsortedfirst, no warp divergence)
# ─────────────────────────────────────────────────────────────────────────────

@inline function _log_index(x          :: Float64,
                                log_vmin   :: Float64,
                                log_vmax   :: Float64,
                                n          :: Int32) :: Int32
    t   = (log10(x) - log_vmin) / (log_vmax - log_vmin)
    idx = Int32(1) + Int32(floor(t * Float64(n - Int32(1))))
    return clamp(idx, Int32(1), n)
end


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernel — one thread per domain, no atomics, single launch
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# CPU helper: build lookup table (used only on the GPU path)
# ─────────────────────────────────────────────────────────────────────────────

function _build_lookup(irrad_cond, gsm2, type_AT)
    Rp  = irrad_cond[1].Rp
    Rc  = irrad_cond[1].Rc
    Kp  = irrad_cond[1].Kp
    Rk  = Rp

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


# ─────────────────────────────────────────────────────────────────────────────
# GPU dispatch: single upload + single kernel launch
# ─────────────────────────────────────────────────────────────────────────────

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

    println("   Upload $(Np) particles | $(nblocks) blocks × $(BLOCK_SIZE) threads | $(total_domains) domains")

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


# ─────────────────────────────────────────────────────────────────────────────
# MC_dose_fast! — top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df_copy,
                    df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg;
                    x_cb=0., y_cb=0.)

Automatic CPU/GPU dispatch:

    Npar >= GPU_PARTICLE_THRESHOLD AND CUDA available
    → GPU path: DataFrames converted to matrices, single CUDA kernel launch,
        no atomics, O(1) log-spaced index.

    Npar <  GPU_PARTICLE_THRESHOLD OR no CUDA
    → CPU path: calls your original MC_loop_ions_domain_tsc_fast! /
        MC_loop_ions_domain_fast! directly on DataFrames.
        Zero conversion overhead — identical performance to your original code.
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
    println("    Backend selected : $(use_gpu ? "GPU 🚀" : "CPU 🖥")")
    println("    track_seg        : $track_seg")
    println("───────────────────────────────────────────────")

    t_start = time()
    gsm2    = gsm2_cycle[1]

    # ── helper: select representative cells for a given filter ───────────────
    function _get_representatives(cell_df, energy_step_filter=nothing)
        if energy_step_filter === nothing
            cell_df_is = filter(row -> row.is_cell == 1, cell_df)
        else
            cell_df_is = filter(row -> row.is_cell == 1 &&
                                        row.energy_step == energy_step_filter, cell_df)
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

    # ─────────────────────────────────────────────────────────────────────────
    # TRACK-SEGMENT branch
    # ─────────────────────────────────────────────────────────────────────────
    if track_seg

        _, cell_df_single_x, cell_df_single_y, at_single =
            _get_representatives(cell_df_copy)

        if cell_df_single_x === nothing
            @warn "No cells with is_cell=1 → skipping."
            return
        end

        if use_gpu
            # ── GPU path ──────────────────────────────────────────────────────
            println("• [GPU] TSC mode")

            dose_vec, core_dose,
            core_radius_sq, mid_radius_sq, penumbra_radius_sq,
            Kp, log_impact_min, log_impact_max =
                _build_lookup([irrad_cond[1]], gsm2, type_AT)

            mat_x, mat_y, mat_at =
                dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

            num_cells, num_domains_per_cell = size(mat_x)
            total_domains = num_cells * num_domains_per_cell

            cu_dom_x = CuArray(vec(mat_x'))
            cu_dom_y = CuArray(vec(mat_y'))
            cu_dose  = CuArray(dose_vec)
            cu_out   = CUDA.zeros(Float64, total_domains)

            Np = rand(Poisson(Npar))
            println("   Sampling $Np particles (Poisson($Npar))...")
            x_list = Vector{Float64}(undef, Np)
            y_list = Vector{Float64}(undef, Np)
            Threads.@threads for ip in 1:Np
                x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
            end

            _gpu_run_single!(
                cu_out, cu_dom_x, cu_dom_y, x_list, y_list, cu_dose,
                core_radius_sq, mid_radius_sq, penumbra_radius_sq,
                core_dose, Kp, log_impact_min, log_impact_max
            )

            mat_at .= reshape(Array(cu_out), num_domains_per_cell, num_cells)'

            matrix_to_dataframe!(at_single, mat_at)

        else
            # ── CPU path — your original function, zero overhead ──────────────
            println("• [CPU] TSC mode")

            MC_loop_ions_domain_tsc_fast!(
                Npar, x_cb, y_cb,
                [irrad_cond[1]], gsm2,
                cell_df_single_x, cell_df_single_y, at_single,
                R_beam, type_AT, ion
            )
        end

        MC_loop_copy_dose_domain_fast!(cell_df_copy, at_single, at)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYERED (non-TSC) branch
    # ─────────────────────────────────────────────────────────────────────────
    else

        if use_gpu
            # ── GPU: sample particles once, reuse across layers ───────────────
            println("• [GPU] Layered mode")

            Np = rand(Poisson(Npar))
            println("• Sampling $Np particles...")
            x_list = Vector{Float64}(undef, Np)
            y_list = Vector{Float64}(undef, Np)
            Threads.@threads for ip in 1:Np
                x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
            end

            for id in unique(cell_df_copy.energy_step)
                println("  → Layer $id")

                _, cell_df_single_x, cell_df_single_y, at_single =
                    _get_representatives(cell_df_copy, id)
                cell_df_single_x === nothing && (println("    (empty)"); continue)

                dose_vec, core_dose,
                core_radius_sq, mid_radius_sq, penumbra_radius_sq,
                Kp, log_impact_min, log_impact_max =
                    _build_lookup([irrad_cond[id]], gsm2, type_AT)

                mat_x, mat_y, mat_at =
                    dataframes_to_matrices(cell_df_single_x, cell_df_single_y, at_single)

                num_cells, num_domains_per_cell = size(mat_x)
                total_domains = num_cells * num_domains_per_cell

                cu_dom_x = CuArray(vec(mat_x'))
                cu_dom_y = CuArray(vec(mat_y'))
                cu_dose  = CuArray(dose_vec)
                cu_out   = CUDA.zeros(Float64, total_domains)

                _gpu_run_single!(
                    cu_out, cu_dom_x, cu_dom_y, x_list, y_list, cu_dose,
                    core_radius_sq, mid_radius_sq, penumbra_radius_sq,
                    core_dose, Kp, log_impact_min, log_impact_max
                )

                mat_at .= reshape(Array(cu_out), num_domains_per_cell, num_cells)'
                matrix_to_dataframe!(at_single, mat_at)
                MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, at_single, at, id)
            end

        else
            # ── CPU: your original function, zero overhead ────────────────────
            println("• [CPU] Layered mode")

            Np = rand(Poisson(Npar))
            println("• Sampling $Np particles...")
            x_list = Vector{Float64}(undef, Np)
            y_list = Vector{Float64}(undef, Np)
            Threads.@threads for ip in 1:Np
                x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
            end

            for id in unique(cell_df_copy.energy_step)
                println("  → Layer $id")

                _, cell_df_single_x, cell_df_single_y, at_single =
                    _get_representatives(cell_df_copy, id)
                cell_df_single_x === nothing && (println("    (empty)"); continue)

                MC_loop_ions_domain_fast!(
                    x_list, y_list,
                    [irrad_cond[id]], gsm2,
                    cell_df_single_x, cell_df_single_y, at_single,
                    type_AT, ion
                )

                MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, at_single, at, id)
            end
        end
    end

    dt = round(time() - t_start; digits=3)
    println("───────────────────────────────────────────────")
    println("🎉  MC_dose_fast! finished in $(dt)s")
    println("───────────────────────────────────────────────\n")
end
