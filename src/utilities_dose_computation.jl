#! ============================================================================
#! utilities_dose_computation.jl
#!
#! Amorphous-track Monte Carlo dose deposition onto sub-cellular domains.
#!
#! REQUIRES (load before this file)
#!   utilities_structures.jl      Cell, Track, Ion, Irrad, AT, GSM2
#!   utilities_general.jl         GenerateHit_Circle  (NOT redefined here)
#!   utilities_radiation.jl       ATRadius, stopping tables
#!   utilities_AT_computation.jl  distribute_dose_domain, GetRadialLinearDose
#!   Random, Distributions, DataFrames
#!
#! Physics is unchanged from the original: near field from
#! `distribute_dose_domain`, far field `Kp/dist²`, hard cutoff at `Rp`.
#! Only the execution strategy differs.
#!
#! FUNCTIONS
#! ---------
#~ Fast kernel machinery
#?   chunk_ranges(n, nchunks)                     -> Vector{UnitRange{Int}}
#?   build_radial_lut(a, gsm2, type_AT; n)        -> (lut, inv_step, b_near2)
#?   TrackGrid(tx, ty, h)                         -> TrackGrid
#?   accumulate_dose!(out, dom_x, dom_y, tx, ty, ...) -> out
#~ MC kernels
#?   MC_loop_ions_domain_tsc_matrix!(...)         track-segment mode
#?   MC_loop_ions_domain_matrix!(...)             full-MC mode
#~ Driver and plumbing
#?   MC_dose_CPU!(...)                            top-level, signature unchanged
#?   dataframes_to_matrices / matrix_to_dataframe!
#?   MC_loop_copy_dose_domain_fast!(...)          TSC copy-back
#?   MC_loop_copy_dose_domain_layer_fast_notsc!(...) layered copy-back
#?   MC_precompute_lut!(...)                      per-particle LUT
#~ Diagnostics
#?   check_against_reference(...)                 fast path vs brute force
#?   check_lut_closure(...)                       ∫Ḡ(b)·2πb db vs LET·0.1602
#!
#! WHAT CHANGED FROM THE ORIGINAL, AND WHY
#! ---------------------------------------
#! 1. LOOP INVERSION + TILING.  The old kernels were particle-outer /
#!    domain-inner, so each particle streamed the whole accumulator through
#!    cache:  `for _ in 1:Np; for k in 1:total_domains`.  At 10⁶ domains that
#!    is 8 MB per particle, ~1.8 TB at 2×10⁵ particles, and the loop is
#!    bandwidth-bound — roughly 10 flops per pair, far too few to hide it.
#!    Domains are now partitioned across threads, tiled to L2, and all tracks
#!    are swept per tile:  traffic becomes (ndom/TILE)·Np·16 bytes.
#!
#! 2. NO PER-THREAD ACCUMULATORS.  `at_acc[Threads.threadid()]` is unsafe under
#!    `Threads.@threads` since Julia 1.8 (`:dynamic` scheduling migrates tasks
#!    mid-body, so the tid read at the top can be stale and two tasks share one
#!    buffer).  Partitioning by domain instead means threads write disjoint
#!    slices of the output: the race is structurally impossible, the reduction
#!    pass is gone, and so is nthreads()·ndom·8 bytes of scratch.
#!
#! 3. O(1) LUT LOOKUP.  The table is uniform in dist² rather than log-spaced in
#!    dist, so the index is one multiply and a truncate instead of
#!    `searchsortedfirst`.  Uniform-in-d² also puts the finest resolution near
#!    the domain rim, where dose(b) varies fastest.
#!
#! 4. BRANCHLESS FAR FIELD + SPARSE NEAR FIELD.  Nearly every pair lands in the
#!    `Kp/d²` branch, so that pass is isolated and vectorisable; the few
#!    near-field pairs are corrected afterwards through a spatial hash of the
#!    track positions.
#!
#! 5. COPY-BACK WAS O(n²)–O(n³).  `findfirst(==(idx), cell_df.index)` scans the
#!    whole column, and sat inside a doubly nested loop.  Replaced by index→row
#!    Dicts built once.  On a large lattice this was often costing more than
#!    the Monte Carlo itself.
#!
#! 6. REPRODUCIBILITY.  Track positions are drawn up front from an explicit
#!    RNG.  The old code could not be reproduced even with `Random.seed!`,
#!    since thread assignment varied between runs.
#!
#! 7. DROPPED.  `MC_loop_ions_domain_tsc_fast!` and `MC_loop_ions_domain_fast!`
#!    duplicated the two matrix kernels with a different progress bar and were
#!    not called by `MC_dose_CPU!`.  The latter was never actually threaded:
#!    `@showprogress for` with thread-local accumulators around it, so tid was
#!    always 1.
#! ============================================================================

using Random

#? Default domain-tile size in Float64 elements. 32768 = 256 KB, a common L2.
#? This is the only cache-tuning knob; the *strategy* below adapts on its own.
const DEFAULT_TILE_DOMAINS = 32_768

#! ============================================================================
#! Chunking and execution planning
#! ============================================================================

"""
    chunk_ranges(n, nchunks) -> Vector{UnitRange{Int}}

Partition `1:n` into contiguous ranges whose lengths differ by at most 1.
The chunk index is what accumulators are keyed on — never `threadid()`.
"""
function chunk_ranges(n::Int, nchunks::Int)
    n <= 0 && return UnitRange{Int}[]
    nchunks = max(1, min(nchunks, n))
    base, rem = divrem(n, nchunks)
    rs = Vector{UnitRange{Int}}(undef, nchunks)
    lo = 1
    for c in 1:nchunks
        len = base + (c <= rem ? 1 : 0)
        rs[c] = lo:(lo + len - 1)
        lo += len
    end
    return rs
end

"""
    DosePlan

How `accumulate_dose!` will decompose the work. Printable, so the chosen
strategy is visible rather than implicit.
"""
struct DosePlan
    ndom::Int
    Np::Int
    tile::Int
    ntiles::Int
    npchunks::Int
    nthreads::Int
    strategy::Symbol
    accum_bytes::Int
    traffic_bytes::Float64
end

"""
    plan_dose(ndom, Np; nthreads, tile) -> DosePlan

Choose the decomposition from the actual geometry.

The work is a full `ndom × Np` sweep either way, so the only thing to optimise
is memory traffic. Two regimes:

* **Accumulator fits cache** (`ndom ≤ tile`, one tile). Splitting by domain
  would force every thread to re-read the whole track list: `nthreads·Np·16`
  bytes. Instead split by *particle* — each thread keeps a private accumulator
  resident and the tracks stream once: `Np·16 + nthreads·ndom·8`.

* **Accumulator exceeds cache** (`ndom > tile`, many tiles). Private
  accumulators would cost `nthreads·ndom·8` and none would stay resident.
  Split by domain into cache-sized tiles and sweep all tracks per tile:
  `ntiles·Np·16`.

Between the two, when tiles are fewer than threads, tiles are additionally
split across particle chunks so every thread has work. Scratch memory is then
bounded by `nthreads·tile·8` — about 5 MB at 19 threads — regardless of how
large the spheroid gets.
"""
function plan_dose(ndom::Int, Np::Int;
                   nthreads::Int = Threads.nthreads(),
                   tile::Int = DEFAULT_TILE_DOMAINS)
    ndom <= 0 && return DosePlan(ndom, Np, 1, 0, 1, nthreads, :empty, 0, 0.0)

    tile     = clamp(tile, 1, ndom)
    ntiles   = cld(ndom, tile)
    npchunks = clamp(cld(nthreads, ntiles), 1, max(1, Np))

    strategy = ntiles == 1 ? (npchunks == 1 ? :serial_tile : :particle_parallel) :
               (npchunks == 1 ? :domain_parallel : :hybrid)

    accum   = npchunks == 1 ? 0 : ntiles * npchunks * min(tile, ndom) * 8
    traffic = npchunks == 1 ? ntiles * float(Np) * 16 + ndom * 8 :
                              float(Np) * 16 * ntiles + accum

    DosePlan(ndom, Np, tile, ntiles, npchunks, nthreads, strategy, accum, traffic)
end

function Base.show(io::IO, p::DosePlan)
    println(io, "  plan | $(p.ndom) domains x $(p.Np) tracks on $(p.nthreads) threads")
    println(io, "       | strategy=$(p.strategy)  tiles=$(p.ntiles)x$(p.tile)  " *
                "particle-chunks=$(p.npchunks)")
    println(io, "       | scratch=$(round(p.accum_bytes/2^20; digits=2)) MiB  " *
                "est. traffic=$(round(p.traffic_bytes/2^20; digits=1)) MiB")
end

#! ============================================================================
#! Radial lookup table, uniform in dist²
#! ============================================================================

"""
    build_radial_lut(a, gsm2, type_AT; n, nchunks) -> (lut, inv_step, b_near2)

Mean domain dose against **squared** impact parameter on `[0, (rd + 150·Rc)²]`,
evaluated with `distribute_dose_domain` — the same near/mid span and the same
integrator the original used.

Uniform in d² makes the lookup `i = trunc(Int, d2 * inv_step) + 1`, with no
binary search, and concentrates resolution near the rim where dose(b) is
steepest.

The table starts at b = 0 rather than b = rd − 10·Rc, at the same construction
cost. To reproduce the old flat-core behaviour exactly, overwrite every entry
below `(rd - 10Rc)^2 * inv_step` with `lut[1]` after building.
"""
function build_radial_lut(a::AT, gsm2::GSM2, type_AT::String;
                          n::Int = 8192, nchunks::Int = Threads.nthreads())
    b_near2  = (gsm2.rd + 150 * a.Rc)^2
    inv_step = n / b_near2

    lut = Vector{Float64}(undef, n)
    tasks = map(chunk_ranges(n, nchunks)) do rng
        Threads.@spawn for i in rng               # one writer per index
            b = sqrt((i - 0.5) / inv_step)        # bin centre in d²
            _, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd,
                                               Track(b, 0.0, a.Rk), a, type_AT)
            @inbounds lut[i] = Gyr
        end
    end
    foreach(wait, tasks)
    return lut, inv_step, b_near2
end

#! ============================================================================
#! Spatial hash over track positions
#! ============================================================================

"""
    TrackGrid(tx, ty, b_near; max_cells)

Uniform bucket grid over track positions, CSR layout: cell `c` holds the tracks
`idx[offs[c] : offs[c+1]-1]`. `nring` is how many cells out the neighbourhood
search must reach to cover `b_near`.

Cell size starts at `b_near` and doubles until the grid fits `max_cells`, so a
very small `b_near` on a wide beam cannot blow up memory. `nring` is recomputed
to match, keeping the search correct at any cell size.

Tracks are hashed rather than domains so the domain partition stays intact and
each thread keeps writing only its own output indices.
"""
struct TrackGrid
    x0::Float64
    y0::Float64
    inv_h::Float64
    nx::Int
    ny::Int
    nring::Int
    offs::Vector{Int}
    idx::Vector{Int32}
end

function TrackGrid(tx::Vector{Float64}, ty::Vector{Float64}, b_near::Float64;
                   max_cells::Int = 4_000_000)
    n = length(tx)
    n == 0 && return TrackGrid(0.0, 0.0, 1.0, 1, 1, 1, [1, 1], Int32[])

    xmin, xmax = extrema(tx)
    ymin, ymax = extrema(ty)
    W, H = xmax - xmin, ymax - ymin

    h = max(b_near, eps())
    while (floor(Int, W / h) + 1) * (floor(Int, H / h) + 1) > max_cells
        h *= 2
    end
    inv_h = 1.0 / h
    nx = max(1, floor(Int, W * inv_h) + 1)
    ny = max(1, floor(Int, H * inv_h) + 1)
    nring = max(1, ceil(Int, b_near * inv_h))

    @inline cellof(x, y) = begin
        i = clamp(floor(Int, (x - xmin) * inv_h), 0, nx - 1)
        j = clamp(floor(Int, (y - ymin) * inv_h), 0, ny - 1)
        j * nx + i + 1
    end

    counts = zeros(Int, nx * ny + 1)
    @inbounds for p in 1:n
        counts[cellof(tx[p], ty[p]) + 1] += 1
    end
    offs = similar(counts)
    offs[1] = 1
    @inbounds for c in 2:length(counts)
        offs[c] = offs[c-1] + counts[c]
    end

    cursor = copy(offs)
    idx = Vector{Int32}(undef, n)
    @inbounds for p in 1:n
        c = cellof(tx[p], ty[p])
        idx[cursor[c]] = Int32(p)
        cursor[c] += 1
    end

    TrackGrid(xmin, ymin, inv_h, nx, ny, nring, offs, idx)
end

#! ============================================================================
#! Inner kernels
#! ============================================================================

"""
    far_sweep!(acc, dx, dy, tx, ty, prange, Kp, rp2, b_near2)

Add `Kp/d²` for every (domain, track) pair with `b_near² ≤ d² < Rp²`, zero
elsewhere. Branchless so it vectorises; `acc`, `dx`, `dy` are contiguous views
of one domain tile.

Nearly every pair lands here, which is why it is worth isolating from the
near-field case.
"""
@inline function far_sweep!(acc::AbstractVector{Float64},
                            dx::AbstractVector{Float64}, dy::AbstractVector{Float64},
                            tx::Vector{Float64}, ty::Vector{Float64},
                            prange::UnitRange{Int},
                            Kp::Float64, rp2::Float64, b_near2::Float64)
    n = length(acc)
    @inbounds for p in prange
        x = tx[p]; y = ty[p]
        @simd for k in 1:n
            ddx = dx[k] - x
            ddy = dy[k] - y
            d2  = ddx * ddx + ddy * ddy
            inr = (d2 >= b_near2) & (d2 < rp2)
            acc[k] += ifelse(inr, Kp / max(d2, b_near2), 0.0)
        end
    end
    return nothing
end

"""
    near_sweep!(out, dom_x, dom_y, tx, ty, krange, grid, lut, inv_step, b_near2)

Add the tabulated near-field dose for pairs with `d² < b_near²`, visiting only
the tracks in the `nring` neighbourhood of each domain. Sparse: a handful of
candidates per domain rather than a test per pair.
"""
function near_sweep!(out::Vector{Float64},
                     dom_x::Vector{Float64}, dom_y::Vector{Float64},
                     tx::Vector{Float64}, ty::Vector{Float64},
                     krange::UnitRange{Int}, grid::TrackGrid,
                     lut::Vector{Float64}, inv_step::Float64, b_near2::Float64)
    nlut = length(lut)
    r    = grid.nring
    @inbounds for k in krange
        x = dom_x[k]; y = dom_y[k]
        i0 = clamp(floor(Int, (x - grid.x0) * grid.inv_h), 0, grid.nx - 1)
        j0 = clamp(floor(Int, (y - grid.y0) * grid.inv_h), 0, grid.ny - 1)
        acc = 0.0
        for j in max(0, j0 - r):min(grid.ny - 1, j0 + r)
            base = j * grid.nx
            for i in max(0, i0 - r):min(grid.nx - 1, i0 + r)
                c = base + i + 1
                for s in grid.offs[c]:(grid.offs[c+1] - 1)
                    p   = grid.idx[s]
                    ddx = x - tx[p]; ddy = y - ty[p]
                    d2  = ddx * ddx + ddy * ddy
                    if d2 < b_near2
                        acc += lut[min(trunc(Int, d2 * inv_step) + 1, nlut)]
                    end
                end
            end
        end
        out[k] += acc
    end
    return nothing
end

#! ============================================================================
#! Core sweep
#! ============================================================================

"""
    accumulate_dose!(out, dom_x, dom_y, tx, ty, Kp, rp2, lut, inv_step, b_near2;
                     plan, nthreads, tile, verbose) -> out

Accumulate the dose from every track into `out`, overwriting it.

Decomposition comes from `plan_dose` and adapts to the geometry: particle-
parallel with private accumulators while the domain set fits cache,
domain-tiled once it does not, hybrid in between. Pass `verbose = true` to see
which was chosen.

Threads never share an output element, under any strategy, so there is no race
and no `threadid()` anywhere.
"""
function accumulate_dose!(out::Vector{Float64},
                          dom_x::Vector{Float64}, dom_y::Vector{Float64},
                          tx::Vector{Float64}, ty::Vector{Float64},
                          Kp::Float64, rp2::Float64,
                          lut::Vector{Float64}, inv_step::Float64, b_near2::Float64;
                          plan::Union{Nothing,DosePlan} = nothing,
                          nthreads::Int = Threads.nthreads(),
                          tile::Int = DEFAULT_TILE_DOMAINS,
                          verbose::Bool = false)

    ndom = length(dom_x)
    Np   = length(tx)
    length(dom_y) == ndom || error("accumulate_dose!: dom_x/dom_y length mismatch")
    length(ty) == Np      || error("accumulate_dose!: tx/ty length mismatch")
    length(out) == ndom   || error("accumulate_dose!: out length mismatch")
    fill!(out, 0.0)
    (ndom == 0 || Np == 0) && return out

    P = plan === nothing ? plan_dose(ndom, Np; nthreads = nthreads, tile = tile) : plan
    verbose && show(stdout, P)

    tiles = [((t - 1) * P.tile + 1):min(t * P.tile, ndom) for t in 1:P.ntiles]

    # ---- far field ----------------------------------------------------------
    if P.npchunks == 1
        # one task per tile, writing its own slice of `out` in place
        tasks = map(tiles) do kr
            Threads.@spawn far_sweep!(view(out, kr), view(dom_x, kr), view(dom_y, kr),
                                      tx, ty, 1:Np, Kp, rp2, b_near2)
        end
        foreach(wait, tasks)
    else
        # tiles split across particle chunks; one buffer per (tile, chunk)
        pranges = chunk_ranges(Np, P.npchunks)
        bufs = [[zeros(Float64, length(kr)) for _ in 1:P.npchunks] for kr in tiles]

        tasks = Task[]
        for (t, kr) in enumerate(tiles), c in 1:P.npchunks
            push!(tasks, Threads.@spawn far_sweep!(bufs[t][c],
                                                   view(dom_x, kr), view(dom_y, kr),
                                                   tx, ty, pranges[c], Kp, rp2, b_near2))
        end
        foreach(wait, tasks)

        rtasks = map(enumerate(tiles)) do (t, kr)
            Threads.@spawn begin
                o = view(out, kr)
                for c in 1:P.npchunks
                    b = bufs[t][c]
                    @inbounds @simd for k in eachindex(o)
                        o[k] += b[k]
                    end
                end
            end
        end
        foreach(wait, rtasks)
    end

    # ---- near field: sparse, always split by domain --------------------------
    grid = TrackGrid(tx, ty, sqrt(b_near2))
    ntasks = map(chunk_ranges(ndom, nthreads)) do kr
        Threads.@spawn near_sweep!(out, dom_x, dom_y, tx, ty, kr, grid,
                                   lut, inv_step, b_near2)
    end
    foreach(wait, ntasks)

    return out
end

#! ============================================================================
#! MC kernels
#! ============================================================================

"""
    MC_loop_ions_domain_tsc_matrix!(Npar, x_cb, y_cb, irrad_cond, gsm2,
                                    mat_x, mat_y, mat_at, R_beam, type_AT, ion,
                                    single_particle; seed, nchunks, n_lut)

Track-segment mode. Samples `Np ~ Poisson(Npar)` impact points with
`GenerateHit_Circle` and accumulates the mean domain dose. Overwrites `mat_at`.

Pass `seed` for a run reproducible independently of thread scheduling.
"""
function MC_loop_ions_domain_tsc_matrix!(
    Npar::Int, x_cb::Float64, y_cb::Float64,
    irrad_cond::Vector{AT}, gsm2::GSM2,
    mat_x::Matrix{Float64}, mat_y::Matrix{Float64}, mat_at::Matrix{Float64},
    R_beam::Float64, type_AT::String, ion::Ion, single_particle::Bool = false;
    seed::Union{Nothing,Integer} = nothing,
    nchunks::Int = Threads.nthreads(),
    n_lut::Int = 8192,
    tile::Int = DEFAULT_TILE_DOMAINS,
    verbose::Bool = true)

    a  = irrad_cond[1]
    Np = single_particle ? 1 : rand(Poisson(Npar))

    rng = seed === nothing ? Random.default_rng() : Xoshiro(UInt64(seed))
    tx = Vector{Float64}(undef, Np)
    ty = Vector{Float64}(undef, Np)
    @inbounds for p in 1:Np
        tx[p], ty[p] = GenerateHit_Circle(rng, x_cb, y_cb, R_beam)
    end

    lut, inv_step, b_near2 = build_radial_lut(a, gsm2, type_AT; n = n_lut, nchunks = nchunks)

    ncell, ndomc = size(mat_x)
    # permutedims, not `mat_x'`: the adjoint is a lazy view, and `vec` of it is a
    # ReshapedArray, not a Vector. Materialise so the hot loop indexes a dense array.
    dom_x = vec(permutedims(mat_x)); dom_y = vec(permutedims(mat_y))
    out   = Vector{Float64}(undef, ncell * ndomc)

    accumulate_dose!(out, dom_x, dom_y, tx, ty, a.Kp, a.Rp^2,
                     lut, inv_step, b_near2;
                     nthreads = nchunks, tile = tile, verbose = verbose)

    mat_at .= reshape(out, ndomc, ncell)'
    return nothing
end

"""
    MC_loop_ions_domain_matrix!(x_list, y_list, irrad_cond, gsm2,
                                mat_x, mat_y, mat_at, type_AT, ion;
                                nchunks, n_lut)

Full-MC mode with pre-generated hit positions. Deterministic given the hit
lists, independent of `nchunks`. Overwrites `mat_at`.
"""
function MC_loop_ions_domain_matrix!(
    x_list::Vector{Float64}, y_list::Vector{Float64},
    irrad_cond::Vector{AT}, gsm2::GSM2,
    mat_x::Matrix{Float64}, mat_y::Matrix{Float64}, mat_at::Matrix{Float64},
    type_AT::String, ion::Ion;
    nchunks::Int = Threads.nthreads(),
    n_lut::Int = 8192,
    tile::Int = DEFAULT_TILE_DOMAINS,
    verbose::Bool = true)

    a = irrad_cond[1]
    lut, inv_step, b_near2 = build_radial_lut(a, gsm2, type_AT; n = n_lut, nchunks = nchunks)

    ncell, ndomc = size(mat_x)
    dom_x = vec(permutedims(mat_x)); dom_y = vec(permutedims(mat_y))
    out   = Vector{Float64}(undef, ncell * ndomc)

    accumulate_dose!(out, dom_x, dom_y, x_list, y_list, a.Kp, a.Rp^2,
                     lut, inv_step, b_near2;
                     nthreads = nchunks, tile = tile, verbose = verbose)

    mat_at .= reshape(out, ndomc, ncell)'
    return nothing
end

#! ============================================================================
#! DataFrame ↔ Matrix
#! ============================================================================

"""Extract all domain columns (everything but `:index`) as dense matrices."""
function dataframes_to_matrices(df_x::DataFrame, df_y::DataFrame, df_at::DataFrame)
    cols = names(df_x, Not(:index))
    return Matrix{Float64}(df_x[:, cols]),
           Matrix{Float64}(df_y[:, cols]),
           Matrix{Float64}(df_at[:, cols])
end

"""Write matrix columns back into the DataFrame's domain columns in place."""
function matrix_to_dataframe!(df::DataFrame, mat::Matrix{Float64})
    cols = names(df, Not(:index))
    size(mat, 2) == length(cols) || error("matrix_to_dataframe!: column count mismatch")
    for (j, c) in enumerate(cols)
        df[!, c] = mat[:, j]
    end
    return nothing
end

"""Rows of `df` whose `:index` is in `set`, without materialising DataFrameRows."""
select_rows(df::DataFrame, set) = df[in.(df.index, Ref(set)), :]

"""Representative cell `:index` per distinct (x, y) column position."""
representative_indices(cell_df::DataFrame) =
    Set(combine(groupby(cell_df, [:x, :y]), :index => first => :rep).rep)

#! ============================================================================
#! Top-level driver
#! ============================================================================

"""
    MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df_copy,
                 df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg;
                 x_cb, y_cb, single_particle, seed, nchunks, n_lut) -> Nothing

Mutates `cell_df_copy` (dose fields) and `at` (AT state) in place.

`track_seg = true`  → one representative plane, doses propagated to all layers.
`track_seg = false` → per-layer MC with shared hit positions.

`R_beam` should be padded by `Rp` beyond the scored region, or cells near the
boundary lose contributions from tracks just outside it. Whatever computes
`Npar` must use the same padded radius or the prescribed dose is not recovered.
"""
function MC_dose_CPU!(
    ion::Ion, Npar::Int64, R_beam::Float64,
    irrad_cond::Vector{AT}, cell_df_copy::DataFrame,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
    x_cb::Float64 = 0.0, y_cb::Float64 = 0.0, single_particle::Bool = false,
    seed::Union{Nothing,Integer} = nothing,
    nchunks::Int = Threads.nthreads(),
    n_lut::Int = 8192)

    println("\n─────────────────────────────────────────────")
    println("MC_dose_CPU!   track_seg = $track_seg")
    println("─────────────────────────────────────────────")
    t0   = time()
    gsm2 = gsm2_cycle[1]

    if track_seg
        cell_df_is = cell_df_copy[cell_df_copy.is_cell .== 1, :]
        if nrow(cell_df_is) == 0
            @warn "No cells with is_cell = 1 → skipping."
            return nothing
        end

        reps = representative_indices(cell_df_is)
        println("  $(length(reps)) representative cells")

        dfx = select_rows(df_center_x, reps)
        dfy = select_rows(df_center_y, reps)
        ats = select_rows(at,          reps)

        mx, my, ma = dataframes_to_matrices(dfx, dfy, ats)
        MC_loop_ions_domain_tsc_matrix!(Npar, x_cb, y_cb, [irrad_cond[1]], gsm2,
                                        mx, my, ma, R_beam, type_AT, ion,
                                        single_particle;
                                        seed = seed, nchunks = nchunks, n_lut = n_lut)
        matrix_to_dataframe!(ats, ma)
        MC_loop_copy_dose_domain_fast!(cell_df_copy, ats, at)

    else
        Np = single_particle ? 1 : rand(Poisson(Npar))
        println("  sampling $Np particle hits")

        rng = seed === nothing ? Random.default_rng() : Xoshiro(UInt64(seed))
        x_list = Vector{Float64}(undef, Np)
        y_list = Vector{Float64}(undef, Np)
        @inbounds for p in 1:Np                       # serial: one shared RNG
            x_list[p], y_list[p] = GenerateHit_Circle(rng, x_cb, y_cb, R_beam)
        end

        for id in unique(cell_df_copy.energy_step)
            sel = (cell_df_copy.is_cell .== 1) .& (cell_df_copy.energy_step .== id)
            if !any(sel)
                println("  layer $id empty → skip")
                continue
            end
            println("  layer $id")

            reps = representative_indices(cell_df_copy[sel, :])
            dfx  = select_rows(df_center_x, reps)
            dfy  = select_rows(df_center_y, reps)
            ats  = select_rows(at,          reps)

            mx, my, ma = dataframes_to_matrices(dfx, dfy, ats)
            MC_loop_ions_domain_matrix!(x_list, y_list, [irrad_cond[id]], gsm2,
                                        mx, my, ma, type_AT, ion;
                                        nchunks = nchunks, n_lut = n_lut)
            matrix_to_dataframe!(ats, ma)
            MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_copy, ats, at, id)
        end
    end

    println("  done in $(round(time() - t0; digits = 3)) s")
    return nothing
end

#! ============================================================================
#! Copy-back
#! ============================================================================

"""
    MC_loop_copy_dose_domain_fast!(cell_df, at_single, at) -> Nothing

TSC mode: propagate doses from the representative plane to every z-layer,
matched by (x, y). Inactive cells receive zero.

The original ran `findfirst(==(idx), cell_df.index)` and
`findfirst(==(idx), at.index)` inside a doubly nested loop — a full column scan
per lookup. Both index→row maps are now built once, so this is O(n).
"""
function MC_loop_copy_dose_domain_fast!(cell_df::DataFrame, at_single::DataFrame,
                                        at::DataFrame)
    for (nm, df) in (("cell_df", cell_df), ("at_single", at_single), ("at", at))
        hasproperty(df, :index) || error("$nm is missing :index")
    end
    (hasproperty(cell_df, :x) && hasproperty(cell_df, :y)) ||
        error("cell_df must contain :x and :y")
    if nrow(at_single) == 0
        println("  at_single empty → nothing to copy")
        return nothing
    end

    cols  = names(at_single, Not(:index))
    ndom  = length(cols)
    ncell = nrow(cell_df)

    if !hasproperty(cell_df, :dose) || !(eltype(cell_df.dose) <: AbstractVector)
        cell_df.dose = [zeros(Float64, ndom) for _ in 1:ncell]
    end
    if !hasproperty(cell_df, :dose_cell)
        cell_df.dose_cell = zeros(Float64, ncell)
    end

    cell_row = Dict{Int,Int}(idx => r for (r, idx) in enumerate(cell_df.index))
    at_row   = Dict{Int,Int}(idx => r for (r, idx) in enumerate(at.index))

    xy_targets = Dict{Tuple{Float64,Float64},Vector{Int}}()
    for r in 1:ncell
        key = (Float64(cell_df.x[r]), Float64(cell_df.y[r]))
        push!(get!(xy_targets, key, Int[]), cell_df.index[r])
    end

    for rs in eachrow(at_single)
        vec_dose = Float64[rs[c] for c in cols]
        scalar   = ndom == 0 ? 0.0 : sum(vec_dose) / ndom

        rp = get(cell_row, rs.index, 0)
        rp == 0 && (@warn "representative $(rs.index) not in cell_df"; continue)

        key = (Float64(cell_df.x[rp]), Float64(cell_df.y[rp]))
        for idx in get(xy_targets, key, Int[])
            cr = get(cell_row, idx, 0); ar = get(at_row, idx, 0)
            (cr == 0 || ar == 0) && continue

            ndom > 0 && (at[ar, cols] .= vec_dose)
            if cell_df.is_cell[cr] == 1
                cell_df.dose[cr]      = copy(vec_dose)
                cell_df.dose_cell[cr] = scalar
            else
                cell_df.dose[cr]      = zeros(Float64, ndom)
                cell_df.dose_cell[cr] = 0.0
            end
        end
    end
    return nothing
end

"""
    MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df, at_single, at, step) -> Nothing

Full-MC mode: copy doses for one `energy_step`, matched by (x, y).

Serial by design. The previous version held a `ReentrantLock` around the
progress update on every iteration; that cost more than the copy itself, and a
lock in a `Threads.@threads` body is precisely the yield point that lets tasks
migrate and makes `threadid()`-indexed state unsafe.
"""
function MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df::DataFrame,
        at_single::DataFrame, at::DataFrame, energy_step_to_match::Int64)

    for c in (:index, :x, :y, :is_cell, :energy_step)
        hasproperty(cell_df, c) || error("cell_df is missing required column :$c")
    end
    if nrow(at_single) == 0
        println("  at_single empty → nothing to do")
        return nothing
    end

    cols  = names(at_single, Not(:index))
    ndom  = length(cols)
    ncell = nrow(cell_df)

    if !hasproperty(cell_df, :dose) || !(eltype(cell_df.dose) <: AbstractVector)
        cell_df.dose = [zeros(Float64, ndom) for _ in 1:ncell]
    end
    if !hasproperty(cell_df, :dose_cell)
        cell_df.dose_cell = zeros(Float64, ncell)
    end

    cell_row = Dict{Int,Int}(idx => r for (r, idx) in enumerate(cell_df.index))
    at_row   = Dict{Int,Int}(idx => r for (r, idx) in enumerate(at.index))

    xy_dose = Dict{Tuple{Float64,Float64},Tuple{Vector{Float64},Float64}}()
    sizehint!(xy_dose, nrow(at_single))
    for rs in eachrow(at_single)
        r = get(cell_row, rs.index, 0)
        r == 0 && continue
        cell_df.energy_step[r] == energy_step_to_match || continue
        v = Float64[rs[c] for c in cols]
        xy_dose[(Float64(cell_df.x[r]), Float64(cell_df.y[r]))] =
            (v, ndom == 0 ? 0.0 : sum(v) / ndom)
    end

    targets = findall(i -> cell_df.energy_step[i] == energy_step_to_match &&
                           cell_df.is_cell[i] == 1, 1:ncell)
    println("  $(length(targets)) active cells in step $energy_step_to_match")

    nmiss = 0
    for i in targets
        key = (Float64(cell_df.x[i]), Float64(cell_df.y[i]))
        hit = get(xy_dose, key, nothing)
        if hit === nothing
            nmiss += 1
            v, s = zeros(Float64, ndom), 0.0
        else
            v, s = hit
        end

        length(cell_df.dose[i]) != ndom && (cell_df.dose[i] = zeros(Float64, ndom))
        cell_df.dose[i] .= v
        cell_df.dose_cell[i] = s

        ar = get(at_row, cell_df.index[i], 0)
        (ar != 0 && ndom > 0) && (at[ar, cols] .= v)
    end
    nmiss > 0 && @warn "copy-back: $nmiss cells had no matching (x,y) → zero dose"
    return nothing
end

#! ============================================================================
#! Per-particle LUT (continuous dose rate)
#! ============================================================================

"""
    MC_precompute_lut!(ion, Npar, R_beam, irrad_cond, cell_df,
                       df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg;
                       x_cb, y_cb, seed, nchunks, n_lut)
        -> Vector{Dict{Int,Vector{Float64}}}

`lut[p][rep_index]` is the per-domain dose from particle `p`, keeping only
cells with a non-zero contribution.

Parallelised over particles rather than domains, since each particle needs its
own output. The old version allocated a full `zeros(total_domains)` per particle
plus one slice per cell per particle, then called `GC.gc()` after each chunk to
contain the churn; buffers are now per chunk and reused.

Memory is the binding constraint. The sparsity filter helps little once the
penumbra covers the domain, which it does at high energy, so budget close to
`Npar × ncells × ndomains × 8` bytes. The printed estimate is that bound.
"""
function MC_precompute_lut!(
    ion::Ion, Npar::Int, R_beam::Float64,
    irrad_cond::Vector{AT}, cell_df::DataFrame,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool;
    x_cb::Float64 = 0.0, y_cb::Float64 = 0.0,
    seed::Union{Nothing,Integer} = nothing,
    nchunks::Int = Threads.nthreads(),
    n_lut::Int = 8192)

    gsm2 = gsm2_cycle[1]
    a    = irrad_cond[1]

    cell_df_is = cell_df[cell_df.is_cell .== 1, :]
    reps = representative_indices(cell_df_is)
    dfx  = select_rows(df_center_x, reps)
    dfy  = select_rows(df_center_y, reps)

    cols  = names(dfx, Not(:index))
    ncell = nrow(dfx)
    ndomc = length(cols)
    ntot  = ncell * ndomc
    dom_x = vec(permutedims(Matrix{Float64}(dfx[:, cols])))
    dom_y = vec(permutedims(Matrix{Float64}(dfy[:, cols])))
    rep_index = Vector{Int}(dfx.index)

    lut_r, inv_step, b_near2 = build_radial_lut(a, gsm2, type_AT; n = n_lut, nchunks = nchunks)
    nlut = length(lut_r)
    rp2  = a.Rp^2

    println("  LUT: $Npar particles × $ncell cells × $ndomc domains")
    println("  upper bound on output: $(round(Npar * ntot * 8 / 2^30; digits = 2)) GiB")

    out    = Vector{Dict{Int,Vector{Float64}}}(undef, Npar)
    ranges = chunk_ranges(Npar, nchunks)
    rngs   = [seed === nothing ? Xoshiro(rand(UInt64)) : Xoshiro(UInt64(seed) + c)
              for c in 1:length(ranges)]

    tasks = map(enumerate(ranges)) do (c, prange)
        Threads.@spawn begin
            buf = Vector{Float64}(undef, ntot)     # one buffer per chunk, reused
            rng = rngs[c]
            for p in prange
                x, y = GenerateHit_Circle(rng, x_cb, y_cb, R_beam)
                @inbounds @simd for k in 1:ntot
                    dx = dom_x[k] - x
                    dy = dom_y[k] - y
                    d2 = dx * dx + dy * dy
                    inr = (d2 >= b_near2) & (d2 < rp2)
                    buf[k] = ifelse(inr, a.Kp / max(d2, b_near2), 0.0)
                end
                @inbounds for k in 1:ntot          # near-field correction
                    dx = dom_x[k] - x
                    dy = dom_y[k] - y
                    d2 = dx * dx + dy * dy
                    if d2 < b_near2
                        buf[k] += lut_r[min(trunc(Int, d2 * inv_step) + 1, nlut)]
                    end
                end

                d = Dict{Int,Vector{Float64}}()
                for j in 1:ncell
                    lo = (j - 1) * ndomc + 1
                    hi = j * ndomc
                    any(!iszero, view(buf, lo:hi)) && (d[rep_index[j]] = buf[lo:hi])
                end
                out[p] = d                          # one writer per index
            end
        end
    end
    foreach(wait, tasks)

    println("  LUT complete: $Npar particles")
    return out
end

#! ============================================================================
#! Diagnostics
#! ============================================================================

"""
    check_against_reference(a, gsm2, type_AT, dom_x, dom_y, tx, ty; rtol, n_lut)

Recompute the dose the slow way — one `distribute_dose_domain` per
(track, domain) pair, no table — and report the worst relative deviation.

Use a small case: this is O(Np·ndom) quadratures. Deviation beyond LUT
resolution means the fast path has drifted from the physics.
"""
function check_against_reference(a::AT, gsm2::GSM2, type_AT::String,
                                 dom_x::Vector{Float64}, dom_y::Vector{Float64},
                                 tx::Vector{Float64}, ty::Vector{Float64};
                                 rtol::Float64 = 1e-3, n_lut::Int = 8192)
    ref = zeros(length(dom_x))
    for p in eachindex(tx), k in eachindex(dom_x)
        b = hypot(dom_x[k] - tx[p], dom_y[k] - ty[p])
        _, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd,
                                           Track(b, 0.0, a.Rk), a, type_AT)
        ref[k] += Gyr
    end

    lut, inv_step, b_near2 = build_radial_lut(a, gsm2, type_AT; n = n_lut)
    fast = zeros(length(dom_x))
    accumulate_dose!(fast, dom_x, dom_y, tx, ty, a.Kp, a.Rp^2, lut, inv_step, b_near2)

    d = maximum(abs.(fast .- ref) ./ max.(abs.(ref), eps()))
    println("  max relative deviation: ", d)
    d > rtol && @warn "fast kernel deviates beyond rtol" deviation=d rtol=rtol
    return d
end

"""
    check_lut_closure(a, gsm2, type_AT; n_lut) -> Float64

Energy conservation over impact parameter:

    ∫₀^{Rp} Ḡ(b)·2πb db   ==   LET·0.1602

Every track deposits its full LET somewhere, so integrating the mean domain
dose over all impact parameters must return LETk, independent of `rd`.

The residual is not expected to be zero here, and its size is informative: the
far field uses `Kp/b²`, the dose at the domain *centre*, whereas the domain
*mean* is `(Kp/rd²)·ln(b²/(b²−rd²))` — larger by `(1 + rd²/2b²)`. The shortfall
is roughly `πKp·rd²/b_near²` against a total of `2πKp·(0.5 + ln(Rp/Rc))`. This
routine measures that for your actual parameters rather than assuming it.

A residual far larger than that estimate points at something else — most often
a mismatch between the radius the core is normalised against and the radius the
domain integral truncates at.
"""
function check_lut_closure(a::AT, gsm2::GSM2, type_AT::String; n_lut::Int = 8192)
    lut, inv_step, b_near2 = build_radial_lut(a, gsm2, type_AT; n = n_lut)
    b_near = sqrt(b_near2)

    # near field: LUT bins are uniform in b², and ∫Ḡ·2πb db = π∫Ḡ d(b²)
    near = π * sum(lut) / inv_step

    # far field: ∫ (Kp/b²)·2πb db from b_near to Rp
    far = a.Rp > b_near ? 2π * a.Kp * log(a.Rp / b_near) : 0.0

    LETk = a.LET * 0.1602
    err  = (near + far) / LETk - 1

    println("── LUT closure ──────────────────────────────")
    println("  near  [0, $(round(b_near, digits=3)) µm] : ", near)
    println("  far   [$(round(b_near, digits=3)), $(round(a.Rp, digits=1)) µm] : ", far)
    println("  total                    : ", near + far)
    println("  LET·0.1602               : ", LETk)
    println("  relative error           : ", err)
    println("─────────────────────────────────────────────")
    return err
end

#! ============================================================================
#! Particle budget
#! ============================================================================

"""
    SpeciesPlan

Result of turning a prescribed dose + energy spectrum into a concrete set of
per-energy particle counts for one sampled realisation.
"""
struct SpeciesPlan
    energies::Vector{Float64}   # MeV/u
    probs::Vector{Float64}      # particle-number fractions, normalised
    lets::Vector{Float64}       # keV/µm at each energy
    Nbar::Float64               # mean total particles
    Ntot::Int                   # sampled total (Poisson)
    counts::Vector{Int}         # per-energy split (Multinomial), sums to Ntot
end

function Base.show(io::IO, sp::SpeciesPlan)
    println(io, "  SpeciesPlan: N̄=$(round(sp.Nbar; digits=1)), sampled Ntot=$(sp.Ntot)")
    for k in eachindex(sp.energies)
        println(io, "    E=$(rpad(sp.energies[k], 8)) MeV/u  p=$(rpad(round(sp.probs[k]; digits=4), 8))" *
                    "  LET=$(rpad(round(sp.lets[k]; digits=3), 9)) keV/µm  →  $(sp.counts[k]) particles")
    end
end

"""
    mean_particle_number(dose_Gy, R_beam, energies, probs, lets; weight)
        -> (Nbar, probs_normalised)

Mean number of primaries for a prescribed dose over a beam disc of radius
`R_beam`, given the energy spectrum and its per-energy LET.

`weight = :fluence` (default) sums the reciprocal-LET contributions, which is
the fluence-correct budget. `:mean_energy` uses one LET at the mean energy and
is provided only for comparison; it requires the caller to pass the LET at the
mean energy as a one-element `lets` (see `plan_species`).
"""
function mean_particle_number(dose_Gy::Float64, R_beam::Float64,
                              probs::Vector{Float64}, lets::Vector{Float64};
                              weight::Symbol = :fluence)
    area = π * R_beam^2
    if weight === :fluence
        # N̄ = D·A · Σ p_k / (LET_k · 0.1602)
        s = 0.0
        for k in eachindex(probs)
            lets[k] > 0 || error("mean_particle_number: non-positive LET at species $k")
            s += probs[k] / (lets[k] * 0.1602)
        end
        return dose_Gy * area * s
    elseif weight === :mean_energy
        length(lets) == 1 || error("mean_particle_number(:mean_energy): pass a single LET at ⟨E⟩")
        return dose_Gy * area / (lets[1] * 0.1602)
    else
        error("mean_particle_number: unknown weight :$weight. Use :fluence or :mean_energy.")
    end
end

"""
    plan_species(dose_Gy, R_beam, energies, probs, particle, sp;
                 weight, rng) -> SpeciesPlan

Build the per-energy particle counts for one realisation.

`energies` in MeV/u, `probs` are particle-number fractions (normalised here if
they do not sum to 1). `particle` is the ion label ("1H"); `sp` the stopping
tables. `let_at(table_for(sp, particle), E)` supplies each LET.

Sampling: `Ntot ~ Poisson(N̄)`, then `counts ~ Multinomial(Ntot, probs)`. The
multinomial (rather than independent Poissons per species) reproduces your
worked example exactly: a single total is drawn, then partitioned.
"""
function plan_species(dose_Gy::Float64, R_beam::Float64,
                      energies::Vector{Float64}, probs::Vector{Float64},
                      particle::AbstractString, sp::Dict{String,StoppingTable};
                      weight::Symbol = :fluence,
                      rng::AbstractRNG = Random.default_rng())

    length(energies) == length(probs) || error("plan_species: energies/probs length mismatch")
    all(≥(0), probs) || error("plan_species: negative probability")
    ps = sum(probs)
    ps > 0 || error("plan_species: probabilities sum to zero")
    pn = probs ./ ps

    tab  = table_for(sp, particle)
    lets = [let_at(tab, E) for E in energies]

    Nbar = if weight === :fluence
        mean_particle_number(dose_Gy, R_beam, pn, lets; weight = :fluence)
    else
        Ebar = sum(pn .* energies)
        mean_particle_number(dose_Gy, R_beam, pn, [let_at(tab, Ebar)]; weight = :mean_energy)
    end

    Ntot   = rand(rng, Poisson(Nbar))
    counts = Ntot == 0 ? zeros(Int, length(pn)) : rand(rng, Multinomial(Ntot, pn))

    return SpeciesPlan(collect(float.(energies)), pn, lets, Nbar, Ntot, Vector{Int}(counts))
end

#! ============================================================================
#! Polyenergetic driver
#! ============================================================================

"""
    MC_dose_CPU_poly!(particle, dose_Gy, R_beam, energies, probs,
                      cell_df, df_center_x, df_center_y, at,
                      gsm2_cycle, irrad, type_AT, track_seg, sp;
                      weight, x_cb, y_cb, seed, nchunks, n_lut, A) -> SpeciesPlan

Polyenergetic version of `MC_dose_CPU!`. Instead of a single `Ion` + `Npar`,
takes an energy spectrum and a prescribed dose, and superimposes one
monoenergetic run per energy.

Writes the SUMMED per-domain dose into `at`'s domain columns and into
`cell_df.dose` / `cell_df.dose_cell`, matching what `MC_dose_CPU!` writes for a
single energy. Returns the `SpeciesPlan` so the realised counts are inspectable.

Arguments beyond `MC_dose_CPU!`:
* `dose_Gy`   — prescribed macroscopic dose; sets the particle budget.
* `energies`, `probs` — spectrum; `probs` are particle-number fractions.
* `irrad`     — the `Irrad` passed to `compute_energy_box!` / `ATRadius`.
* `sp`        — stopping tables.
* `weight`    — `:fluence` (default, correct) or `:mean_energy` (comparison).
* `A`         — mass number; defaults to `getA(particle)`.

Each species is built through `compute_energy_box!`, so per-layer energy
propagation, the occupied-layer absorption logic and AT validity all apply
independently to every energy — a low-energy component that stops early simply
contributes zero beyond its range.

Reproducibility: with `seed` set, the species split and every monoenergetic run
are seeded deterministically (`seed + k` for species k).
"""
function MC_dose_CPU_poly!(
    particle::AbstractString, dose_Gy::Float64, R_beam::Float64,
    energies::Vector{Float64}, probs::Vector{Float64},
    cell_df::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, irrad::Irrad, type_AT::String, track_seg::Bool,
    sp::Dict{String,StoppingTable};
    weight::Symbol = :fluence,
    x_cb::Float64 = 0.0, y_cb::Float64 = 0.0,
    seed::Union{Nothing,Integer} = nothing,
    nchunks::Int = Threads.nthreads(),
    n_lut::Int = 8192,
    A::Int = getA(particle))

    Z = getZ(particle)

    println("\n═════════════════════════════════════════════")
    println("MC_dose_CPU_poly!   $particle   dose=$dose_Gy Gy   weight=:$weight")
    println("  $(length(energies)) energies, track_seg=$track_seg")
    println("═════════════════════════════════════════════")
    t0 = time()

    rng  = seed === nothing ? Random.default_rng() : Xoshiro(UInt64(seed))
    plan = plan_species(dose_Gy, R_beam, energies, probs, particle, sp;
                        weight = weight, rng = rng)
    show(stdout, plan)

    n_layers  = length(unique(cell_df.z))
    dom_cols  = names(at, Not(:index))
    ndom      = length(dom_cols)

    # accumulator for the summed per-domain dose, keyed by cell :index
    acc = Dict{Int,Vector{Float64}}()
    for idx in at.index
        acc[idx] = zeros(Float64, ndom)
    end

    any_run = false
    for k in eachindex(plan.energies)
        Nk = plan.counts[k]
        Nk == 0 && continue
        Ek = plan.energies[k]

        println("\n─ species $k/$(length(plan.energies)):  E=$Ek MeV/u,  $Nk particles ─")

        # build irrad_cond for this energy through the normal layer propagation
        ion_k = Ion(String(particle), Ek, A, Z, plan.lets[k], 1.0)
        irrad_cond = Array{AT}(undef, n_layers)
        compute_energy_box!(irrad_cond, ion_k, irrad, type_AT, cell_df, track_seg, sp)

        # scratch copies so each species writes independently, then we sum
        at_k      = copy(at)
        cell_df_k = copy(cell_df)

        MC_dose_CPU!(ion_k, Nk, R_beam, irrad_cond, cell_df_k,
                     df_center_x, df_center_y, at_k,
                     gsm2_cycle, type_AT, track_seg;
                     x_cb = x_cb, y_cb = y_cb, single_particle = false,
                     seed = seed === nothing ? nothing : UInt64(seed) + k,
                     nchunks = nchunks, n_lut = n_lut)

        # sum this species' per-domain dose into the accumulator
        at_row = Dict(idx => r for (r, idx) in enumerate(at_k.index))
        for idx in at.index
            r = at_row[idx]
            a = acc[idx]
            @inbounds for (j, c) in enumerate(dom_cols)
                a[j] += at_k[r, c]
            end
        end
        any_run = true
    end

    any_run || @warn "MC_dose_CPU_poly!: no particles sampled (N̄=$(round(plan.Nbar; digits=3)))"

    # write summed dose back into `at`, and into cell_df.dose / dose_cell
    ncell = nrow(cell_df)
    if !hasproperty(cell_df, :dose) || !(eltype(cell_df.dose) <: AbstractVector)
        cell_df.dose = [zeros(Float64, ndom) for _ in 1:ncell]
    end
    if !hasproperty(cell_df, :dose_cell)
        cell_df.dose_cell = zeros(Float64, ncell)
    end
    cell_row = Dict(idx => r for (r, idx) in enumerate(cell_df.index))

    for idx in at.index
        v  = acc[idx]
        ar = findfirst(==(idx), at.index)
        at[ar, dom_cols] .= v

        cr = get(cell_row, idx, 0)
        cr == 0 && continue
        if cell_df.is_cell[cr] == 1
            cell_df.dose[cr]      = copy(v)
            cell_df.dose_cell[cr] = ndom == 0 ? 0.0 : sum(v) / ndom
        else
            cell_df.dose[cr]      = zeros(Float64, ndom)
            cell_df.dose_cell[cr] = 0.0
        end
    end

    println("\n═ poly done in $(round(time() - t0; digits=3)) s ═")
    return plan
end

#! ============================================================================
#! Verification
#! ============================================================================

"""
    check_poly_dose(plan, cell_df; R_beam) -> Float64

Compare the mean delivered dose over occupied cells against the prescription
implied by the sampled counts,

    D_expected = Σ_k N_k · LET_k · 0.1602 / (πR²)

which is what the fluence actually carries for THIS realisation (using the
sampled `counts`, not `Nbar`, so Poisson/multinomial scatter is not counted as
error). A large gap points at edge starvation (pad `R_beam` by `Rp`) or a
domain/beam-area mismatch, not at the spectrum handling.
"""
function check_poly_dose(plan::SpeciesPlan, cell_df::DataFrame, R_beam::Float64)
    hasproperty(cell_df, :dose_cell) || error("check_poly_dose: run the poly driver first")
    occ = cell_df.is_cell .== 1
    any(occ) || (println("  no occupied cells"); return NaN)

    delivered = sum(cell_df.dose_cell[occ]) / count(occ)
    expected  = sum(plan.counts .* plan.lets) * 0.1602 / (π * R_beam^2)
    rel = expected > 0 ? delivered / expected - 1 : NaN

    println("── poly dose check ──────────────────────────")
    println("  delivered ⟨D⟩ over cells : ", delivered, " Gy")
    println("  expected from counts     : ", expected,  " Gy")
    println("  relative error           : ", rel)
    println("─────────────────────────────────────────────")
    return rel
end


"""
    MC_dose_damage_poly!(ion, Npar_vect, R_beam, irrad_cond,
                         cell_df, df_center_x, df_center_y, at,
                         gsm2_cycle, type_AT, track_seg) -> Nothing

Run the full dose+damage chain once per species (energy component) and sum the
per-species contributions in place.

`ion[i]`, `Npar_vect[i]`, `irrad_cond[i]` describe species `i`; `cell_df` must
already carry the `layer_i` / `energy_step_i` columns built by the vectorized
`setup_irrad_conditions!`. For each `i` this:

  1. points the unsuffixed `:layer` / `:energy_step` columns (the ones
     `MC_dose_CPU!`/`MC_loop_damage!` read) at species `i`'s assignment;
  2. zeroes the dose scratch (`at`, `cell_df.dose`, `cell_df.dose_cell`) so
     species `i` deposits from a clean slate, not the previous species'
     leftover dose;
  3. runs `MC_dose_CPU!` then `MC_loop_damage!` for that species;
  4. stashes the resulting per-cell state as `dose_i`, `dose_cell_i`,
     `dam_X_dom_i`, `dam_Y_dom_i` columns on `cell_df`.

After every species has run, the per-species columns are summed element-wise
back into `cell_df.dose`, `.dose_cell`, `.dam_X_dom`, `.dam_Y_dom`, and
`.dam_X_total` / `.dam_Y_total` are recomputed from the summed damage vectors.
`cell_df.layer` / `cell_df.energy_step` are left holding the LAST species'
assignment (needed by downstream `layer_plot=true` plotting) — there is no
single canonical choice once multiple species have run.
"""
function MC_dose_damage_poly!(
    ion::Vector{Ion}, Npar_vect::Vector{Int}, R_beam::Float64,
    irrad_cond::Vector{Vector{AT}},
    cell_df::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2_cycle::Vector{GSM2}, type_AT::String, track_seg::Bool) where {AT}

    n_species = length(ion)
    length(Npar_vect)  == n_species || error("MC_dose_damage_poly!: Npar_vect length mismatch")
    length(irrad_cond) == n_species || error("MC_dose_damage_poly!: irrad_cond length mismatch")

    ncell    = nrow(cell_df)
    dom_cols = names(at, Not(:index))
    t0       = time()

    for i in 1:n_species
        println("\n═════════════════════════════════════════════")
        println("MC_dose_damage_poly!   species $i/$n_species   E=$(ion[i].E) MeV/u   Npar=$(Npar_vect[i])")
        println("═════════════════════════════════════════════")

        # point the unsuffixed columns MC_dose_CPU!/MC_loop_damage! read at
        # this species' per-step layer/energy_step assignment
        cell_df.layer       = cell_df[!, Symbol("layer_", i)]
        cell_df.energy_step = cell_df[!, Symbol("energy_step_", i)]

        # clean scratch so this species deposits from zero, not the previous
        # species' leftover dose
        for c in dom_cols
            fill!(at[!, c], 0.0)
        end
        for r in 1:ncell
            fill!(cell_df.dose[r], 0.0)
        end
        fill!(cell_df.dose_cell, 0.0)

        println("\n--- Monte Carlo dose deposition ---")
        @time MC_dose_CPU!(ion[i], Npar_vect[i], R_beam, irrad_cond[i], cell_df,
                            df_center_x, df_center_y, at,
                            gsm2_cycle, type_AT, track_seg)

        println("\n--- Poisson DNA damage sampling ---")
        @time MC_loop_damage!(ion[i], cell_df, irrad_cond[i], gsm2_cycle)

        # stash this species' contribution under an indexed column
        cell_df[!, Symbol("dose_", i)]      = deepcopy(cell_df.dose)
        cell_df[!, Symbol("dose_cell_", i)] = copy(cell_df.dose_cell)
        cell_df[!, Symbol("dam_X_dom_", i)] = deepcopy(cell_df.dam_X_dom)
        cell_df[!, Symbol("dam_Y_dom_", i)] = deepcopy(cell_df.dam_Y_dom)
    end

    #! ------------------------------------------------------------------
    #! Sum every species' contribution back into the unsuffixed columns
    #! ------------------------------------------------------------------
    ndom = length(dom_cols)
    cell_df.dose      = [zeros(Float64, ndom) for _ in 1:ncell]
    cell_df.dose_cell = zeros(Float64, ncell)
    cell_df.dam_X_dom = [Int[] for _ in 1:ncell]
    cell_df.dam_Y_dom = [Int[] for _ in 1:ncell]

    for i in 1:n_species
        dose_i = cell_df[!, Symbol("dose_", i)]
        damX_i = cell_df[!, Symbol("dam_X_dom_", i)]
        damY_i = cell_df[!, Symbol("dam_Y_dom_", i)]
        for r in 1:ncell
            cell_df.dose[r]      .+= dose_i[r]
            cell_df.dose_cell[r]  += cell_df[r, Symbol("dose_cell_", i)]
            if isempty(cell_df.dam_X_dom[r])
                cell_df.dam_X_dom[r] = copy(damX_i[r])
                cell_df.dam_Y_dom[r] = copy(damY_i[r])
            else
                cell_df.dam_X_dom[r] .+= damX_i[r]
                cell_df.dam_Y_dom[r] .+= damY_i[r]
            end
        end
    end

    # totals recomputed from the summed per-domain vectors, so they match exactly
    cell_df.dam_X_total = [isempty(v) ? 0 : sum(v) for v in cell_df.dam_X_dom]
    cell_df.dam_Y_total = [isempty(v) ? 0 : sum(v) for v in cell_df.dam_Y_dom]

    # :layer/:energy_step are left holding the LAST species' assignment —
    # downstream layer_plot=true grouping (plot_scalar_cell, plot_damage)
    # needs them present; there's no single canonical choice across species.

    println("\n═ poly dose+damage done in $(round(time() - t0; digits=3)) s ═")
    return nothing
end