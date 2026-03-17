#! ============================================================================
#! utilities_ABM.jl
#!
#! FUNCTIONS
#! ---------
#~ Radiation Damage & Repair
#?   compute_times_domain!(cell_df, gsm2_cycle; nat_apo, terminal_time, verbose, print_every, summary)
#       Parallel per-cell GSM2 survival + stochastic repair scheduling.
#       Updates sp, death_time, recover_time, cycle_time, is_cell in place.
#       Dispatches gsm2 by cell_cycle phase. Deferred neighbor updates applied serially.
#?   compute_repair_domain(X, Y, gsm2; terminal_time, rng, verbose, max_events_log)
#           -> (death_time, recover_time, death_code, X_final, Y_final)
#       Gillespie simulation of X-lesion repair under GSM2 rates (a, b, r).
#       death_code: 1=lethal, 0=recovered, -1=timeout.
#       Immediate lethal if any Y>0; immediate recovery if sum(X)==0.
#
#~ Cell Cycle Helpers
#?   generate_cycle_time(phase) -> Float64
#       Samples a Gamma-distributed cycle duration from PHASE_DURATIONS[phase].
#?   assign_random_phase() -> String
#       Returns "G1"/"S"/"G2"/"M" weighted by phase duration (12/8/3/1 hours).
#       G0 is never assigned.
#
#~ CellPopulation ↔ DataFrame Conversion
#?   CellPopulation(df::DataFrame) -> CellPopulation
#       Constructs a SoA CellPopulation from a row-oriented DataFrame.
#       Optional columns: :is_stem, :is_death_rad, :x, :y.
#?   to_dataframe(pop; alive_only=false) -> DataFrame
#       Converts CellPopulation back to DataFrame. alive_only=true filters is_cell==1.
#?   update_dataframe!(df, pop) -> Nothing
#       Writes final CellPopulation state back into an existing DataFrame in place.
#
#~ Lattice / Neighbor Utilities
#?   is_valid_index(idx, n) -> Bool
#       Bounds check: 1 ≤ idx ≤ n.
#?   is_time_due(t, eps, treat_neg) -> Bool
#       Returns true if time t has elapsed (within eps, with optional negative-time support).
#?   recount_empty_neighbors(pop, cell_idx) -> Int16
#       Counts empty (is_cell==0) neighbors for a cell. Ground-truth recount.
#?   recalculate_number_nei!(df) -> Nothing
#       Recomputes number_nei for all cells in a DataFrame from scratch.
#
#~ ABM Event Handlers
#?   _perform_division!(pop, parent_idx, nat_apo) -> Nothing
#       Divides parent into first empty neighbor. Resets both to G1.
#       Updates number_nei and can_divide for all affected neighbors.
#?   _handle_cell_removal!(pop, removed_idx, is_natural_apoptosis) -> Nothing
#       Marks cell dead, decrements n_alive, unblocks neighbors (sets G1 + cycle_time).
#       Respects recover_time when computing new cycle_time for unblocked neighbors.
#
#~ ABM Time Stepping
#?   compute_next_event(pop) -> (Float64, Int32, String)
#       Returns (min_time, cell_idx, event_name) across all alive cells.
#       event_name is "death_time" or "cycle_time".
#?   update_time!(pop, elapsed) -> Nothing
#       Subtracts elapsed from death_time, cycle_time, recover_time for all alive cells.
#       Clamps at 0.
#?   update_ABM!(pop, next_time, event, idx, nat_apo) -> Nothing
#       Advances time, processes one event (death or cycle transition/division),
#       sweeps for any other cells whose death_time hit 0.
#?   check_time!(pop, nat_apo; eps=0.0) -> Nothing
#       Post-step cleanup: removes cells with death_time≈0, clears expired recover/cycle times.
#
#~ Simulation Recording
#?   record_timepoint!(ts, t, pop) -> Nothing
#       Appends current state to SimulationTimeSeries (total, G0/G1/S/G2/M, stem counts).
#?   print_initial_stats(pop) -> Nothing
#       Prints n_alive to stdout. Lightweight pre-run header.
#?   count_phase_alive(df; phase_col, phases) -> Dict{Symbol,Int}
#       Returns Dict(phase=>count) for each phase in phases.
#
#~ Top-Level Simulation Runner
#?   run_simulation_abm!(pop; nat_apo, terminal_time, snapshot_times, print_interval, verbose)
#           -> (ts::SimulationTimeSeries, snapshots::Dict{Int, CellPopulation})
#       Main event loop on CellPopulation. Stops at terminal_time or no more events.
#       Snapshots taken at floor(t) ∈ snapshot_times (each hour captured once).
#?   run_simulation_abm!(cell_df; nat_apo, terminal_time, return_dataframes, update_input, kwargs...)
#           -> (ts, snapshots::Dict{Int, DataFrame|CellPopulation})
#       DataFrame wrapper: AoS→SoA, runs simulation, optionally writes back and converts snapshots.
#! ============================================================================

"""
    compute_times_domain!(cell_df, gsm2_cycle;
                            nat_apo=1e-10,
                            terminal_time=Inf, verbose=false, print_every=0, summary=true)
        -> Nothing

Parallel per-cell GSM2 survival + stochastic repair scheduling for all active
cells (`is_cell==1`). Updates `cell_df` in place.

For each cell:
1. Computes `sp` via `domain_GSM2`.
2. Runs `compute_repair_domain` → one of four outcomes:
   - **Immediate removal** (`death_time==0`) → `is_cell=0`, deferred neighbor update.
   - **Radiogenic death** (finite `death_time`) → schedules death; clears damage arrays.
   - **Survivor** (finite `recover_time`) → schedules recovery + phase-dependent cycle_time.
   - **Timeout** (`death_time=Inf, recover_time=Inf`) → writes back residual X/Y damage.

GSM2 model selected per phase: G1/G0→gsm2_cycle[1], S→gsm2_cycle[2], G2/M→gsm2_cycle[3].
Neighbor updates are deferred per-thread and merged serially at the end.

Required columns: `:index, :is_cell, :dam_X_dom, :dam_Y_dom, :sp, :apo_time,
:death_time, :recover_time, :cycle_time, :is_death_rad, :death_type,
:cell_cycle, :is_stem, :number_nei, :nei`.

# Example
```julia
compute_times_domain!(cell_df, gsm2_cycle, 0.01; terminal_time=24.0, summary=true)
```
"""
function compute_times_domain!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
                                nat_apo::Float64 = 1e-10,
                                terminal_time::Float64 = Inf,
                                verbose::Bool = false, print_every::Int = 0, summary::Bool = true)

    (0.0 < nat_apo < 1.0) || error("nat_apo must be in (0, 1); got $nat_apo")

    λ     = -log(nat_apo)
    inv_λ = 1.0 / λ

    active_cells = @view cell_df.index[cell_df.is_cell .== 1]
    n_active     = length(active_cells)

    # Reset output columns
    cell_df.sp          .= 1.0
    cell_df.apo_time    .= Inf
    cell_df.death_time  .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time  .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type  .= -1

    if n_active == 0
        println("[compute_times_domain!] No active cells. Nothing to do.")
        return nothing
    end

    start_t    = time()
    max_threads = Threads.maxthreadid()
    println("[compute_times_domain!] Starting | active=$n_active | threads=$(Threads.nthreads()) | nat_apo=$nat_apo")

    neighbor_updates = [Int[] for _ in 1:max_threads]
    plock            = ReentrantLock()

    immediate_removed = Threads.Atomic{Int}(0)
    scheduled_death   = Threads.Atomic{Int}(0)
    survived          = Threads.Atomic{Int}(0)
    timeout_cells     = Threads.Atomic{Int}(0)

    for k in eachindex(active_cells)
        i   = active_cells[k]
        tid = Threads.threadid()
        tid <= max_threads || error("Thread ID $tid exceeds max_threads $max_threads")

        # Select GSM2 by phase
        phase = cell_df.cell_cycle[i]
        gsm2  = if phase == "G1" || phase == "G0"
            gsm2_cycle[1]
        elseif phase == "S"
            gsm2_cycle[2]
        elseif phase == "G2" || phase == "M"
            gsm2_cycle[3]
        else
            println("Unknown phase '$phase' at cell $i → using gsm2_cycle[4]")
            gsm2_cycle[4]
        end

        # 1) GSM2 survival probability
        SP_cell        = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)
        cell_df.sp[i]  = SP_cell

        # 2) Stochastic repair
        death_time, recover_time_sample, death_type, X_gsm2, Y_gsm2 =
            compute_repair_domain(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2;
                                    terminal_time=terminal_time)

        # ── Immediate removal ────────────────────────────────────────────────
        if death_time == 0.0
            cell_df.is_cell[i]    = 0
            cell_df.death_type[i] = death_type
            cell_df.apo_time[i]   = Inf
            append!(neighbor_updates[tid], cell_df.nei[i])
            Threads.atomic_add!(immediate_removed, 1)
            verbose && print_every > 0 && k % print_every == 0 &&
                lock(plock) do
                    println("[k=$k|cell=$i|thr=$tid] SP=$(round(SP_cell,digits=4)) | immediate removal | type=$death_type")
                end
            continue
        end

        # ── Radiation death scheduled ───────────────────────────────────────
        if isfinite(death_time)
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i]   = Inf
            cell_df.death_type[i]   = death_type
            cell_df.dam_X_dom[i]  .*= 0
            cell_df.dam_Y_dom[i]  .*= 0

            cell_df.death_time[i] = (phase == "M" && cell_df.number_nei[i] > 0) ?
                min(death_time, rand(Gamma(2*1., 0.5))) : death_time

            Threads.atomic_add!(scheduled_death, 1)
            verbose && print_every > 0 && k % print_every == 0 &&
                lock(plock) do
                    println("[k=$k|cell=$i|thr=$tid] SP=$(round(SP_cell,digits=4)) | death=$(round(cell_df.death_time[i],digits=4)) | type=$death_type")
                end

        # ── Survivor ─────────────────────────────────────────────────────────
        elseif isfinite(recover_time_sample)
            cell_df.death_time[i]   = Inf
            cell_df.recover_time[i] = recover_time_sample
            cell_df.death_type[i]   = death_type
            cell_df.dam_X_dom[i]  .*= 0
            cell_df.dam_Y_dom[i]  .*= 0

            if cell_df.number_nei[i] > 0
                if phase == "M"
                    ct = rand(Gamma(2*1., 0.5))
                    if ct < recover_time_sample
                        cell_df.recover_time[i] = Inf
                        cell_df.cycle_time[i]   = Inf
                        cell_df.death_time[i]   = ct
                    else
                        cell_df.cycle_time[i] = ct
                    end
                elseif phase == "G2"
                    ct = rand(Gamma(2*5, 0.5))
                    cell_df.cycle_time[i]   = max(ct, recover_time_sample)
                elseif phase == "S"
                    ct = rand(Gamma(2*7, 0.5))
                    cell_df.cycle_time[i]   = max(ct, recover_time_sample)
                elseif phase == "G1"
                    ct = rand(Gamma(2*11, 0.5))
                    cell_df.cycle_time[i]   = max(ct, recover_time_sample)
                end
            else
                cell_df.cycle_time[i] = Inf
            end

            Threads.atomic_add!(survived, 1)
            verbose && print_every > 0 && k % print_every == 0 &&
                lock(plock) do
                    println("[k=$k|cell=$i|thr=$tid] SP=$(round(SP_cell,digits=4)) | survive | recover=$(recover_time_sample) | cycle=$(cell_df.cycle_time[i])")
                end

        # ── Timeout ───────────────────────────────────────────────────────────
        else
            cell_df.death_time[i]   = Inf
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i]   = Inf
            cell_df.death_type[i]   = death_type
            cell_df.dam_X_dom[i]    = X_gsm2
            cell_df.dam_Y_dom[i]    = Y_gsm2
            cell_df.dam_X_total[i]  = sum(X_gsm2)
            cell_df.dam_Y_total[i]  = sum(Y_gsm2)
            Threads.atomic_add!(timeout_cells, 1)
            verbose && print_every > 0 && k % print_every == 0 &&
                lock(plock) do
                    println("[k=$k|cell=$i|thr=$tid] SP=$(round(SP_cell,digits=4)) | timeout T=$terminal_time")
                end
        end

        cell_df.apo_time[i] = Inf
    end

    # Serial neighbor merge
    affected = Set{Int}()
    for updates in neighbor_updates, nei_idx in updates
        push!(affected, nei_idx)
    end
    for i in affected
        cell_df.number_nei[i] = length(cell_df.nei[i]) - sum(cell_df.is_cell[cell_df.nei[i]])
    end

    if summary
        elapsed = time() - start_t
        mean_sp = mean(cell_df.sp[active_cells])
        println("\n[compute_times_domain!] Summary")
        println("  Active cells          : $n_active")
        println("  Immediate removals    : $(immediate_removed[])")
        println("  Scheduled death       : $(scheduled_death[])")
        println("  Survivors             : $(survived[])")
        println("  Timeout (T=$terminal_time) : $(timeout_cells[])")
        println("  Mean SP               : $(round(mean_sp, digits=6))")
        println("  Elapsed               : $(round(elapsed, digits=3)) s")
    end
    return nothing
end

"""
    compute_repair_domain(X, Y, gsm2; terminal_time=Inf, rng=default_rng(), verbose=false,
                            max_events_log=0)
        -> (death_time, recover_time, death_code, X_final, Y_final)

Gillespie simulation of X-lesion repair under GSM2 rates.
- Immediate lethal if any `Y[j] > 0` → `(0.0, Inf, 1, ...)`.
- Immediate recovery if `sum(X) == 0` → `(Inf, 0.0, 0, ...)`.
- Rates per domain j: repair=r·X[j], misrepair=a·X[j], interaction=b·X[j]·(X[j]-1).
- Time scaled by `au=4.0`. Stops at `terminal_time` → `(Inf, Inf, -1, ...)`.

`death_code`: 1=lethal, 0=recovered, -1=timeout. Inputs X, Y are not mutated.

# Example
```julia
dt, rt, code, Xf, Yf = compute_repair_domain(X, Y, gsm2; terminal_time=24.0)
```
"""
function compute_repair_domain(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2;
                                terminal_time::Float64   = Inf,
                                rng::AbstractRNG          = Random.default_rng(),
                                verbose::Bool             = false,
                                max_events_log::Int       = 0,
                                au::Float64               = 4.0)

    if any(>(0), Y)
        verbose && println("[compute_repair_domain] Y-lesion detected → immediate lethal.")
        return (0.0, Inf, 1, copy(X), copy(Y))
    end

    sum_X = sum(X)
    if sum_X == 0
        verbose && println("[compute_repair_domain] No X-lesions → recovered.")
        return (Inf, 0.0, 0, copy(X), copy(Y))
    end

    X_work = copy(X)
    Y_work = copy(Y)
    a, b, r  = gsm2.a, gsm2.b, gsm2.r
    n        = length(X_work)
    t        = 0.0
    printed  = 0

    rates_r = Vector{Float64}(undef, n)
    rates_a = Vector{Float64}(undef, n)
    rates_b = Vector{Float64}(undef, n)

    while sum_X > 0 && t < terminal_time
        # Build propensities
        a0 = 0.0
        @inbounds for i in 1:n
            xi = X_work[i]
            rr = r * xi
            ra = a * xi
            rb = xi > 1 ? b * xi * (xi - 1) : 0.0
            rates_r[i] = rr; rates_a[i] = ra; rates_b[i] = rb
            a0 += rr + ra + rb
        end

        if a0 <= 0.0
            verbose && println("[compute_repair_domain] a0=0 with X remaining → recovery.")
            return (Inf, t, 0, X_work, Y_work)
        end

        # Select reaction
        threshold  = rand(rng) * a0
        cumulative = 0.0
        reac_type  = 0
        reac_domain = 0

        @inbounds for i in 1:n
            cumulative += rates_r[i]
            if cumulative >= threshold; reac_type = 1; reac_domain = i; break; end
        end
        if reac_type == 0
            @inbounds for i in 1:n
                cumulative += rates_a[i]
                if cumulative >= threshold; reac_type = 2; reac_domain = i; break; end
            end
        end
        if reac_type == 0
            @inbounds for i in 1:n
                cumulative += rates_b[i]
                if cumulative >= threshold; reac_type = 3; reac_domain = i; break; end
            end
        end

        # Time step
        new_time = t + au * (-log(rand(rng)) / a0)
        if new_time >= terminal_time
            verbose && println("[compute_repair_domain] Timeout at t=$(round(t,digits=4))")
            return (Inf, Inf, -1, X_work, Y_work)
        end
        t = new_time

        # Apply reaction
        if reac_type == 1
            @inbounds X_work[reac_domain] -= 1
            sum_X -= 1
            verbose && (max_events_log <= 0 || printed < max_events_log) &&
                (printed += 1; println("[t=$(round(t,digits=4))] REPAIR domain $reac_domain → X=$(X_work[reac_domain]) remaining=$sum_X"))
        else
            verbose && (max_events_log <= 0 || printed < max_events_log) &&
                (printed += 1; println("[t=$(round(t,digits=4))] $(reac_type==2 ? "MISREPAIR" : "INTERACTION") domain $reac_domain → LETHAL"))
            return (t, Inf, 1, X_work, Y_work)
        end
    end

    # Loop exit: either all repaired or timeout
    if t >= terminal_time
        verbose && println("[compute_repair_domain] Timeout with sum_X=$sum_X remaining")
        return (Inf, Inf, -1, X_work, Y_work)
    end
    verbose && println("[compute_repair_domain] All X repaired at t=$(round(t,digits=4))")
    return (Inf, t, 0, X_work, Y_work)
end

#! ============================================================================
#! Cell Cycle Helpers
#! ============================================================================

"""
    generate_cycle_time(phase::String) -> Float64

Samples a Gamma-distributed cycle duration from `PHASE_DURATIONS[phase]`.
Requires globally defined `PHASE_DURATIONS` with `.shape` and `.scale` fields.
"""
@inline function generate_cycle_time(phase::String)::Float64
    params = PHASE_DURATIONS[phase]
    return rand(Gamma(params.shape, params.scale))
end

"""
    assign_random_phase() -> String

Returns "G1"/"S"/"G2"/"M" weighted by phase hours (12/8/3/1).
G0 is never assigned.
"""
@inline function assign_random_phase()::String
    ru = rand() * 24.0
    return ru <= 12.0 ? "G1" : ru <= 20.0 ? "S" : ru <= 23.0 ? "G2" : "M"
end

#! ============================================================================
#! CellPopulation ↔ DataFrame Conversion
#! ============================================================================

"""
    CellPopulation(df::DataFrame) -> CellPopulation

Constructs a SoA `CellPopulation` from a row-oriented DataFrame.
Optional columns: `:is_stem, :is_death_rad, :x, :y`.

# Example
```julia
pop = CellPopulation(cell_df)
```
"""
function CellPopulation(df::DataFrame)
    n = nrow(df)

    get_int8(name)  = hasproperty(df, name) ? Vector{Int8}(df[!, name])  : nothing
    get_int32(name) = hasproperty(df, name) ? Vector{Int32}(df[!, name]) : nothing

    return CellPopulation(
        Vector{Int8}(df.is_cell),
        Vector{Int8}(df.can_divide),
        get_int8(:is_stem),
        get_int8(:is_death_rad),
        Vector{Float64}(df.death_time),
        Vector{Float64}(df.cycle_time),
        Vector{Float64}(df.recover_time),
        [String7(string(s)) for s in df.cell_cycle],
        Vector{Int16}(df.number_nei),
        [Vector{Int32}(nei) for nei in df.nei],
        get_int32(:x),
        get_int32(:y),
        Int32(n),
        Int32(count(df.is_cell .== 1)),
        hasproperty(df, :index) ? Vector{Int32}(df.index) : Vector{Int32}(1:n)
    )
end

"""
    to_dataframe(pop::CellPopulation; alive_only=false) -> DataFrame

Converts CellPopulation SoA back to a DataFrame.
`alive_only=true` filters to rows where `is_cell==1`.
Optional columns added if present: `:is_stem, :is_death_rad, :x, :y`.

# Example
```julia
df = to_dataframe(pop; alive_only=true)
```
"""
function to_dataframe(pop::CellPopulation; alive_only::Bool = false)::DataFrame
    indices = alive_only ? findall(pop.is_cell .== 1) : eachindex(pop.is_cell)

    df = DataFrame(
        index       = pop.indices[indices],
        is_cell     = pop.is_cell[indices],
        can_divide  = pop.can_divide[indices],
        death_time  = pop.death_time[indices],
        cycle_time  = pop.cycle_time[indices],
        recover_time = pop.recover_time[indices],
        cell_cycle  = String.(pop.cell_cycle[indices]),
        number_nei  = pop.number_nei[indices],
        nei         = pop.nei[indices]
    )

    !isnothing(pop.is_stem)      && (df.is_stem      = pop.is_stem[indices])
    !isnothing(pop.is_death_rad) && (df.is_death_rad = pop.is_death_rad[indices])
    !isnothing(pop.x)            && (df.x            = pop.x[indices])
    !isnothing(pop.y)            && (df.y            = pop.y[indices])

    return df
end

"""
    update_dataframe!(df::DataFrame, pop::CellPopulation) -> Nothing

Writes final CellPopulation state back into an existing DataFrame in place.
Errors if row count mismatches. Updates optional columns if both sides have them.

# Example
```julia
update_dataframe!(cell_df, pop)
```
"""
function update_dataframe!(df::DataFrame, pop::CellPopulation)
    nrow(df) == pop.n_cells ||
        error("DataFrame has $(nrow(df)) rows but CellPopulation has $(pop.n_cells) cells")

    df.is_cell    .= pop.is_cell
    df.can_divide .= pop.can_divide
    df.death_time .= pop.death_time
    df.cycle_time .= pop.cycle_time
    df.recover_time .= pop.recover_time
    df.cell_cycle .= String.(pop.cell_cycle)
    df.number_nei .= pop.number_nei
    df.nei        .= pop.nei

    hasproperty(df, :is_stem)      && !isnothing(pop.is_stem)      && (df.is_stem      .= pop.is_stem)
    hasproperty(df, :is_death_rad) && !isnothing(pop.is_death_rad) && (df.is_death_rad .= pop.is_death_rad)
    hasproperty(df, :x)            && !isnothing(pop.x)            && (df.x            .= pop.x)
    hasproperty(df, :y)            && !isnothing(pop.y)            && (df.y            .= pop.y)

    return nothing
end

#! ============================================================================
#! Lattice / Neighbor Utilities
#! ============================================================================

@inline is_valid_index(idx::Int32, n::Int32)::Bool = 1 <= idx <= n

@inline function is_time_due(t::Float64, eps::Float64, treat_neg::Bool)::Bool
    (ismissing(t) || isinf(t)) && return false
    return treat_neg ? (t <= eps) : (abs(t) <= eps)
end

"""
    recount_empty_neighbors(pop, cell_idx) -> Int16

Counts neighbors with `is_cell==0`. Ground-truth recount — call after structural changes.
"""
function recount_empty_neighbors(pop::CellPopulation, cell_idx::Int32)::Int16
    n        = pop.n_cells
    occupied = Int16(0)
    @inbounds for nb in pop.nei[cell_idx]
        is_valid_index(nb, n) && (occupied += pop.is_cell[nb])
    end
    return Int16(length(pop.nei[cell_idx])) - occupied
end

"""
    recalculate_number_nei!(df::DataFrame) -> Nothing

Recomputes `number_nei` for all cells in a DataFrame from scratch.
Call this to fix any accumulated drift.
"""
function recalculate_number_nei!(df::DataFrame)
    for i in 1:nrow(df)
        df.number_nei[i] = length(df.nei[i]) - sum(df.is_cell[j] for j in df.nei[i])
    end
end

#! ============================================================================
#! ABM Event Handlers
#! ============================================================================

"""
    _perform_division!(pop, parent_idx, nat_apo) -> Nothing

Divides parent cell into first empty neighbor. Resets both to G1 with fresh
cycle times. Recounts `number_nei` for parent, daughter, and all their neighbors.
Blocks neighbors with `number_nei==0` (sets `cycle_time=Inf, can_divide=0`).

# Example
```julia
_perform_division!(pop, Int32(42), nat_apo)
```
"""
function _perform_division!(pop::CellPopulation, parent_idx::Int32, nat_apo::Float64)
    n               = pop.n_cells
    parent_neighbors = pop.nei[parent_idx]
    is_cell         = pop.is_cell
    daughter_idx    = Int32(0)

    @inbounds for n_idx in parent_neighbors
        if is_valid_index(n_idx, n) && is_cell[n_idx] == 0
            daughter_idx = n_idx; break
        end
    end

    daughter_idx == 0 && (@warn "Division attempted but no empty neighbor" parent=parent_idx; return nothing)

    # Place daughter
    @inbounds begin
        pop.is_cell[daughter_idx]    = 1
        pop.cell_cycle[daughter_idx] = String7("G1")
        pop.cycle_time[daughter_idx] = generate_cycle_time("G1")
        pop.death_time[daughter_idx] = Inf
        pop.recover_time[daughter_idx] = Inf
        pop.can_divide[daughter_idx] = 1
    end

    # Reset parent to G1
    @inbounds begin
        pop.cell_cycle[parent_idx]   = String7("G1")
        pop.cycle_time[parent_idx]   = generate_cycle_time("G1")
        pop.death_time[parent_idx]   = Inf
        pop.recover_time[parent_idx] = Inf
        pop.can_divide[parent_idx]   = 1
    end

    daughter_neighbors = pop.nei[daughter_idx]
    number_nei = pop.number_nei
    can_divide = pop.can_divide
    cycle_times = pop.cycle_time

    pop.number_nei[parent_idx]   = recount_empty_neighbors(pop, parent_idx)
    pop.number_nei[daughter_idx] = recount_empty_neighbors(pop, daughter_idx)

    for neighbors in (parent_neighbors, daughter_neighbors)
        @inbounds for n_idx in neighbors
            is_valid_index(n_idx, n) && is_cell[n_idx] == 1 &&
            n_idx != parent_idx && n_idx != daughter_idx || continue
            number_nei[n_idx] = recount_empty_neighbors(pop, n_idx)
            if number_nei[n_idx] == 0 && can_divide[n_idx] == 1
                cycle_times[n_idx] = Inf
                can_divide[n_idx]  = 0
            end
        end
    end

    pop.n_alive += 1
    return nothing
end

"""
    _handle_cell_removal!(pop, removed_idx, is_natural_apoptosis) -> Nothing

Marks cell dead (`is_cell=0`), decrements `n_alive`, clears timers.
For each alive neighbor: recounts `number_nei` and unblocks cells with
`number_nei>0 && can_divide==0` → resets to G1 (respects existing `recover_time`).

# Example
```julia
_handle_cell_removal!(pop, Int32(17), false)
```
"""
function _handle_cell_removal!(pop::CellPopulation, removed_idx::Int32,
                                is_natural_apoptosis::Bool)
    pop.is_cell[removed_idx] == 0 && return nothing

    @inbounds begin
        pop.is_cell[removed_idx]     = 0
        pop.death_time[removed_idx]  = Inf
        pop.cycle_time[removed_idx]  = Inf
        pop.recover_time[removed_idx] = Inf
        pop.can_divide[removed_idx]  = 0
        pop.number_nei[removed_idx]  = Int16(0)
    end

    is_natural_apoptosis && !isnothing(pop.is_death_rad) &&
        (pop.is_death_rad[removed_idx] = 0)

    pop.n_alive -= 1

    n           = pop.n_cells
    is_cell     = pop.is_cell
    number_nei  = pop.number_nei
    can_divide  = pop.can_divide
    cell_cycle  = pop.cell_cycle
    cycle_times = pop.cycle_time
    recover_times = pop.recover_time

    @inbounds for n_idx in pop.nei[removed_idx]
        is_valid_index(n_idx, n) && is_cell[n_idx] == 1 || continue
        number_nei[n_idx] = recount_empty_neighbors(pop, n_idx)
        if number_nei[n_idx] > 0 && can_divide[n_idx] == 0
            cell_cycle[n_idx] = String7("G1")
            new_ct = generate_cycle_time("G1")
            cycle_times[n_idx] = isinf(recover_times[n_idx]) ?
                new_ct : max(new_ct, recover_times[n_idx])
            pop.death_time[n_idx] = Inf
            can_divide[n_idx]     = 1
        end
    end
    return nothing
end

#! ============================================================================
#! ABM Time Stepping
#! ============================================================================

"""
    compute_next_event(pop::CellPopulation) -> (Float64, Int32, String)

Returns `(min_time, cell_idx, event_name)` across all alive cells.
`event_name` is `"death_time"` or `"cycle_time"`.
Returns `(Inf, 0, "")` if no finite events exist.

# Example
```julia
next_time, idx, event = compute_next_event(pop)
```
"""
function compute_next_event(pop::CellPopulation)::Tuple{Float64, Int32, String}
    min_time  = Inf
    min_idx   = Int32(0)
    min_event = ""

    is_cell    = pop.is_cell
    death_times = pop.death_time
    cycle_times = pop.cycle_time
    n = pop.n_cells

    @inbounds for i in Int32(1):n
        is_cell[i] == 0 && continue
        dt = death_times[i]
        if !isinf(dt) && dt < min_time
            min_time = dt; min_idx = i; min_event = "death_time"
        end
        ct = cycle_times[i]
        if !isinf(ct) && ct < min_time
            min_time = ct; min_idx = i; min_event = "cycle_time"
        end
    end
    return (min_time, min_idx, min_event)
end

"""
    update_time!(pop::CellPopulation, elapsed::Float64) -> Nothing

Subtracts `elapsed` from `death_time`, `cycle_time`, `recover_time` for all
alive cells. Clamps each at 0.

# Example
```julia
update_time!(pop, 0.5)
```
"""
function update_time!(pop::CellPopulation, elapsed::Float64)
    @assert elapsed >= 0 "elapsed must be non-negative"

    is_cell       = pop.is_cell
    death_times   = pop.death_time
    cycle_times   = pop.cycle_time
    recover_times = pop.recover_time
    n = pop.n_cells

    @inbounds for i in Int32(1):n
        is_cell[i] == 0 && continue
        dt = death_times[i];   !isinf(dt) && (death_times[i]   = max(0.0, dt - elapsed))
        ct = cycle_times[i];   !isinf(ct) && (cycle_times[i]   = max(0.0, ct - elapsed))
        rt = recover_times[i]; !isinf(rt) && (recover_times[i] = max(0.0, rt - elapsed))
    end
    return nothing
end

"""
    update_ABM!(pop, next_time, event, idx, nat_apo) -> Nothing

Advances time via `update_time!`, sweeps for other cells whose `death_time`
hit 0, then processes the main event:
- `"death_time"` → `_handle_cell_removal!(pop, idx, false)`
- `"cycle_time"`:
    - M + space → `_perform_division!`
    - M + no space → enter G0 (cycle_time=Inf, can_divide=0)
    - G1/S/G2 + space → transition to next phase, new cycle_time
    - G1/S/G2 + no space → enter G0

Phase transitions follow `PHASE_TRANSITION` dict.

# Example
```julia
update_ABM!(pop, next_time, "cycle_time", idx, nat_apo)
```
"""
function update_ABM!(pop::CellPopulation, next_time::Float64, event::String,
                        idx::Int32, nat_apo::Float64)
    update_time!(pop, next_time)

    # Sweep: remove cells whose death_time hit 0 (other than the main event cell)
    @inbounds for i in Int32(1):pop.n_cells
        pop.is_cell[i] == 0 && continue
        i == idx         && continue
        dt = pop.death_time[i]
        !isinf(dt) && dt == 0.0 && _handle_cell_removal!(pop, i, false)
    end

    # Main event
    if event == "death_time"
        _handle_cell_removal!(pop, idx, false)

    elseif event == "cycle_time"
        @inbounds begin
            current_phase = String(pop.cell_cycle[idx])
            has_space     = pop.number_nei[idx] > 0

            if current_phase == "M"
                if has_space
                    _perform_division!(pop, idx, nat_apo)
                else
                    pop.cell_cycle[idx]   = String7("G0")
                    pop.cycle_time[idx]   = Inf
                    pop.death_time[idx]   = Inf
                    pop.recover_time[idx] = Inf
                    pop.can_divide[idx]   = 0
                end
            else
                next_phase = PHASE_TRANSITION[current_phase]
                pop.cell_cycle[idx] = String7(next_phase)
                if has_space
                    pop.cycle_time[idx]   = generate_cycle_time(next_phase)
                    pop.death_time[idx]   = Inf
                    pop.recover_time[idx] = Inf
                    pop.can_divide[idx]   = 1
                else
                    pop.cell_cycle[idx]   = String7("G0")
                    pop.cycle_time[idx]   = Inf
                    pop.death_time[idx]   = Inf
                    pop.recover_time[idx] = Inf
                    pop.can_divide[idx]   = 0
                end
            end
        end
    end
    return nothing
end

"""
    check_time!(pop, nat_apo; eps=0.0) -> Nothing

Post-step cleanup: removes cells with `|death_time| ≤ eps`,
clears `recover_time` and `cycle_time` if `|t| ≤ eps`.

# Example
```julia
check_time!(pop, nat_apo; eps=1e-10)
```
"""
function check_time!(pop::CellPopulation, nat_apo::Float64; eps::Float64 = 0.0)
    is_cell       = pop.is_cell
    death_times   = pop.death_time
    cycle_times   = pop.cycle_time
    recover_times = pop.recover_time
    n = pop.n_cells

    @inbounds for i in Int32(1):n
        is_cell[i] == 0 && continue
        dt = death_times[i]
        if !isinf(dt) && abs(dt) <= eps
            _handle_cell_removal!(pop, i, false); continue
        end
        rt = recover_times[i]; !isinf(rt) && abs(rt) <= eps && (recover_times[i] = Inf)
        ct = cycle_times[i];   !isinf(ct) && abs(ct) <= eps && (cycle_times[i]   = Inf)
    end
    return nothing
end

#! ============================================================================
#! Simulation Recording
#! ============================================================================

"""
    record_timepoint!(ts::SimulationTimeSeries, t::Float64, pop::CellPopulation) -> Nothing

Appends current population state to `ts`: time, total alive, G0/G1/S/G2/M counts,
stem/non-stem counts (0 if `pop.is_stem` is nothing).

# Example
```julia
record_timepoint!(ts, t, pop)
```
"""
function record_timepoint!(ts::SimulationTimeSeries, t::Float64, pop::CellPopulation)
    push!(ts.time, t)
    push!(ts.total_cells, pop.n_alive)

    g0 = g1 = s = g2 = m = Int32(0)
    @inbounds for i in Int32(1):pop.n_cells
        if pop.is_cell[i] == 1
            ph = pop.cell_cycle[i]
            if     ph == "G0"; g0 += 1
            elseif ph == "G1"; g1 += 1
            elseif ph == "S";  s  += 1
            elseif ph == "G2"; g2 += 1
            elseif ph == "M";  m  += 1
            end
        end
    end
    push!(ts.g0_cells, g0); push!(ts.g1_cells, g1); push!(ts.s_cells, s)
    push!(ts.g2_cells, g2); push!(ts.m_cells, m)

    if !isnothing(pop.is_stem)
        stem = non_stem = Int32(0)
        @inbounds for i in Int32(1):pop.n_cells
            if pop.is_cell[i] == 1
                pop.is_stem[i] == 1 ? (stem += 1) : (non_stem += 1)
            end
        end
        push!(ts.stem_cells, stem); push!(ts.non_stem_cells, non_stem)
    else
        push!(ts.stem_cells, Int32(0)); push!(ts.non_stem_cells, Int32(0))
    end
end

"""
    print_initial_stats(pop::CellPopulation) -> Nothing

Prints `n_alive` to stdout as a pre-run header.

# Example
```julia
print_initial_stats(pop)
```
"""
function print_initial_stats(pop::CellPopulation)
    println("\n", "="^70)
    println("INITIAL STATE")
    println("="^70)
    println("Active cells: $(pop.n_alive)")
    println("="^70, "\n")
end

#! ============================================================================
#! Top-Level Simulation Runner
#! ============================================================================

"""
    run_simulation_abm!(pop::CellPopulation; nat_apo=1e-10, terminal_time=48.0,
                        snapshot_times=[1,6,12,24], print_interval=1.0, verbose=true)
        -> (ts::SimulationTimeSeries, snapshots::Dict{Int, CellPopulation})

Main event loop on `CellPopulation`. Mutates `pop` in place.
Records snapshot at `t=0` always; additional snapshots at `floor(t) ∈ snapshot_times`
(each hour captured once). Stops when no events remain or `t + next_time > terminal_time`.

# Example
```julia
ts, snaps = run_simulation_abm!(pop; terminal_time=72.0, snapshot_times=[6,24,72])
```
"""
function run_simulation_abm!(pop::CellPopulation;
                                nat_apo::Float64         = 1e-10,
                                terminal_time::Float64   = 48.0,
                                snapshot_times::Vector{Int} = [1, 6, 12, 24],
                                print_interval::Float64  = 1.0,
                                verbose::Bool            = true)
    verbose && print_initial_stats(pop)

    ts        = SimulationTimeSeries()
    snapshots = Dict{Int, CellPopulation}()

    verbose && println("Creating snapshot for t = 0h")
    snapshots[0] = create_snapshot(pop)

    t           = 0.0
    record_timepoint!(ts, t, pop)
    event_count   = Int32(0)
    next_print    = print_interval
    snap_set      = Set(snapshot_times)

    verbose && println("\n", "="^70, "\nSTARTING SIMULATION\n", "="^70)

    while t < terminal_time
        next_time, idx, event = compute_next_event(pop)
        (isinf(next_time) || t + next_time > terminal_time) && break

        t           += next_time
        event_count += 1

        if verbose && t >= next_print
            println("t=$(round(t,digits=2))h | $event | Cells=$(pop.n_alive) | Events=$event_count")
            next_print += print_interval
        end

        current_hour = round(Int, floor(t))
        if current_hour in snap_set && !haskey(snapshots, current_hour)
            verbose && println("  └─ Snapshot at t=$(current_hour)h")
            snapshots[current_hour] = create_snapshot(pop)
        end

        update_ABM!(pop, next_time, event, idx, nat_apo)
        record_timepoint!(ts, t, pop)
    end

    if verbose
        println("="^70, "\nSIMULATION COMPLETE\n", "="^70)
        println("Final time : $(round(t, digits=2))h")
        println("Events     : $event_count")
        println("Final cells: $(pop.n_alive)")
        println("="^70, "\n")
    end
    return (ts, snapshots)
end

"""
    run_simulation_abm!(cell_df::DataFrame; nat_apo=1e-10, terminal_time=48.0,
                        return_dataframes=true, update_input=true, kwargs...)
        -> (ts, snapshots::Dict{Int, DataFrame|CellPopulation})

DataFrame convenience wrapper: converts AoS→SoA, runs simulation, optionally
writes final state back to `cell_df`, and optionally converts snapshots to DataFrames.

`return_dataframes=true` → `Dict{Int, DataFrame}` (alive_only=true).
`update_input=true` → `cell_df` is mutated in place at the end.

# Example
```julia
ts, snaps = run_simulation_abm!(cell_df; terminal_time=72.0)
```
"""
function run_simulation_abm!(cell_df::DataFrame;
                                nat_apo::Float64         = 1e-10,
                                terminal_time::Float64   = 48.0,
                                return_dataframes::Bool  = true,
                                update_input::Bool       = true,
                                kwargs...)
    pop = CellPopulation(cell_df)
    ts, snapshots_soa = run_simulation_abm!(pop; nat_apo=nat_apo,
                                            terminal_time=terminal_time, kwargs...)

    update_input && update_dataframe!(cell_df, pop)

    if return_dataframes
        return (ts, Dict{Int,DataFrame}(k => to_dataframe(v, alive_only=true)
                                        for (k, v) in snapshots_soa))
    else
        return (ts, snapshots_soa)
    end
end


"""
    count_phase_alive(df::DataFrame;
                        phase_col::Symbol = :cell_cycle,
                        phases = ["G0","G1","S","G2","M"])

Return the number of **alive cells** in each cell-cycle phase.

A cell is considered *alive* if:
- `df.is_cell == 1`  (the agent is still present), and
- `df.death_time` is not finite (i.e. the model has not assigned a death time).

The function inspects the `phase_col` column (default: `:cell_cycle`)
and counts how many alive cells are in each of the phases listed in `phases`.
Phases that do not appear among alive cells are included with a count of `0`.

# Arguments
- `df::DataFrame`: the table containing the cell population.
- `phase_col::Symbol`: the column containing the cell-cycle phase (`"G0"`, `"G1"`, `"S"`, `"G2"`, `"M"`).
- `phases`: an ordered list that determines which phases are reported.

# Returns
A `Dict{String,Int}` mapping each phase to the number of alive cells. Example:

```julia
Dict("G0" => 12, "G1" => 58, "S" => 41, "G2" => 23, "M" => 6)
```
"""
function count_phase_alive(df::DataFrame; phase_col::Symbol = :cell_cycle,
                                            phases=["G0", "G1", "S", "G2", "M"])
                                            # Alive = present AND no finite death_time
    alive_mask = (df.is_cell .== 1) .& .!isfinite.(df.death_time)

    # Initialize result dictionary
    counts = Dict(p => 0 for p in phases)

    if !isempty(df)
        phase_vals = df[alive_mask, phase_col]
        for p in phase_vals
            if p in phases
                counts[p] += 1
            end
        end
    end

    return counts
end