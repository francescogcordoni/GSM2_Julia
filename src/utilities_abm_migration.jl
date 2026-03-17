#! ============================================================================
#! migration_ABM.jl
#!
#! Simple event-driven random walk migration.
#!
#! THEORY
#! ------
#! Each cell hops to a uniformly chosen empty neighbour at rate λ = D/h²,
#! where D [µm²/h] is the random-motility coefficient and h [µm] is the
#! lattice spacing (= cell diameter).  The waiting time to the next hop is
#!
#!     t_migrate ~ Exponential(1 / Λ),   Λ = n_empty · D/h²
#!
#! where n_empty is the number of currently empty neighbours.
#! This is the exact continuum limit of unbiased diffusion on a lattice.
#! If a cell has no empty neighbours Λ = 0 → t_migrate = ∞ (cell is stuck).
#!
#! Migration events compete in the same priority queue as division and death —
#! no fixed time step is introduced.
#!
#! USAGE
#! -----
#!   1.  include("migration_ABM.jl") after utilities_ABM.jl
#!   2.  Call run_simulation_abm_migration!(pop, D, h; ...) instead of
#!       run_simulation_abm!(pop; ...)
#!   3.  Set D = 0.0 to disable migration entirely (recovers original behaviour)
#!
#! FUNCTIONS
#! ---------
#?  sample_migration_time(D, h, n_empty) -> Float64
#?  schedule_migration!(mig_times, pop, i, D, h) -> Nothing
#?  schedule_all_migrations!(mig_times, pop, D, h) -> Nothing
#?  compute_next_event_migration(pop, mig_times) -> (Float64, Int32, String)
#?  update_time_migration!(pop, mig_times, elapsed) -> Nothing
#?  perform_migration!(pop, mig_times, idx, D, h) -> Nothing
#?  update_ABM_migration!(pop, mig_times, next_time, event, idx, nat_apo, D, h) -> Nothing
#?  run_simulation_abm_migration!(pop, D, h; kwargs...) -> (ts, snapshots)
#?  run_simulation_abm_migration!(cell_df, D, h; kwargs...) -> (ts, snapshots)
#! ============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sample migration time
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_migration_time(D, h, n_empty) -> Float64

Draw the next hop time from Exp(1/Λ) where Λ = n_empty · D/h².
Returns Inf when n_empty == 0 (cell surrounded).
"""
@inline function sample_migration_time(D::Float64, h::Float64,
                                        n_empty::Integer)::Float64
    n_empty == 0 && return Inf
    Λ = D / h^2 * n_empty
    return -log(rand()) / Λ
end


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Schedule migration for one cell
# ─────────────────────────────────────────────────────────────────────────────

"""
    schedule_migration!(mig_times, pop, i, D, h) -> Nothing

Resample mig_times[i] for cell i.
"""
@inline function schedule_migration!(mig_times::Vector{Float64},
                                      pop::CellPopulation,
                                      i::Int32,
                                      D::Float64, h::Float64)
    if pop.is_cell[i] == 1
        mig_times[i] = sample_migration_time(D, h, Int(pop.number_nei[i]))
    else
        mig_times[i] = Inf
    end
    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Initialise migration times for the whole population
# ─────────────────────────────────────────────────────────────────────────────

"""
    schedule_all_migrations!(mig_times, pop, D, h) -> Nothing

Set mig_times[i] for every alive cell. Call once after initialisation.
"""
function schedule_all_migrations!(mig_times::Vector{Float64},
                                   pop::CellPopulation,
                                   D::Float64, h::Float64)
    @inbounds for i in Int32(1):pop.n_cells
        schedule_migration!(mig_times, pop, i, D, h)
    end
    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Find the next event (death / cycle / migration)
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_next_event_migration(pop, mig_times) -> (Float64, Int32, String)

Drop-in replacement for compute_next_event that also considers migration.
event_name ∈ {"death_time", "cycle_time", "migration_time"}.
"""
function compute_next_event_migration(pop::CellPopulation,
                                       mig_times::Vector{Float64}
                                       )::Tuple{Float64, Int32, String}
    min_time  = Inf
    min_idx   = Int32(0)
    min_event = ""

    @inbounds for i in Int32(1):pop.n_cells
        pop.is_cell[i] == 0 && continue

        dt = pop.death_time[i]
        if !isinf(dt) && dt < min_time
            min_time = dt; min_idx = i; min_event = "death_time"
        end
        ct = pop.cycle_time[i]
        if !isinf(ct) && ct < min_time
            min_time = ct; min_idx = i; min_event = "cycle_time"
        end
        mt = mig_times[i]
        if !isinf(mt) && mt < min_time
            min_time = mt; min_idx = i; min_event = "migration_time"
        end
    end
    return (min_time, min_idx, min_event)
end


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Advance all timers (including migration)
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_time_migration!(pop, mig_times, elapsed) -> Nothing

Decrement death_time, cycle_time, recover_time, and mig_times by elapsed.
"""
function update_time_migration!(pop::CellPopulation,
                                 mig_times::Vector{Float64},
                                 elapsed::Float64)
    @assert elapsed >= 0.0
    @inbounds for i in Int32(1):pop.n_cells
        pop.is_cell[i] == 0 && continue
        dt = pop.death_time[i];   !isinf(dt) && (pop.death_time[i]   = max(0.0, dt - elapsed))
        ct = pop.cycle_time[i];   !isinf(ct) && (pop.cycle_time[i]   = max(0.0, ct - elapsed))
        rt = pop.recover_time[i]; !isinf(rt) && (pop.recover_time[i] = max(0.0, rt - elapsed))
        mt = mig_times[i];        !isinf(mt) && (mig_times[i]         = max(0.0, mt - elapsed))
    end
    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Execute a migration hop
# ─────────────────────────────────────────────────────────────────────────────

"""
    perform_migration!(pop, mig_times, idx, D, h) -> Nothing

Move cell idx to a uniformly chosen empty neighbour.
Updates neighbor counts, unblocks G0 cells that gain free space,
and resamples migration times for the moved cell and affected neighbors.
"""
function perform_migration!(pop::CellPopulation,
                             mig_times::Vector{Float64},
                             idx::Int32,
                             D::Float64, h::Float64)
    n = pop.n_cells

    # collect empty neighbours
    empty_neis = Int32[]
    @inbounds for nb in pop.nei[idx]
        is_valid_index(nb, n) && pop.is_cell[nb] == 0 && push!(empty_neis, nb)
    end

    if isempty(empty_neis)
        mig_times[idx] = Inf    # surrounded — wait until space opens
        return nothing
    end

    target = empty_neis[rand(1:length(empty_neis))]

    # copy cell state to target slot
    @inbounds begin
        pop.is_cell[target]      = 1
        pop.cell_cycle[target]   = pop.cell_cycle[idx]
        pop.cycle_time[target]   = pop.cycle_time[idx]
        pop.death_time[target]   = pop.death_time[idx]
        pop.recover_time[target] = pop.recover_time[idx]
        pop.can_divide[target]   = pop.can_divide[idx]
        mig_times[target]        = mig_times[idx]

        !isnothing(pop.is_stem)      && (pop.is_stem[target]      = pop.is_stem[idx])
        !isnothing(pop.is_death_rad) && (pop.is_death_rad[target] = pop.is_death_rad[idx])
    end

    # vacate source slot
    @inbounds begin
        pop.is_cell[idx]      = 0
        pop.death_time[idx]   = Inf
        pop.cycle_time[idx]   = Inf
        pop.recover_time[idx] = Inf
        pop.can_divide[idx]   = 0
        pop.number_nei[idx]   = Int16(0)
        mig_times[idx]        = Inf
    end

    # recount neighbors for target and all cells that share a neighborhood
    pop.number_nei[target] = recount_empty_neighbors(pop, target)

    affected = union(pop.nei[idx], pop.nei[target])
    @inbounds for nb in affected
        is_valid_index(nb, n) || continue
        pop.is_cell[nb] == 1  || continue
        nb == target           && continue

        pop.number_nei[nb] = recount_empty_neighbors(pop, nb)

        # unblock a G0 cell that now has free space
        if pop.number_nei[nb] > 0 && pop.can_divide[nb] == 0
            pop.cell_cycle[nb] = String7("G1")
            new_ct             = generate_cycle_time("G1")
            pop.cycle_time[nb] = isinf(pop.recover_time[nb]) ?
                                      new_ct : max(new_ct, pop.recover_time[nb])
            pop.death_time[nb] = Inf
            pop.can_divide[nb] = 1
        end

        schedule_migration!(mig_times, pop, nb, D, h)
    end

    # resample for the moved cell at its new position
    schedule_migration!(mig_times, pop, target, D, h)

    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main event dispatcher
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_ABM_migration!(pop, mig_times, next_time, event, idx, nat_apo, D, h)

Drop-in replacement for update_ABM!.
Handles death, cycle, and migration events.
"""
function update_ABM_migration!(pop::CellPopulation,
                                mig_times::Vector{Float64},
                                next_time::Float64,
                                event::String,
                                idx::Int32,
                                nat_apo::Float64,
                                D::Float64,
                                h::Float64)

    # advance all timers
    update_time_migration!(pop, mig_times, next_time)

    # sweep secondary deaths (same as original update_ABM!)
    @inbounds for i in Int32(1):pop.n_cells
        pop.is_cell[i] == 0 && continue
        i == idx              && continue
        dt = pop.death_time[i]
        if !isinf(dt) && dt == 0.0
            _handle_cell_removal!(pop, i, false)
            mig_times[i] = Inf
            for nb in pop.nei[i]
                is_valid_index(nb, pop.n_cells) && pop.is_cell[nb] == 1 &&
                    schedule_migration!(mig_times, pop, nb, D, h)
            end
        end
    end

    if event == "death_time"
        _handle_cell_removal!(pop, idx, false)
        mig_times[idx] = Inf
        for nb in pop.nei[idx]
            is_valid_index(nb, pop.n_cells) && pop.is_cell[nb] == 1 &&
                schedule_migration!(mig_times, pop, nb, D, h)
        end

    elseif event == "cycle_time"
        # reuse the original cycle handler (time already advanced to 0)
        update_ABM!(pop, 0.0, event, idx, nat_apo)
        # resample migration for idx and its neighbourhood
        schedule_migration!(mig_times, pop, idx, D, h)
        for nb in pop.nei[idx]
            is_valid_index(nb, pop.n_cells) && pop.is_cell[nb] == 1 &&
                schedule_migration!(mig_times, pop, nb, D, h)
        end

    elseif event == "migration_time"
        perform_migration!(pop, mig_times, idx, D, h)
    end

    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Top-level simulation runner (CellPopulation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_simulation_abm_migration!(pop, D, h; kwargs...) -> (ts, snapshots)

Drop-in replacement for run_simulation_abm! with random-walk migration.

# Arguments
- `D`  : random-motility coefficient [µm²/h].  Set D=0 to disable migration.
- `h`  : lattice spacing [µm]  (typically 2·R_cell)

All other keyword arguments are identical to run_simulation_abm!.
"""
function run_simulation_abm_migration!(pop::CellPopulation,
                                        D::Float64,
                                        h::Float64;
                                        nat_apo::Float64            = 1e-10,
                                        terminal_time::Float64      = 48.0,
                                        snapshot_times::Vector{Int} = [1, 6, 12, 24],
                                        print_interval::Float64     = 1.0,
                                        verbose::Bool               = true)

    # if migration is disabled fall back to the original runner
    if D == 0.0
        return run_simulation_abm!(pop;
                   nat_apo        = nat_apo,
                   terminal_time  = terminal_time,
                   snapshot_times = snapshot_times,
                   print_interval = print_interval,
                   verbose        = verbose)
    end

    mig_times = fill(Inf, pop.n_cells)
    schedule_all_migrations!(mig_times, pop, D, h)

    verbose && print_initial_stats(pop)
    verbose && @printf("Migration ON: D=%.1f µm²/h  h=%.1f µm  hop_rate=%.4f /h\n",
                        D, h, D/h^2)

    ts        = SimulationTimeSeries()
    snapshots = Dict{Int, CellPopulation}()
    snapshots[0] = create_snapshot(pop)

    t           = 0.0
    record_timepoint!(ts, t, pop)
    event_count = 0
    next_print  = print_interval
    snap_set    = Set(snapshot_times)

    verbose && println("\n", "="^70, "\nSTARTING SIMULATION (random-walk migration)\n", "="^70)

    while t < terminal_time
        next_time, idx, event = compute_next_event_migration(pop, mig_times)
        (isinf(next_time) || t + next_time > terminal_time) && break

        t           += next_time
        event_count += 1

        if verbose && t >= next_print
            println("t=$(round(t,digits=2))h | $event | cells=$(pop.n_alive) | events=$event_count")
            next_print += print_interval
        end

        hr = floor(Int, t)
        if hr in snap_set && !haskey(snapshots, hr)
            verbose && println("  └─ snapshot t=$(hr)h")
            snapshots[hr] = create_snapshot(pop)
        end

        update_ABM_migration!(pop, mig_times, next_time, event, idx, nat_apo, D, h)
        record_timepoint!(ts, t, pop)
    end

    if verbose
        println("="^70, "\nSIMULATION COMPLETE")
        println("  Final time : $(round(t, digits=2))h")
        println("  Events     : $event_count")
        println("  Final cells: $(pop.n_alive)")
        println("="^70)
    end
    return (ts, snapshots)
end


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DataFrame convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_simulation_abm_migration!(cell_df, D, h; kwargs...)

DataFrame wrapper: AoS → SoA → run → write back.
Mirrors the DataFrame signature of run_simulation_abm!.
"""
function run_simulation_abm_migration!(cell_df::DataFrame,
                                        D::Float64,
                                        h::Float64;
                                        nat_apo::Float64        = 1e-10,
                                        terminal_time::Float64  = 48.0,
                                        return_dataframes::Bool = true,
                                        update_input::Bool      = true,
                                        kwargs...)
    pop = CellPopulation(cell_df)
    ts, snapshots_soa = run_simulation_abm_migration!(pop, D, h;
                             nat_apo       = nat_apo,
                             terminal_time = terminal_time,
                             kwargs...)
    update_input && update_dataframe!(cell_df, pop)

    if return_dataframes
        return (ts, Dict{Int, DataFrame}(
            k => to_dataframe(v; alive_only=true) for (k, v) in snapshots_soa))
    else
        return (ts, snapshots_soa)
    end
end
