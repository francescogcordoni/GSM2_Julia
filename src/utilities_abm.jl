#! ============================================================================
#! utilities_abm.jl  —  single ABM utility file (fast event engine is the standard)
#!
#! Replaces utilities_abm.jl + utilities_abm_migration.jl + utilities_abm_fast.jl
#! + utilities_checkpoint.jl. In load_utilities.jl keep ONLY:
#!     include(joinpath(@__DIR__, "..", "src", "utilities_abm.jl"))
#! (after utilities_biology.jl; utilities_plot_abm.jl may follow).
#!
#! FUNCTIONS
#! ---------
#~ Radiation damage, repair and fate
#?   set_fate_model!(:legacy | :incremental; death_delay = true)
#       :legacy (DEFAULT) = previous behaviour used by the calibration scripts:
#       direct-Y cells removed at t = 0, lethal events at the SSA time, all
#       timers reset at each call. :incremental = only damaged, non-doomed cells
#       processed, doomed cells kept with delayed death, checkpoint modes.
#?   compute_times_domain!(cell_df, gsm2_cycle; nat_apo, terminal_time, verbose,
#?                         print_every, summary, model = FATE_MODEL[], checkpoint = CHECKPOINT[])
#?   set_checkpoint!(:all | :g2)
#       :all = damaged cells held in G1/S/G2 until repaired (default);
#       :g2  = only the G2/M checkpoint holds damaged cells (→ inverse dose-rate effect).
#?   sample_death_delay(phase), _doom_cell!(cell_df, i, t_commit, phase, type)
#?   set_death_delay!(; scale, shape)
#       Mean delay = scale × DEATH_DELAY_MEAN_CYCLES[phase] × cycle length (Gamma, shape k).
#       Defaults scale = 1, k = 3. Used by the :incremental fate model.
#?   compute_repair_domain(X, Y, gsm2; terminal_time, rng, verbose, max_events_log, au)
#           -> (death_time, recover_time, death_code, X_final, Y_final)
#       death_code: 1 = direct Y, 2 = misrepair (a), 3 = interaction (b), 0 = recovered, -1 = timeout.
#?   compute_repair_domain_trajectory(X, Y, gsm2; rng, au)
#
#~ Cell cycle
#?   generate_cycle_time(phase), assign_random_phase()
#?   set_division_limit!(cell_df; R50, w, center, rmax, rq), clear_division_limit!(),
#?   division_probability(idx), beyond_rq(idx)
#       rq = quiescence radius: a cell farther than rq from the centre enters G0 at its
#       next cycle event and is never unblocked by neighbours (default rq = Inf, off).
#?   _cycle_event!(pop, idx, nat_apo)        single source of truth for cycle events
#
#~ CellPopulation ↔ DataFrame, lattice utilities, event handlers
#?   CellPopulation(df), to_dataframe(pop), update_dataframe!(df, pop)
#?   is_valid_index, is_time_due, recount_empty_neighbors, recalculate_number_nei!
#?   _sync_block_state!, _perform_division!, _handle_cell_removal!
#
#~ Contact inhibition
#?   set_quiescence_threshold!(k; cell_df), apply_quiescence!(cell_df)
#       G0 when ≤ k of the 26 neighbours are empty (default k = 0: fully surrounded).
#
#~ Lineage tracking
#?   set_founder_tracking!(cell_df), clear_founder_tracking!(), founders()
#       Daughters inherit the founder of their parent; migration carries it.
#
#~ Migration (random walk on the lattice)
#?   sample_migration_time, schedule_migration!, schedule_all_migrations!,
#?   perform_migration!, compute_next_event_migration, update_ABM_migration!
#
#~ Simulation (fast engine = standard)
#?   run_simulation_abm!(pop | cell_df; nat_apo, terminal_time, snapshot_times,
#?                       print_interval, verbose, record_dt, fast = true)
#?   run_simulation_abm_migration!(pop | cell_df, D, h; ..., fast = true)
#       Heap-based engine: cost per event independent of population size.
#       fast = false → original loops (run_simulation_abm_original!,
#       run_simulation_abm_migration_original!), kept for validation.
#?   compare_engines(cell_df; seed, terminal_time, D, h) -> Bool
#
#~ Continuous irradiation
#?   add_uniform_damage!(cell_df, dD, K_tot, L_tot, rng)
#?   irradiate_continuous_abm!(cell_df, gsm2_cycle; dose, dose_rate_gyh, K_tot, L_tot,
#?                             checkpoint, max_pulse_dose, max_pulse_h, seed) -> (ts, summary)
#
#~ Recording
#?   record_timepoint!(ts, t, pop), print_initial_stats(pop), count_phase_alive(df)
#! ============================================================================

using Printf

#! ============================================================================

#! ---------------------------------------------------------------------------
#! CONSTANTS + HELPERS
#! ---------------------------------------------------------------------------

"""
Current mean cycle length (h) from PHASE_DURATIONS. Evaluated at call time, so the
death delays follow any change of the cycle length (e.g. set_cycle_length! in the
calibration scripts).
"""
current_cycle_length() = sum(PHASE_DURATIONS[p].shape * PHASE_DURATIONS[p].scale
                             for p in ("G1", "S", "G2", "M"))

"""
Shape of the Gamma death-delay distribution. k = 3 gives CV = 1/√3 ≈ 0.58,
i.e. a broad, right-skewed spread of death times, as seen in time-lapse data.
"""
const DEATH_DELAY_SHAPE = 3.0

"""
Mean delay between lethal commitment and removal, in units of T_CYCLE.
Cells committed in G2/M reach (abortive) mitosis soonest and die first;
quiescent G0 cells have no mitosis to trigger death and persist longest.
"""
const DEATH_DELAY_MEAN_CYCLES = Dict(
    "G2" => 1.0,
    "M"  => 1.0,
    "S"  => 1.25,
    "G1" => 1.5,
    "G0" => 2.5,
)

"""
    sample_death_delay(phase) -> Float64

Random delay (h) between lethal commitment and removal of a doomed cell.
"""
@inline function sample_death_delay(phase::AbstractString)::Float64
    DEATH_DELAY_ON[] || return 0.0
    k = DEATH_DELAY_K[]
    m = DEATH_DELAY_SCALE[] * get(DEATH_DELAY_MEAN_CYCLES, String(phase), 1.5) * current_cycle_length()
    m > 0 || return 0.0
    return rand(Gamma(k, m / k))
end

"""Multiplier of the mean death delay (1 = DEATH_DELAY_MEAN_CYCLES as given) and Gamma shape."""
const DEATH_DELAY_SCALE = Ref(1.0)
const DEATH_DELAY_K     = Ref(DEATH_DELAY_SHAPE)

"""
    set_death_delay!(; scale = 1.0, shape = 3.0) -> Nothing

Death delay of doomed cells (:incremental fate model): Gamma with shape `shape` and mean
`scale × DEATH_DELAY_MEAN_CYCLES[phase] × cycle length` (G2/M 1, S 1.25, G1 1.5, G0 2.5
cycles at scale 1). Global: set it before a threaded loop, never inside it.
"""
function set_death_delay!(; scale::Real = 1.0, shape::Real = DEATH_DELAY_SHAPE)
    (scale >= 0 && shape > 0) || error("need scale ≥ 0 and shape > 0")
    DEATH_DELAY_SCALE[] = Float64(scale); DEATH_DELAY_K[] = Float64(shape)
    return nothing
end

"""
    _doom_cell!(cell_df, i, t_commit, phase, death_type) -> Nothing

Marks cell `i` as doomed: removal at `t_commit + delay`, no further cycling,
no division (`can_divide = 0` also protects it from being unblocked by
`_sync_block_state!`, which requires `death_time == Inf`).
"""
@inline function _doom_cell!(cell_df::DataFrame, i::Int, t_commit::Float64,
                             phase::AbstractString, death_type::Int)
    cell_df.death_time[i]   = t_commit + sample_death_delay(phase)
    cell_df.cycle_time[i]   = Inf
    cell_df.recover_time[i] = Inf
    cell_df.can_divide[i]   = 0
    cell_df.is_death_rad[i] = 1
    cell_df.death_type[i]   = death_type
    cell_df.dam_X_dom[i]  .*= 0
    cell_df.dam_Y_dom[i]  .*= 0
    return nothing
end

#! ---------------------------------------------------------------------------
#! REPLACEMENT for compute_times_domain!
#! ---------------------------------------------------------------------------


#! ============================================================================
#! Checkpoint model
#! ============================================================================

const CHECKPOINT = Ref(:all)

"""Select the checkpoint model (:all = current, :g2 = G2/M checkpoint only)."""
function set_checkpoint!(mode::Symbol)
    mode in (:all, :g2) || error("checkpoint must be :all or :g2, got $mode")
    CHECKPOINT[] = mode
    println("[checkpoint] mode = $mode")
    return nothing
end

@inline _gsm2_for(phase, g) = (phase == "G1" || phase == "G0") ? g[1] :
                              phase == "S" ? g[2] : g[3]

@inline _fresh_phase_time(phase::AbstractString) =
    generate_cycle_time(phase == "G0" ? "G1" : String(phase))   # follows PHASE_DURATIONS


"""
Adds the bookkeeping columns used by compute_times_domain! / _doom_cell! if a
cell_df does not have them yet (the previous version created some of them
implicitly by resetting them at every call).
"""
function _ensure_fate_columns!(cell_df::DataFrame)
    n = nrow(cell_df)
    for (col, val) in ((:sp, 1.0), (:apo_time, Inf), (:death_time, Inf), (:recover_time, Inf),
                       (:cycle_time, Inf), (:cycle_hold, Inf), (:death_type, -1),
                       (:is_death_rad, 0), (:dam_X_total, 0), (:dam_Y_total, 0))
        hasproperty(cell_df, col) || (cell_df[!, col] = fill(val, n))
    end
    return nothing
end

"""
Legacy fate assignment (the behaviour all existing calibration scripts were built on):
resets the timers of every alive cell, removes cells with direct Y lesions at t = 0
(`is_cell = 0`), schedules other lethal events at the SSA time (no extra delay),
gives survivors a fresh phase time with the hard-coded durations (G1 11 h, S 7 h,
G2 5 h, M 1 h), and holds unrepaired cells (timeout) with `cycle_time = Inf`.
"""
function _compute_times_domain_legacy!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
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

    @Threads.threads for k in eachindex(active_cells)
        i   = active_cells[k]
        tid = Threads.threadid()

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

            cell_df.death_time[i] = (phase == "M" && !_quiescent(cell_df.number_nei[i])) ?
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

            if !_quiescent(cell_df.number_nei[i])
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
                    cell_df.cycle_time[i]   = ct < recover_time_sample ? recover_time_sample : (ct + recover_time_sample)
                elseif phase == "S"
                    ct = rand(Gamma(2*7, 0.5))
                    cell_df.cycle_time[i]   = ct < recover_time_sample ? recover_time_sample : (ct + recover_time_sample)
                elseif phase == "G1"
                    ct = rand(Gamma(2*11, 0.5))
                    cell_df.cycle_time[i]   = ct < recover_time_sample ? recover_time_sample : (ct + recover_time_sample)
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

    # Serial neighbor merge — also syncs cell_cycle/can_divide, not just
    # number_nei: a neighbor that started G0 (fully packed at setup) can gain
    # its first-ever empty slot right here, from an immediate removal, before
    # the ABM loop (and _sync_block_state!) ever runs on it.
    affected = Set{Int}()
    for updates in neighbor_updates, nei_idx in updates
        push!(affected, nei_idx)
    end
    for i in affected
        cell_df.number_nei[i] = length(cell_df.nei[i]) - sum(cell_df.is_cell[cell_df.nei[i]])
        cell_df.is_cell[i] == 1 || continue
        if _quiescent(cell_df.number_nei[i]) && cell_df.can_divide[i] == 1
            cell_df.cell_cycle[i] = "G0"
            cell_df.cycle_time[i] = Inf
            cell_df.can_divide[i] = 0
        elseif !_quiescent(cell_df.number_nei[i]) && cell_df.can_divide[i] == 0 && isinf(cell_df.death_time[i])
            cell_df.cell_cycle[i] = "G1"
            new_ct = generate_cycle_time("G1")
            cell_df.cycle_time[i] = isinf(cell_df.recover_time[i]) ?
                new_ct : max(new_ct, cell_df.recover_time[i])
            cell_df.can_divide[i] = 1
        end
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

const FATE_MODEL = Ref(:legacy)

"""
    set_fate_model!(:legacy | :incremental; death_delay = true)

Selects how `compute_times_domain!` assigns fates.
- `:legacy` (default): previous behaviour, used by the calibration scripts
  (direct-Y cells removed at t = 0, lethal events at the SSA time, every alive
  cell's timers reset at each call). Delays such as tdetA/tdetB/tdetY are then
  added by the scripts themselves.
- `:incremental`: only alive, non-doomed, damaged cells are processed; other
  cells keep their timers; lethal cells are doomed with `_doom_cell!` (delayed
  death if `death_delay = true`, otherwise death at the commitment time); the
  checkpoint mode (`set_checkpoint!`) applies. Needed for repeated calls
  (continuous / fractionated irradiation) and for the :g2 checkpoint.
"""
const DEATH_DELAY_ON = Ref(true)
function set_fate_model!(model::Symbol; death_delay::Bool = true)
    model in (:legacy, :incremental) || error("fate model must be :legacy or :incremental, got $model")
    FATE_MODEL[] = model; DEATH_DELAY_ON[] = death_delay
    println("[fate] model = $model", model === :incremental ? "  (death delay $(death_delay ? "on" : "off"))" : "")
    return nothing
end

"""
    compute_times_domain!(cell_df, gsm2_cycle; nat_apo = 1e-10, terminal_time = Inf,
                          verbose = false, print_every = 0, summary = true,
                          model = FATE_MODEL[], checkpoint = CHECKPOINT[]) -> Nothing

GSM2 repair and fate for the cells of `cell_df` (see `set_fate_model!`).
death_type: 1 = direct Y (or mitotic death), 2 = misrepair a, 3 = interaction b.
"""
function compute_times_domain!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
                               nat_apo::Float64 = 1e-10, terminal_time::Float64 = Inf,
                               verbose::Bool = false, print_every::Int = 0, summary::Bool = true,
                               model::Symbol = FATE_MODEL[], checkpoint::Symbol = CHECKPOINT[])
    if model === :legacy
        checkpoint === :all || error("the :$checkpoint checkpoint needs set_fate_model!(:incremental)")
        return _compute_times_domain_legacy!(cell_df, gsm2_cycle; nat_apo = nat_apo,
                   terminal_time = terminal_time, verbose = verbose,
                   print_every = print_every, summary = summary)
    end
    return _compute_times_domain_incremental!(cell_df, gsm2_cycle; nat_apo = nat_apo,
               terminal_time = terminal_time, verbose = verbose, print_every = print_every,
               summary = summary, checkpoint = checkpoint)
end

"""
    _compute_times_domain_incremental!(cell_df, gsm2_cycle; ..., checkpoint) -> Nothing

Incremental fate assignment. Processes only cells that are alive, not already doomed, and carry damage.
`terminal_time` = repair window (Inf: resolve all damage). Outcomes:
- lethal event (or direct Y) → doomed via `_doom_cell!` (delayed death);
- repaired within the window → `recover_time`, cycle per checkpoint mode;
- not repaired within the window (timeout) → residual damage kept, cycle per mode.
"""
function _compute_times_domain_incremental!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
                               nat_apo::Float64 = 1e-10,
                               terminal_time::Float64 = Inf,
                               verbose::Bool = false, print_every::Int = 0,
                               summary::Bool = true,
                               checkpoint::Symbol = CHECKPOINT[])
    (0.0 < nat_apo < 1.0) || error("nat_apo must be in (0, 1); got $nat_apo")
    _ensure_fate_columns!(cell_df)
    cell_df.sp       .= 1.0               # diagnostics only, recomputed for damaged cells
    cell_df.apo_time .= Inf
    g2mode = checkpoint === :g2

    todo = Int[]
    @inbounds for i in cell_df.index
        cell_df.is_cell[i] == 1          || continue
        isfinite(cell_df.death_time[i])  && continue            # doomed: keep schedule
        (any(!iszero, cell_df.dam_X_dom[i]) || any(!iszero, cell_df.dam_Y_dom[i])) || continue
        push!(todo, i)
    end

    n_doom = Threads.Atomic{Int}(0); n_mit = Threads.Atomic{Int}(0)
    n_surv = Threads.Atomic{Int}(0); n_tout = Threads.Atomic{Int}(0)
    start_t = time()

    @Threads.threads for k in eachindex(todo)
        i     = todo[k]
        phase = cell_df.cell_cycle[i]
        gsm2  = _gsm2_for(phase, gsm2_cycle)
        cell_df.sp[i] = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

        dt, rt, dtype, Xf, Yf = compute_repair_domain(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i],
                                                      gsm2; terminal_time = terminal_time)
        # Remaining time of the current phase, if one is running or held
        old_ct = cell_df.cycle_time[i]
        rem = isfinite(old_ct) ? old_ct :
              isfinite(cell_df.cycle_hold[i]) ? cell_df.cycle_hold[i] : _fresh_phase_time(phase)
        has_space = !_quiescent(cell_df.number_nei[i])

        if isfinite(dt)
            # ── lethal commitment ──────────────────────────────────────────
            t_commit = (phase == "M" && has_space) ? min(dt, rem) : dt
            _doom_cell!(cell_df, i, t_commit, phase, dtype)
            cell_df.cycle_hold[i] = Inf
            Threads.atomic_add!(n_doom, 1)

        elseif isfinite(rt)
            # ── repaired within the window ─────────────────────────────────
            cell_df.death_time[i]   = Inf
            cell_df.recover_time[i] = rt
            cell_df.death_type[i]   = dtype
            cell_df.dam_X_dom[i]  .*= 0
            cell_df.dam_Y_dom[i]  .*= 0
            cell_df.cycle_hold[i]   = Inf
            if !has_space || phase == "G0"
                cell_df.cycle_time[i] = Inf
            elseif phase == "M"
                c = g2mode ? rem : _fresh_phase_time("M")
                if c < rt
                    _doom_cell!(cell_df, i, c, "M", 1); Threads.atomic_add!(n_mit, 1); continue
                end
                cell_df.cycle_time[i] = c
            elseif g2mode
                # G1/S continue at their running clock; G2 is held until repaired
                cell_df.cycle_time[i] = phase == "G2" ? max(rem, rt) : rem
            else
                ct = _fresh_phase_time(phase)                  # original :all rule
                cell_df.cycle_time[i] = ct < rt ? rt : ct + rt
            end
            Threads.atomic_add!(n_surv, 1)

        else
            # ── not repaired within the window: keep residual damage ──────
            cell_df.death_time[i]   = Inf
            cell_df.recover_time[i] = Inf
            cell_df.death_type[i]   = dtype
            cell_df.dam_X_dom[i]    = Xf
            cell_df.dam_Y_dom[i]    = Yf
            cell_df.dam_X_total[i]  = sum(Xf)
            cell_df.dam_Y_total[i]  = sum(Yf)
            if !g2mode || !has_space || phase == "G0"
                cell_df.cycle_time[i] = Inf                    # original :all rule (held)
            elseif phase == "G2"
                cell_df.cycle_hold[i] = rem                    # held at the G2 checkpoint
                cell_df.cycle_time[i] = Inf
            elseif phase == "M"
                if rem < terminal_time                         # mitosis ends with damage
                    _doom_cell!(cell_df, i, rem, "M", 1); Threads.atomic_add!(n_mit, 1); continue
                end
                cell_df.cycle_time[i] = rem
            else
                cell_df.cycle_time[i] = rem                    # G1/S keep cycling
            end
            Threads.atomic_add!(n_tout, 1)
        end
        cell_df.apo_time[i] = Inf
    end

    if summary
        println("[compute_times_domain! | $checkpoint] damaged=$(length(todo))  doomed=$(n_doom[])  " *
                "mitotic=$(n_mit[])  repaired=$(n_surv[])  pending=$(n_tout[])  " *
                "($(round(time() - start_t, digits = 2)) s)")
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
                                au::Float64               = 8.0)

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
            return (t, Inf, reac_type, X_work, Y_work)   # 2 = a (X→Y), 3 = b (X+X→Y)
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

"""
    compute_repair_domain_trajectory(X, Y, gsm2; rng=default_rng(), au=4.0)
        -> (times, X_traj, Y_traj, death_code)

Gillespie simulation of X-lesion repair under GSM2 rates, recording the full
temporal trajectory. The simulation runs until all X lesions are consumed
(sum(X) == 0), regardless of whether lethal events (misrepair, interaction) occur.

Reactions per domain j:
- Repair      (rate r·X[j]):          X[j] -= 1
- Misrepair   (rate a·X[j]):          X[j] -= 1, Y[j] += 1
- Interaction (rate b·X[j]·(X[j]-1)): X[j] -= 2, Y[j] += 1

At each reaction event the current time and the full X/Y state vectors are
appended to the trajectory. The initial state (t=0) is always the first entry.

Returns:
- `times::Vector{Float64}` — time of each recorded event (including t=0)
- `X_traj::Vector{Vector{Int}}` — X state at each event
- `Y_traj::Vector{Vector{Int}}` — Y state at each event
- `death_code::Int` — 0 = no Y accumulated (all repaired), 1 = Y > 0 (lethal)

Inputs X, Y are not mutated.
"""
function compute_repair_domain_trajectory(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2;
                                           rng::AbstractRNG = Random.default_rng(),
                                           au::Float64      = 8.0)

    times  = Float64[]
    X_traj = Vector{Int}[]
    Y_traj = Vector{Int}[]

    X_work = copy(X)
    Y_work = copy(Y)

    # Record initial state
    push!(times,  0.0)
    push!(X_traj, copy(X_work))
    push!(Y_traj, copy(Y_work))

    sum_X = sum(X_work)
    if sum_X == 0
        return (times, X_traj, Y_traj, any(>(0), Y_work) ? 1 : 0)
    end

    a, b, r = gsm2.a, gsm2.b, gsm2.r
    n       = length(X_work)
    t       = 0.0

    rates_r = Vector{Float64}(undef, n)
    rates_a = Vector{Float64}(undef, n)
    rates_b = Vector{Float64}(undef, n)

    while sum_X > 0
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
            break   # no reactions possible, exit with current state
        end

        # Select reaction
        threshold   = rand(rng) * a0
        cumulative  = 0.0
        reac_type   = 0
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

        # Advance time
        t += au * (-log(rand(rng)) / a0)

        # Apply reaction
        if reac_type == 1
            # Repair: one X removed
            @inbounds X_work[reac_domain] -= 1
            sum_X -= 1
        elseif reac_type == 2
            # Misrepair: one X consumed, one Y created
            @inbounds X_work[reac_domain] -= 1
            @inbounds Y_work[reac_domain] += 1
            sum_X -= 1
        else
            # Interaction: two X consumed, one Y created
            @inbounds X_work[reac_domain] -= 2
            @inbounds Y_work[reac_domain] += 1
            sum_X -= 2
        end

        push!(times,  t)
        push!(X_traj, copy(X_work))
        push!(Y_traj, copy(Y_work))
    end

    death_code = any(>(0), Y_work) ? 1 : 0
    return (times, X_traj, Y_traj, death_code)
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
#! Contact inhibition (quiescence threshold)
#! ============================================================================

"""
A cell is quiescent (G0, cannot divide) when it has at most
`QUIESCENCE_EMPTY_MAX[]` empty sites among its lattice neighbours (26 on the
cubic lattice). 0 (default) = original rule: G0 only when completely surrounded.
k > 0: cells with only a few holes around them stay quiescent, so the deaths
scattered through an irradiated spheroid do not reactivate the whole core.
"""
const QUIESCENCE_EMPTY_MAX = Ref(0)

@inline _quiescent(n_empty::Integer) = n_empty <= QUIESCENCE_EMPTY_MAX[]

"""
    set_quiescence_threshold!(k; cell_df = nothing) -> Nothing

Sets the threshold; if `cell_df` is given, re-applies it to the population
(see `apply_quiescence!`). Call before irradiation / simulation.
"""
function set_quiescence_threshold!(k::Integer; cell_df::Union{Nothing,DataFrame} = nothing)
    k >= 0 || error("threshold must be ≥ 0")
    QUIESCENCE_EMPTY_MAX[] = k
    println("[quiescence] G0 when empty neighbours ≤ $k")
    isnothing(cell_df) || apply_quiescence!(cell_df)
    return nothing
end

"""
    apply_quiescence!(cell_df) -> Nothing

Recounts empty neighbours and sets G0 / cycling according to the current
threshold for alive, non-doomed cells (the setup routine uses the k = 0 rule).
"""
function apply_quiescence!(cell_df::DataFrame)
    n = nrow(cell_df); nG0 = 0
    @inbounds for i in 1:n
        cell_df.is_cell[i] == 1 || continue
        e = 0
        for j in cell_df.nei[i]
            1 <= j <= n && cell_df.is_cell[j] == 0 && (e += 1)
        end
        cell_df.number_nei[i] = e
        isfinite(cell_df.death_time[i]) && continue
        if _quiescent(e)
            if cell_df.cell_cycle[i] != "G0"
                cell_df.cell_cycle[i] = "G0"; cell_df.cycle_time[i] = Inf; cell_df.can_divide[i] = 0
            end
            nG0 += 1
        elseif cell_df.cell_cycle[i] == "G0"
            cell_df.cell_cycle[i] = "G1"
            cell_df.cycle_time[i] = generate_cycle_time("G1"); cell_df.can_divide[i] = 1
        end
    end
    println("[quiescence] G0 cells: $nG0 / $(count(==(1), cell_df.is_cell))")
    return nothing
end

#! ============================================================================
#! Lineage (founder) tracking
#! ============================================================================

"""
Founder of the cell on each lattice site (0 = none). When active, a daughter
inherits its parent's founder and a migrating cell carries it to its new site.
Global setting: one tracked simulation at a time (the ABM loop is sequential).
"""
const FOUNDER = Ref(Int32[])

"""
    set_founder_tracking!(cell_df) -> Nothing

Starts lineage tracking: every alive cell becomes the founder of its own clone
(founder id = its lattice index). Call after irradiation, before the ABM run.
"""
function set_founder_tracking!(cell_df::DataFrame)
    f = zeros(Int32, nrow(cell_df))
    @inbounds for i in 1:nrow(cell_df)
        cell_df.is_cell[i] == 1 && (f[i] = Int32(cell_df.index[i]))
    end
    FOUNDER[] = f
    return nothing
end

"""Stops lineage tracking."""
clear_founder_tracking!() = (FOUNDER[] = Int32[]; nothing)

"""Founder id of every site (copy); 0 for empty sites or when tracking is off."""
founders() = copy(FOUNDER[])

@inline function _inherit_founder!(to::Integer, from::Integer)
    F = FOUNDER[]
    isempty(F) || (@inbounds F[to] = F[from])
    return nothing
end

#! ============================================================================
#! Distance-limited proliferation (spheroid growth plateau)
#! ============================================================================

"""
Settings of the distance-dependent division probability. Lattice sites never move,
so the distance of every site from the centre is computed once (`dist[site]`) and the
probability is a table lookup at division time. Global setting: set it once before
any threaded loop.
"""
mutable struct DivisionLimit
    active::Bool
    R50::Float64            # radius (µm) where p = 0.5
    w::Float64              # width (µm) of the transition; 0 = hard cut at R50
    rmax::Float64           # p = 0 beyond this radius (µm)
    rq::Float64             # quiescence radius (µm): beyond it cells go to G0 (Inf = off)
    center::NTuple{3,Float64}
    dist::Vector{Float64}   # distance of each lattice site from the centre (µm)
end

const DIVISION_LIMIT = Ref(DivisionLimit(false, Inf, 0.0, Inf, Inf, (0.0, 0.0, 0.0), Float64[]))

"""
    set_division_limit!(cell_df; R50, w = 0.1 * R50, center = :centroid, rmax = R50 + 3w)

A cell completing M with an empty neighbour divides with probability
p(r) = 1 / (1 + exp((r − R50) / w)) for r ≤ rmax, 0 beyond; r = distance of its site
from `center` (`:centroid` of the alive cells, `:origin`, or a tuple in µm).
A refused division sends the cell back to G1 (retry one cycle later).
"""
function set_division_limit!(cell_df::DataFrame; R50::Real = Inf, w::Real = isfinite(R50) ? 0.1 * R50 : 0.0,
                             center = :centroid, rmax::Real = R50 + 3 * w, rq::Real = Inf)
    (isfinite(R50) || isfinite(rq)) || error("set R50 (division probability) and/or rq (quiescence radius)")
    (R50 > 0 && w >= 0 && rmax >= R50 && rq > 0) || error("need R50 > 0, w ≥ 0, rmax ≥ R50, rq > 0 (got $R50, $w, $rmax, $rq)")
    hasproperty(cell_df, :x) && hasproperty(cell_df, :y) || error("cell_df needs :x and :y columns")
    x = Float64.(cell_df.x); y = Float64.(cell_df.y)
    z = hasproperty(cell_df, :z) ? Float64.(cell_df.z) : zeros(nrow(cell_df))
    c = if center === :centroid
        alive = cell_df.is_cell .== 1
        any(alive) || error("no alive cells to compute the centroid")
        (mean(x[alive]), mean(y[alive]), mean(z[alive]))
    elseif center === :origin
        (0.0, 0.0, 0.0)
    else
        Tuple(Float64.(center))
    end
    d = sqrt.((x .- c[1]) .^ 2 .+ (y .- c[2]) .^ 2 .+ (z .- c[3]) .^ 2)
    DIVISION_LIMIT[] = DivisionLimit(true, Float64(R50), Float64(w), Float64(rmax), Float64(rq), c, d)
    return nothing
end

"""Switches distance-limited proliferation off (every cell with space divides)."""
clear_division_limit!() = (DIVISION_LIMIT[] = DivisionLimit(false, Inf, 0.0, Inf, Inf, (0.0, 0.0, 0.0), Float64[]); nothing)

"""
    beyond_rq(idx) -> Bool

True if the site `idx` lies farther than the quiescence radius `rq` from the centre
(always false when no limit or `rq = Inf`). Such cells are put in G0 at their next
cycle event and are not unblocked by `_sync_block_state!`.
"""
@inline function beyond_rq(idx::Integer)::Bool
    L = DIVISION_LIMIT[]
    (L.active && isfinite(L.rq)) || return false
    return @inbounds L.dist[idx] > L.rq
end

"""Probability that the cell on lattice site `idx` divides (1.0 if no limit is set)."""
@inline function division_probability(idx::Integer)::Float64
    L = DIVISION_LIMIT[]
    L.active || return 1.0
    idx <= length(L.dist) || error("division limit was set for $(length(L.dist)) sites, got site $idx: " *
                                   "call set_division_limit! on the lattice being simulated")
    isfinite(L.R50) || return 1.0                 # only a quiescence radius was set
    r = @inbounds L.dist[idx]
    r > L.rmax && return 0.0
    L.w == 0.0 && return r <= L.R50 ? 1.0 : 0.0
    return 1.0 / (1.0 + exp((r - L.R50) / L.w))
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

"""
    is_valid_index(idx, n) -> Bool

Returns `true` if `1 ≤ idx ≤ n`. Used as a bounds check before accessing neighbor
arrays to guard against out-of-range neighbor indices in lattice operations.
"""
@inline is_valid_index(idx::Int32, n::Int32)::Bool = 1 <= idx <= n

"""
    is_time_due(t, eps, treat_neg) -> Bool

Returns `true` if time `t` has elapsed (i.e. is due), within tolerance `eps`.
- Returns `false` immediately for `Inf` or `missing` values (not yet scheduled).
- `treat_neg=false`: checks `abs(t) ≤ eps` (clears both near-zero and small-negative).
- `treat_neg=true`:  checks `t ≤ eps` (also fires for any negative t, useful if
  clamping is not guaranteed upstream).

In practice `update_time!` clamps timers at 0, so negative values should not occur;
`treat_neg=false` is the safe default.
"""
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
        nei = df.nei[i]
        df.number_nei[i] = length(nei) - sum(df.is_cell[j] for j in nei)
    end
end

#! ============================================================================
#! ABM Event Handlers
#! ============================================================================

"""
    _sync_block_state!(pop, idx) -> Nothing

Recomputes `number_nei[idx]` from the ground truth and syncs `can_divide` /
`cell_cycle` against it:
- loses its last empty neighbor (`number_nei==0`, currently cycling) → blocked
  into `"G0"` (`cycle_time=Inf, can_divide=0`).
- gains an empty neighbor (`number_nei>0`, currently `can_divide==0`) AND has
  no pending scheduled death (`death_time==Inf`) → unblocked back to `"G1"`
  with a fresh `cycle_time` (respects an existing `recover_time`, taking
  whichever is later). A cell with a finite `death_time` is left blocked —
  it's going to die on schedule regardless, so resurrecting its cycle isn't
  worth the behavior change.

Call this on `idx` after ANY event that can change its empty-neighbor count —
a division nearby (fills a slot) or a neighbor's death (frees one) — so both
directions stay in sync regardless of which kind of event caused the change.
Previously the block direction only ran from `_perform_division!` and the
(gated) unblock direction only from `_handle_cell_removal!`, so a cell that
gained space purely as a side effect of a nearby division (not a death) had
no path back to G1 even once its `death_time` was `Inf`.

# Example
```julia
_sync_block_state!(pop, Int32(17))
```
"""
function _sync_block_state!(pop::CellPopulation, idx::Int32)
    pop.number_nei[idx] = recount_empty_neighbors(pop, idx)
    if _quiescent(pop.number_nei[idx]) && pop.can_divide[idx] == 1
        pop.cell_cycle[idx] = String7("G0")
        pop.cycle_time[idx] = Inf
        pop.can_divide[idx] = 0
    elseif !_quiescent(pop.number_nei[idx]) && pop.can_divide[idx] == 0 && isinf(pop.death_time[idx]) &&
           !beyond_rq(idx)
        pop.cell_cycle[idx] = String7("G1")
        new_ct = generate_cycle_time("G1")
        pop.cycle_time[idx] = isinf(pop.recover_time[idx]) ?
            new_ct : max(new_ct, pop.recover_time[idx])
        pop.can_divide[idx] = 1
    end
    return nothing
end

"""
    _perform_division!(pop, parent_idx, nat_apo) -> Nothing

Divides parent cell into random empty neighbor. Resets both to G1 with fresh
cycle times. Recounts `number_nei` for parent, daughter, and all their
neighbors, syncing each via `_sync_block_state!` — a neighbor left with
`number_nei==0` is blocked into `"G0"` (the primary path into quiescence: a
neighbor losing its last empty slot to THIS division, not just a cell finding
no space when its own cycle timer fires), while one that regains space is
unblocked back to `"G1"`.

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

    empty_neighbors = Int32[]
    @inbounds for n_idx in parent_neighbors
        is_valid_index(n_idx, n) && is_cell[n_idx] == 0 && push!(empty_neighbors, n_idx)
    end
    daughter_idx = isempty(empty_neighbors) ? Int32(0) : rand(empty_neighbors)

    daughter_idx == 0 && (@warn "Division attempted but no empty neighbor" parent=parent_idx; return Int32(0))

    # Place daughter
    _inherit_founder!(daughter_idx, parent_idx)
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

    pop.number_nei[parent_idx]   = recount_empty_neighbors(pop, parent_idx)
    pop.number_nei[daughter_idx] = recount_empty_neighbors(pop, daughter_idx)

    # Parent/daughter can themselves end up with zero empty neighbors right
    # after this division (e.g. the parent had exactly one free slot, now
    # taken by the daughter) — block them into G0 immediately rather than
    # leaving the unconditional G1/can_divide=1 set above in place.
    for idx in (parent_idx, daughter_idx)
        if _quiescent(pop.number_nei[idx])
            pop.cell_cycle[idx] = String7("G0")
            pop.cycle_time[idx] = Inf
            pop.can_divide[idx] = 0
        end
    end

    for neighbors in (parent_neighbors, daughter_neighbors)
        @inbounds for n_idx in neighbors
            is_valid_index(n_idx, n) && is_cell[n_idx] == 1 &&
            n_idx != parent_idx && n_idx != daughter_idx || continue
            _sync_block_state!(pop, n_idx)
        end
    end

    pop.n_alive += 1
    return daughter_idx
end

"""
    _handle_cell_removal!(pop, removed_idx, is_natural_apoptosis) -> Nothing

Marks cell dead (`is_cell=0`), decrements `n_alive`, clears timers.
For each alive neighbor: syncs block state via `_sync_block_state!` — a
neighbor left with `number_nei==0` is blocked into `"G0"`; one that gains
space (and has no pending scheduled death, `death_time==Inf`) is unblocked
back to `"G1"` (respects an existing `recover_time`).

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

    n       = pop.n_cells
    is_cell = pop.is_cell

    @inbounds for n_idx in pop.nei[removed_idx]
        is_valid_index(n_idx, n) && is_cell[n_idx] == 1 || continue
        _sync_block_state!(pop, n_idx)
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


#! ============================================================================
#! Cycle event (single source of truth: fast engine and original loops)
#! ============================================================================

"""
Cycle event with checkpoint. In :g2 mode a cell entering G2 while repair is
still pending (recover_time > 0) is held until it completes, and recover_time
is not cleared by G1→S / S→G2 transitions. Otherwise identical to the engine's
original handler (same RNG use).
"""
function _cycle_event!(pop::CellPopulation, idx::Int32, nat_apo::Float64)
    if beyond_rq(idx)                         # too far from the centre: quiescent
        pop.cell_cycle[idx] = String7("G0"); pop.cycle_time[idx] = Inf
        pop.death_time[idx] = Inf; pop.recover_time[idx] = Inf; pop.can_divide[idx] = 0
        return nothing
    end
    current_phase = String(pop.cell_cycle[idx])
    has_space     = !_quiescent(pop.number_nei[idx])
    g2mode        = CHECKPOINT[] === :g2
    @inbounds if current_phase == "M"
        divides = has_space && rand() < division_probability(idx)
        if divides
            _perform_division!(pop, idx, nat_apo)
        elseif has_space
            pop.cell_cycle[idx] = String7("G1"); pop.cycle_time[idx] = generate_cycle_time("G1")
            pop.death_time[idx] = Inf; pop.recover_time[idx] = Inf; pop.can_divide[idx] = 1
        else
            pop.cell_cycle[idx] = String7("G0"); pop.cycle_time[idx] = Inf
            pop.death_time[idx] = Inf; pop.recover_time[idx] = Inf; pop.can_divide[idx] = 0
        end
    else
        if has_space
            next_phase = PHASE_TRANSITION[current_phase]
            pending    = g2mode ? pop.recover_time[idx] : Inf
            pop.cell_cycle[idx] = String7(next_phase)
            ct = generate_cycle_time(next_phase)
            pop.cycle_time[idx] = (next_phase == "G2" && isfinite(pending) && pending > 0) ?
                                  max(ct, pending) : ct                       # G2 checkpoint
            pop.death_time[idx]   = Inf
            pop.recover_time[idx] = (isfinite(pending) && pending > 0) ? pending : Inf
            pop.can_divide[idx]   = 1
        else
            pop.cell_cycle[idx] = String7("G0"); pop.cycle_time[idx] = Inf
            pop.death_time[idx] = Inf; pop.recover_time[idx] = Inf; pop.can_divide[idx] = 0
        end
    end
    return nothing
end


"""
    update_ABM!(pop, next_time, event, idx, nat_apo) -> Nothing

Original-loop step: advances every timer by `next_time`, removes other cells whose
death_time reached 0, then processes the event (death → `_handle_cell_removal!`,
cycle → `_cycle_event!`, which applies the division limit and the checkpoint mode).
Used by the original loops (fast = false) and by update_ABM_migration!.
"""
function update_ABM!(pop::CellPopulation, next_time::Float64, event::String,
                     idx::Int32, nat_apo::Float64)
    update_time!(pop, next_time)
    @inbounds for i in Int32(1):pop.n_cells
        pop.is_cell[i] == 0 && continue
        i == idx && continue
        dt = pop.death_time[i]
        !isinf(dt) && dt == 0.0 && _handle_cell_removal!(pop, i, false)
    end
    if event == "death_time"
        _handle_cell_removal!(pop, idx, false)
    elseif event == "cycle_time"
        _cycle_event!(pop, idx, nat_apo)
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
#! Migration (random walk on the lattice)
#! ============================================================================

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
Updates neighbor counts via _sync_block_state! (unblocks G0 cells that gain
free space unless they carry a scheduled death; blocks cells left without space),
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
        _inherit_founder!(target, idx)
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

    # Recount neighbours for the moved cell and every cell sharing a
    # neighbourhood, and sync block state with the shared handler.
    # _sync_block_state! only unblocks cells with death_time == Inf, so doomed
    # cells (and G0 cells with a scheduled radiation death) are never
    # resurrected by a neighbour's hop. It also blocks a cycling cell into G0
    # if the hop left it with no empty neighbour.
    _sync_block_state!(pop, target)

    affected = union(pop.nei[idx], pop.nei[target])
    @inbounds for nb in affected
        is_valid_index(nb, n) || continue
        pop.is_cell[nb] == 1  || continue
        nb == target           && continue

        _sync_block_state!(pop, nb)
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


#! ============================================================================
#! Fast event engine (standard) + original loops
#! ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Binary min-heap of (time, cell, kind)
# ─────────────────────────────────────────────────────────────────────────────
const EV_DEATH = Int8(1)
const EV_CYCLE = Int8(2)
const EV_MIG   = Int8(3)

struct EventHeap
    t::Vector{Float64}
    i::Vector{Int32}
    k::Vector{Int8}
end
EventHeap() = EventHeap(Float64[], Int32[], Int8[])
Base.isempty(H::EventHeap) = isempty(H.t)

@inline function _hswap!(H::EventHeap, a::Int, b::Int)
    @inbounds begin
        H.t[a], H.t[b] = H.t[b], H.t[a]
        H.i[a], H.i[b] = H.i[b], H.i[a]
        H.k[a], H.k[b] = H.k[b], H.k[a]
    end
    return nothing
end

function hpush!(H::EventHeap, t::Float64, i::Int32, k::Int8)
    push!(H.t, t); push!(H.i, i); push!(H.k, k)
    j = length(H.t)
    @inbounds while j > 1
        p = j >> 1
        H.t[p] <= H.t[j] && break
        _hswap!(H, p, j); j = p
    end
    return nothing
end

@inline htop(H::EventHeap) = @inbounds (H.t[1], H.i[1], H.k[1])

function hpop!(H::EventHeap)
    n = length(H.t)
    _hswap!(H, 1, n)
    pop!(H.t); pop!(H.i); pop!(H.k)
    n -= 1
    j = 1
    @inbounds while true
        l = 2j
        l > n && break
        r = l + 1
        c = (r <= n && H.t[r] < H.t[l]) ? r : l
        H.t[j] <= H.t[c] && break
        _hswap!(H, j, c); j = c
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Engine state
# ─────────────────────────────────────────────────────────────────────────────
mutable struct FastEngine
    pop::CellPopulation
    mig::Vector{Float64}          # migration timers (absolute); all Inf if D == 0
    H::EventHeap
    sched::Matrix{Float64}        # 3 × n: time currently queued per (kind, site); NaN = none
    mark::Vector{Int32}           # visit stamps for the 2-ring
    stamp::Int32
    ring::Vector{Int32}
    absb::Matrix{Float64}         # 4 × |ring| timers before the handler (absolute)
    relb::Matrix{Float64}         # 4 × |ring| same, relative
    g0::Int32; g1::Int32; s::Int32; g2::Int32; m::Int32; stem::Int32
    migrate::Bool
end

@inline function _timer(E::FastEngine, k::Integer, i::Int32)
    k == 1 ? E.pop.death_time[i] : k == 2 ? E.pop.cycle_time[i] :
    k == 3 ? E.pop.recover_time[i] : E.mig[i]
end
@inline function _set_timer!(E::FastEngine, k::Integer, i::Int32, v::Float64)
    if k == 1;     E.pop.death_time[i]   = v
    elseif k == 2; E.pop.cycle_time[i]   = v
    elseif k == 3; E.pop.recover_time[i] = v
    else           E.mig[i]              = v
    end
    return nothing
end

"""Phase/stem counters, ± one cell (only if alive)."""
@inline function _count!(E::FastEngine, i::Int32, sgn::Int32)
    pop = E.pop
    pop.is_cell[i] == 1 || return nothing
    ph = pop.cell_cycle[i]
    if     ph == "G0"; E.g0 += sgn
    elseif ph == "G1"; E.g1 += sgn
    elseif ph == "S";  E.s  += sgn
    elseif ph == "G2"; E.g2 += sgn
    elseif ph == "M";  E.m  += sgn
    end
    !isnothing(pop.is_stem) && pop.is_stem[i] == 1 && (E.stem += sgn)
    return nothing
end

"""Sites within two lattice steps of `idx` (deduplicated), into E.ring."""
function _collect_ring!(E::FastEngine, idx::Int32)
    pop = E.pop; n = pop.n_cells
    E.stamp += Int32(1); st = E.stamp
    empty!(E.ring)
    E.mark[idx] = st; push!(E.ring, idx)
    @inbounds for nb in pop.nei[idx]
        is_valid_index(nb, n) || continue
        E.mark[nb] != st && (E.mark[nb] = st; push!(E.ring, nb))
    end
    n1 = length(E.ring)
    @inbounds for q in 2:n1
        for nb in pop.nei[E.ring[q]]
            is_valid_index(nb, n) || continue
            E.mark[nb] != st && (E.mark[nb] = st; push!(E.ring, nb))
        end
    end
    nr = length(E.ring)
    if size(E.absb, 2) < nr
        E.absb = Matrix{Float64}(undef, 4, 2nr)
        E.relb = Matrix{Float64}(undef, 4, 2nr)
    end
    return nothing
end

"""Ring timers → relative to `t` (what the original handlers expect)."""
function _ring_to_rel!(E::FastEngine, t::Float64)
    @inbounds for (q, i) in enumerate(E.ring), k in 1:4
        v = _timer(E, k, i)
        E.absb[k, q] = v
        r = isfinite(v) ? v - t : v
        E.relb[k, q] = r
        _set_timer!(E, k, i, r)
    end
    return nothing
end

"""Ring timers → absolute. Unchanged timers get their exact previous value back."""
function _ring_to_abs!(E::FastEngine, t::Float64)
    @inbounds for (q, i) in enumerate(E.ring), k in 1:4
        r = _timer(E, k, i)
        v = r === E.relb[k, q] || r == E.relb[k, q] ? E.absb[k, q] :
            (isfinite(r) ? r + t : r)
        _set_timer!(E, k, i, v)
    end
    return nothing
end

"""Queue the current death / cycle / migration timers of site `i` if they changed."""
@inline function _reschedule!(E::FastEngine, i::Int32)
    alive = E.pop.is_cell[i] == 1
    @inbounds for (row, k) in ((1, 1), (2, 2), (3, 4))      # (sched row, timer id)
        v = _timer(E, k, i)
        if alive && isfinite(v)
            if v != E.sched[row, i]
                hpush!(E.H, v, i, Int8(row)); E.sched[row, i] = v
            end
        else
            E.sched[row, i] = NaN
        end
    end
    return nothing
end

function _record_fast!(ts::SimulationTimeSeries, t::Float64, E::FastEngine)
    pop = E.pop
    push!(ts.time, t); push!(ts.total_cells, pop.n_alive)
    push!(ts.g0_cells, E.g0); push!(ts.g1_cells, E.g1); push!(ts.s_cells, E.s)
    push!(ts.g2_cells, E.g2); push!(ts.m_cells, E.m)
    if isnothing(pop.is_stem)
        push!(ts.stem_cells, Int32(0)); push!(ts.non_stem_cells, Int32(0))
    else
        push!(ts.stem_cells, E.stem); push!(ts.non_stem_cells, pop.n_alive - E.stem)
    end
end

"""Snapshot with timers expressed relative to `t` (as the original loops store them)."""
function _snapshot_rel(pop::CellPopulation, t::Float64)
    s = create_snapshot(pop)
    for v in (s.death_time, s.cycle_time, s.recover_time)
        @inbounds for j in eachindex(v)
            isfinite(v[j]) && (v[j] = max(0.0, v[j] - t))
        end
    end
    return s
end

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
"""
    run_simulation_abm_fast!(pop; D=0.0, h=1.0, nat_apo=1e-10, terminal_time=48.0,
                             snapshot_times=[1,6,12,24], print_interval=1.0,
                             verbose=true, record_dt=0.0) -> (ts, snapshots)

Drop-in replacement for `run_simulation_abm!` (D = 0) and
`run_simulation_abm_migration!` (D > 0, µm²/h; h = lattice spacing, µm).
`record_dt = 0` records after every event (as before); `record_dt > 0` records
at most once per `record_dt` hours, which keeps the time series small on long runs.
On return, `pop` timers are relative to the last event time, as in the originals.
"""
function run_simulation_abm_fast!(pop::CellPopulation;
                                  D::Float64 = 0.0, h::Float64 = 1.0,
                                  nat_apo::Float64 = 1e-10,
                                  terminal_time::Float64 = 48.0,
                                  snapshot_times::Vector{Int} = [1, 6, 12, 24],
                                  print_interval::Float64 = 1.0,
                                  verbose::Bool = true,
                                  record_dt::Float64 = 0.0)
    n = pop.n_cells
    migrate = D > 0.0
    mig = fill(Inf, n)
    migrate && schedule_all_migrations!(mig, pop, D, h)

    E = FastEngine(pop, mig, EventHeap(), fill(NaN, 3, n), zeros(Int32, n), Int32(0),
                   Int32[], Matrix{Float64}(undef, 4, 256), Matrix{Float64}(undef, 4, 256),
                   0, 0, 0, 0, 0, 0, migrate)

    # t = 0: relative == absolute
    @inbounds for i in Int32(1):n
        pop.is_cell[i] == 1 || continue
        _count!(E, i, Int32(1))
        _reschedule!(E, i)
    end

    verbose && print_initial_stats(pop)
    verbose && migrate && @printf("Migration ON: D=%.1f µm²/h  h=%.1f µm\n", D, h)

    ts = SimulationTimeSeries()
    snapshots = Dict{Int, CellPopulation}()
    snapshots[0] = create_snapshot(pop)
    t = 0.0
    _record_fast!(ts, t, E)
    next_record = record_dt
    event_count = 0
    next_print  = print_interval
    snap_set    = Set(snapshot_times)
    neg = Int32(-1); pos = Int32(1)

    verbose && println("\n", "="^70, "\nSTARTING SIMULATION (fast engine)\n", "="^70)

    while !isempty(E.H)
        tn, i, k = htop(E.H)
        if !(pop.is_cell[i] == 1 && E.sched[k, i] == tn)   # stale entry
            hpop!(E.H); continue
        end
        tn > terminal_time && break
        hpop!(E.H)
        E.sched[k, i] = NaN
        t = tn
        event_count += 1

        if verbose && t >= next_print
            println("t=$(round(t,digits=2))h | ", k == EV_DEATH ? "death" : k == EV_CYCLE ? "cycle" : "migration",
                    " | cells=$(pop.n_alive) | events=$event_count")
            next_print += print_interval
        end
        hr = floor(Int, t)
        if hr in snap_set && !haskey(snapshots, hr)
            verbose && println("  └─ snapshot t=$(hr)h")
            snapshots[hr] = _snapshot_rel(pop, t)
        end

        # Local update on the 2-ring, in the original handlers' relative frame
        _collect_ring!(E, i)
        for j in E.ring; _count!(E, j, neg); end
        _ring_to_rel!(E, t)

        if k == EV_DEATH
            _handle_cell_removal!(pop, i, false)
            if migrate
                mig[i] = Inf
                for nb in pop.nei[i]
                    is_valid_index(nb, n) && pop.is_cell[nb] == 1 &&
                        schedule_migration!(mig, pop, nb, D, h)
                end
            end
        elseif k == EV_CYCLE
            _cycle_event!(pop, i, nat_apo)
            if migrate
                schedule_migration!(mig, pop, i, D, h)
                for nb in pop.nei[i]
                    is_valid_index(nb, n) && pop.is_cell[nb] == 1 &&
                        schedule_migration!(mig, pop, nb, D, h)
                end
            end
        else
            perform_migration!(pop, mig, i, D, h)
        end

        _ring_to_abs!(E, t)
        for j in E.ring
            _count!(E, j, pos)
            _reschedule!(E, j)
        end

        if record_dt <= 0.0 || t >= next_record
            _record_fast!(ts, t, E)
            record_dt > 0.0 && (next_record = (floor(t / record_dt) + 1) * record_dt)
        end
    end
    record_dt > 0.0 && (isempty(ts.time) || ts.time[end] != t) && _record_fast!(ts, t, E)

    # Back to relative timers (relative to the last event), as the originals leave them
    @inbounds for i in Int32(1):n
        for v in (pop.death_time, pop.cycle_time, pop.recover_time)
            isfinite(v[i]) && (v[i] = max(0.0, v[i] - t))
        end
    end

    if verbose
        println("="^70, "\nSIMULATION COMPLETE (fast engine)")
        println("  Final time : $(round(t, digits=2))h")
        println("  Events     : $event_count")
        println("  Final cells: $(pop.n_alive)")
        println("="^70)
    end
    return (ts, snapshots)
end

"""DataFrame wrapper, same conventions as run_simulation_abm!(cell_df; ...)."""
function run_simulation_abm_fast!(cell_df::DataFrame;
                                  return_dataframes::Bool = true,
                                  update_input::Bool = true,
                                  kwargs...)
    pop = CellPopulation(cell_df)
    ts, snaps = run_simulation_abm_fast!(pop; kwargs...)
    update_input && update_dataframe!(cell_df, pop)
    return_dataframes ?
        (ts, Dict{Int, DataFrame}(k => to_dataframe(v; alive_only = true) for (k, v) in snaps)) :
        (ts, snaps)
end

# ─────────────────────────────────────────────────────────────────────────────
# Validation: same seed → same trajectory as the original loop
# ─────────────────────────────────────────────────────────────────────────────
"""
    compare_engines(cell_df; seed=1, terminal_time=48.0, D=0.0, h=30.0) -> Bool

Runs the original loop and the fast engine from the same initial state and
global RNG seed, and compares the total-cell and phase time series. Prints the
speed-up. Copies `cell_df`, so the input is not modified.
"""
function compare_engines(cell_df::DataFrame; seed::Integer = 1, terminal_time::Float64 = 48.0,
                         D::Float64 = 0.0, h::Float64 = 30.0)
    df1 = deepcopy(cell_df); df2 = deepcopy(cell_df)

    Random.seed!(seed)
    t_old = @elapsed ts_old, _ = D > 0 ?
        run_simulation_abm_migration_original!(CellPopulation(df1), D, h;
                                               terminal_time = terminal_time, verbose = false) :
        run_simulation_abm_original!(CellPopulation(df1); terminal_time = terminal_time,
                                     verbose = false)

    Random.seed!(seed)
    t_new = @elapsed ts_new, _ = run_simulation_abm_fast!(df2; D = D, h = h,
                                      terminal_time = terminal_time, verbose = false)

    same_len = length(ts_old.time) == length(ts_new.time)
    nmin = min(length(ts_old.time), length(ts_new.time))
    dtot = maximum(abs.(Int.(ts_old.total_cells[1:nmin]) .- Int.(ts_new.total_cells[1:nmin])))
    dph  = maximum(abs.(Int.(ts_old.g1_cells[1:nmin]) .- Int.(ts_new.g1_cells[1:nmin])))
    @printf("[compare_engines] events %d vs %d | max |Δ total| = %d | max |Δ G1| = %d | final %d vs %d\n",
            length(ts_old.time), length(ts_new.time), dtot, dph,
            ts_old.total_cells[end], ts_new.total_cells[end])
    @printf("[compare_engines] time: original %.2f s, fast %.2f s  (×%.1f)\n",
            t_old, t_new, t_old / max(t_new, eps()))
    ok = same_len && dtot == 0 && dph == 0
    ok || @warn "Trajectories differ. Small differences after a near-tie in event times are " *
                "expected; large ones from the start point to a bug."
    return ok
end


# ─────────────────────────────────────────────────────────────────────────────
# Original loops, kept for fast = false and for compare_engines
# ─────────────────────────────────────────────────────────────────────────────
function run_simulation_abm_original!(pop::CellPopulation;
                                      nat_apo::Float64 = 1e-10,
                                      terminal_time::Float64 = 48.0,
                                      snapshot_times::Vector{Int} = [1, 6, 12, 24],
                                      print_interval::Float64 = 1.0,
                                      verbose::Bool = true, kwargs...)
    verbose && print_initial_stats(pop)
    ts = SimulationTimeSeries()
    snapshots = Dict{Int, CellPopulation}()
    snapshots[0] = create_snapshot(pop)
    t = 0.0
    record_timepoint!(ts, t, pop)
    event_count = 0; next_print = print_interval; snap_set = Set(snapshot_times)
    while t < terminal_time
        next_time, idx, event = compute_next_event(pop)
        (isinf(next_time) || t + next_time > terminal_time) && break
        t += next_time; event_count += 1
        if verbose && t >= next_print
            println("t=$(round(t,digits=2))h | $event | Cells=$(pop.n_alive) | Events=$event_count")
            next_print += print_interval
        end
        hr = floor(Int, t)
        hr in snap_set && !haskey(snapshots, hr) && (snapshots[hr] = create_snapshot(pop))
        update_ABM!(pop, next_time, event, idx, nat_apo)
        record_timepoint!(ts, t, pop)
    end
    verbose && println("Final time $(round(t, digits=2))h | events $event_count | cells $(pop.n_alive)")
    return (ts, snapshots)
end

function run_simulation_abm_migration_original!(pop::CellPopulation, D::Float64, h::Float64;
                                                nat_apo::Float64 = 1e-10,
                                                terminal_time::Float64 = 48.0,
                                                snapshot_times::Vector{Int} = [1, 6, 12, 24],
                                                print_interval::Float64 = 1.0,
                                                verbose::Bool = true, kwargs...)
    D == 0.0 && return run_simulation_abm_original!(pop; nat_apo = nat_apo,
                           terminal_time = terminal_time, snapshot_times = snapshot_times,
                           print_interval = print_interval, verbose = verbose)
    mig_times = fill(Inf, pop.n_cells)
    schedule_all_migrations!(mig_times, pop, D, h)
    verbose && print_initial_stats(pop)
    ts = SimulationTimeSeries()
    snapshots = Dict{Int, CellPopulation}()
    snapshots[0] = create_snapshot(pop)
    t = 0.0
    record_timepoint!(ts, t, pop)
    event_count = 0; next_print = print_interval; snap_set = Set(snapshot_times)
    while t < terminal_time
        next_time, idx, event = compute_next_event_migration(pop, mig_times)
        (isinf(next_time) || t + next_time > terminal_time) && break
        t += next_time; event_count += 1
        if verbose && t >= next_print
            println("t=$(round(t,digits=2))h | $event | cells=$(pop.n_alive) | events=$event_count")
            next_print += print_interval
        end
        hr = floor(Int, t)
        hr in snap_set && !haskey(snapshots, hr) && (snapshots[hr] = create_snapshot(pop))
        update_ABM_migration!(pop, mig_times, next_time, event, idx, nat_apo, D, h)
        record_timepoint!(ts, t, pop)
    end
    verbose && println("Final time $(round(t, digits=2))h | events $event_count | cells $(pop.n_alive)")
    return (ts, snapshots)
end

# ─────────────────────────────────────────────────────────────────────────────
# Standard entry points (fast engine; fast = false → original loops)
# ─────────────────────────────────────────────────────────────────────────────
function run_simulation_abm!(pop::CellPopulation;
                             nat_apo::Float64 = 1e-10,
                             terminal_time::Float64 = 48.0,
                             snapshot_times::Vector{Int} = [1, 6, 12, 24],
                             print_interval::Float64 = 1.0,
                             verbose::Bool = true,
                             fast::Bool = true,
                             record_dt::Float64 = 0.0)
    fast || return run_simulation_abm_original!(pop; nat_apo = nat_apo,
                       terminal_time = terminal_time, snapshot_times = snapshot_times,
                       print_interval = print_interval, verbose = verbose)
    return run_simulation_abm_fast!(pop; D = 0.0, nat_apo = nat_apo,
               terminal_time = terminal_time, snapshot_times = snapshot_times,
               print_interval = print_interval, verbose = verbose, record_dt = record_dt)
end

function run_simulation_abm_migration!(pop::CellPopulation, D::Float64, h::Float64;
                                       nat_apo::Float64 = 1e-10,
                                       terminal_time::Float64 = 48.0,
                                       snapshot_times::Vector{Int} = [1, 6, 12, 24],
                                       print_interval::Float64 = 1.0,
                                       verbose::Bool = true,
                                       fast::Bool = true,
                                       record_dt::Float64 = 0.0)
    fast || return run_simulation_abm_migration_original!(pop, D, h; nat_apo = nat_apo,
                       terminal_time = terminal_time, snapshot_times = snapshot_times,
                       print_interval = print_interval, verbose = verbose)
    return run_simulation_abm_fast!(pop; D = D, h = h, nat_apo = nat_apo,
               terminal_time = terminal_time, snapshot_times = snapshot_times,
               print_interval = print_interval, verbose = verbose, record_dt = record_dt)
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


#! ============================================================================
#! Continuous irradiation
#! ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Continuous irradiation driver (single damage state per cell)
# ─────────────────────────────────────────────────────────────────────────────
"""
    add_uniform_damage!(cell_df, dD, K_tot, L_tot, rng)

Adds the lesions of a dose increment `dD` (Gy) for a uniform low-LET field
(high-energy protons: ≪ 1 lesion per track per cell). `K_tot`, `L_tot` = X and Y
lesions per cell per Gy; spread uniformly over the domains. Skips dead/doomed cells.
"""
function add_uniform_damage!(cell_df::DataFrame, dD::Float64, K_tot::Float64,
                             L_tot::Float64, rng::AbstractRNG)
    @inbounds for i in cell_df.index
        cell_df.is_cell[i] == 1 || continue
        isfinite(cell_df.death_time[i]) && continue
        X = cell_df.dam_X_dom[i]; Y = cell_df.dam_Y_dom[i]; nd = length(X)
        for _ in 1:rand(rng, Poisson(K_tot * dD)); X[rand(rng, 1:nd)] += 1; end
        for _ in 1:rand(rng, Poisson(L_tot * dD)); Y[rand(rng, 1:nd)] += 1; end
    end
    return nothing
end

"""Advance all running timers by an idle interval δ (no event happened in it)."""
function _advance_idle!(cell_df::DataFrame, δ::Float64)
    δ <= 0 && return nothing
    for col in (cell_df.death_time, cell_df.cycle_time, cell_df.recover_time)
        @inbounds for i in eachindex(col)
            cell_df.is_cell[i] == 1 && isfinite(col[i]) && (col[i] = max(0.0, col[i] - δ))
        end
    end
    return nothing
end

"""
    irradiate_continuous_abm!(cell_df, gsm2_cycle; dose, dose_rate_gyh, K_tot, L_tot,
                              checkpoint = :g2, max_pulse_dose = 0.05, max_pulse_h = 0.1,
                              seed = 1) -> (ts, summary)

Continuous irradiation of the ABM as a train of small pulses (≤ `max_pulse_dose`
Gy and ≤ `max_pulse_h` h each, much shorter than repair and phase times). Each
pulse: add damage → `compute_times_domain!` with a repair window equal to the
pulse interval → ABM for the same interval → advance timers by the idle time at
the end of the interval. ONE damage state per cell (dam_X_dom / dam_Y_dom).
At the end, residual damage is resolved (`terminal_time = Inf`).

Returns the time series during exposure and a NamedTuple with:
`SF_plated` (clonogenic / cells present at the end), `N_rel` (clonogenic cells /
initial cells), and the G1/S/G2/M/G0 fractions at the end of exposure.
"""
function irradiate_continuous_abm!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
                                   dose::Float64, dose_rate_gyh::Float64,
                                   K_tot::Float64, L_tot::Float64,
                                   checkpoint::Symbol = :g2,
                                   max_pulse_dose::Float64 = 0.05,
                                   max_pulse_h::Float64 = 0.1,
                                   seed::Integer = 1, verbose::Bool = false)
    set_checkpoint!(checkpoint)
    _ensure_fate_columns!(cell_df)
    rng  = Xoshiro(seed)
    N0   = count(==(1), cell_df.is_cell)
    Tirr = dose / dose_rate_gyh
    Δ    = min(max_pulse_h, max_pulse_dose / dose_rate_gyh)
    np   = max(1, ceil(Int, Tirr / Δ)); Δ = Tirr / np; dD = dose / np

    ts_all = SimulationTimeSeries(); t0 = 0.0
    for k in 1:np
        add_uniform_damage!(cell_df, dD, K_tot, L_tot, rng)
        compute_times_domain!(cell_df, gsm2_cycle; terminal_time = Δ, summary = false,
                              model = :incremental, checkpoint = checkpoint)
        ts, _ = run_simulation_abm!(cell_df; terminal_time = Δ, snapshot_times = Int[],
                                    verbose = false, return_dataframes = false,
                                    update_input = true)
        t_last = isempty(ts.time) ? 0.0 : ts.time[end]
        _advance_idle!(cell_df, Δ - t_last)              # timers are relative to t_last
        for f in fieldnames(SimulationTimeSeries)
            v = getfield(ts, f); f === :time && (v = v .+ t0)
            append!(getfield(ts_all, f), v)
        end
        t0 += Δ
        verbose && k % max(1, np ÷ 10) == 0 &&
            println("  pulse $k/$np  t=$(round(t0, digits = 2)) h  alive=$(count(==(1), cell_df.is_cell))")
    end

    alive = cell_df.is_cell .== 1
    phases = cell_df.cell_cycle[alive .& .!isfinite.(cell_df.death_time)]
    nph = max(length(phases), 1)
    frac = (G1 = count(==("G1"), phases) / nph, S = count(==("S"), phases) / nph,
            G2 = count(==("G2"), phases) / nph, M = count(==("M"), phases) / nph,
            G0 = count(==("G0"), phases) / nph)

    compute_times_domain!(cell_df, gsm2_cycle; terminal_time = Inf, summary = false,
                          model = :incremental, checkpoint = checkpoint)   # resolve residual damage
    alive  = cell_df.is_cell .== 1
    clono  = count(alive .& .!isfinite.(cell_df.death_time))
    present = count(alive)
    return ts_all, (SF_plated = clono / max(present, 1), N_rel = clono / N0,
                    N_present = present, T_irr = Tirr, pulses = np, phase_frac = frac)
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