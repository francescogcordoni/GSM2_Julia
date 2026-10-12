#! ============================================================================
#! utilities_checkpoint.jl
#!
#! Cell-cycle checkpoint model + continuous irradiation driver for the ABM.
#! Include LAST in load_utilities.jl (after utilities_abm.jl,
#! utilities_abm_migration.jl and utilities_abm_fast.jl):
#!     include(joinpath(@__DIR__, "..", "src", "utilities_checkpoint.jl"))
#!
#! CHECKPOINT MODES  (set_checkpoint!(:g2) / set_checkpoint!(:all))
#!   :all  current model: a damaged cell's cycle clock is held in G1, S and G2
#!         until repair is complete (cycle_time = recover + phase time).
#!   :g2   G2/M checkpoint only: damaged cells keep cycling through G1 and S
#!         while they repair (with the kinetics of the phase they are in), and
#!         are HELD in G2 until repair completes. Cells pile up in the sensitive
#!         G2 phase under protracted exposure → inverse dose-rate effect.
#!   In both modes, mitosis ending with unrepaired damage is lethal (doomed).
#!
#! WHAT THIS FILE REPLACES
#!   compute_times_domain!  — checkpoint-aware; also fixes three problems that
#!       matter whenever it is called repeatedly (continuous / fractionated):
#!       (1) undamaged cells were reset and got a FRESH phase time at every
#!           call → under continuous irradiation the cycle never advanced;
#!       (2) doomed cells (finite death_time) were reset to Inf → resurrected;
#!       (3) in :g2 mode the remaining time of a held G2 cell is kept in the
#!           new column :cycle_hold and resumed after repair.
#!       Cells that are neither damaged nor doomed are now left untouched.
#!   _cycle_event!  (fast engine) — G2 hold on S→G2 entry while recover_time
#!       is pending, and recover_time is kept across phase transitions in :g2.
#!   The fast engine must be in use (default after utilities_abm_fast.jl);
#!   the original loop (fast = false) does not apply the G2 hold on entry.
#!
#! NEW
#!   set_checkpoint!(mode), CHECKPOINT[]
#!   add_uniform_damage!(cell_df, dD, K_tot, L_tot, rng)
#!   irradiate_continuous_abm!(cell_df, gsm2_cycle; dose, dose_rate_gyh, ...)
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

@inline function _fresh_phase_time(phase::AbstractString)
    shape = phase == "G2" ? 2 * 5 : phase == "S" ? 2 * 7 : phase == "M" ? 2 * 1.0 : 2 * 11
    return rand(Gamma(shape, 0.5))
end

"""
    compute_times_domain!(cell_df, gsm2_cycle; nat_apo, terminal_time, verbose,
                          print_every, summary, checkpoint = CHECKPOINT[]) -> Nothing

Processes only cells that are alive, not already doomed, and carry damage.
`terminal_time` = repair window (Inf: resolve all damage). Outcomes:
- lethal event (or direct Y) → doomed via `_doom_cell!` (delayed death);
- repaired within the window → `recover_time`, cycle per checkpoint mode;
- not repaired within the window (timeout) → residual damage kept, cycle per mode.
"""
function compute_times_domain!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
                               nat_apo::Float64 = 1e-10,
                               terminal_time::Float64 = Inf,
                               verbose::Bool = false, print_every::Int = 0,
                               summary::Bool = true,
                               checkpoint::Symbol = CHECKPOINT[])
    (0.0 < nat_apo < 1.0) || error("nat_apo must be in (0, 1); got $nat_apo")
    hasproperty(cell_df, :cycle_hold) || (cell_df.cycle_hold = fill(Inf, nrow(cell_df)))
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
        has_space = cell_df.number_nei[i] > 0

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
Cycle event with checkpoint. In :g2 mode a cell entering G2 while repair is
still pending (recover_time > 0) is held until it completes, and recover_time
is not cleared by G1→S / S→G2 transitions. Otherwise identical to the engine's
original handler (same RNG use).
"""
function _cycle_event!(pop::CellPopulation, idx::Int32, nat_apo::Float64)
    current_phase = String(pop.cell_cycle[idx])
    has_space     = pop.number_nei[idx] > 0
    has_limit     = isdefined(@__MODULE__, :division_probability)
    g2mode        = CHECKPOINT[] === :g2
    @inbounds if current_phase == "M"
        divides = has_space &&
                  (has_limit ? rand() < Base.invokelatest(division_probability, idx) : true)
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
    hasproperty(cell_df, :cycle_hold) || (cell_df.cycle_hold = fill(Inf, nrow(cell_df)))
    rng  = Xoshiro(seed)
    N0   = count(==(1), cell_df.is_cell)
    Tirr = dose / dose_rate_gyh
    Δ    = min(max_pulse_h, max_pulse_dose / dose_rate_gyh)
    np   = max(1, ceil(Int, Tirr / Δ)); Δ = Tirr / np; dD = dose / np

    ts_all = SimulationTimeSeries(); t0 = 0.0
    for k in 1:np
        add_uniform_damage!(cell_df, dD, K_tot, L_tot, rng)
        compute_times_domain!(cell_df, gsm2_cycle; terminal_time = Δ, summary = false,
                              checkpoint = checkpoint)
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
                          checkpoint = checkpoint)          # resolve residual damage
    alive  = cell_df.is_cell .== 1
    clono  = count(alive .& .!isfinite.(cell_df.death_time))
    present = count(alive)
    return ts_all, (SF_plated = clono / max(present, 1), N_rel = clono / N0,
                    N_present = present, T_irr = Tirr, pulses = np, phase_frac = frac)
end

println("[utilities_checkpoint] checkpoint-aware compute_times_domain! loaded (mode = $(CHECKPOINT[]))")
