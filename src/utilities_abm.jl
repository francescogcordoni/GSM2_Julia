"""
    compute_times_domain!(cell_df::DataFrame, gsm2::GSM2, nat_apo::Float64;
                            terminal_time::Float64 = Inf,
                            verbose::Bool = false,
                            print_every::Int = 0,
                            summary::Bool = true)

Compute, **in parallel**, per‑cell survival (`sp`) with the GSM2 model and schedule
stochastic times for radiogenic death, recovery, and cell‑cycle events. Updates
`cell_df` **in place** for all **active** cells (`is_cell == 1`).

# Behavior
- Validates `nat_apo` ∈ (0, 1) and prints a header with run configuration.
- Freezes the set of active cells (`is_cell == 1`) at the start and processes them in a
    `Threads.@threads` loop.
- For each active cell `i`:
    1. Computes `SP_cell = domain_GSM2(dam_X_dom[i], dam_Y_dom[i], gsm2)` and stores it in `cell_df.sp[i]`.
    2. Calls `compute_repair_domain` with a **time limit** `T = terminal_time`:
     - If **immediate removal** occurs (`death_time == 0.0`), the cell is marked as inactive,
       neighbor increments are **deferred** to a per-thread buffer, and `death_type` is saved.
     - If a **radiogenic death** is scheduled (finite `death_time`), sets `death_time` (competing with
        division time in phase `"M"` if stem and neighbors available), clears `dam_X_dom`, `dam_Y_dom`, and sets `death_type`.
     - If **survivor** with finite `recover_time`, sets `recover_time` and possibly `cycle_time`
        based on phase (`"M"`, `"G2"`, `"S"`, `"G1"`) and neighbor availability; clears `dam_X_dom`, `dam_Y_dom`.
     - If **timeout** (`death_time` and `recover_time` are `Inf` with `death_type == -1`),
        writes back the returned `X_gsm2`, `Y_gsm2` and marks the case as timeout.
    3. Sets `apo_time` to `Inf` (placeholder; adjust if a natural apoptosis process should be modeled).
- After the parallel loop, a **serial merge** applies all deferred neighbor increments to `number_nei`.
- A **final summary** (if `summary=true`) prints totals, percentages, mean SP, and elapsed time.

# Thread safety & RNG
- Each thread writes only to its own cell row.
- Neighbor increments are **collected per thread** and applied **serially** at the end.
- A **print lock** avoids interleaved log lines across threads.
- The internal `compute_repair_domain` is called with keyword `T=terminal_time`. If it supports a
    `rng` keyword, you can pass the thread-local RNG (see the comment inside the code).

# Arguments
- `cell_df::DataFrame`: Must contain at least
:index, :is_cell, :dam_X_dom, :dam_Y_dom, :sp, :apo_time, :death_time,
:recover_time, :cycle_time, :is_death_rad, :death_type, :cell_cycle,
:is_stem, :number_nei, :nei
where `:nei[i]` is expected to be a vector of neighbor indices.
- `gsm2::GSM2`: GSM2 model parameters (`a`, `b`, `r`).
- `nat_apo::Float64`: Must satisfy `0 < nat_apo < 1`. (Reserved here; `apo_time` is set to `Inf`.)
- `terminal_time::Float64 = Inf`: Time limit for the repair-domain simulation (passed to `compute_repair_domain`).
- `verbose::Bool = false`: If `true`, enable per‑cell progress logs in English.
- `print_every::Int = 0`: If `> 0`, prints a log line every N processed cells (in the active set).
- `summary::Bool = true`: If `true`, prints a final summary with counts and elapsed time.

# Returns
- `nothing` (modifies `cell_df` in place)
"""
function compute_times_domain!(cell_df::DataFrame, gsm2::GSM2, nat_apo::Float64;
                                terminal_time::Float64 = Inf,
                                verbose::Bool = false, print_every::Int = 0, summary::Bool = true)

    # -----------------------------
    # Basic parameter validation
    # -----------------------------
    if !(0.0 < nat_apo < 1.0)
        error("nat_apo must be in the open interval (0, 1); got nat_apo = $nat_apo")
    end

    # Precompute reciprocal for exponential sampling:
    λ = -log(nat_apo)
    inv_λ = 1.0 / λ

    # Freeze the active set
    active_cells = @view cell_df.index[cell_df.is_cell .== 1]
    n_active = length(active_cells)

    # State reset
    cell_df.sp .= 1.0
    cell_df.apo_time .= Inf
    cell_df.death_time .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type .= -1

    if n_active == 0
        println("[compute_times_domain!] No active cells (is_cell == 1). Nothing to do.")
        return nothing
    end

    # Logging setup
    start_t = time()
    println("[compute_times_domain!] Starting parallel computation")
    println("  Active cells: $n_active | Threads: $(Threads.nthreads()) | nat_apo=$nat_apo (λ=$(round(λ, digits=6)))")
    if verbose && print_every > 0
        println("  Verbose logging enabled (every $print_every processed elements in active set).")
    end

    # IMPORTANTE: Usa il numero MASSIMO di thread possibili, non solo nthreads()
    # Questo previene BoundsError se Julia riassegna i thread
    max_threads = Threads.maxthreadid()  # Usa maxthreadid invece di nthreads
    neighbor_updates = [Int[] for _ in 1:max_threads]

    # Thread-safe print lock
    plock = ReentrantLock()

    # Outcome counters
    immediate_removed = Threads.Atomic{Int}(0)
    scheduled_death   = Threads.Atomic{Int}(0)
    survived          = Threads.Atomic{Int}(0)
    timeout_cells     = Threads.Atomic{Int}(0)

    # --------------------------------
    # Parallel loop on active set
    # --------------------------------
    Threads.@threads for k in eachindex(active_cells)
        i = active_cells[k]
        tid = Threads.threadid()
        
        # SAFETY CHECK: assicurati che tid sia valido
        if tid > max_threads
            error("Thread ID $tid exceeds max_threads $max_threads")
        end

        # 1) GSM2 survival
        SP_cell = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)
        cell_df.sp[i] = SP_cell

        # Initialize defaults
        death_time = Inf
        recover_time_sample = 0.0
        death_type = 0
        X_gsm2 = cell_df.dam_X_dom[i]
        Y_gsm2 = cell_df.dam_Y_dom[i]

        # 2) Repair-driven outcome if SP < 1.0
        death_time, recover_time_sample, death_type, X_gsm2, Y_gsm2 =
            compute_repair_domain(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2; terminal_time=terminal_time)
        if death_time == 0.0
            # Immediate removal
            cell_df.is_cell[i] = 0
            append!(neighbor_updates[tid], cell_df.nei[i])
            cell_df.death_type[i] = death_type
            Threads.atomic_add!(immediate_removed, 1)
            if verbose && print_every > 0 && (k % print_every == 0)
                lock(plock) do
                    println("[t=$k | cell=$i | thr=$tid] SP=$(round(SP_cell, digits=4)) | immediate removal | death_type=$death_type")
                end
            end
            
            # Sample apoptosis and continue to next cell
            cell_df.apo_time[i] = Inf
            continue
        end

        # 3) Schedule radiogenic death or post-survival dynamics
        if isfinite(death_time) && death_time != 0.0 && death_type != -1
            # Radiogenic death scheduled
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i]   = Inf
            cell_df.death_type[i]   = death_type
            cell_df.dam_X_dom[i] .*= 0
            cell_df.dam_Y_dom[i] .*= 0

            if (cell_df.cell_cycle[i] == "M") && (cell_df.is_stem[i] == 1) && (cell_df.number_nei[i] > 0)
                division_time = rand(Gamma(1/2, 1/2))
                cell_df.death_time[i] = min(death_time, division_time)
            else
                cell_df.death_time[i] = death_time
            end

            Threads.atomic_add!(scheduled_death, 1)

            if verbose && print_every > 0 && (k % print_every == 0)
                lock(plock) do
                    println("[t=$k | cell=$i | thr=$tid] SP=$(round(SP_cell, digits=4)) | death_time=$(round(cell_df.death_time[i], digits=4)) | death_type=$death_type")
                end
            end

        elseif isfinite(recover_time_sample)
            # Survives radiation → may recover / divide
            cell_df.death_time[i]   = Inf
            cell_df.recover_time[i] = recover_time_sample
            cell_df.death_type[i]   = death_type

            cell_df.dam_X_dom[i] .*= 0
            cell_df.dam_Y_dom[i] .*= 0

            has_neighbors = (cell_df.number_nei[i] > 0)

            if has_neighbors
                if cell_df.cell_cycle[i] == "M"
                    cycle_time = rand(Gamma(1/2, 1/2))
                    if isfinite(recover_time_sample) && (cycle_time < recover_time_sample)
                        cell_df.recover_time[i] = Inf
                        cell_df.cycle_time[i]   = Inf
                        cell_df.death_time[i]   = cycle_time
                    else
                        cell_df.cycle_time[i] = cycle_time
                    end
                elseif cell_df.cell_cycle[i] == "G2"
                    cycle_time = rand(Gamma(1/4, 1/4))
                    if isfinite(recover_time_sample)
                        cell_df.cycle_time[i]   = max(cycle_time, recover_time_sample)
                        cell_df.recover_time[i] = Inf
                    else
                        cell_df.cycle_time[i] = cycle_time
                    end
                elseif cell_df.cell_cycle[i] == "S"
                    cycle_time = rand(Gamma(1/8, 1/8))
                    if isfinite(recover_time_sample)
                        cell_df.cycle_time[i]   = max(cycle_time, recover_time_sample)
                        cell_df.recover_time[i] = Inf
                    else
                        cell_df.cycle_time[i] = cycle_time
                    end
                elseif cell_df.cell_cycle[i] == "G1"
                    cycle_time = rand(Gamma(1/11, 1/11))
                    if isfinite(recover_time_sample)
                        cell_df.cycle_time[i]   = max(cycle_time, recover_time_sample)
                        cell_df.recover_time[i] = Inf
                    else
                        cell_df.cycle_time[i] = cycle_time
                    end
                end
            end

            Threads.atomic_add!(survived, 1)

            if verbose && print_every > 0 && (k % print_every == 0)
                lock(plock) do
                    println("[t=$k | cell=$i | thr=$tid] SP=$(round(SP_cell, digits=4)) | survive | recover_time=$(recover_time_sample) | cycle_time=$(cell_df.cycle_time[i])")
                end
            end

        elseif !isfinite(recover_time_sample) && !isfinite(death_time)
            # Timeout case
            cell_df.death_time[i]   = Inf
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i]   = Inf
            cell_df.death_type[i]   = death_type
            cell_df.dam_X_dom[i] = X_gsm2
            cell_df.dam_Y_dom[i] = Y_gsm2
            cell_df.dam_X_total[i] = sum(X_gsm2)
            cell_df.dam_Y_total[i] = sum(Y_gsm2)
            
            Threads.atomic_add!(timeout_cells, 1)
            
            if verbose && print_every > 0 && (k % print_every == 0)
                lock(plock) do
                    println("[t=$k | cell=$i | thr=$tid] SP=$(round(SP_cell, digits=4)) | timeout at T=$terminal_time")
                end
            end
        end

        # 4) Natural apoptosis time (Exponential with rate λ)
        cell_df.apo_time[i] = Inf

        if verbose && print_every > 0 && (k % print_every == 0)
            lock(plock) do
                println("[t=$k | cell=$i | thr=$tid] apo_time=$(round(cell_df.apo_time[i], digits=4))")
            end
        end
    end

    # --------------------------------
    # Serial merge of neighbor updates
    # --------------------------------
    for updates in neighbor_updates
        for nei_idx in updates
            cell_df.number_nei[nei_idx] += 1
        end
    end

    # --------------------------------
    # Summary
    # --------------------------------
    if summary
        elapsed = time() - start_t
        sp_vals = cell_df.sp[active_cells]
        mean_sp = mean(sp_vals)
        println("\n[compute_times_domain!] Summary")
        println("  Processed active cells    : $n_active")
        println("  Immediate removals        : $(immediate_removed[])")
        println("  Scheduled radiogenic death: $(scheduled_death[])")
        println("  Survivors                 : $(survived[])")
        println("  Timeout cells (T=$terminal_time)      : $(timeout_cells[])")
        println("  Mean SP (active set)      : $(round(mean_sp, digits=6))")
        println("  Elapsed time              : $(round(elapsed, digits=3)) s")
    end

    return nothing
end

"""
compute_repair_domain(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2;
                            terminal_time::Float64 = Inf,
                            rng::AbstractRNG = Random.default_rng(),
                            verbose::Bool = false,
                            max_events_log::Int = 0)

Simulate, via a **time-limited Gillespie-style** algorithm, the stochastic repair
dynamics of domain-level X-type lesions under the GSM2 model and return:
(death_time, recover_time, death_code, X_final, Y_final)
where:
- `death_time::Float64` — time of radiogenic death (finite if lethal).
- `recover_time::Float64` — total repair time (finite if recovery).
- `death_code::Int`:
  * `1`  → lethal outcome (by misrepair or interaction),
  * `0`  → all X lesions repaired (recovered),
  * `-1` → **timeout**: the time limit `terminal_time` is reached before a terminal event.
- `X_final`, `Y_final` — **final states** of the lesion vectors (copies).

# Model & Behavior
- **Immediate lethal if any Y-type lesion** exists: returns `(0.0, Inf, 1, copy(X), copy(Y))`.
- **Immediate recovery if no X-type lesions** (`sum(X) == 0`): returns `(Inf, 0.0, 0, copy(X), copy(Y))`.
- Otherwise, at each step and for each domain `j`:
  - **Repair** rate:        `r * X[j]`
  - **Misrepair** rate:     `a * X[j]`                (lethal)
  - **Interaction** rate:   `b * X[j] * (X[j]-1)`     (lethal, only if `X[j] > 1`)
- Draw the next reaction by inverse-CDF on the cumulative rates. The time increment is
    `dt = -log(U) / a0`, with `a0` the total propensity. The simulated time accumulates as
  `t ← t + au * dt`, where here `au = 4.0` (time-scaling factor).
- The simulation **stops** when:
  1) a **lethal** (misrepair/interaction) event occurs → `(t, Inf, 1, X_final, Y_final)`,
  2) **all X lesions are repaired** → `(Inf, t, 0, X_final, Y_final)`,
  3) the **time limit** `terminal_time` is reached before applying the next reaction → `(Inf, Inf, -1, X_final, Y_final)`.

# Arguments
- `X::Vector{Int64}`: counts of X-type lesions per domain (input is **not** mutated).
- `Y::Vector{Int64}`: counts of Y-type lesions per domain (input is **not** mutated).
- `gsm2::GSM2`: parameters with fields `a`, `b`, `r`.
- `terminal_time::Float64 = Inf`: hard cap on simulated time.
- `rng::AbstractRNG`: RNG used for sampling; pass a **per-thread RNG** in parallel settings.
- `verbose::Bool = false`: if `true`, prints English log lines for key events.
- `max_events_log::Int = 0`: maximum number of events to print (0 → unlimited).

# Returns
- `(death_time::Float64, recover_time::Float64, death_code::Int,
    X_final::Vector{Int64}, Y_final::Vector{Int64})`

# Notes
- Inputs `X`, `Y` are not modified; internal copies are used and returned.
- If `a0 == 0` while `sum_X > 0`, the process is returned as **recovered** with current time.
"""
function compute_repair_domain(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2;
                                terminal_time::Float64 = Inf,
                                rng::AbstractRNG = Random.default_rng(),
                                verbose::Bool = false,
                                max_events_log::Int = 0)

    # Immediate lethal if any Y-type lesion is present
    au = 4.
    if any(>(0), Y)
        if verbose
            println("[compute_repair_domain] Y-type lesion detected → immediate lethal outcome.")
        end
        return (0.0, Inf, 1, copy(X), copy(Y))
    end

    sum_X = sum(X)
    if sum_X == 0
        if verbose
            println("[compute_repair_domain] No X-type lesions → recovered (no repair time).")
        end
        return (Inf, 0.0, 0, copy(X), copy(Y))
    end

    # Work on a local copy to avoid mutating the input (important for threaded usage)
    X_work = copy(X)
    Y_work = copy(Y)

    a, b, r = gsm2.a, gsm2.b, gsm2.r
    n = length(X_work)
    current_time = 0.0

    # Pre-allocate rate arrays
    rates_r = Vector{Float64}(undef, n)
    rates_a = Vector{Float64}(undef, n)
    rates_b = Vector{Float64}(undef, n)

    # Optional logging control
    printed = 0

    while sum_X > 0 && current_time < terminal_time
        # 1) Build total propensity and component-wise rates
        a0 = 0.0
        @inbounds for i in 1:n
            xi = X_work[i]
            rr = r * xi
            ra = a * xi
            rb = (xi > 1) ? b * xi * (xi - 1) : 0.0
            rates_r[i] = rr
            rates_a[i] = ra
            rates_b[i] = rb
            a0 += rr + ra + rb
        end

        if a0 <= 0.0
            # No possible reactions despite sum_X > 0: return a safe recovery
            if verbose
                println("[compute_repair_domain] a0=0 with remaining X; treating as recovery.")
            end
            return (Inf, current_time, 0, X_work, Y_work)
        end

        # 2) Draw next event and time increment
        threshold = rand(rng) * a0
        cumulative = 0.0
        reac_type = 0  # 1=repair, 2=misrepair, 3=interaction
        reac_domain = 0

        # Repair first
        @inbounds for i in 1:n
            cumulative += rates_r[i]
            if cumulative >= threshold
                reac_type = 1
                reac_domain = i
                break
            end
        end

        # If not found, misrepair
        if reac_type == 0
            @inbounds for i in 1:n
                cumulative += rates_a[i]
                if cumulative >= threshold
                    reac_type = 2
                    reac_domain = i
                    break
                end
            end
        end

        # If still not found, interaction
        if reac_type == 0
            @inbounds for i in 1:n
                cumulative += rates_b[i]
                if cumulative >= threshold
                    reac_type = 3
                    reac_domain = i
                    break
                end
            end
        end

        # Gillespie time step
        dt = -log(rand(rng)) / a0
        new_time = current_time + au*dt
        
        # Check if we would exceed T before applying the reaction
        if new_time >= terminal_time
            if verbose
                println("[compute_repair_domain] Time limit T=$terminal_time reached at t=$(round(current_time, digits=4)) → timeout")
            end
            return (Inf, Inf, -1, X_work, Y_work)
        end
        
        current_time = new_time

        # 3) Apply reaction
        if reac_type == 1
            # Repair: remove one lesion
            @inbounds X_work[reac_domain] -= 1
            sum_X -= 1

            if verbose && (max_events_log <= 0 || printed < max_events_log)
                printed += 1
                println("[compute_repair_domain] t=$(round(current_time, digits=4)) | REPAIR in domain $reac_domain → X[$reac_domain]=$(X_work[reac_domain]); remaining=$sum_X")
            end
        else
            # Misrepair or interaction → lethal outcome
            if verbose && (max_events_log <= 0 || printed < max_events_log)
                printed += 1
                kind = (reac_type == 2 ? "MISREPAIR" : "INTERACTION")
                println("[compute_repair_domain] t=$(round(current_time, digits=4)) | $kind in domain $reac_domain → LETHAL")
            end
            return (current_time, Inf, 1, X_work, Y_work)
        end
    end

    # Check why we exited the loop
    if current_time >= T
        # Time limit reached
        if verbose
            println("[compute_repair_domain] Time limit T=$terminal_time reached with sum_X=$sum_X remaining → timeout")
        end
        return (Inf, Inf, -1, X_work, Y_work)
    else
        # All lesions repaired → recovered, return scaled time
        if verbose
            println("[compute_repair_domain] All X repaired at t=$(round(current_time, digits=4)) → RECOVER")
        end
        return (Inf, current_time, 0, X_work, Y_work)
    end
end

