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
        if isfinite(death_time) && death_time != 0.0
            # Radiogenic death scheduled
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i]   = Inf
            cell_df.death_type[i]   = death_type
            cell_df.dam_X_dom[i] .*= 0
            cell_df.dam_Y_dom[i] .*= 0

            if (cell_df.cell_cycle[i] == "M") && (cell_df.number_nei[i] > 0)
                division_time = rand(Gamma(0.5, 2.))
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
                    cycle_time = rand(Gamma(0.5, 2.))
                    if isfinite(recover_time_sample) && (cycle_time < recover_time_sample)
                        cell_df.recover_time[i] = Inf
                        cell_df.cycle_time[i]   = Inf
                        cell_df.death_time[i]   = cycle_time
                    else
                        cell_df.cycle_time[i] = cycle_time
                    end
                elseif cell_df.cell_cycle[i] == "G2"
                    cycle_time = rand(Gamma(0.5*3, 2.))
                    cell_df.cycle_time[i]   = max(cycle_time, recover_time_sample)
                    cell_df.recover_time[i] = recover_time_sample
                elseif cell_df.cell_cycle[i] == "S"
                    cycle_time = rand(Gamma(0.5*8, 2.))
                    cell_df.cycle_time[i]   = max(cycle_time, recover_time_sample)
                    cell_df.recover_time[i] = recover_time_sample
                elseif cell_df.cell_cycle[i] == "G1"
                    cycle_time = rand(Gamma(0.5*12, 2.))
                    cell_df.cycle_time[i]   = max(cycle_time, recover_time_sample)
                    cell_df.recover_time[i] = recover_time_sample
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

"""
    generate_cycle_time(phase::String) -> Float64

Generate a random cell‑cycle duration for the given `phase` using a Gamma
distribution whose parameters are stored in `PHASE_DURATIONS`.

# Arguments
- `phase::String`: Name of the cell‑cycle phase.  
    Must correspond to a key in the global dictionary `PHASE_DURATIONS`.

# Behavior
The function looks up the entry `PHASE_DURATIONS[phase]`, which must contain:
- `.shape` — the Gamma shape parameter α
- `.scale` — the Gamma scale parameter θ

It then draws a sample from `Gamma(α, θ)` and returns it.

# Returns
- `Float64`: A random duration sampled from the Gamma distribution of that phase.

# Notes
- Marked as `@inline` to encourage compiler inlining for performance.
- Assumes `PHASE_DURATIONS` is globally defined and properly structured.
"""
@inline function generate_cycle_time(phase::String)::Float64
    params = PHASE_DURATIONS[phase]
    return rand(Gamma(params.shape, params.scale))
end

"""
    assign_random_phase() -> String

Randomly assign a cell-cycle phase based on a 24-hour weighted schedule.

A uniform random number `ru` is sampled in `[0, 24)`, and the phase is assigned
according to the following intervals:

- `0  ≤ ru ≤ 12`  → `"G1"`   (50% probability)
- `12 < ru ≤ 20`  → `"S"`    (33% probability)
- `20 < ru ≤ 23`  → `"G2"`   (12.5% probability)
- `23 < ru ≤ 24`  → `"M"`    (4.17% probability)

# Returns
- `String`: One of `"G1"`, `"S"`, `"G2"`, `"M"`.

# Notes
- The probabilities correspond to relative phase durations in hours
    (12h, 8h, 3h, 1h).
- This is a simple, time‑proportional phase initialization often used to
    randomize the initial state of a cell population.
"""
@inline function assign_random_phase()::String
    ru = rand() * 24.0
    return ru <= 12.0 ? "G1" : ru <= 20.0 ? "S" : ru <= 23.0 ? "G2" : "M"
end

@inline is_valid_index(idx::Int32, n::Int32)::Bool = 1 <= idx <= n

@inline function is_time_due(t::Float64, eps::Float64, treat_neg::Bool)::Bool
    (ismissing(t) || isinf(t)) && return false
    return treat_neg ? (t <= eps) : (abs(t) <= eps)
end

function CellPopulation(df::DataFrame)
    n = nrow(df)
    
    # Helper to safely convert optional columns
    get_int8_col(name) = hasproperty(df, name) ? Vector{Int8}(df[!, name]) : nothing
    get_int32_col(name) = hasproperty(df, name) ? Vector{Int32}(df[!, name]) : nothing
    
    # Count alive cells
    n_alive = count(df.is_cell .== 1)
    
    pop = CellPopulation(
        Vector{Int8}(df.is_cell),
        Vector{Int8}(df.can_divide),
        get_int8_col(:is_stem),
        get_int8_col(:is_death_rad),
        Vector{Float64}(df.death_time),
        Vector{Float64}(df.cycle_time),
        Vector{Float64}(df.recover_time),
        [String7(string(s)) for s in df.cell_cycle],
        Vector{Int16}(df.number_nei),
        [Vector{Int32}(nei) for nei in df.nei],
        get_int32_col(:x),
        get_int32_col(:y),
        Int32(n),
        Int32(n_alive),
        hasproperty(df, :index) ? Vector{Int32}(df.index) : Vector{Int32}(1:n)
    )
    
    return pop
end

function to_dataframe(pop::CellPopulation; alive_only::Bool=false)::DataFrame
    if alive_only
        alive_mask = pop.is_cell .== 1
        indices = findall(alive_mask)
        
        df = DataFrame(
            index = pop.indices[indices],
            is_cell = pop.is_cell[indices],
            can_divide = pop.can_divide[indices],
            death_time = pop.death_time[indices],
            cycle_time = pop.cycle_time[indices],
            recover_time = pop.recover_time[indices],
            cell_cycle = String.(pop.cell_cycle[indices]),
            number_nei = pop.number_nei[indices],
            nei = pop.nei[indices]
        )
    else
        df = DataFrame(
            index = pop.indices,
            is_cell = pop.is_cell,
            can_divide = pop.can_divide,
            death_time = pop.death_time,
            cycle_time = pop.cycle_time,
            recover_time = pop.recover_time,
            cell_cycle = String.(pop.cell_cycle),
            number_nei = pop.number_nei,
            nei = pop.nei
        )
    end
    
    # Add optional columns
    !isnothing(pop.is_stem) && (df.is_stem = pop.is_stem)
    !isnothing(pop.is_death_rad) && (df.is_death_rad = pop.is_death_rad)
    !isnothing(pop.x) && (df.x = pop.x)
    !isnothing(pop.y) && (df.y = pop.y)
    
    return df
end

"""
    compute_next_event(cell_df::DataFrame) -> Tuple{Float64, Int64, String}

Compute the next event time across alive cells (`is_cell == 1`), considering the
`death_time` and `cycle_time` columns, and return the earliest event.

# Arguments
- `cell_df::DataFrame`: Must contain:
    - `is_cell` (1 = alive; others ignored)
    - `death_time`, `cycle_time` (may contain `missing`)

# Returns
A 3-tuple `(min_time, min_idx, min_event)` where:
- `min_time::Float64`: the minimum event time among alive cells (or `Inf` if none).
- `min_idx::Int64`: the **row index in the original DataFrame** of the chosen cell (or `0` if none).
- `min_event::String`: `"death_time"` or `"cycle_time"` (or `""` if none).

# Behavior & Notes
- Only considers rows with `is_cell == 1`.
- Ignores `missing` event times.
- Tie-breaking: if the minimum time is equal for multiple candidates,
    the priority order is `["death_time", "cycle_time"]` (first match wins).
- Returns `(Inf, 0, "")` if no alive cells or no finite event times exist.
"""
function compute_next_event(pop::CellPopulation)::Tuple{Float64, Int32, String}
    min_time = Inf
    min_idx = Int32(0)
    min_event = ""
    
    # Cache vectors for faster access
    is_cell = pop.is_cell
    death_times = pop.death_time
    cycle_times = pop.cycle_time
    n = pop.n_cells
    
    @inbounds for i in Int32(1):n
        # Skip dead cells
        is_cell[i] == 0 && continue
        
        # Check death_time
        dt = death_times[i]
        if !isinf(dt) && dt < min_time
            min_time = dt
            min_idx = i
            min_event = "death_time"
        end
        
        # Check cycle_time
        ct = cycle_times[i]
        if !isinf(ct) && ct < min_time
            min_time = ct
            min_idx = i
            min_event = "cycle_time"
        end
    end
    
    return (min_time, min_idx, min_event)
end

"""
    update_time!(cell_df::DataFrame, elapsed_time::Float64; clamp_nonnegative::Bool=false)

Decrease countdown timers by `elapsed_time` for all alive cells (`is_cell == 1`).

The function updates the following columns if present (and if values are finite and non-missing):
- `death_time`
- `cycle_time`
- `recover_time` (optional column)

# Arguments
- `cell_df::DataFrame`: Simulation table with at least `is_cell` and any subset of
    the time columns (`death_time`, `cycle_time`, `recover_time`).
- `elapsed_time::Float64`: Positive amount of time to subtract from each applicable timer.

# Keyword Arguments
- `clamp_nonnegative::Bool=false`: If `true`, times are clamped at `0.0` after subtraction.
    If `false`, times can become negative (useful if another handler triggers exactly at/below 0).

# Behavior & Notes
- Only rows with `is_cell == 1` are updated.
- `missing` and `Inf` values are ignored (left unchanged).
- If a time column is not present, it is silently skipped.
- This function modifies `cell_df` **in place**.

# Returns
- `nothing`
"""
function update_time!(pop::CellPopulation, elapsed::Float64)
    @assert elapsed >= 0 "elapsed_time must be non-negative"
    
    # Cache vectors
    is_cell = pop.is_cell
    death_times = pop.death_time
    cycle_times = pop.cycle_time
    recover_times = pop.recover_time
    n = pop.n_cells
    
    @inbounds for i in Int32(1):n
        is_cell[i] == 0 && continue
        
        # Update death_time
        dt = death_times[i]
        !isinf(dt) && (death_times[i] = dt - elapsed)
        
        # Update cycle_time
        ct = cycle_times[i]
        !isinf(ct) && (cycle_times[i] = ct - elapsed)
        
        # Update recover_time
        rt = recover_times[i]
        !isinf(rt) && (recover_times[i] = rt - elapsed)
    end
    
    return nothing
end

"""
_perform_division!(cell_df::DataFrame, parent_idx::Int64, nat_apo::Float64)

Perform a cell division for the cell at `parent_idx` into one empty neighboring
site. The function:

1. Finds the first empty neighbor (where `is_cell == 0`) in `cell_df.nei[parent_idx]`.
2. Creates a **daughter** at that neighbor:
    - `is_cell = 1`
    - `cell_cycle = "G1"`
    - `cycle_time = generate_cycle_time("G1")`
    - `death_time = Inf`
    - `can_divide = 1`
3. Resets the **parent** to `"G1"` with a freshly generated cycle time and sets
    `can_divide = 1`.
4. Updates neighbor-availability counts (`number_nei`) for affected cells:
    - For all cells that neighbor either the parent or the daughter, decrement
        `number_nei` by 1 (since one empty spot is now occupied).
    - If a neighbor’s `number_nei` becomes `0` and it had `can_divide == 1`,
        then set its `cycle_time = Inf` and `can_divide = 0` (blocked division).
5. Updates the parent and daughter’s own `number_nei`:
    - Parent loses one free neighbor (`-1`) because the daughter occupies a spot.
    - Daughter’s `number_nei` becomes the count of empty neighbors around her.

# Arguments
- `cell_df::DataFrame`: Simulation data. Must include at least:
    - `is_cell::Vector{Int}` (1=alive, 0=empty),
    - `nei::Vector{<:AbstractVector{Int}}` (neighbors by row),
    - `cell_cycle::Vector{String}`,
    - `cycle_time::Vector{<:Union{Missing, Float64}}`,
    - `death_time::Vector{<:Union{Missing, Float64}}`,
    - `can_divide::Vector{Int}`,
    - `number_nei::Vector{Int}`.
- `parent_idx::Int64`: Row index of the dividing parent cell.
- `nat_apo::Float64`: Natural apoptosis parameter (passed for API symmetry;
    not used directly here, but may be used by downstream hooks if extended).

# Returns
- `nothing`. Mutates `cell_df` in-place.

# Notes
- If no empty neighbor is found, a warning is issued and the function returns.
- This function assumes `number_nei[i]` tracks the **count of empty neighbor spots**
    for cell `i`. If your convention differs, adjust the updates accordingly.
- Tie-breaking for the daughter position is “first empty neighbor in `nei[parent]`”.
- Calls `generate_cycle_time("G1")` to seed new G1 durations.

"""
function _perform_division!(pop::CellPopulation, parent_idx::Int32, nat_apo::Float64)
    n = pop.n_cells
    parent_neighbors = pop.nei[parent_idx]
    
    # Find empty neighbor
    is_cell = pop.is_cell
    daughter_idx = Int32(0)
    
    @inbounds for n_idx in parent_neighbors
        if is_valid_index(n_idx, n) && is_cell[n_idx] == 0
            daughter_idx = n_idx
            break
        end
    end
    
    if daughter_idx == 0
        @warn "Division attempted but no empty neighbor" parent=parent_idx
        return nothing
    end
    
    # Create daughter cell
    @inbounds begin
        pop.is_cell[daughter_idx] = 1
        pop.cell_cycle[daughter_idx] = String7("G1")
        pop.cycle_time[daughter_idx] = generate_cycle_time("G1")
        pop.death_time[daughter_idx] = Inf
        pop.can_divide[daughter_idx] = 1
    end
    
    # Reset parent to G1
    @inbounds begin
        pop.cell_cycle[parent_idx] = String7("G1")
        pop.cycle_time[parent_idx] = generate_cycle_time("G1")
        pop.can_divide[parent_idx] = 1
    end
    
    # Update neighbor counts
    daughter_neighbors = pop.nei[daughter_idx]
    number_nei = pop.number_nei
    can_divide = pop.can_divide
    cycle_times = pop.cycle_time
    
    # Use boolean mask for small neighbor lists
    if length(parent_neighbors) + length(daughter_neighbors) < 20
        updated = falses(n)
        
        @inbounds for n_idx in parent_neighbors
            if is_valid_index(n_idx, n) && is_cell[n_idx] == 1 && !updated[n_idx]
                number_nei[n_idx] -= 1
                updated[n_idx] = true
                
                if number_nei[n_idx] == 0 && can_divide[n_idx] == 1
                    cycle_times[n_idx] = Inf
                    can_divide[n_idx] = 0
                end
            end
        end
        
        @inbounds for n_idx in daughter_neighbors
            if is_valid_index(n_idx, n) && is_cell[n_idx] == 1 && !updated[n_idx]
                number_nei[n_idx] -= 1
                updated[n_idx] = true
                
                if number_nei[n_idx] == 0 && can_divide[n_idx] == 1
                    cycle_times[n_idx] = Inf
                    can_divide[n_idx] = 0
                end
            end
        end
    else
        # Use Set for large neighbor lists
        affected = Set{Int32}()
        sizehint!(affected, length(parent_neighbors) + length(daughter_neighbors))
        
        @inbounds begin
            for i in parent_neighbors
                push!(affected, i)
            end
            for i in daughter_neighbors
                push!(affected, i)
            end
        end
        
        @inbounds for n_idx in affected
            if is_valid_index(n_idx, n) && is_cell[n_idx] == 1
                number_nei[n_idx] -= 1
                
                if number_nei[n_idx] == 0 && can_divide[n_idx] == 1
                    cycle_times[n_idx] = Inf
                    can_divide[n_idx] = 0
                end
            end
        end
    end
    
    # Update cells' own neighbor counts
    pop.number_nei[parent_idx] -= 1
    
    # Count empty neighbors for daughter
    empty_count = Int16(0)
    @inbounds for nb in daughter_neighbors
        if is_valid_index(nb, n) && is_cell[nb] == 0
            empty_count += 1
        end
    end
    pop.number_nei[daughter_idx] = empty_count
    
    # Update alive count
    pop.n_alive += 1
    
    return nothing
end

"""
    _handle_cell_removal!(cell_df::DataFrame, removed_idx::Int64, is_natural_apoptosis::Bool)

Remove a cell from the lattice/graph by marking its slot empty and updating
dependent state for the removed cell and its neighbors.

# Behavior
1. If the cell is already empty (`is_cell == 0`), the function returns immediately.
2. For the removed cell (`removed_idx`):
    - `is_cell = 0`
    - `death_time = Inf`, `cycle_time = Inf`, `can_divide = 0`
    - If present: `apo_time = Inf`, `recover_time = Inf`
    - If `is_natural_apoptosis` and `is_death_rad` exists, set `is_death_rad = 0`
3. For each alive neighbor `n` in `nei[removed_idx]`:
    - Increment `number_nei[n]` by 1 (one additional free spot now exists).
    - If `number_nei[n]` becomes `1` and `can_divide[n] == 0`, the neighbor is
        unblocked and reinitialized:
        - Assign a new phase via `assign_random_phase()`
        - Set `cell_cycle[n]` to that phase
        - Set `cycle_time[n] = generate_cycle_time(new_phase)`
        - Set `can_divide[n] = 1`

# Arguments
- `cell_df::DataFrame`: Simulation table. Expected columns:
    - `is_cell::Vector{Int}`, `nei::Vector{<:AbstractVector{Int}}`,
    - `cell_cycle::Vector{String}`, `cycle_time`, `death_time`,
    - `can_divide::Vector{Int}`, `number_nei::Vector{Int}`.
    - Optional: `apo_time`, `recover_time`, `is_death_rad`.
- `removed_idx::Int64`: Row index of the cell to remove.
- `is_natural_apoptosis::Bool`: If `true`, marks `is_death_rad=0` (if present).

# Returns
- `nothing` (mutates `cell_df` in-place).

# Notes
- Assumes `number_nei` stores the count of **empty** neighboring positions.
- Neighbor indices are checked for bounds; out-of-range neighbors are ignored.
- Reinitialization policy when a neighbor becomes unblocked can be adapted to
    your model (currently random phase + gamma cycle time).
"""
function _handle_cell_removal!(pop::CellPopulation, removed_idx::Int32, 
                                is_natural_apoptosis::Bool)
    pop.is_cell[removed_idx] == 0 && return nothing
    
    # Mark as dead
    @inbounds begin
        pop.is_cell[removed_idx] = 0
        pop.death_time[removed_idx] = Inf
        pop.cycle_time[removed_idx] = Inf
        pop.recover_time[removed_idx] = Inf
        pop.can_divide[removed_idx] = 0
    end
    
    # Update optional fields
    if is_natural_apoptosis && !isnothing(pop.is_death_rad)
        pop.is_death_rad[removed_idx] = 0
    end
    
    # Update alive count
    pop.n_alive -= 1
    
    # Update neighbors
    n = pop.n_cells
    neighbors = pop.nei[removed_idx]
    
    # Cache vectors
    is_cell = pop.is_cell
    number_nei = pop.number_nei
    can_divide = pop.can_divide
    cell_cycle = pop.cell_cycle
    cycle_times = pop.cycle_time
    
    @inbounds for n_idx in neighbors
        if is_valid_index(n_idx, n) && is_cell[n_idx] == 1
            number_nei[n_idx] += 1
            
            # Unblock if this creates first free space
            if number_nei[n_idx] == 1 && can_divide[n_idx] == 0
                new_phase = assign_random_phase()
                cell_cycle[n_idx] = String7(new_phase)
                cycle_times[n_idx] = generate_cycle_time(new_phase)
                can_divide[n_idx] = 1
            end
        end
    end
    
    return nothing
end

"""
update_ABM!(cell_df::DataFrame, next_time::Float64, event::String, idx::Int64, nat_apo::Float64)

Advance the simulation by `next_time`, then handle the event that occurred at row `idx`
according to an ABM (agent-based model) cell-cycle and death logic. Mutates `cell_df` in-place.

# Arguments
- `cell_df::DataFrame`: Simulation state. Expected columns include:
    - `is_cell::Vector{Int}` (1=alive, 0=empty),
    - `cell_cycle::Vector{String}` (e.g., "G1","S","G2","M"),
    - `cycle_time`, `death_time` (countdown timers; may include `missing`/`Inf`),
    - `can_divide::Vector{Int}`,
    - `number_nei::Vector{Int}` (count of **empty** neighboring sites),
    - `nei::Vector{<:AbstractVector{Int}}` (neighbor indices).
    Global structures used:
    - `PHASE_TRANSITION::Dict{String,String}` mapping current phase → next phase,
    - `generate_cycle_time(phase::String)`.
- `next_time::Float64`: Time increment to subtract from all applicable timers via `update_time!`.
- `event::String`: The event that fired at `idx`. Supported values:
    - `"death_time"`
    - `"cycle_time"`
- `idx::Int64`: Row index where the event happened (1-based).
- `nat_apo::Float64`: Natural apoptosis parameter passed to downstream hooks (used in `check_time!` or division routines).

# Behavior
1. **Global time update**: `update_time!(cell_df, next_time)` subtracts `next_time` from timers
    for alive cells, skipping `missing`/`Inf`.
2. **Event handling**:
   - **death_time**: remove the cell via `_handle_cell_removal!(..., is_natural_apoptosis=false)`.
   - **cycle_time**:
        - Read `current_phase = cell_df.cell_cycle[idx]` and `has_space = cell_df.number_nei[idx] > 0`.
        - If `current_phase == "M"`:
            - If `has_space`: perform division via `_perform_division!(cell_df, idx, nat_apo)`.
            - Else: block progression (`cell_cycle="G1"`, `cycle_time=Inf`, `can_divide=0`).
        - Else (`"G1"`, `"S"`, `"G2"`):
            - Transition to `next_phase = PHASE_TRANSITION[current_phase]`.
            - If `has_space`: set `cycle_time = generate_cycle_time(next_phase)`, `can_divide=1`.
            - Else: block (`cycle_time=Inf`, `can_divide=0`).
3. **Cleanup**: `check_time!(cell_df, nat_apo)` to resolve any timers that hit exactly zero
        and enforce model consistency (e.g., cascading deaths).

# Returns
- `nothing` — `cell_df` is modified in place.

# Notes
- Assumes `idx` refers to a **currently alive** cell when `event=="cycle_time"`.
- If your model uses a different policy for “blocked M” (e.g., `"M_blocked"`), adjust the assignment.
- If multiple event types can fire at equal time, ensure upstream selection (e.g., `compute_next_event`)
    uses your desired tie-breaking rules.
"""
function update_ABM!(pop::CellPopulation, next_time::Float64, event::String, 
                    idx::Int32, nat_apo::Float64)
    # Update times
    update_time!(pop, next_time)
    
    # Handle event
    if event == "death_time"
        _handle_cell_removal!(pop, idx, false)
        
    elseif event == "cycle_time"
        @inbounds begin
            current_phase = String(pop.cell_cycle[idx])
            has_space = pop.number_nei[idx] > 0
            
            if current_phase == "M"
                if has_space
                    _perform_division!(pop, idx, nat_apo)
                else
                    pop.cell_cycle[idx] = String7("G1")
                    pop.cycle_time[idx] = Inf
                    pop.can_divide[idx] = 0
                end
            else
                next_phase = PHASE_TRANSITION[current_phase]
                pop.cell_cycle[idx] = String7(next_phase)
                
                if has_space
                    pop.cycle_time[idx] = generate_cycle_time(next_phase)
                    pop.can_divide[idx] = 1
                else
                    pop.cycle_time[idx] = Inf
                    pop.can_divide[idx] = 0
                end
            end
        end
    else
        @warn "Unhandled event" event idx
    end
    
    # Clean up zero times
    check_time!(pop, nat_apo)
    
    return nothing
end

"""
check_time!(cell_df::DataFrame, nat_apo::Float64;
                eps::Float64=0.0, treat_negatives_as_due::Bool=false)

Resolve timers that have reached (approximately) zero for all **alive** cells
(`is_cell == 1`) and enforce model side-effects.

# Behavior
For each alive row `i`:
- If `death_time[i]` is due (see below), remove the cell:
    `_handle_cell_removal!(cell_df, i, false)`.
- If `recover_time` exists and is due, set `recover_time[i] = Inf`.
- If `cycle_time[i]` is due, set `cycle_time[i] = Inf`.

A time `t` is considered **due** if:
- `eps == 0`: exactly `t == 0.0`
- `eps > 0`: `abs(t) ≤ eps`
- If `treat_negatives_as_due == true`, then `t ≤ eps` (allowing negative times to trigger)

`missing` and `Inf` are ignored.

# Arguments
- `cell_df::DataFrame`: Simulation state with at least `is_cell`, `death_time`, `cycle_time`.
    Optionally: `recover_time`.
- `nat_apo::Float64`: Natural apoptosis parameter (not used directly here; included for API
    symmetry with other update functions).

# Keyword Arguments
- `eps::Float64=0.0`: Numerical tolerance for deciding if a timer is due.
- `treat_negatives_as_due::Bool=false`: If `true`, times ≤ `eps` trigger handling.

# Returns
- `nothing` — mutates `cell_df` in-place.

# Notes
- This function focuses on **post-step cleanups** after `update_time!`. It is typically
    called at the end of an event update (e.g., in `update_ABM!`).
- Removal may change neighbor-related fields; ensure your invariants (e.g., `number_nei ≥ 0`)
    are maintained elsewhere or add debug assertions if needed.
"""
function check_time!(pop::CellPopulation, nat_apo::Float64; eps::Float64=0.0)
    # Cache vectors
    is_cell = pop.is_cell
    death_times = pop.death_time
    cycle_times = pop.cycle_time
    recover_times = pop.recover_time
    n = pop.n_cells
    
    @inbounds for i in Int32(1):n
        is_cell[i] == 0 && continue
        
        # Handle death_time
        dt = death_times[i]
        if !isinf(dt) && abs(dt) <= eps
            _handle_cell_removal!(pop, i, false)
            continue
        end
        
        # Handle recover_time
        rt = recover_times[i]
        !isinf(rt) && abs(rt) <= eps && (recover_times[i] = Inf)
        
        # Handle cycle_time
        ct = cycle_times[i]
        !isinf(ct) && abs(ct) <= eps && (cycle_times[i] = Inf)
    end
    
    return nothing
end

"""
record_timepoint!(ts::SimulationTimeSeries, t::Float64, cell_df::DataFrame;
                        phases::Vector{String}=["G1","S","G2","M"],
                        validate_lengths::Bool=false)

Append a snapshot of the current simulation state at time `t` to the
`SimulationTimeSeries` buffers. Counts are computed among **alive** cells
(`is_cell == 1`).

# Recorded fields
- `ts.time`: push `t`
- `ts.total_cells`: number of alive cells
- `ts.stem_cells`: alive cells with `is_stem == 1` (0 if the column is missing)
- `ts.non_stem_cells`: alive cells with `is_stem == 0` (0 if the column is missing)
- Phase counts among alive cells:
    - `ts.g1_cells`, `ts.s_cells`, `ts.g2_cells`, `ts.m_cells`
    (or whichever `phases` you request, if you adapt your `ts` fields accordingly)

# Arguments
- `ts::SimulationTimeSeries`: Accumulator of time-series vectors.
- `t::Float64`: Current simulation time to record.
- `cell_df::DataFrame`: Simulation state containing (at least) columns:
    - `is_cell::Vector{Int}` (1=alive, 0=empty),
    - `cell_cycle::Vector{String}` (e.g., "G1","S","G2","M"),
    - optionally `is_stem::Vector{Int}`.

# Keyword Arguments
- `phases::Vector{String}=["G1","S","G2","M"]`: Which phases to count. The default
    assumes standard phases and that `ts` has corresponding fields.
- `validate_lengths::Bool=false`: If `true`, performs a lightweight consistency check
    so all `ts` vectors remain the same length after the push.

# Returns
- `nothing` — mutates `ts` in-place.

# Notes
- If `is_stem` is absent, both `ts.stem_cells` and `ts.non_stem_cells` record `0`.
- For performance, counts are computed with boolean summation.
- If you change `phases`, ensure `ts` has corresponding fields and adjust the code
    below (or switch to a dictionary of phase series).
"""

function record_timepoint!(ts::SimulationTimeSeries, t::Float64, pop::CellPopulation)
    push!(ts.time, t)
    push!(ts.total_cells, pop.n_alive)  # Use cached count!
    
    # Count by phase (still need to iterate, but fast)
    g1_count = Int32(0)
    s_count = Int32(0)
    g2_count = Int32(0)
    m_count = Int32(0)
    
    @inbounds for i in Int32(1):pop.n_cells
        if pop.is_cell[i] == 1
            phase = pop.cell_cycle[i]
            if phase == "G1"
                g1_count += 1
            elseif phase == "S"
                s_count += 1
            elseif phase == "G2"
                g2_count += 1
            elseif phase == "M"
                m_count += 1
            end
        end
    end
    
    push!(ts.g1_cells, g1_count)
    push!(ts.s_cells, s_count)
    push!(ts.g2_cells, g2_count)
    push!(ts.m_cells, m_count)
    
    # Stem cells (if present)
    if !isnothing(pop.is_stem)
        stem_count = Int32(0)
        non_stem_count = Int32(0)
        @inbounds for i in Int32(1):pop.n_cells
            if pop.is_cell[i] == 1
                if pop.is_stem[i] == 1
                    stem_count += 1
                else
                    non_stem_count += 1
                end
            end
        end
        push!(ts.stem_cells, stem_count)
        push!(ts.non_stem_cells, non_stem_count)
    else
        push!(ts.stem_cells, Int32(0))
        push!(ts.non_stem_cells, Int32(0))
    end
end

"""
print_initial_stats(cell_df::DataFrame; io::IO=stdout, prefix::AbstractString="")

Print a summary of the initial simulation state with counts and simple time statistics,
computed among **alive** cells (`is_cell == 1`).

# Printed Summary
- Total active (alive) cells
- Cells marked to die (finite `death_time`)
- Cells scheduled for repair/progression (finite `recover_time` OR finite `cycle_time`)
- Unaffected cells (active - dying - recovering)
- Surviving fraction = (recovering + unaffected) / total_active
- Max death time (hours)
- Max recovery time (hours; prints 0 if recover_time column is absent)
- Median recovery time (hours; prints 0 if recover_time column is absent)

# Arguments
- `cell_df::DataFrame`: Simulation table. Expected columns:
    - `is_cell::Vector{Int}` (1=alive, 0=empty)
    - `death_time`, `cycle_time` (may include `missing`/`Inf`)
    - Optional: `recover_time`

# Keyword Arguments
- `io::IO=stdout`: Where to print.
- `prefix::AbstractString=""`: Optional prefix added to each printed line (e.g., "[Init] ").

# Returns
A `NamedTuple` with fields:
- `max_death::Float64`
- `max_recover::Float64`
- `median_recover::Float64`
- `surviving_fraction::Float64`

# Notes
- `missing` and `Inf` are ignored for statistics.
- If no alive cells are present, all counts are 0 and the surviving fraction is 0.0.
- If `recover_time` is absent, recovery stats are reported as 0.0 and counts derive only from `cycle_time`.
"""
function print_initial_stats(pop::CellPopulation)
    println("\n", "="^70)
    println("INITIAL STATE")
    println("="^70)
    println("Active cells: $(pop.n_alive)")
    println("="^70, "\n")
end

"""
run_simulation_abm!(cell_df::DataFrame, nat_apo::Float64;
                        terminal_time::Float64=48.0,
                        snapshot_times::Vector{Int}=[1, 6, 12, 24],
                        print_interval::Float64=1.0,
                        max_events::Int=typemax(Int),
                        eps_time::Float64=1e-9,
                        verbose::Bool=true,
                        snapshots_at_start::Bool=true)

Run the ABM simulation loop until `terminal_time` (in hours) or until no further
events are available, updating `cell_df` in-place. Returns a time series and a
dictionary of snapshots.

# Arguments
- `cell_df::DataFrame`: Simulation state. Mutated in place by event handlers.
- `nat_apo::Float64`: Natural apoptosis parameter passed to update/check functions.

# Keyword Arguments
- `terminal_time::Float64=48.0`: Stop time (inclusive within a small tolerance `eps_time`).
- `snapshot_times::Vector{Int}=[1, 6, 12, 24]`: Whole-hour marks to save population snapshots.
- `print_interval::Float64=1.0`: Console progress print step in hours. If `<= 0`, disables periodic prints.
- `max_events::Int=typemax(Int)`: Hard cap to avoid runaway loops.
- `eps_time::Float64=1e-9`: Time comparison tolerance (e.g., handle floating-point accumulation).
- `verbose::Bool=true`: If true, prints initial stats, progress, and final summary.
- `snapshots_at_start::Bool=true`: If true, saves a snapshot at `t=0`.

# Behavior
- Initializes a `SimulationTimeSeries`, records the initial state at `t=0`.
- In each loop:
    1. Finds `(next_time, idx, event) = compute_next_event(cell_df)`.
    2. If `next_time` is `Inf`, stops (no events remaining).
    3. Advances time by `next_time`. If this would exceed `terminal_time` with tolerance,
        truncates final step to stop at `terminal_time` and ends afterwards.
    4. Processes the event via `update_ABM!`.
    5. Records a timepoint into the time series.
    6. Optionally prints progress at fixed wall-clock intervals.
    7. Creates snapshots when crossing integer hours in `snapshot_times`.
- On completion, prints a final summary (if `verbose`) and returns `(ts, snapshots)`.

# Returns
- `(ts::SimulationTimeSeries, snapshots::Dict{Int, DataFrame})`

# Notes
- Uses `create_snapshot(cell_df)` to copy alive rows into each snapshot.
- Assumes `compute_next_event`, `update_ABM!`, `record_timepoint!`, and `create_snapshot`
    are defined and consistent with your model.
"""
function run_simulation_abm!(pop::CellPopulation, nat_apo::Float64;
                            terminal_time::Float64=48.0,
                            snapshot_times::Vector{Int}=[1, 6, 12, 24],
                            print_interval::Float64=1.0,
                            verbose::Bool=true)
    
    verbose && print_initial_stats(pop)
    
    # Initialize outputs
    ts = SimulationTimeSeries()
    snapshots = Dict{Int, CellPopulation}()
    
    # Initial snapshot
    if verbose
        println("Creating snapshot for t = 0h")
    end
    snapshots[0] = create_snapshot(pop)
    
    # Record initial state
    t = 0.0
    record_timepoint!(ts, t, pop)
    
    # Setup
    event_count = Int32(0)
    next_print_time = print_interval
    snap_set = Set(snapshot_times)
    
    verbose && println("\n", "="^70, "\nSTARTING SIMULATION\n", "="^70)
    
    # Main loop
    while t < terminal_time
        # Find next event
        next_time, idx, event = compute_next_event(pop)
        
        isinf(next_time) && break
        
        t + next_time > terminal_time && break
        
        t += next_time
        event_count += 1
        
        # Progress
        if verbose && t >= next_print_time
            println("t=$(round(t, digits=2))h | $event | Cells=$(pop.n_alive) | Events=$event_count")
            next_print_time += print_interval
        end
        
        # Snapshots
        current_hour = round(Int, floor(t))
        if current_hour in snap_set && !haskey(snapshots, current_hour)
            verbose && println("  └─ Snapshot at t=$(current_hour)h")
            snapshots[current_hour] = create_snapshot(pop)
        end
        
        # Process
        update_ABM!(pop, next_time, event, idx, nat_apo)
        record_timepoint!(ts, t, pop)
    end
    
    # Summary
    if verbose
        println("="^70, "\nSIMULATION COMPLETE\n", "="^70)
        println("Final time: $(round(t, digits=2))h")
        println("Events: $event_count")
        println("Final cells: $(pop.n_alive)")
        println("="^70, "\n")
    end
    
    return (ts, snapshots)
end

function run_simulation_abm!(cell_df::DataFrame, nat_apo::Float64; 
                            return_dataframes::Bool=true, kwargs...)
    # Convert to SoA
    pop = CellPopulation(cell_df)
    
    # Run simulation
    ts, snapshots_soa = run_simulation_abm!(pop, nat_apo; kwargs...)
    
    # Convert snapshots back to DataFrames if requested
    if return_dataframes
        snapshots_df = Dict{Int, DataFrame}()
        for (time, snap_pop) in snapshots_soa
            snapshots_df[time] = to_dataframe(snap_pop, alive_only=true)
        end
        return (ts, snapshots_df)
    else
        return (ts, snapshots_soa)
    end
end