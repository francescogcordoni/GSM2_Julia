function sigmoid(x, C, K)
    return C .+ (K .- C) / (1 + exp(-0.05 * (x - 250)))
end

function p53_network(dydt, y, p, t)

    (ATM_tot, Akt_tot, Pip_tot, DBS_c, CytoC_tot, Casp3_tot, j_nc) = p

    k_dim = 10.
    k_undim = 1.
    k_acatm = 2.
    j_acatm = 1.
    k_deatm = 1.3
    j_deatm = 2.3

    k_dmdm2n0 = 0.003
    k_dmdm2n1 = 0.05
    j_atm = 1.0
    k_acp531 = 0.2
    k_dep53 = 0.1
    k_dp53s = 0.01
    k_sp53 = 0.2
    k_dp53n = 0.05
    k_dp53 = 0.7
    j_1p53n = 0.1
    k_smdm20 = 0.002
    k_smdm2 = 0.024
    j_smdm2 = 1.0
    k_dmdm2c = 0.003
    k_1mdm2s = 0.3
    j_1mdm2s = 0.1
    k_mdm2s = 8.0
    j_mdm2s = 0.3
    k_i = 0.06
    k_0 = 0.09
    k_acakt = 0.25
    j_acakt = 0.1
    k_deakt = 0.1
    j_deakt = 0.2
    k_p2 = 0.1
    j_p2 = 0.2
    k_p3 = 0.5
    j_p3 = 0.4
    k_p46 = 0.6
    j_p46 = 0.5
    k_dp46 = 0.3
    j_dp46 = 0.2
    k_swip10 = 0.01
    k_swip1 = 0.09
    j_swip1 = 0.5
    k_dwip1 = 0.05
    k_sdinp10 = 0.001
    k_sdinp11 = 0.01
    j_sdinp11 = 0.7
    k_sdinp12 = 0.07
    j_sdinp12 = 0.3
    k_ddinp1 = 0.01
    k_spten0 = 0.01
    k_spten = 0.5
    j_spten = 1.0
    k_dpten = 0.1
    k_sp210 = 0.01
    k_sp21 = 0.2
    j_sp21 = 0.6
    k_dp21 = 0.1
    k_saip10 = 0.01
    k_saip1 = 0.32
    j_saip1 = 1.5
    k_daip1 = 0.1
    k_accytoc0 = 0.001
    k_accytoc1 = 0.9
    j_casp3 = 0.5
    k_decytoc = 0.05
    k_accasp30 = 0.001
    k_accasp31 = 0.9
    j_cytoc = 0.5
    k_decasp3 = 0.07

    dydt[1] = 0.5*k_dim*(ATM_tot - 2*y[1] - y[2])^2 - k_undim*y[1]
    dydt[2] = k_acatm*((DBS_c)/(DBS_c + j_nc))*y[2]*((ATM_tot - 2*y[1] - y[2])/(ATM_tot - 2*y[1] - y[2] + j_acatm)) - k_deatm*((y[2])/(y[2] + j_deatm))*(1 + y[11])
    dydt[3] = k_acp531*((y[2])/(y[2] + j_atm))*y[4] - k_dep53*y[3] - k_dp53s*y[7]*((y[3])/(y[3] + j_1p53n))
    dydt[4] = k_sp53 - k_dp53n*y[4] - k_dp53*y[7]*((y[4])/(y[4] + j_1p53n)) - k_acp531*((y[2])/(y[2] + j_atm))*y[4] + k_dep53*y[3]
    dydt[5] = k_smdm20 + k_smdm2*((y[3]^4)/(y[3]^4 + j_smdm2^4)) + k_1mdm2s*((y[6])/(y[6] + j_1mdm2s)) - k_dmdm2c*y[5] - k_mdm2s*(y[8])*((y[5])/(y[5] + j_mdm2s))
    dydt[6] = k_mdm2s*(y[8])*((y[5])/(y[5] + j_mdm2s)) - k_1mdm2s*((y[6])/(y[6] + j_1mdm2s)) - k_i*y[6] + k_0*y[7] - k_dmdm2c*y[6]
    dydt[7] = k_i*y[6] - k_0*y[7] - y[7]*(k_dmdm2n0 + k_dmdm2n1*((y[2])/(y[2] + j_atm)))
    dydt[8] = k_acakt*y[9]*((Akt_tot - y[8])/(Akt_tot - y[8] + j_acakt)) - k_deakt*((y[8])/(y[8] + j_deakt))
    dydt[9] = k_p2*((Pip_tot - y[9])/(Pip_tot - y[9] + j_p2)) - k_p3*y[13]*((y[9])/(y[9] + j_p3))
    dydt[10] = k_p46*y[12]*((y[3] - y[10])/(y[3] - y[10] + j_p46)) - k_dp46*y[11]*((y[10])/(y[10] + j_dp46))
    dydt[11] = k_swip10 + k_swip1*(((y[3] - y[10])^3)/((y[3] - y[10])^3 + j_swip1^3)) - k_dwip1*y[11]
    dydt[12] = k_sdinp10 + k_sdinp11*(((y[3] - y[10])^3)/((y[3] - y[10])^3 + j_sdinp11^3)) + k_sdinp12*(((y[10])^3)/((y[10])^3 + j_sdinp12^3)) - k_ddinp1*y[12]
    dydt[13] = k_spten0 + k_spten*(((y[10])^3)/((y[10])^3 + j_spten^3)) - k_dpten*y[13]
    dydt[14] = k_sp210 + k_sp21*(((y[3] - y[10])^3)/((y[3] - y[10])^3 + j_sp21^3)) - k_dp21*y[14]
    dydt[15] = k_saip10 + k_saip1*(((y[10])^3)/((y[10])^3 + j_saip1^3)) - k_daip1*y[15]
    dydt[16] = (k_accytoc0 + k_accytoc1*y[15]*((y[17]^4)/(y[17]^4 + j_casp3^4)))*(CytoC_tot - y[16]) - k_decytoc*y[16]
    dydt[17] = (k_accasp30 + k_accasp31*((y[16]^4)/(y[16]^4 + j_cytoc^4)))*(Casp3_tot - y[17]) - k_decasp3*y[17]
end

function p53_network_senescense(dydt, y, p, t)

    (ATM_tot, Akt_tot, Pip_tot, DBS_c, CytoC_tot, Casp3_tot, j_nc) = p

    k_dim = 10.
    k_undim = 1.
    k_acatm = 2.
    j_acatm = 1.
    k_deatm = 1.3
    j_deatm = 2.3

    k_dmdm2n0 = 0.003
    k_dmdm2n1 = 0.05
    j_atm = 1.0
    k_acp531 = 0.2
    k_dep53 = 0.1
    k_dp53s = 0.01
    k_sp53 = 0.2
    k_dp53n = 0.05
    k_dp53 = 0.7
    j_1p53n = 0.1
    k_smdm20 = 0.002
    k_smdm2 = 0.024
    j_smdm2 = 1.0
    k_dmdm2c = 0.003
    k_1mdm2s = 0.3
    j_1mdm2s = 0.1
    k_mdm2s = 8.0
    j_mdm2s = 0.3
    k_i = 0.06
    k_0 = 0.09
    k_acakt = 0.25
    j_acakt = 0.1
    k_deakt = 0.1
    j_deakt = 0.2
    k_p2 = 0.1
    j_p2 = 0.2
    k_p3 = 0.5
    j_p3 = 0.4
    k_p46 = 0.6
    j_p46 = 0.5
    k_dp46 = 0.3
    j_dp46 = 0.2
    k_swip10 = 0.01
    k_swip1 = 0.09
    j_swip1 = 0.5
    k_dwip1 = 0.05
    k_sdinp10 = 0.001
    k_sdinp11 = 0.01
    j_sdinp11 = 0.7
    k_sdinp12 = 0.07
    j_sdinp12 = 0.3
    k_ddinp1 = 0.01
    k_spten0 = 0.01
    k_spten = 0.5
    j_spten = 1.0
    k_dpten = 0.1
    k_sp210 = 0.01
    k_sp21 = 0.2
    j_sp21 = 0.6
    k_dp21 = 0.1
    k_saip10 = 0.01
    k_saip1 = 0.32
    j_saip1 = 1.5
    k_daip1 = 0.1
    k_accytoc0 = 0.001
    k_accytoc1 = 0.9
    j_casp3 = 0.5
    k_decytoc = 0.05
    k_accasp30 = 0.001
    k_accasp31 = 0.9
    j_cytoc = 0.5
    k_decasp3 = 0.07

    q_arr = 0.3
    deltaS = 1/2000
    kS = 0.1

    dydt[1] = 0.5*k_dim*(ATM_tot - 2*y[1] - y[2])^2 - k_undim*y[1]
    dydt[2] = k_acatm*((DBS_c)/(DBS_c + j_nc))*y[2]*((ATM_tot - 2*y[1] - y[2])/(ATM_tot - 2*y[1] - y[2] + j_acatm)) - k_deatm*((y[2])/(y[2] + j_deatm))*(1 + y[11])
    dydt[3] = k_acp531*((y[2])/(y[2] + j_atm))*y[4] - k_dep53*y[3] - k_dp53s*y[7]*((y[3])/(y[3] + j_1p53n))
    dydt[4] = k_sp53 - k_dp53n*y[4] - k_dp53*y[7]*((y[4])/(y[4] + j_1p53n)) - k_acp531*((y[2])/(y[2] + j_atm))*y[4] + k_dep53*y[3]
    dydt[5] = k_smdm20 + k_smdm2*((y[3]^4)/(y[3]^4 + j_smdm2^4)) + k_1mdm2s*((y[6])/(y[6] + j_1mdm2s)) - k_dmdm2c*y[5] - k_mdm2s*(y[8])*((y[5])/(y[5] + j_mdm2s))
    dydt[6] = k_mdm2s*(y[8])*((y[5])/(y[5] + j_mdm2s)) - k_1mdm2s*((y[6])/(y[6] + j_1mdm2s)) - k_i*y[6] + k_0*y[7] - k_dmdm2c*y[6]
    dydt[7] = k_i*y[6] - k_0*y[7] - y[7]*(k_dmdm2n0 + k_dmdm2n1*((y[2])/(y[2] + j_atm)))
    dydt[8] = k_acakt*y[9]*((Akt_tot - y[8])/(Akt_tot - y[8] + j_acakt)) - k_deakt*((y[8])/(y[8] + j_deakt))
    dydt[9] = k_p2*((Pip_tot - y[9])/(Pip_tot - y[9] + j_p2)) - k_p3*y[13]*((y[9])/(y[9] + j_p3))
    dydt[10] = k_p46*y[12]*((y[3] - y[10])/(y[3] - y[10] + j_p46)) - k_dp46*y[11]*((y[10])/(y[10] + j_dp46))
    dydt[11] = k_swip10 + k_swip1*(((y[3] - y[10])^3)/((y[3] - y[10])^3 + j_swip1^3)) - k_dwip1*y[11]
    dydt[12] = k_sdinp10 + k_sdinp11*(((y[3] - y[10])^3)/((y[3] - y[10])^3 + j_sdinp11^3)) + k_sdinp12*(((y[10])^3)/((y[10])^3 + j_sdinp12^3)) - k_ddinp1*y[12]
    dydt[13] = k_spten0 + k_spten*(((y[10])^3)/((y[10])^3 + j_spten^3)) - k_dpten*y[13]
    dydt[14] = k_sp210 + k_sp21*(((y[3] - y[10])^3)/((y[3] - y[10])^3 + j_sp21^3)) - k_dp21*y[14]
    dydt[15] = k_saip10 + k_saip1*(((y[10])^3)/((y[10])^3 + j_saip1^3)) - k_daip1*y[15]
    dydt[16] = (k_accytoc0 + k_accytoc1*y[15]*((y[17]^4)/(y[17]^4 + j_casp3^4)))*(CytoC_tot - y[16]) - k_decytoc*y[16]
    dydt[17] = (k_accasp30 + k_accasp31*((y[16]^4)/(y[16]^4 + j_cytoc^4)))*(Casp3_tot - y[17]) - k_decasp3*y[17]
    dydt[18] = kS * max(y[14] - q_arr, 0) - deltaS * y[18]
end


function compute_times_domain_p53!(cell_df::DataFrame, gsm2_cycle::Vector{GSM2};
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
            compute_repair_domain_p53(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2;
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

function compute_repair_domain_p53_history(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2;
                                    terminal_time::Float64 = Inf,
                                    au::Float64            = 6.0,
                                    p53_scale::Float64     = 120.0)

    X = copy(X)
    Y = copy(Y)
    # ── Early exits (return signature must match: dt, rt, death_type, X, Y, df_markov, df_mol) ──
    if sum(Y) > 0
        df_markov = DataFrame(time=[0.0], sum_X=[sum(X)], sum_Y=[sum(Y)])
        df_mol    = DataFrame()
    end

    if sum(X) == 0
        df_markov = DataFrame(time=[0.0], sum_X=[0], sum_Y=[0])
        df_mol    = DataFrame()
        return (Inf, 0.0, 0, X, Y, df_markov, df_mol)
    end

    # ── Constants ────────────────────────────────────────────────────────────
    ATM_tot = 5.0; Akt_tot = 1.0; Pip_tot = 1.0; CytoC_tot = 3.0; Casp3_tot = 3.0

    ATM2_0=2.17; ATMs_0=0.01; p53_0=0.8; p53s_0=0.0; Mdm2c_0=0.01; Mdm2cp_0=0.4
    Mdm2n_0=0.26; Akts_0=0.94; Pip3_0=0.89; PTEN_0=0.1; p53killer_0=0.0; p21_0=0.1
    Wip1_0=0.2; p53DINP1_0=0.0; p53AIP1_0=0.1; CytoC_0=0.06; Casp3_0=0.05

    Sen_0 = 0.

    net  = ["ATM2","ATMs","p53s","p53","Mdm2c","Mdm2cp","Mdm2n","Akts","Pip3",
            "p53killer","Wip1","p53DINP1","PTEN","p21","p53AIP1","CytoC","Casp3",
            "Sen"]
    net_ = [net; "time"]

    state  = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0,
                p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0,
                Sen_0]

    j_nc = 3
    a = gsm2.a;  b = gsm2.b;  r = gsm2.r

    # ── Initialise storage DataFrames ────────────────────────────────────────
    # df_mol: concatenated ODE solutions with cumulative wall-clock time
    df_mol = DataFrame()
    for (i, name) in enumerate(net_)
        df_mol[!, name] = [i <= length(state) ? state[i] : 0.0]
    end

    # df_markov: Markov chain trajectory (sum_X, sum_Y at each reaction)
    markov_times = Float64[0.0]
    markov_sumX  = Int[sum(X)]
    markov_sumY  = Int[sum(Y)]

    # ── Initial rates (coupled to p53 state) ─────────────────────────────────
    aC = a .* sigmoid(sum(X), 1 - df_mol.p53s[1]/(1+df_mol.p53s[1]),
                                1 + df_mol.p53s[1]/(1+df_mol.p53s[1]))
    rC = r .* sigmoid(sum(X), 1 + df_mol.p53s[1]/(1+df_mol.p53s[1]),
                                1 - df_mol.p53s[1]/(1+df_mol.p53s[1]))

    current_time      = 0.0
    dt_result         = Inf
    rt_result         = Inf
    death_type_result = -1

    n_domains = length(X)

    # ══════════════════════════════════════════════════════════════════════════
    # Main Gillespie + ODE loop
    # ══════════════════════════════════════════════════════════════════════════
    while sum(X) > 0

        # ── 1) Gillespie: draw next reaction time & channel ──────────────────
        r1 = rand()
        r2 = rand()

        aX = aC .* X                          # misrepair propensities
        bX = max.(b .* X .* (X .- 1), 0)      # interaction propensities
        rX = rC .* X                           # repair propensities

        a0 = sum(rX) + sum(aX) + sum(bX)
        if a0 ≤ 0.0
            break  # no more possible reactions
        end

        dt = au * (1.0 / a0) * log(1.0 / r1)       # inter-arrival time (hours)

        # ── 2) Terminal-time cutoff ──────────────────────────────────────────
        if current_time + dt > terminal_time
            # Solve ODE for remaining window, then exit
            dt_remaining = terminal_time - current_time
            if dt_remaining > 0
                tmin_ode = dt_remaining * p53_scale
                p_ode = (ATM_tot, Akt_tot, Pip_tot, sum(X), CytoC_tot, Casp3_tot, j_nc)
                prob  = ODEProblem(p53_network_senescense, state, (0.0, tmin_ode), p_ode)
                sol   = solve(prob, RK4(), isoutofdomain=(u,p,t)->any(x->x<0,u),
                                save_everystep=false, save_start=false, saveat=tmin_ode/10)
                if length(sol.u) > 0
                    sol_mat = hcat(sol.u...)
                    df_sol  = DataFrame(sol_mat', :auto)
                    rename!(df_sol, Symbol.(names(df_sol)) .=> Symbol.(net))
                    df_sol[!, :time] = sol.t ./ p53_scale .+ current_time
                    append!(df_mol, df_sol)
                    state = Vector(df_sol[end, net])
                end
            end
            current_time = terminal_time
            push!(markov_times, current_time)
            push!(markov_sumX, sum(X))
            push!(markov_sumY, sum(Y))
            break
        end

        # ── 3) Solve p53 ODE from current_time to current_time + dt ─────────
        tmin_ode = dt * p53_scale                  # ODE uses a scaled time
        p_ode    = (ATM_tot, Akt_tot, Pip_tot, sum(X), CytoC_tot, Casp3_tot, j_nc)

        prob = ODEProblem(p53_network_senescense, state, (0.0, tmin_ode), p_ode)
        sol  = solve(prob, RK4(), isoutofdomain=(u,p,t)->any(x->x<0,u),
                        save_everystep=false, save_start=false, saveat=tmin_ode/10)

        # Store ODE chunk with correct wall-clock times
        if length(sol.u) > 0
            sol_mat = hcat(sol.u...)
            df_sol  = DataFrame(sol_mat', :auto)
            rename!(df_sol, Symbol.(names(df_sol)) .=> Symbol.(net))
            # *** FIX: offset by current_time BEFORE advancing it ***
            df_sol[!, :time] = sol.t ./ p53_scale .+ current_time
            append!(df_mol, df_sol)
            # Update ODE state to terminal value
            state = Vector(df_sol[end, net])
        end

        # ── 4) Advance wall-clock time (AFTER ODE solve) ─────────────────────
        current_time += dt

        # ── 5) Update Markov rates from p53 state ───────────────────────────
        p53s_now = df_mol.p53s[end]
        aC = a .* sigmoid(sum(X), 1 - p53s_now/(1+p53s_now),
                                    1 + p53s_now/(1+p53s_now))
        rC = r .* sigmoid(sum(X), 1 + p53s_now/(1+p53s_now),
                                    1 - p53s_now/(1+p53s_now))

        # ── 6) Check for Caspase-3 triggered apoptosis ──────────────────────
        if death_type_result == -1 && maximum(df_mol.Casp3[max(1,end-10):end]) > 1.0
            dt_result         = current_time
            rt_result         = Inf
            death_type_result = 3
        end

        # ── 7) Identify and apply the fired reaction ────────────────────────
        props    = vcat(rX, aX, bX)
        cum      = cumsum(props)
        reac_idx = findfirst(x -> x >= r2 * a0, cum)

        if reac_idx <= n_domains
            # ── Repair: remove 1 X ──────────────────────────────────────────
            dom = reac_idx
            X[dom] -= 1

            # If all damage repaired, record survival
            if death_type_result == -1 && sum(X) == 0
                dt_result         = Inf
                rt_result         = current_time
                death_type_result = 0
            end

        elseif reac_idx <= 2 * n_domains
            # ── Misrepair: remove 1 X, add 1 Y ─────────────────────────────
            dom = reac_idx - n_domains
            X[dom] -= 1
            Y[dom] += 1

            if death_type_result == -1
                th    = 0.1
                p_sen = df_mol.p21[end] / (th + df_mol.p21[end])
                dt_result         = current_time
                rt_result         = Inf
                death_type_result = rand() < p_sen ? 2 : 1
            end

        else
            # ── Interaction: remove 2 X, add 1 Y ───────────────────────────
            dom = reac_idx - 2 * n_domains
            X[dom] -= 2
            Y[dom] += 1

            if death_type_result == -1
                th    = 0.1
                p_sen = df_mol.p21[end] / (th + df_mol.p21[end])
                dt_result         = current_time
                rt_result         = Inf
                death_type_result = rand() < p_sen ? 2 : 1
            end
        end

        # ── 8) Record Markov state ──────────────────────────────────────────
        push!(markov_times, current_time)
        push!(markov_sumX,  sum(X))
        push!(markov_sumY,  sum(Y))
    end

    df_markov = DataFrame(time=markov_times, sum_X=markov_sumX, sum_Y=markov_sumY)

    return (dt_result, rt_result, death_type_result, X, Y, df_markov, df_mol)
end


function compute_repair_domain_p53(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2;
                                    terminal_time::Float64 = Inf,
                                    au::Float64            = 6.0,
                                    p53_scale::Float64     = 120.0)

    X = copy(X);  Y = copy(Y)

    any(>(0), Y) && return (0.0, Inf, 1, X, Y)
    sum(X) == 0  && return (Inf, 0.0, 0, X, Y)

    ATM_tot=5.0; Akt_tot=1.0; Pip_tot=1.0; CytoC_tot=3.0; Casp3_tot=3.0
    j_nc = 3
    state = [2.17,0.01,0.0,0.8,0.01,0.4,0.26,0.94,0.89,
                0.0,0.2,0.0,0.1,0.1,0.1,0.06,0.05]
    p53s_idx = 3;  p21_idx = 14;  casp3_idx = 17

    a = gsm2.a;  b = gsm2.b;  r = gsm2.r
    n_domains = length(X)

    p53s_now  = state[p53s_idx]
    p53s_term = p53s_now / (1.0 + p53s_now)
    aC = a * sigmoid(sum(X), 1 - p53s_term, 1 + p53s_term)
    rC = r * sigmoid(sum(X), 1 + p53s_term, 1 - p53s_term)

    current_time = 0.0

    while sum(X) > 0
        r1 = rand();  r2 = rand()

        aX = aC .* X
        bX = max.(b .* X .* (X .- 1), 0)
        rX = rC .* X
        a0 = sum(rX) + sum(aX) + sum(bX)
        a0 ≤ 0.0 && break

        dt = au * (1.0 / a0) * log(1.0 / r1)
        current_time + dt > terminal_time && return (Inf, Inf, -1, X, Y)

        # Solve p53 ODE over this inter-arrival interval
        tmin_ode = dt * p53_scale
        p_ode    = (ATM_tot, Akt_tot, Pip_tot, Float64(sum(X)), CytoC_tot, Casp3_tot, j_nc)
        sol      = solve(ODEProblem(p53_network, state, (0.0, tmin_ode), p_ode),
                            RK4(), isoutofdomain=(u,p,t)->any(x->x<0,u),
                            save_everystep=false, save_start=false, dense=false)

        if !isempty(sol.u)
            state = sol.u[end]
        end

        current_time += dt

        # Caspase-3 apoptosis
        if state[casp3_idx] > 1.5
            return (current_time, Inf, 3, X, Y)
        end

        # Update p53-coupled rates
        p53s_now  = state[p53s_idx]
        p53s_term = p53s_now / (1.0 + p53s_now)
        aC = a * sigmoid(sum(X), 1 - p53s_term, 1 + p53s_term)
        rC = r * sigmoid(sum(X), 1 + p53s_term, 1 - p53s_term)

        # Select and apply reaction
        props    = vcat(rX, aX, bX)
        reac_idx = findfirst(x -> x >= r2 * a0, cumsum(props))

        if reac_idx <= n_domains
            X[reac_idx] -= 1
            sum(X) == 0 && return (Inf, current_time, 0, X, Y)

        elseif reac_idx <= 2 * n_domains
            dom = reac_idx - n_domains
            X[dom] -= 1;  Y[dom] += 1
            th    = 0.1
            p_sen = state[p21_idx] / (th + state[p21_idx])
            return (current_time, Inf, rand() < p_sen ? 2 : 1, X, Y)

        else
            dom = reac_idx - 2 * n_domains
            X[dom] -= 2;  Y[dom] += 1
            th    = 0.1
            p_sen = state[p21_idx] / (th + state[p21_idx])
            return (current_time, Inf, rand() < p_sen ? 2 : 1, X, Y)
        end
    end

    return (Inf, current_time, 0, X, Y)
end




