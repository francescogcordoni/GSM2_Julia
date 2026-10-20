# ============================================================================
# example/benchmark_computation.jl
#
# Computation time vs spheroid size (radius, number of cells) for several ions,
# split by stage, using the pipeline of example/spheroid.jl:
#
#   setup    lattice + population + oxygen (once per radius, shared by the ions)
#   dose     MC_dose_CPU!                 (track-structure dose)
#   damage   MC_loop_damage!              (Poisson lesions)
#   fate     compute_times_domain!        (GSM2 repair / death times)
#   abm      run_simulation_abm!          (T_ABM hours of cell dynamics)
#   total    dose + damage + fate + abm   (setup reported separately)
#
# A warm-up run at the smallest radius (discarded) removes compilation time.
# Each (radius, ion) is repeated N_REP times; median and range are reported.
# Results are appended to the CSV after every run, so a long benchmark can be
# stopped and the data kept.
#
# HOW TO RUN
#   julia --threads auto example/benchmark_computation.jl
# Stress test: STRESS_RADII (up to 1000 µm) for the worst case (1H 100 MeV) and
# 12C 80 MeV/u, one repeat; radii whose estimated memory exceeds the free memory
# are skipped. Stress-test points are drawn as open symbols.
# Outputs: example/output/benchmark/  timings.csv, timings_summary.csv, figures (PDF)
# ============================================================================

include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using StatsPlots
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using CUDA
using Statistics: mean
using SparseArrays
using Printf
using Dates

println("Threads available: ", nthreads())
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()

# ============================================================================
# 1. PARAMETERS
# ============================================================================
const RADII  = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]   # µm
const MARGIN = 150.0                 # µm of free lattice around the spheroid
const IONS = [
    (tag = "1H_100MeV",  particle = "1H",  E = 100.0),
    (tag = "1H_30MeV",   particle = "1H",  E = 30.0),
    (tag = "12C_80MeVu", particle = "12C", E = 80.0),
    (tag = "12C_10MeVu", particle = "12C", E = 10.0),
]
# Stress test: large spheroids, worst-case (1H 100 MeV) and carbon only, one repeat each.
# A radius is skipped if its estimated memory exceeds MEM_FRACTION of the free memory.
const STRESS_RADII = [600.0, 800.0, 1000.0]                     # µm
const STRESS_IONS  = ["1H_100MeV", "12C_80MeVu"]
const N_REP_STRESS = 1
const MEM_FRACTION = 0.7
const DOSE   = 2.0
const T_ABM  = 72.0
const N_REP  = 3
const SEED   = 1

R_CELL = 15.0; RD = 0.8; RN = 7.2
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657
const TYPE_AT     = "KC"
const TRACK_SEG   = true
const TARGET_GEOM = "circle"
const CALC_TYPE   = "full"
const NAT_APO     = 1e-10

const OUTDIR = joinpath(@__DIR__, "output", "benchmark")
mkpath(OUTDIR)
const CSVFILE = joinpath(OUTDIR, "timings.csv")

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(R_G1, A_G1, B_G1, RD, RN)
gsm2_cycle[2] = GSM2(R_S,  A_S,  B_S,  RD, RN)
gsm2_cycle[3] = GSM2(R_G2, A_G2, B_G2, RD, RN)
gsm2_cycle[4] = GSM2(R_AVG, A_AVG, B_AVG, RD, RN)
setup_GSM2!(R_AVG, A_AVG, B_AVG, RD, RN)
isdefined(Main, :set_fate_model!) && set_fate_model!(:incremental)

G(name::Symbol) = Base.invokelatest(getfield, Main, name)

# ============================================================================
# 2. SETUP OF ONE SPHEROID (as example/spheroid.jl)
# ============================================================================
"""Builds the spheroid of radius R. Returns (pristine cell_df, domain tables, setup time)."""
function build(R::Float64)
    X_box   = ceil((R + MARGIN) / (2R_CELL)) * 2R_CELL
    X_voxel = 2X_box
    t = @elapsed begin
        Random.seed!(SEED)
        setup_IonIrrad!(DOSE, 100.0, "1H"; type_AT = TYPE_AT)
        N_sideVox   = Int(floor(2 * X_box / X_voxel))
        N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_CELL)))
        setup_cell_lattice!(TARGET_GEOM, X_box, R_CELL, N_sideVox, N_CellsSide;
                            ParIrr = "false", track_seg = TRACK_SEG, full_cycle = true)
        @eval Main begin
            tumor_radius = $R
            X_voxel      = $X_voxel
            X_box        = $X_box
        end
        setup_cell_population!(TARGET_GEOM, X_box, R_CELL, N_sideVox, N_CellsSide,
                               gsm2_cycle[4]; ParIrr = "false")
        df = G(:cell_df)
        setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
        set_oxygen_diffusion!(df; density = :mean, plot_oxygen = false, verbose = false)
        hasproperty(df, :death_type)  || (df.death_type  = fill(-1, nrow(df)))
        hasproperty(df, :dam_X_total) || (df.dam_X_total = zeros(Int, nrow(df)))
        hasproperty(df, :dam_Y_total) || (df.dam_Y_total = zeros(Int, nrow(df)))
    end
    return deepcopy(G(:cell_df)), G(:df_center_x), G(:df_center_y), G(:at), X_box, t
end

"""One irradiation + ABM run; returns the stage times (s) and sizes."""
function run_once(ion_, R, X_box, pristine, dfx, dfy, at0, rep)
    df = deepcopy(pristine)
    X_voxel = 2X_box
    setup_IonIrrad!(DOSE, ion_.E, ion_.particle; type_AT = TYPE_AT)
    R_beam, _, _ = calculate_beam_properties(CALC_TYPE, TARGET_GEOM, X_box, X_voxel, R)
    setup_irrad_conditions!(G(:ion), G(:irrad), TYPE_AT, df, TRACK_SEG)
    Npar = round(Int, G(:irrad).dose / (1.602e-9 * G(:LET)) * π * R_beam^2 * 1e-8)
    at_  = deepcopy(at0)
    GC.gc()
    t_dose = @elapsed MC_dose_CPU!(G(:ion), Npar, R_beam, G(:irrad_cond), df, dfx, dfy, at_,
                                   gsm2_cycle, TYPE_AT, TRACK_SEG; seed = hash((SEED, ion_.tag, R, rep)))
    Random.seed!(hash((SEED, ion_.tag, R, rep, :damage)))
    t_dam  = @elapsed MC_loop_damage!(G(:ion), df, G(:irrad_cond), gsm2_cycle; show_progress = false)
    Random.seed!(hash((SEED, ion_.tag, R, rep, :fate)))
    t_fate = @elapsed compute_times_domain!(df, gsm2_cycle; nat_apo = NAT_APO, terminal_time = Inf,
                                            verbose = false, summary = false)
    Random.seed!(hash((SEED, ion_.tag, R, rep, :abm)))
    t_abm  = @elapsed run_simulation_abm!(df; nat_apo = NAT_APO, terminal_time = T_ABM,
                                          snapshot_times = Int[], verbose = false,
                                          return_dataframes = false, update_input = true)
    return (t_dose = t_dose, t_damage = t_dam, t_fate = t_fate, t_abm = t_abm,
            t_total = t_dose + t_dam + t_fate + t_abm, Npar = Npar, LET = G(:LET),
            N_final = count(==(1), df.is_cell))
end

# ============================================================================
# 3. RUN
# ============================================================================
# The repo's setup functions create globals with @eval. In Julia ≥ 1.12 a binding
# created during a call is not visible to code already running, so the first
# setup is done statement by statement at TOP LEVEL (as in example/spheroid.jl);
# later calls of build() only update globals that already exist.
println("\n--- Warm-up (top-level setup + compilation, discarded) ---")
const R_WARM  = minimum(RADII)
const XB_WARM = ceil((R_WARM + MARGIN) / (2R_CELL)) * 2R_CELL
setup_IonIrrad!(DOSE, 100.0, "1H"; type_AT = TYPE_AT)
setup_cell_lattice!(TARGET_GEOM, XB_WARM, R_CELL, Int(floor(2XB_WARM / (2XB_WARM))),
                    2 * convert(Int64, floor(XB_WARM / (2 * R_CELL)));
                    ParIrr = "false", track_seg = TRACK_SEG, full_cycle = true)
@eval Main begin
    tumor_radius = $R_WARM
    X_voxel      = $(2XB_WARM)
    X_box        = $XB_WARM
end
setup_cell_population!(TARGET_GEOM, XB_WARM, R_CELL, Int(floor(2XB_WARM / (2XB_WARM))),
                       2 * convert(Int64, floor(XB_WARM / (2 * R_CELL))), gsm2_cycle[4]; ParIrr = "false")
setup_irrad_conditions!(ion, irrad, TYPE_AT, cell_df, TRACK_SEG)
set_oxygen_diffusion!(cell_df; density = :mean, plot_oxygen = false, verbose = false)

warm = build(R_WARM)
for ion_ in IONS
    run_once(ion_, R_WARM, warm[5], warm[1], warm[2], warm[3], warm[4], 0)
end
# memory per lattice site (cell_df with dose and damage vectors), for the stress-test estimate
const BYTES_PER_SITE = Base.summarysize(warm[1]) / nrow(warm[1])
@printf("memory per lattice site ≈ %.1f kB\n", BYTES_PER_SITE / 1024)
warm = nothing; GC.gc()

"""Estimated memory (GB) for radius R: ~3 copies of the lattice (setup, pristine, run)."""
function mem_estimate_GB(R)
    X_box = ceil((R + MARGIN) / (2R_CELL)) * 2R_CELL
    nsites = (2 * floor(Int, X_box / (2R_CELL)))^3
    return 3.0 * nsites * BYTES_PER_SITE / 2^30, nsites
end

isfile(CSVFILE) && mv(CSVFILE, CSVFILE * "." * Dates.format(now(), "yyyymmdd_HHMMSS") * ".bak")
for R in vcat(RADII, STRESS_RADII)
    stress = R in STRESS_RADII
    memGB, nsites_est = mem_estimate_GB(R)
    freeGB = Sys.free_memory() / 2^30
    println("\n=== radius $(R) µm", stress ? " (stress test)" : "", " ===")
    @printf("  estimated memory %.1f GB for %d lattice sites (free %.1f GB)\n", memGB, nsites_est, freeGB)
    if memGB > MEM_FRACTION * freeGB
        @warn "radius $R µm skipped: estimated memory $(round(memGB, digits = 1)) GB > $(MEM_FRACTION) × free memory"
        continue
    end
    ions_R = stress ? filter(i -> i.tag in STRESS_IONS, IONS) : IONS
    nrep_R = stress ? N_REP_STRESS : N_REP
    GC.gc()
    pristine, dfx, dfy, at0, X_box, t_setup = build(R)
    Ncells = count(==(1), pristine.is_cell)
    @printf("  %d cells, %d lattice sites, box half-size %.0f µm, setup %.1f s\n",
            Ncells, nrow(pristine), X_box, t_setup)
    for ion_ in ions_R, rep in 1:nrep_R
        r = run_once(ion_, R, X_box, pristine, dfx, dfy, at0, rep)
        row = DataFrame(radius_um = R, stress = stress, n_cells = Ncells, n_sites = nrow(pristine), X_box_um = X_box,
                        mem_lattice_GB = Base.summarysize(pristine) / 2^30,
                        ion = ion_.tag, LET = r.LET, Npar = r.Npar, rep = rep, threads = nthreads(),
                        t_setup = t_setup, t_dose = r.t_dose, t_damage = r.t_damage,
                        t_fate = r.t_fate, t_abm = r.t_abm, t_total = r.t_total, N_final = r.N_final)
        CSV.write(CSVFILE, row; append = isfile(CSVFILE))
        @printf("  %-11s rep %d | Npar %9d | dose %7.2f s  damage %6.2f s  fate %6.2f s  abm %6.2f s  total %7.2f s\n",
                ion_.tag, rep, r.Npar, r.t_dose, r.t_damage, r.t_fate, r.t_abm, r.t_total)
    end
    pristine = nothing; GC.gc()
end

# ============================================================================
# 4. SUMMARY + FIGURES (vector PDF, no titles)
# ============================================================================
tim = CSV.read(CSVFILE, DataFrame)
stages = [:t_dose, :t_damage, :t_fate, :t_abm, :t_total]
summ = combine(groupby(tim, [:ion, :radius_um, :n_cells, :Npar]),
               [s => median => Symbol(s, :_med) for s in stages]...,
               [s => minimum => Symbol(s, :_min) for s in stages]...,
               [s => maximum => Symbol(s, :_max) for s in stages]...,
               :t_setup => first => :t_setup, :stress => first => :stress,
               :mem_lattice_GB => first => :mem_lattice_GB)
sort!(summ, [:ion, :radius_um])
CSV.write(joinpath(OUTDIR, "timings_summary.csv"), summ)

default(framestyle = :box, lw = 1.8, guidefontsize = 12, tickfontsize = 10,
        legendfontsize = 9, size = (600, 450), dpi = 300,
        left_margin = 5Plots.mm, bottom_margin = 5Plots.mm, right_margin = 3Plots.mm)
paper_style!()
const ION_COL = Dict("1H_100MeV" => :steelblue, "1H_30MeV" => :deepskyblue,
                     "12C_80MeVu" => :firebrick, "12C_10MeVu" => :darkorange)
nice(t) = ion_label(t)
const STAGE_LAB = Dict(:t_dose => "dose (MC)", :t_damage => "damage", :t_fate => "fate (GSM2)",
                       :t_abm => "ABM ($(Int(T_ABM)) h)", :t_total => "total")
save(p, name) = (savefig(p, joinpath(OUTDIR, name)); println("  ", name))

for (xcol, xlab, tagx) in ((:n_cells, "Number of cells", "cells"), (:radius_um, tex("Spheroid radius (\$\\mu\$m)"), "radius"))
    panels = []
    for st in stages
        p = plot(xscale = :log10, yscale = :log10, xlabel = xlab, ylabel = "Time (s)",
                 legend = :topleft, legendtitle = tex(STAGE_LAB[st]))
        for ion_ in IONS
            g = summ[summ.ion .== ion_.tag, :]
            med = g[!, Symbol(st, :_med)]
            lo  = med .- g[!, Symbol(st, :_min)]; hi = g[!, Symbol(st, :_max)] .- med
            plot!(p, g[!, xcol], med; yerror = (lo, hi), marker = :circle, ms = 4,
                  color = ION_COL[ion_.tag], label = nice(ion_.tag))
            st_ = g.stress
            any(st_) && scatter!(p, g[st_, xcol], med[st_]; marker = :circle, ms = 6, mc = :white,
                                 msc = ION_COL[ion_.tag], msw = 1.5, label = "")
        end
        any(summ.stress) && scatter!(p, [NaN], [NaN]; marker = :circle, mc = :white, msc = :black,
                                     msw = 1.5, label = "stress test")
        save(p, "time_$(st)_vs_$(tagx).pdf")
        push!(panels, p)
    end
    save(plot(panels...; layout = (2, 3), size = (1800, 900)), "time_all_vs_$(tagx).pdf")
end

# Stage breakdown: share of the total per radius, one panel per ion
bars = map(IONS) do ion_
    g = summ[summ.ion .== ion_.tag, :]
    M = hcat([g[!, Symbol(st, :_med)] for st in stages[1:4]]...)
    groupedbar(string.(Int.(g.radius_um)), M ./ sum(M, dims = 2); bar_position = :stack,
               label = permutedims([STAGE_LAB[s] for s in stages[1:4]]),
               xlabel = tex("Spheroid radius (\$\\mu\$m)"), ylabel = "Share of computation time",
               legend = :outerright, legendtitle = nice(ion_.tag), ylims = (0, 1),
               color = [:steelblue :seagreen :darkorange :firebrick])
end
save(plot(bars...; layout = (2, 2), size = (1500, 900)), "time_share_by_stage.pdf")

# Dose time per particle (cost of the track-structure step)
p = plot(xscale = :log10, yscale = :log10, xlabel = "Number of particles", ylabel = "Dose time (s)",
         legend = :topleft)
for ion_ in IONS
    g = summ[summ.ion .== ion_.tag, :]
    plot!(p, g.Npar, g.t_dose_med; marker = :circle, color = ION_COL[ion_.tag], label = nice(ion_.tag))
end
save(p, "dose_time_vs_particles.pdf")

# Power-law exponents: time ∝ cells^k
println("\nScaling exponents k (time ∝ cells^k, log-log fit):")
for ion_ in IONS, st in stages
    g = summ[summ.ion .== ion_.tag, :]
    nrow(g) >= 3 || continue
    slope(gg) = (x = log.(gg.n_cells); y = log.(max.(gg[!, Symbol(st, :_med)], 1e-6)); cov(x, y) / var(x))
    gr_ = g[.!g.stress, :]
    @printf("  %-11s %-12s k = %.2f (standard sizes)%s\n", ion_.tag, STAGE_LAB[st], slope(gr_),
            any(g.stress) ? @sprintf("   k = %.2f (incl. stress test)", slope(g)) : "")
end
println("\nSaved in $OUTDIR")
