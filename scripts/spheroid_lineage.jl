# ============================================================================
# spheroid_lineage.jl
#
# Where does regrowth come from? Every cell alive after irradiation is tagged
# as the founder of its own clone (set_founder_tracking!); daughters inherit
# the tag. After T_FOLLOW we count the clonogenic descendants of every initial
# cell and relate them to its radial position and phase at irradiation.
#
# Run: julia -t auto scripts/spheroid_lineage.jl
# Outputs: data/spheroid_lineage/{initial_cells,final_cells}.csv + figures
# ============================================================================
include(joinpath(@__DIR__, "..", "src", "plot_style.jl"))   # paper figure style
include(joinpath(@__DIR__, "spheroid_common.jl"))

# ── Settings ────────────────────────────────────────────────────────────────
const TUMOR_RADIUS = 150.0
const X_BOX        = 350.0
const N_REP        = 5
const T_FOLLOW     = 240.0                     # h (10 days)
# Doses chosen to give a similar initial clonogenic fraction (adjust after a first run
# using the printed SF0 so the two ions are compared at iso-effect)
const CONDITIONS = [
    (tag = "1H_100MeV",  particle = "1H",  E = 100.0, dose = 6.0),
    (tag = "12C_80MeVu", particle = "12C", E = 80.0,  dose = 4.0),
    (tag = "control",    particle = "1H",  E = 100.0, dose = 0.0),
]
const OUTDIR = joinpath(@__DIR__, "..", "data", "spheroid_lineage")
mkpath(OUTDIR)

# ── Run ─────────────────────────────────────────────────────────────────────
pristine = build_spheroid(TUMOR_RADIUS, X_BOX)
alive0   = findall(==(1), pristine.is_cell)
zc = hasproperty(pristine, :z) ? mean(pristine.z[alive0]) : 0.0
cx, cy = mean(pristine.x[alive0]), mean(pristine.y[alive0])
radius(df, i) = sqrt((df.x[i] - cx)^2 + (df.y[i] - cy)^2 +
                     (hasproperty(df, :z) ? (df.z[i] - zc)^2 : 0.0))

init_rows  = DataFrame(tag = String[], rep = Int[], cell = Int[], x = Float64[], y = Float64[],
                       z = Float64[], r = Float64[], phase0 = String[], survived = Bool[],
                       n_desc = Int[])
final_rows = DataFrame(tag = String[], rep = Int[], x = Float64[], y = Float64[], z = Float64[],
                       founder = Int[], founder_r = Float64[], founder_phase = String[])

for c in CONDITIONS
    println("\n== $(c.tag)  $(c.dose) Gy ==")
    dosed = dose_copy(c.E, c.particle, c.dose, pristine; tumor_radius = TUMOR_RADIUS, X_box = X_BOX)
    for rep in 1:N_REP
        df = deepcopy(dosed)
        Random.seed!(hash((c.tag, rep)))
        sample_damage!(df, c.dose)
        compute_times_domain!(df, gsm2_cycle; terminal_time = Inf, summary = false)
        phase0   = copy(df.cell_cycle)
        survived = (df.is_cell .== 1) .& .!isfinite.(df.death_time)

        set_founder_tracking!(df)
        run_simulation_abm!(df; terminal_time = T_FOLLOW, snapshot_times = Int[], verbose = false,
                            return_dataframes = false, update_input = true, record_dt = 1.0)
        F = founders(); clear_founder_tracking!()

        final = findall((df.is_cell .== 1) .& .!isfinite.(df.death_time))
        ndesc = zeros(Int, nrow(df))
        for i in final
            F[i] > 0 && (ndesc[F[i]] += 1)
        end
        for i in alive0
            push!(init_rows, (c.tag, rep, i, pristine.x[i], pristine.y[i],
                              hasproperty(pristine, :z) ? pristine.z[i] : 0.0,
                              radius(pristine, i), String(phase0[i]), survived[i], ndesc[i]))
        end
        for i in final
            f = F[i]
            push!(final_rows, (c.tag, rep, df.x[i], df.y[i], hasproperty(df, :z) ? df.z[i] : 0.0,
                               f, f > 0 ? radius(pristine, f) : NaN, f > 0 ? String(phase0[f]) : ""))
        end
        @printf("  rep %d  SF0=%.4f  founders=%d  final clonogenic=%d\n", rep,
                count(survived[alive0]) / length(alive0), count(>(0), ndesc), length(final))
    end
end
CSV.write(joinpath(OUTDIR, "initial_cells.csv"), init_rows)
CSV.write(joinpath(OUTDIR, "final_cells.csv"), final_rows)

# ── Summary numbers ─────────────────────────────────────────────────────────
println("\nShare of the regrown spheroid descending from initially quiescent (G0) cells:")
for g in groupby(init_rows, :tag)
    tot = sum(g.n_desc)
    @printf("  %-12s  G0 founders give %.1f%% of final cells (G0 were %.1f%% of initial cells)\n",
            g.tag[1], 100 * sum(g.n_desc[g.phase0 .== "G0"]) / max(tot, 1),
            100 * count(==("G0"), g.phase0) / nrow(g))
end

# ── Plots ───────────────────────────────────────────────────────────────────
default(framestyle = :box, lw = 1.8)
paper_style!()
tags  = [c.tag for c in CONDITIONS]
tcol  = Dict("1H_100MeV" => :steelblue, "12C_80MeVu" => :firebrick, "control" => :gray40)
slab  = R_CELL                                  # equatorial slice half-thickness (µm)
Rmax  = maximum(init_rows.r)

# A. Equatorial slice of the irradiated spheroid: founders coloured by clone size (rep 1)
pA = map(tags) do t
    g = init_rows[(init_rows.tag .== t) .& (init_rows.rep .== 1) .& (abs.(init_rows.z .- zc) .<= slab), :]
    p = scatter(g.x, g.y; color = :lightgray, ms = 3, msw = 0, label = "", aspect_ratio = 1,
                title = t, xlabel = tex("x (\$\\mu\$m)"), ylabel = tex("y (\$\\mu\$m)"))
    f = g[g.n_desc .> 0, :]
    isempty(f) || scatter!(p, f.x, f.y; zcolor = log10.(f.n_desc), color = :plasma, ms = 4, msw = 0,
                           colorbar_title = tex("log\$_{10}\$ descendants"), label = "")
    p
end
savefig(plot(pA...; layout = (1, length(pA)), size = (520 * length(pA), 470)),
        joinpath(OUTDIR, "founder_maps.pdf"))

# B. Radial profiles: P(survive), P(founder), mean descendants per initial cell
edges = range(0, Rmax + 1e-6; length = 9)
mid   = (edges[1:end-1] .+ edges[2:end]) ./ 2
prof(v, r) = [mean(v[(r .>= edges[k]) .& (r .< edges[k + 1])]) for k in 1:length(mid)]
pB1 = plot(xlabel = tex("Radial position at irradiation (\$\\mu\$m)"), ylabel = "Fraction of cells",
           legend = :topleft, title = "Survival and founding")
pB2 = plot(xlabel = tex("Radial position at irradiation (\$\\mu\$m)"), ylabel = "Share of regrown spheroid",
           legend = :topleft, title = "Origin of the regrown spheroid")
for t in tags
    g = init_rows[init_rows.tag .== t, :]
    plot!(pB1, mid, prof(Float64.(g.survived), g.r); color = tcol[t], ls = :dash, marker = :circle,
          label = "$t survived")
    plot!(pB1, mid, prof(Float64.(g.n_desc .> 0), g.r); color = tcol[t], marker = :diamond,
          label = "$t founder")
    share = [sum(g.n_desc[(g.r .>= edges[k]) .& (g.r .< edges[k + 1])]) for k in 1:length(mid)]
    plot!(pB2, mid, share ./ max(sum(share), 1); color = tcol[t], marker = :circle, label = t)
end
savefig(plot(pB1, pB2; layout = (1, 2), size = (1300, 450), left_margin = 6Plots.mm,
             bottom_margin = 6Plots.mm), joinpath(OUTDIR, "radial_origin.pdf"))

# C. Composition of the regrown spheroid by founder phase vs initial composition
phs = ["G0", "G1", "S", "G2", "M"]
comp_init  = [count(==(p), init_rows.phase0[(init_rows.tag .== tags[1]) .& (init_rows.rep .== 1)]) for p in phs]
comp_init  = comp_init ./ sum(comp_init)
bars = hcat(comp_init, [begin g = init_rows[init_rows.tag .== t, :]
                          v = [sum(g.n_desc[g.phase0 .== p]) for p in phs]; v ./ max(sum(v), 1) end
                        for t in tags]...)
pC = groupedbar(phs, bars; bar_position = :dodge, xlabel = "Phase of the founder at irradiation",
                ylabel = "Fraction", label = permutedims(vcat("initial spheroid", tags)),
                color = permutedims(vcat(:black, [tcol[t] for t in tags])), legend = :topright)

# D. Clone-size distribution (complementary CDF)
pD = plot(xscale = :log10, yscale = :log10, xlabel = "Clone size (descendants)",
          ylabel = tex("P(size \$\\geq\$ n)"), legend = :bottomleft)
for t in tags
    s = sort(init_rows.n_desc[(init_rows.tag .== t) .& (init_rows.n_desc .> 0)])
    isempty(s) && continue
    n = length(s)
    plot!(pD, s, 1 .- (0:n-1) ./ n; seriestype = :steppost, color = tcol[t], label = t)
end
savefig(plot(pC, pD; layout = (1, 2), size = (1300, 450), left_margin = 6Plots.mm,
             bottom_margin = 6Plots.mm), joinpath(OUTDIR, "phase_and_clone_size.pdf"))
println("\nResults in ", OUTDIR)
