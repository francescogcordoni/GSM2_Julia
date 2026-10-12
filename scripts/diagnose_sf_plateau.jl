# ============================================================================
# diagnose_sf_plateau.jl
#
# Why does the clonogenic fraction stop at ~0.6 at high dose? For one ion and
# dose: dose, lesions, GSM2 survival probability and fate of EVERY cell, split
# by phase at irradiation (G0 vs cycling) and by radial position.
#
# Run: julia -t auto scripts/diagnose_sf_plateau.jl
# Output: data/diagnose_sf_plateau/cells.csv + printed tables
# ============================================================================
include(joinpath(@__DIR__, "spheroid_common.jl"))

const TUMOR_RADIUS = 150.0
const X_BOX        = 350.0
const PARTICLE, ENERGY, DOSE = "1H", 100.0, 8.0
const OUTDIR = joinpath(@__DIR__, "..", "data", "diagnose_sf_plateau"); mkpath(OUTDIR)

println("fate model = $(FATE_MODEL[])   checkpoint = $(CHECKPOINT[])   quiescence k = $(QUIESCENCE_EMPTY_MAX[])")

pristine = build_spheroid(TUMOR_RADIUS, X_BOX)
df = dose_copy(ENERGY, PARTICLE, DOSE, pristine; tumor_radius = TUMOR_RADIUS, X_box = X_BOX)
Random.seed!(1)
sample_damage!(df, DOSE)

alive = findall(==(1), df.is_cell)
phase0 = String.(df.cell_cycle)
r = [sqrt(df.x[i]^2 + df.y[i]^2 + (hasproperty(df, :z) ? df.z[i]^2 : 0.0)) for i in 1:nrow(df)]
gsm_for(ph) = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
spv = [domain_GSM2(df.dam_X_dom[i], df.dam_Y_dom[i], gsm_for(phase0[i])) for i in 1:nrow(df)]

compute_times_domain!(df, gsm2_cycle; terminal_time = Inf, summary = true)

cells = DataFrame(index = df.index[alive], phase0 = phase0[alive], r = r[alive],
                  dose_cell = hasproperty(df, :dose_cell) ? df.dose_cell[alive] : fill(NaN, length(alive)),
                  dose_mean = [isempty(df.dose[i]) ? 0.0 : mean(df.dose[i]) for i in alive],
                  X = [sum(df.dam_X_dom[i]) for i in alive],    # cleared for repaired/doomed cells
                  Y = [sum(df.dam_Y_dom[i]) for i in alive],
                  sp = spv[alive],
                  alive_after = df.is_cell[alive] .== 1,
                  doomed = isfinite.(df.death_time[alive]),
                  death_type = df.death_type[alive])
cells.clonogenic = cells.alive_after .& .!cells.doomed
CSV.write(joinpath(OUTDIR, "cells.csv"), cells)

println("\n== $(PARTICLE) $(ENERGY) MeV, $(DOSE) Gy: $(nrow(cells)) cells ==")
@printf("mean cell dose %.3f Gy | cells with zero dose: %d | mean GSM2 survival %.4f | observed clonogenic %.4f\n",
        mean(cells.dose_mean), count(==(0.0), cells.dose_mean), mean(cells.sp), mean(cells.clonogenic))

println("\nBy phase at irradiation:")
show(stdout, combine(groupby(cells, :phase0), nrow => :n,
     :dose_mean => mean => :dose, :sp => mean => :GSM2_survival,
     :clonogenic => mean => :observed_survival, :doomed => mean => :doomed_frac,
     :alive_after => (x -> mean(.!x)) => :removed_frac)); println()

cells.shell = floor.(Int, cells.r ./ 30)
println("\nBy radial shell (30 µm):")
show(stdout, combine(groupby(sort(cells, :shell), :shell), nrow => :n,
     :dose_mean => mean => :dose, :sp => mean => :GSM2_survival,
     :clonogenic => mean => :observed_survival)); println()

println("\nUnexpected survivors (GSM2 survival < 0.01 but clonogenic): ",
        count((cells.sp .< 0.01) .& cells.clonogenic))
println("Death types: ", countmap(cells.death_type))
