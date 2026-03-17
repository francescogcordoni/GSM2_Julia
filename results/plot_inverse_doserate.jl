using CSV, DataFrames
using Plots
using Statistics

#~ ============================================================
#~ Configuration — must match run_split_dose.jl
#~ ============================================================
indir  = "results"
outdir = "results"

# Condition metadata: (label, display_name, color, linestyle)
CONDITIONS = [
    (label="1H_2",   name="1H  2 MeV",    color=:steelblue,  ls=:solid),
    (label="1H_10",  name="1H  10 MeV",   color=:dodgerblue, ls=:dash),
    (label="1H_100", name="1H  100 MeV",  color=:skyblue,    ls=:dot),
    (label="12C_15", name="12C 15 MeV/u", color=:red,        ls=:solid),
    (label="12C_20", name="12C 20 MeV/u", color=:orangered,  ls=:dash),
    (label="12C_80", name="12C 80 MeV/u", color=:darkorange, ls=:dot),
]

#~ ============================================================
#~ Load data
#~ ============================================================

# Try per-condition files first; fall back to the combined master CSV
function load_condition_data(label::String)
    per_file = joinpath(indir, "phase_times_$(label).csv")
    if isfile(per_file)
        return CSV.read(per_file, DataFrame)
    end
    master = joinpath(indir, "phase_times_all.csv")
    if isfile(master)
        df = CSV.read(master, DataFrame)
        sub = filter(r -> r.type == label, df)
        nrow(sub) > 0 && return sub
    end
    @warn "Data not found for condition: $label"
    return nothing
end

data = Dict{String, DataFrame}()
for cond in CONDITIONS
    df = load_condition_data(cond.label)
    isnothing(df) || (data[cond.label] = df)
end
println("Loaded $(length(data)) / $(length(CONDITIONS)) conditions.")

#~ ============================================================
#~ PLOT 1: Survival probability vs inter-fraction time
#~ ============================================================
p_surv = plot(;
    xlabel  = "Inter-fraction time (h)",
    ylabel  = "Survival fraction",
    title   = "Split-dose: survival vs gap — 1.5 Gy × 2",
    legend  = :topright,
    size    = (900, 550),
    dpi     = 150)

for cond in CONDITIONS
    haskey(data, cond.label) || continue
    df = data[cond.label]
    plot!(p_surv, df.time, df.surv_prob;
          label     = cond.name,
          color     = cond.color,
          linestyle = cond.ls,
          lw        = 2,
          marker    = :circle,
          markersize = 4)
end
display(p_surv)
savefig(p_surv, joinpath(outdir, "split_dose_survival.png"))
println("Saved: split_dose_survival.png")

#~ ============================================================
#~ PLOT 2: Phase fractions at time of second fraction
#~   One panel per condition, stacked area of G0/G1/S/G2/M
#~ ============================================================
phase_cols   = [:G0, :G1, :S, :G2, :M]
phase_colors = [:gray :steelblue :green :orange :red]
phase_labels = ["G0" "G1" "S" "G2" "M"]

valid_conds = [c for c in CONDITIONS if haskey(data, c.label)]
n = length(valid_conds)

p_phases = plot(layout = (2, ceil(Int, n/2)), size = (500 * ceil(Int, n/2), 800), dpi = 150)

for (pi, cond) in enumerate(valid_conds)
    df  = data[cond.label]
    tot = max.(df.Nalive, 1)
    for (fi, col) in enumerate(phase_cols)
        hasproperty(df, col) || continue
        plot!(p_phases, df.time, df[!, col] ./ tot;
              subplot   = pi,
              label     = phase_labels[fi],
              color     = phase_colors[fi],
              lw        = 1.5,
              fill      = (0, phase_colors[fi], 0.25),
              title     = cond.name,
              xlabel    = pi > ceil(Int, n/2) ? "Time (h)" : "",
              ylabel    = pi % ceil(Int, n/2) == 1 ? "Phase fraction" : "",
              ylims     = (0, 1))
    end
end
display(p_phases)
savefig(p_phases, joinpath(outdir, "split_dose_phases.png"))
println("Saved: split_dose_phases.png")

#~ ============================================================
#~ PLOT 3: Alive cells vs inter-fraction time
#~ ============================================================
p_alive = plot(;
    xlabel  = "Inter-fraction time (h)",
    ylabel  = "Mean alive cells (pre-2nd fraction)",
    title   = "Cells alive at time of 2nd fraction",
    legend  = :topright,
    size    = (900, 500),
    dpi     = 150)

for cond in CONDITIONS
    haskey(data, cond.label) || continue
    df = data[cond.label]
    plot!(p_alive, df.time, df.Nalive;
          label     = cond.name,
          color     = cond.color,
          linestyle = cond.ls,
          lw        = 2)
end
display(p_alive)
savefig(p_alive, joinpath(outdir, "split_dose_alive.png"))
println("Saved: split_dose_alive.png")

#~ ============================================================
#~ PLOT 4: Proton vs Carbon comparison — survival only
#~ ============================================================
p_compare = plot(;
    xlabel  = "Inter-fraction time (h)",
    ylabel  = "Survival fraction",
    title   = "1H vs 12C — split-dose survival",
    legend  = :topright,
    size    = (900, 500),
    dpi     = 150)

proton_conds  = [c for c in CONDITIONS if startswith(c.label, "1H")]
carbon_conds  = [c for c in CONDITIONS if startswith(c.label, "12C")]

for cond in proton_conds
    haskey(data, cond.label) || continue
    df = data[cond.label]
    plot!(p_compare, df.time, df.surv_prob;
          label = cond.name, color = cond.color, ls = cond.ls, lw = 2)
end
for cond in carbon_conds
    haskey(data, cond.label) || continue
    df = data[cond.label]
    plot!(p_compare, df.time, df.surv_prob;
          label = cond.name, color = cond.color, ls = cond.ls, lw = 2)
end
display(p_compare)
savefig(p_compare, joinpath(outdir, "split_dose_1H_vs_12C.png"))
println("Saved: split_dose_1H_vs_12C.png")

#~ ============================================================
#~ FINAL PRINT
#~ ============================================================
println("\n", "="^60)
println("ALL PLOTS SAVED TO $outdir/")
println("="^60)
for f in filter(f -> endswith(f, ".png"), sort(readdir(outdir)))
    println("  $outdir/$f")
end
