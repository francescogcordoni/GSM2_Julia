using Base.Threads
using Distributed
using CSV, DataFrames
using Distributions
using Random
using Plots
using ProgressBars
using GLM
using JLD2
using DelimitedFiles
using StatsPlots
using Statistics
using StatsBase
using Optim
using LsqFit
using ProgressMeter
using InlineStrings
using InlineStrings
using CUDA
using Statistics: mean

nthreads()

#~ ============================================================
#~ Load functions
#~ ============================================================
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))

#&Stopping power
sp = load_stopping_power()

df = CSV.read("data/phase_times_async.csv", DataFrame)

df_12C_15  = filter(:type => ==("12C_15"),  df)
df_12C_20  = filter(:type => ==("12C_20"),  df)
df_12C_80  = filter(:type => ==("12C_80"),  df)
df_1H_2   = filter(:type => ==("1H_2"),   df)
df_1H_10   = filter(:type => ==("1H_10"),   df)
df_1H_100  = filter(:type => ==("1H_100"),  df)

p1 = plot(df_12C_15.time, df_12C_15.surv_prob;
    lw=2, color=:black, label="Survival probability",
    xlabel="Time [h]", ylabel="Survival probability")

p2 = plot(df_12C_15.time, df_12C_15.Nalive;
    lw=2, color=:black, label="G1",
    xlabel="Survival probability", ylabel="N alive")
plot!(df_12C_15.time, df_12C_15.G1;
    lw=2, color=:steelblue, label="G1")
plot!(df_12C_15.time, df_12C_15.S;  lw=2, color="#D55E00", label="S")
plot!(df_12C_15.time, df_12C_15.G2; lw=2, color=:forestgreen, label="G2")
plot!(df_12C_15.time, df_12C_15.M;  lw=2, color=:purple, label="M")

p_12C_15 = plot(p1, p2; layout=@layout([a; b]), size=(900,700),
        title="12C 15 MeV/u")


p1 = plot(df_12C_20.time, df_12C_20.surv_prob;
    lw=2, color=:black, label="Survival probability",
    xlabel="Time [h]", ylabel="Survival probability")

p2 = plot(df_12C_20.time, df_12C_20.Nalive;
    lw=2, color=:black, label="G1",
    xlabel="Survival probability", ylabel="N alive")
plot!(df_12C_20.time, df_12C_20.G1;
    lw=2, color=:steelblue, label="G1")
plot!(df_12C_20.time, df_12C_20.S;  lw=2, color="#D55E00", label="S")
plot!(df_12C_20.time, df_12C_20.G2; lw=2, color=:forestgreen, label="G2")
plot!(df_12C_20.time, df_12C_20.M;  lw=2, color=:purple, label="M")

p_12C_20 = plot(p1, p2; layout=@layout([a; b]), size=(900,700),
        title="12C 20 MeV/u")


p1 = plot(df_12C_80.time, df_12C_80.surv_prob;
    lw=2, color=:black, label="Survival probability",
    xlabel="Time [h]", ylabel="Survival probability")

p2 = plot(df_12C_80.time, df_12C_80.Nalive;
    lw=2, color=:black, label="G1",
    xlabel="Survival probability", ylabel="N alive")
plot!(df_12C_80.time, df_12C_80.G1;
    lw=2, color=:steelblue, label="G1")
plot!(df_12C_80.time, df_12C_80.S;  lw=2, color="#D55E00", label="S")
plot!(df_12C_80.time, df_12C_80.G2; lw=2, color=:forestgreen, label="G2")
plot!(df_12C_80.time, df_12C_80.M;  lw=2, color=:purple, label="M")

p_12C_80 = plot(p1, p2; layout=@layout([a; b]), size=(900,700),
        title="12C 80 MeV/u")


p1 = plot(df_1H_100.time, df_1H_100.surv_prob;
    lw=2, color=:black, label="Survival probability",
    xlabel="Time [h]", ylabel="Survival probability")

p2 = plot(df_1H_100.time, df_1H_100.Nalive;
    lw=2, color=:black, label="G1",
    xlabel="Survival probability", ylabel="N alive")
plot!(df_1H_100.time, df_1H_100.G1;
    lw=2, color=:steelblue, label="G1")
plot!(df_1H_100.time, df_1H_100.S;  lw=2, color="#D55E00", label="S")
plot!(df_1H_100.time, df_1H_100.G2; lw=2, color=:forestgreen, label="G2")
plot!(df_1H_100.time, df_1H_100.M;  lw=2, color=:purple, label="M")

p_1H_100 = plot(p1, p2; layout=@layout([a; b]), size=(900,700),
        title="1H 100 MeV/u")


p1 = plot(df_1H_10.time, df_1H_10.surv_prob;
    lw=2, color=:black, label="Survival probability",
    xlabel="Time [h]", ylabel="Survival probability")

p2 = plot(df_1H_10.time, df_1H_10.Nalive;
    lw=2, color=:black, label="G1",
    xlabel="Survival probability", ylabel="N alive")
plot!(df_1H_10.time, df_1H_10.G1;
    lw=2, color=:steelblue, label="G1")
plot!(df_1H_10.time, df_1H_10.S;  lw=2, color="#D55E00", label="S")
plot!(df_1H_10.time, df_1H_10.G2; lw=2, color=:forestgreen, label="G2")
plot!(df_1H_10.time, df_1H_10.M;  lw=2, color=:purple, label="M")

p_1H_10 = plot(p1, p2; layout=@layout([a; b]), size=(900,700),
        title="1H 10 MeV/u")


p1 = plot(df_1H_2.time, df_1H_2.surv_prob;
    lw=2, color=:black, label="Survival probability",
    xlabel="Time [h]", ylabel="Survival probability")

p2 = plot(df_1H_2.time, df_1H_2.Nalive;
    lw=2, color=:black, label="G1",
    xlabel="Survival probability", ylabel="N alive")
plot!(df_1H_2.time, df_1H_2.G1;
    lw=2, color=:steelblue, label="G1")
plot!(df_1H_2.time, df_1H_2.S;  lw=2, color="#D55E00", label="S")
plot!(df_1H_2.time, df_1H_2.G2; lw=2, color=:forestgreen, label="G2")
plot!(df_1H_2.time, df_1H_2.M;  lw=2, color=:purple, label="M")

p_1H_2 = plot(p1, p2; layout=@layout([a; b]), size=(900,700),
        title="1H 2 MeV/u")


p1 = plot(df_1H_100.time, df_1H_100.surv_prob;
    lw=2, label="1H 100 MeV/u") 
plot!(df_1H_10.time, df_1H_10.surv_prob;
    lw=2, label="1H 10 MeV/u")
plot!(df_1H_2.time, df_1H_2.surv_prob;
    lw=2, label="1H 2 MeV/u")
plot!(df_12C_80.time, df_12C_80.surv_prob;
    lw=2, label="12C 80 MeV/u")
plot!(df_12C_20.time, df_12C_20.surv_prob;
    lw=2, label="12C 20 MeV/u")
plot!(df_12C_15.time, df_12C_15.surv_prob;
    lw=2, label="12C 15 MeV/u", legend = :best)


#####histograms
using DataFrames
using Plots

# Colors (S in your rust-orange #D55E00)
phase_colors = Dict(
    "G1" => :steelblue,
    "S"  => "#D55E00",
    "G2" => :forestgreen,
    "M"  => :purple
)

"""
    phase_fraction_bar(df, target_time, label)

Compute the phase fractions at a given time and return a bar plot.

- df must have columns: :Nalive, :G1, :S, :G2, :M, :time
- target_time is the value of `time` you want
- label is the title of the histogram
"""
function phase_fraction_bar(df::DataFrame, target_time::Real, label::String)

    # Select rows at target time
    df_t = df[df.time .== target_time, :]

    if nrow(df_t) == 0
        error("No data for time = $target_time in $label")
    end

    # Aggregate if multiple rows exist for that time
    agg = combine(groupby(df_t, :time),
                  :Nalive => sum => :Nalive_sum,
                  :G1     => sum => :G1_sum,
                  :S      => sum => :S_sum,
                  :G2     => sum => :G2_sum,
                  :M      => sum => :M_sum)

    N = agg.Nalive_sum[1]

    if N == 0
        error("Nalive == 0 at time $target_time in $label, cannot compute fractions.")
    end

    fracs = Dict(
        "G1" => agg.G1_sum[1] / N,
        "S"  => agg.S_sum[1]  / N,
        "G2" => agg.G2_sum[1] / N,
        "M"  => agg.M_sum[1]  / N
    )

    labels = ["G1", "S", "G2", "M"]
    values = [fracs[p] for p in labels]

    bar(labels, values;
        ylim=(0,1), xlabel="", ylabel="",
        title="$label at t = $target_time",
        color=getindex.(Ref(phase_colors), labels),
        legend=false,
        size=(500,400))
end

target_time = 10.0  

p_12C_15_10h  = phase_fraction_bar(df_12C_15,  target_time, "12C_15")
p_12C_20_10h  = phase_fraction_bar(df_12C_20,  target_time, "12C_20")
p_12C_80_10h  = phase_fraction_bar(df_12C_80,  target_time, "12C_80")

p_1H_2_10h    = phase_fraction_bar(df_1H_2,    target_time, "1H_2")
p_1H_10_10h   = phase_fraction_bar(df_1H_10,   target_time, "1H_10")
p_1H_100_10h  = phase_fraction_bar(df_1H_100,  target_time, "1H_100")

p_10h = plot(p_12C_15_10h, p_12C_20_10h, p_12C_80_10h,
    p_1H_2_10h,   p_1H_10_10h,  p_1H_100_10h;
    layout=(2,3), size=(1800,1000))


target_time = 24.0  

p_12C_15_24h  = phase_fraction_bar(df_12C_15,  target_time, "12C_15")
p_12C_20_24h  = phase_fraction_bar(df_12C_20,  target_time, "12C_20")
p_12C_80_24h  = phase_fraction_bar(df_12C_80,  target_time, "12C_80")

p_1H_2_24h    = phase_fraction_bar(df_1H_2,    target_time, "1H_2")
p_1H_10_24h   = phase_fraction_bar(df_1H_10,   target_time, "1H_10")
p_1H_100_24h  = phase_fraction_bar(df_1H_100,  target_time, "1H_100")

# Combine in a 2×3 grid
p_24h = plot(p_12C_15_24h, p_12C_20_24h, p_12C_80_24h,
    p_1H_2_24h,   p_1H_10_24h,  p_1H_100_24h;
    layout=(2,3), size=(1800,1000))
