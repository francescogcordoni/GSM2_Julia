using Plots

xrow = collect(df_center_x[1, 1:(end - 1)])
yrow = collect(df_center_y[1, 1:(end - 1)])

p = scatter(xrow, yrow;
    label = "domain centers",
    aspect_ratio = :equal,
    xlabel = "x",
    ylabel = "y"
)

x = collect(df_center_x[1, 1:(end-1)])
y = collect(df_center_y[1, 1:(end-1)])
θ = range(0, 2π; length=400)

for (xi, yi) in zip(x, y)
    xc = xi .+ rd .* cos.(θ)
    yc = yi .+ rd .* sin.(θ)
    plot!(p, xc, yc; label="", lw=1.5, linecolor=:gray, alpha=0.7)
end

px = cell_df.x[1]
py = cell_df.y[1]
scatter!(p, [px], [py]; color=:red, markersize=8, label="center")

x0 = px
y0 = py

θ = range(0, 2π; length=400)
xc = x0 .+ Rn .* cos.(θ)
yc = y0 .+ Rn .* sin.(θ)

plot!(p, xc, yc; lw=2, linecolor=:black, label="R = $Rn", legend = :best)

display(p)
