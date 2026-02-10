"""
    GenerateHit_BOX(X_box::Float64; verbose::Bool = false) -> (x::Float64, y::Float64)
    GenerateHit_BOX(rng::AbstractRNG, X_box::Float64; verbose::Bool = false) -> (x, y)

Sample a **uniform random hit** inside a square box with side `X_box`,
i.e. `x, y ~ Uniform(0, X_box)` independently.

# Arguments
- `X_box::Float64` : Box side length.

# Keywords
- `verbose::Bool=false` : If `true`, prints the sampled coordinates.

# RNG overload
- Provide an explicit `rng::AbstractRNG` for reproducible sampling.

# Returns
- `(x, y)` sampled uniformly in `[0, X_box] × [0, X_box]`.

# Notes
- Uses independent uniform sampling on each axis.
"""
function GenerateHit_BOX(rng::AbstractRNG, X_box::Float64; verbose::Bool = false)
    x0 = rand(rng, Uniform(0, 1)) * X_box
    y0 = rand(rng, Uniform(0, 1)) * X_box
    if verbose
        println("[GenerateHit_BOX] X_box=", X_box, " → (x,y)=(", x0, ", ", y0, ")")
    end
    return x0, y0
end

function GenerateHit_BOX(X_box::Float64; verbose::Bool = false)
    return GenerateHit_BOX(Random.default_rng(), X_box; verbose=verbose)
end

"""
    GenerateHit_Circle(x_::Float64, y_::Float64, R_beam::Float64; verbose::Bool = false) -> (x, y)
    GenerateHit_Circle(rng::AbstractRNG, x_::Float64, y_::Float64, R_beam::Float64; verbose::Bool = false) -> (x, y)

Sample a **uniform random hit** inside a circle (disk) centered at `(x_, y_)` with radius `R_beam`.
Uniformity over **area** is ensured by sampling `radius = R_beam * sqrt(U)`, `theta = 2πV`,
with `U, V ~ Uniform(0,1)`.

# Arguments
- `x_, y_::Float64` : Disk center.
- `R_beam::Float64` : Disk radius.

# Keywords
- `verbose::Bool=false` : If `true`, prints the sampled polar and Cartesian coordinates.

# RNG overload
- Provide `rng::AbstractRNG` for reproducibility.

# Returns
- `(x, y)` uniformly distributed in the disk.

# Notes
- The `sqrt(U)` factor is essential for uniformity over the area (otherwise si sovracampiona il bordo).
"""
function GenerateHit_Circle(
    rng::AbstractRNG,
    x_::Float64,
    y_::Float64,
    R_beam::Float64;
    verbose::Bool = false
)
    radius = R_beam * sqrt(rand(rng, Uniform(0, 1)))
    theta  = 2π * rand(rng, Uniform(0, 1))
    x0 = radius * cos(theta) + x_
    y0 = radius * sin(theta) + y_
    if verbose
        println("[GenerateHit_Circle] center=(", x_, ", ", y_, "), R=", R_beam,
                " → radius=", radius, " theta=", theta, " → (x,y)=(", x0, ", ", y0, ")")
    end
    return x0, y0
end

function GenerateHit_Circle(x_::Float64, y_::Float64, R_beam::Float64; verbose::Bool = false)
    return GenerateHit_Circle(Random.default_rng(), x_, y_, R_beam; verbose=verbose)
end

"""
    GenerateHit(cell::Cell, Rk::Float64; verbose::Bool = false) -> (x, y)
    GenerateHit(rng::AbstractRNG, cell::Cell, Rk::Float64; verbose::Bool = false) -> (x, y)

Sample a **uniform random hit** inside a circle of radius `(cell.r + Rk)` centered at the origin.
This is typically used to sample an impact around a cell with additional halo `Rk`.

# Arguments
- `cell::Cell`     : Object providing field `r` (cell radius).
- `Rk::Float64`    : Additional radial margin (e.g., track range).

# Keywords
- `verbose::Bool=false` : If `true`, prints the sampled polar and Cartesian coordinates.

# RNG overload
- Provide `rng::AbstractRNG` for reproducible sequences.

# Returns
- `(x, y)` uniformly distributed in the disk of radius `cell.r + Rk`.

# Notes
- Uses `radius = (cell.r + Rk) * sqrt(U)` and `theta = 2πV` for uniform area sampling.
"""
function GenerateHit(rng::AbstractRNG, cell::Cell, Rk::Float64; verbose::Bool = false)
    R = (cell.r + Rk)
    radius = R * sqrt(rand(rng, Uniform(0, 1)))
    theta  = 2π * rand(rng, Uniform(0, 1))
    x0 = radius * cos(theta)
    y0 = radius * sin(theta)
    if verbose
        println("[GenerateHit] R_cell+Rk=", R, " → radius=", radius, " theta=", theta,
                " → (x,y)=(", x0, ", ", y0, ")")
    end
    return x0, y0
end

function GenerateHit(cell::Cell, Rk::Float64; verbose::Bool = false)
    return GenerateHit(Random.default_rng(), cell, Rk; verbose=verbose)
end