#! ============================================================================
#! utilities_general.jl
#!
#! FUNCTIONS
#! ---------
#~ Hit Generation
#?   GenerateHit_BOX(X_box; verbose)
#?   GenerateHit_BOX(rng, X_box; verbose)
#       Uniform random hit in a square box [0, X_box] × [0, X_box].
#
#?   GenerateHit_Circle(x_, y_, R_beam; verbose)
#?   GenerateHit_Circle(rng, x_, y_, R_beam; verbose)
#       Uniform random hit inside a disk centered at (x_, y_) with radius R_beam.
#       Uses sqrt(U) sampling for correct area uniformity.
#
#?   GenerateHit(cell, Rk; verbose)
#?   GenerateHit(rng, cell, Rk; verbose)
#       Uniform random hit inside a disk of radius (cell.r + Rk) centered at origin.
#       Used to sample ion impacts around a cell with track halo Rk.
#! ============================================================================

"""
    GenerateHit_BOX(X_box::Float64; verbose::Bool = false) -> (x, y)
    GenerateHit_BOX(rng::AbstractRNG, X_box::Float64; verbose::Bool = false) -> (x, y)

Uniform random hit in a square box: `x, y ~ Uniform(0, X_box)` independently.

# Example
```julia
x, y = GenerateHit_BOX(900.0)
x, y = GenerateHit_BOX(MersenneTwister(42), 900.0)
```
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
    GenerateHit_Circle(x_, y_, R_beam::Float64; verbose::Bool = false) -> (x, y)
    GenerateHit_Circle(rng, x_, y_, R_beam::Float64; verbose::Bool = false) -> (x, y)

Uniform random hit inside a disk centered at `(x_, y_)` with radius `R_beam`.
Uses `radius = R_beam * sqrt(U)` for correct area uniformity (naive sampling oversamples the rim).

# Example
```julia
x, y = GenerateHit_Circle(0.0, 0.0, 450.0)
```
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

Uniform random hit inside a disk of radius `cell.r + Rk` centered at the origin.
Used to sample ion impact points around a cell including track halo `Rk`.

# Example
```julia
x, y = GenerateHit(cell, track.Rk)
```
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
