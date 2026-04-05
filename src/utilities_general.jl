#! ============================================================================
#! utilities_general.jl
#!
#! FUNCTIONS
#! ---------
#~ Hit Generation
#!   Each function has two overloads: with and without an explicit RNG.
#!   The no-RNG overload delegates to Random.default_rng().
#!   All disk-sampling functions use the inverse-CDF (sqrt) method for
#!   correct area uniformity (avoids the rim bias of naive r = R*U sampling).
#
#?   GenerateHit_BOX(X_box; verbose) -> (Float64, Float64)
#?   GenerateHit_BOX(rng, X_box; verbose) -> (Float64, Float64)
#       Uniform random hit in a square box [0, X_box] × [0, X_box].
#       Origin is at the lower-left corner (not centered).
#
#?   GenerateHit_Circle(x_, y_, R_beam; verbose) -> (Float64, Float64)
#?   GenerateHit_Circle(rng, x_, y_, R_beam; verbose) -> (Float64, Float64)
#       Uniform random hit inside a disk centered at (x_, y_) with radius R_beam.
#       Sampling: r = R_beam * sqrt(U), θ = 2π*V, U,V ~ Uniform(0,1).
#
#?   GenerateHit(cell, Rk; verbose) -> (Float64, Float64)
#?   GenerateHit(rng, cell, Rk; verbose) -> (Float64, Float64)
#       Uniform random hit inside a disk of radius (cell.r + Rk) centered at origin.
#       Used to sample ion impacts around a cell including track halo Rk.
#! ============================================================================

"""
    GenerateHit_BOX(X_box::Float64; verbose::Bool = false) -> Tuple{Float64,Float64}
    GenerateHit_BOX(rng::AbstractRNG, X_box::Float64; verbose::Bool = false) -> Tuple{Float64,Float64}

Uniform random hit in a square box: `x, y ~ Uniform(0, X_box)` independently.
The origin is at the lower-left corner; the box spans `[0, X_box] × [0, X_box]`.

# Example
```julia
x, y = GenerateHit_BOX(900.0)
x, y = GenerateHit_BOX(MersenneTwister(42), 900.0)
```
"""
function GenerateHit_BOX(rng::AbstractRNG, X_box::Float64; verbose::Bool = false)
    x0 = rand(rng) * X_box
    y0 = rand(rng) * X_box
    if verbose
        println("[GenerateHit_BOX] X_box=", X_box, " → (x,y)=(", x0, ", ", y0, ")")
    end
    return x0, y0
end

function GenerateHit_BOX(X_box::Float64; verbose::Bool = false)
    return GenerateHit_BOX(Random.default_rng(), X_box; verbose=verbose)
end

"""
    GenerateHit_Circle(x_, y_, R_beam::Float64; verbose::Bool = false) -> Tuple{Float64,Float64}
    GenerateHit_Circle(rng, x_, y_, R_beam::Float64; verbose::Bool = false) -> Tuple{Float64,Float64}

Uniform random hit inside a disk centered at `(x_, y_)` with radius `R_beam`.

Uses the inverse-CDF method: `r = R_beam * sqrt(U)`, `θ = 2π V` with `U, V ~ Uniform(0,1)`.
The `sqrt` corrects for the Jacobian of polar coordinates (area element is `r dr dθ`),
so that the probability of landing in any area element is proportional to its size.

# Example
```julia
x, y = GenerateHit_Circle(0.0, 0.0, 450.0)
x, y = GenerateHit_Circle(MersenneTwister(1), 100.0, 200.0, 300.0)
```
"""
function GenerateHit_Circle(
    rng::AbstractRNG,
    x_::Float64,
    y_::Float64,
    R_beam::Float64;
    verbose::Bool = false
)
    radius = R_beam * sqrt(rand(rng))
    theta  = 2π * rand(rng)
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
    GenerateHit(cell::Cell, Rk::Float64; verbose::Bool = false) -> Tuple{Float64,Float64}
    GenerateHit(rng::AbstractRNG, cell::Cell, Rk::Float64; verbose::Bool = false) -> Tuple{Float64,Float64}

Uniform random hit inside a disk of radius `cell.r + Rk` centered at the **origin**.
Used to sample ion impact coordinates relative to a cell, including the outer track
halo of radius `Rk` (penumbra of the ion track structure).

Same inverse-CDF polar sampling as `GenerateHit_Circle` — `r = R * sqrt(U)`,
`θ = 2π V` — so the hit density is uniform over the disk area.

# Example
```julia
x, y = GenerateHit(cell, Rk)
x, y = GenerateHit(MersenneTwister(7), cell, Rk)
```
"""
function GenerateHit(rng::AbstractRNG, cell::Cell, Rk::Float64; verbose::Bool = false)
    R      = cell.r + Rk
    radius = R * sqrt(rand(rng))
    theta  = 2π * rand(rng)
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
