"""
calculate_centers(x0::Float64, y0::Float64, radius::Float64, nucleus::Float64)

Compute the positions of the centers of domains of radius radius in a circle of radius nucleus

# Arguments
- `x0::Float64`: the x-coordinate of the center of the cell
- `y0::Float64`: the y-coordinate of the center of the cell
- `radius::Float64`: the radius of the domain
- `nucleus::Float64`: the radius of the nucleus

# Returns
- `center_x::Vector{Float64}`: the x-coordinates of the centers of the domains
- `center_y::Vector{Float64}`: the y-coordinates of the centers of the domains
"""
function calculate_centers(x0::Float64, y0::Float64, radius::Float64, nucleus::Float64)

    # Initialize the vectors to store the positions of the centers of the spheres
    center_x = Vector{Float64}(undef, 0)
    center_y = Vector{Float64}(undef, 0)

    # Add the center of the lattice
    push!(center_x, x0)
    push!(center_y, y0)

    # Loop over the layers of the lattice
    for rho in 1:(floor(Int64, nucleus / (2 * radius)))
        # Calculate the radius of the current layer
        center_rho = (rho) * 2 * radius

        # Calculate the number of spheres in the current layer
        n_circle = floor(Int64, (pi * center_rho) / radius)

        # Calculate the positions of the spheres in the current layer
        theta = range(0, 2π, length=n_circle)
        xi = x0 .+ center_rho .* cos.(theta[1:(end-1)])
        yi = y0 .+ center_rho .* sin.(theta[1:(end-1)])

        # Add the positions of the spheres in the current layer to the vectors
        center_x = vcat(center_x, xi)
        center_y = vcat(center_y, yi)
    end

    return (center_x, center_y)
end

"""
create_domain_dataframes(
        cell_df::DataFrame,
        rel_center_x::Vector{Float64},
        rel_center_y::Vector{Float64}
    ) -> (df_center_x, df_center_y, at)

Generate the absolute domain-center coordinates for every cell and create the
corresponding DataFrames used by the AT (Amorphous Track) computation.

# Purpose
Each cell has a template of **relative** domain-center positions (`rel_center_x`,
`rel_center_y`) produced by `calculate_centers`. This function:

1. Shifts those relative coordinates by each cell’s (x, y) position → **absolute positions**
2. Builds two DataFrames:
    - `df_center_x`: x‑coordinates of all domains for all cells
    - `df_center_y`: y‑coordinates of all domains for all cells
3. Initializes an empty `at` DataFrame used to store domain-level track quantities.

# Arguments
- `cell_df`:
    DataFrame containing at least `:x` and `:y` cell center coordinates.
- `rel_center_x`, `rel_center_y`:
    Vectors of relative offsets (domain layout template).

# Returns
- `df_center_x`: absolute domain center x-positions per cell
- `df_center_y`: absolute domain center y-positions per cell
- `at`: AT dataframe initialized with zeros, matching domain dimensionality

# Notes
- Domain counts ≡ `length(rel_center_x)`  
- Uses broadcasting for fast vectorized transformations  
- Each DataFrame row corresponds to a **cell**, each column to a **domain**  
"""
function create_domain_dataframes(cell_df::DataFrame,
                                    rel_center_x::Vector{Float64},
                                    rel_center_y::Vector{Float64})

    println("... Creating domain dataframes ...")

    # Number of domains per cell (i.e., number of relative center coordinates)
    num_cols = length(rel_center_x)

    # ------------------------------------------------------------------
    # Compute absolute domain center coordinates
    #
    # For each cell:
    #   absolute_x = cell.x + rel_center_x[:]
    #   absolute_y = cell.y + rel_center_y[:]
    #
    # Broadcasting + transpose rel_center vectors gives a matrix:
    #   rows   = cells
    #   cols   = domain centers
    # ------------------------------------------------------------------
    mat_center_x = cell_df.x .+ transpose(rel_center_x)
    mat_center_y = cell_df.y .+ transpose(rel_center_y)

    # ------------------------------------------------------------------
    # Create DataFrames for the domain centers
    # Column names are :center_1, :center_2, ..., :center_N
    # ------------------------------------------------------------------
    df_center_x = DataFrame(mat_center_x, Symbol.("center_$i" for i in 1:num_cols))
    df_center_x.index = cell_df.index   # preserve cell indexing

    df_center_y = DataFrame(mat_center_y, Symbol.("center_$i" for i in 1:num_cols))
    df_center_y.index = cell_df.index

    # ------------------------------------------------------------------
    # Create the AT dataframe
    #
    # AT (Amorphous Track) table will hold domain-level track quantities.
    # It is initialized as a zero matrix with:
    #
    #   rows = number of cells
    #   cols = (num_cols - 1)   # historical convention from original code
    #
    # Rename columns to match the domain naming convention.
    # ------------------------------------------------------------------
    at = DataFrame(zeros(size(df_center_y, 1), (size(df_center_y, 2) - 1)), :auto)
    rename!(at, Symbol.("center_$i" for i in 1:(size(df_center_y, 2) - 1)))
    at.index = df_center_y.index

    return df_center_x, df_center_y, at
end

"""
    MC_loop_damage!(ion, cell_df, irrad_cond, gsm2;
                                        verbose=false, show_progress=true)

Compute DNA damage (X and Y components) for each cell from domain-level
dose vectors using a Poisson process.

- LET is taken from irrad_cond[cell.energy_step].LET
- Oxygen O is taken from cell_df.O
- Dose distribution for each cell is taken from cell_df.dose::Vector{Float64}

Damage per domain uses:
    λX = kappa_base * d
    λY = lambda_base * d
with:
    kappa_base  = 9 * kappa_yield / (n_repeat * N_domains)
    lambda_base = kappa_base * 1e-3
"""
function MC_loop_damage!(
    ion::Ion,
    cell_df::DataFrame,
    irrad_cond::Vector{AT},
    gsm2::GSM2;
    verbose::Bool = false,
    show_progress::Bool = true
) where {AT}

    vprintln(args...) = (verbose ? println("[DEBUG] ", args...) : nothing)

    println("\n-----------------------------------------------------------")
    println("     MC_loop_damage! (FAST DAMAGE MODEL)")
    println("-----------------------------------------------------------")

    num_cells = nrow(cell_df)
    n_repeat  = floor(Int, gsm2.Rn / gsm2.rd)

    println("Cells       : $num_cells")
    println("n_repeat    : $n_repeat\n")

    # Check required fields
    @assert hasproperty(cell_df, :dose) "Missing column: dose"
    @assert hasproperty(cell_df, :energy_step) "Missing column: energy_step"
    @assert hasproperty(cell_df, :O) "Missing column: O"
    @assert hasproperty(cell_df, :is_cell) "Missing column: is_cell"

    # Determine expected output vector length
    first_nonempty = findfirst(x -> !isempty(x), cell_df.dose)
    expected_len = if first_nonempty === nothing
        0
    else
        length(cell_df.dose[first_nonempty]) * n_repeat
    end

    # Initialize X and Y damage vectors
    for cname in (:dam_X_dom, :dam_Y_dom)
        if !hasproperty(cell_df, cname)
            cell_df[!, cname] = [zeros(Int, expected_len) for _ in 1:num_cells]
        else
            # Verifica che sia del tipo corretto
            col = cell_df[!, cname]
            if !(col isa Vector{Vector{Int}})
                @warn "$cname had wrong type ($(typeof(col))), reinitializing"
                cell_df[!, cname] = [zeros(Int, expected_len) for _ in 1:num_cells]
            end
        end
    end

    # Initialize total damage columns (scalars)
    if !hasproperty(cell_df, :dam_X_total)
        cell_df[!, :dam_X_total] = zeros(Int, num_cells)
    end
    if !hasproperty(cell_df, :dam_Y_total)
        cell_df[!, :dam_Y_total] = zeros(Int, num_cells)
    end

    # Progress bar
    if show_progress
        p = Progress(num_cells, 1, "Damage: ")
    end

    # Pre-allocate Poisson sample buffers (riutilizzati per evitare allocazioni)
    buffer_X = Vector{Int}(undef, n_repeat)
    buffer_Y = Vector{Int}(undef, n_repeat)

    # ---------------------------------------------------------
    # Main loop over cells
    # ---------------------------------------------------------
    for i in 1:num_cells
        show_progress && next!(p)
        vprintln("Processing cell $i")

        # Skip inactive or zero-dose cells
        if cell_df.is_cell[i] != 1 || isempty(cell_df.dose[i])
            fill!(cell_df.dam_X_dom[i], 0)
            fill!(cell_df.dam_Y_dom[i], 0)
            cell_df.dam_X_total[i] = 0
            cell_df.dam_Y_total[i] = 0
            continue
        end
        
        if hasproperty(cell_df, :dose_cell) && cell_df.dose_cell[i] <= 0.0
            fill!(cell_df.dam_X_dom[i], 0)
            fill!(cell_df.dam_Y_dom[i], 0)
            cell_df.dam_X_total[i] = 0
            cell_df.dam_Y_total[i] = 0
            continue
        end

        dose_vector = cell_df.dose[i]
        L = length(dose_vector)
        O = cell_df.O[i]
        LET = irrad_cond[cell_df.energy_step[i]].LET

        # YIELD
        kappa_yield = calculate_kappa(ion.ion, LET, O)
        kappa_base  = 9.0 * kappa_yield / (n_repeat * L)
        lambda_base = kappa_base * 1e-3

        vprintln("  LET=", LET, "  O=", O,
                    "  kappa_base=", kappa_base,
                    "  lambda_base=", lambda_base)

        # Ensure vector sizes
        row_len = L * n_repeat
        Xrow = cell_df.dam_X_dom[i]
        Yrow = cell_df.dam_Y_dom[i]
        
        # Ridimensiona se necessario
        if length(Xrow) != row_len
            resize!(Xrow, row_len)
            cell_df.dam_X_dom[i] = Xrow
        end
        if length(Yrow) != row_len
            resize!(Yrow, row_len)
            cell_df.dam_Y_dom[i] = Yrow
        end

        # Fill the vector
        pos = 1
        @inbounds for d in dose_vector
            λX = max(0.0, kappa_base * d)
            λY = max(0.0, lambda_base * d)

            # Campiona in buffer pre-allocati
            if λX > 0.0
                rand!(Poisson(λX), buffer_X)
            else
                fill!(buffer_X, 0)
            end
            
            if λY > 0.0
                rand!(Poisson(λY), buffer_Y)
            else
                fill!(buffer_Y, 0)
            end

            # Copia nei vettori finali
            copyto!(Xrow, pos, buffer_X, 1, n_repeat)
            copyto!(Yrow, pos, buffer_Y, 1, n_repeat)

            pos += n_repeat
        end

        # Calcola i totali (somma di tutti gli elementi del vettore)
        cell_df.dam_X_total[i] = sum(Xrow)
        cell_df.dam_Y_total[i] = sum(Yrow)
        
        vprintln("  dam_X_total=", cell_df.dam_X_total[i], 
                "  dam_Y_total=", cell_df.dam_Y_total[i])
    end

    println("\n✔ Finished damage computation.")
    println("  Total X damages across all cells: ", sum(cell_df.dam_X_total))
    println("  Total Y damages across all cells: ", sum(cell_df.dam_Y_total))
    println()
    
    return nothing
end

"""
    calculate_OER(LET, O)

Compute the Oxygen Enhancement Ratio (OER) using LET and oxygenation O,
following the model in the original calculate_OER function.

Inputs:
- LET :: Float64
- O   :: Float64   (oxygen level for the cell)

Returns:
- OER :: Float64
"""
function calculate_OER(LET::Float64, O::Float64)
    M0 = 3.4
    b  = 0.41
    a  = 8.27e5
    g  = 3.0

    return (b * (a*M0 + LET^g) / (a + LET^g) + O) / (b + O)
end


"""
    calculate_kappa(ion_name, LET, O; OER_bool=true)

Compute the yield of damage per unit Gy using:
- ion species (p1..p5 parameters)
- LET of the cell (layer-dependent)
- O2 of that cell
"""
function calculate_kappa(ion_name::String, LET::Float64, O::Float64; OER_bool::Bool = true)

    # Ion-specific parameters
    if ion_name == "12C"
        p1,p2,p3,p4,p5 = 6.8, 0.156, 0.9214, 0.005245, 1.395
    elseif ion_name == "4He" || ion_name == "3He"
        p1,p2,p3,p4,p5 = 6.8, 0.1471, 1.038, 0.006239, 1.582
    elseif ion_name == "1H" || ion_name == "2H"
        p1,p2,p3,p4,p5 = 6.8, 0.1773, 0.9314, 0.0, 1.0
    elseif ion_name == "16O"
        p1,p2,p3,p4,p5 = 6.8, 0.1749, 0.8722, 0.004987, 1.347
    else
        @warn "Unknown ion species '$ion_name'"
        return 0.0
    end

    # Basic yield
    yield = (p1 + (p2 * LET)^p3) / (1 + (p4 * LET)^p5)

    # Apply OER correction if required
    if OER_bool
        OER = calculate_OER(LET, O)
        yield /= OER
    end

    return yield
end

"""
    compute_cell_survival_GSM2!(cell_df, gsm2; NFrac::Int = 1)

Compute per‑cell survival probabilities using the GSM2 model and update
the corresponding fields of `cell_df` **in place**.

This function evaluates the radiobiological survival probability for each
active cell (`is_cell == 1`) based on its accumulated X- and Y-type damage,
using the GSM2 model.  
The final survival probability accounts for a specified number of fractions
(`NFrac`), which defaults to **1**.

# Behavior
- Resets/initializes the following per-cell state variables:
    * `sp`             → survival probability (set to 1.0 initially)
    * `apo_time`       → apoptotic trigger time (set to `Inf`)
    * `death_time`     → total death time (set to `Inf`)
    * `recover_time`   → repair/recovery time (set to `Inf`)
    * `cycle_time`     → cell-cycle time (set to `Inf`)
    * `is_death_rad`   → radiogenic death flag (set to 0)
    * `death_type`     → categorical death mode (set to -1)

- For each active cell (`is_cell == 1`):
    * Retrieves X- and Y-type domain damage (`dam_X_dom`, `dam_Y_dom`)
    * Evaluates the GSM2 survival function:
        `SP_cell = domain_GSM2(dam_X_dom[i], dam_Y_dom[i], gsm2)`
    * Applies the number of fractions:
        `SP_total = SP_cell ^ NFrac`
    * Stores `SP_total` into `cell_df.sp[i]`

# Arguments
- `cell_df::DataFrame`: table containing per‑cell radiobiological information  
- `gsm2::GSM2`: instance of the GSM2 model  
- `NFrac::Int = 1`: **number of fractions**, default = *1*  

# Notes
- This function **modifies `cell_df` in place**.
- Only rows with `is_cell == 1` are processed.

# Returns
- Nothing (`nothing`)
"""
function compute_cell_survival_GSM2!(cell_df::DataFrame, gsm2::GSM2; NFrac::Int64 = 1)

    cell_df.sp .= 1.
    cell_df.apo_time .= Inf
    cell_df.death_time .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type .= -1

    for i in cell_df.index[cell_df.is_cell .== 1]
        # survival using GSM2
        SP_cell = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

        # apply number of fractions (default NFrac = 1)
        cell_df.sp[i] = SP_cell ^ NFrac
    end
end

"""
domain_GSM2(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2)

Compute the per-domain survival probability predicted by the GSM2 model,
based on the number of X- and Y-type DNA damage clusters in a cell domain.

This function evaluates the GSM2 survival model **for a single cell**,
given the lists of domain-level damages `X` (single-track lesions) and `Y`
(multi-track interactions). The GSM2 model assumes that:

- **Any Y-type lesion is lethal**, i.e. if `sum(Y) > 0`, the survival
  probability is immediately **0**.
- The surviving probability is the **product** of survival probabilities
    for each domain, computed iteratively from the number of X-lesions.

# Mathematical model

For each domain `j`, let `X[j]` be the number of X-lesions.

The survival probability contribution of that domain is:
p_j = ∏_{i = 1 to X[j]}  ( i * r ) / ( (r + a)i + bi*(i-1) )
with GSM2 parameters:

- `r`  → repair probability  
- `a`  → lethal conversion parameter  
- `b`  → sublethal interaction parameter  

The total cell survival is:
SP = ∏_{j} p_j
# Arguments
- `X::Vector{Int64}`  
  Number of X-type lesions **per domain**.

- `Y::Vector{Int64}`  
  Number of Y-type lesions **per domain**.  
  If `sum(Y) > 0`, the model returns **0**.

- `gsm2::GSM2`  
    Object containing GSM2 parameters `r`, `a`, `b`.

# Returns
- `p_cell::Float64`  
    Survival probability predicted by the GSM2 domain model.

"""
function domain_GSM2(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2)

    # Any Y-type cluster is lethal
    if sum(Y) > 0
        return 0.
    end

    p_cell = 1.

    for j in 1:length(X)
        p = 1.0
        for i in X[j]:-1:1
            p *= (i * gsm2.r) /
                 ((gsm2.r + gsm2.a) * i + gsm2.b * i * (i - 1))
        end
        p_cell *= p
    end

    return p_cell
end

