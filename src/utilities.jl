

#######################




##############


function set_energy_step!(cell_df::DataFrame, irrad_cond_original_layers::Vector{AT})

    layer_actual_energies = getfield.(irrad_cond_original_layers, :E)
    positive_energies_in_sim = filter(e -> e > 0, layer_actual_energies)

    if isempty(positive_energies_in_sim)
        @warn "No positive Energies found in irrad_cond_original_layers. Assigning energy_step = 1 to all cells."
        cell_df.energy_step .= 1

        if !isempty(irrad_cond_original_layers)
            # The new irrad_cond will have 1 element: the AT object from the first original layer.
            global irrad_cond = [irrad_cond_original_layers[1]]
        else
            @error "irrad_cond_original_layers is empty. Cannot form final irrad_cond."
            # Fallback: create an empty AT array. Subsequent code must handle this.
            global irrad_cond = Array{AT}(undef, 0)
        end
    else
        min_ref_energy = minimum(positive_energies_in_sim)

        # Calculate energy_step for each cell in cell_df
        new_energy_steps_for_cells = similar(cell_df.layer, Int)
        for (i, layer_idx) in enumerate(cell_df.layer) # layer_idx is 1-based original layer index
            if !(1 <= layer_idx <= length(layer_actual_energies))
                @error "Invalid layer_idx $layer_idx for cell $i when accessing layer_actual_energies. Max index is $(length(layer_actual_energies)). Assigning default energy_step = 1."
                new_energy_steps_for_cells[i] = 1
                continue
            end
            cell_energy = layer_actual_energies[layer_idx]

            if cell_energy <= 0
                new_energy_steps_for_cells[i] = 1
            else
                ratio = cell_energy / min_ref_energy
                # Handle floating point precision for Energies that are effectively equal to min_ref_energy
                if ratio < 1.0 && isapprox(ratio, 1.0)
                    ratio = 1.0
                end

                if ratio < 1.0 # Implies cell_energy is positive but significantly less than min_ref_energy.
                    # This should not happen if min_ref_energy is the true minimum of positive Energies.
                    @warn "Cell Energy $cell_energy at original layer $layer_idx is positive but less than min_ref_energy $min_ref_energy. Assigning step 1. This may indicate an issue with min_ref_energy derivation or data consistency."
                    new_energy_steps_for_cells[i] = 1
                else
                    # Binning: 10 bins per decade
                    new_energy_steps_for_cells[i] = convert(Int, floor(10 * log10(ratio)) + 1)
                end
            end
        end
        cell_df.energy_step = new_energy_steps_for_cells

        # Now, determine the energy_step mapping for each *original layer*.
        # This map will be used to select representative AT objects for the new irrad_cond.
        energy_step_map_for_orig_layers = similar(layer_actual_energies, Int)
        for l_orig in eachindex(layer_actual_energies)
            energy_orig_layer = layer_actual_energies[l_orig]
            if energy_orig_layer <= 0
                energy_step_map_for_orig_layers[l_orig] = 1
            else
                ratio_orig = energy_orig_layer / min_ref_energy
                if ratio_orig < 1.0 && isapprox(ratio_orig, 1.0)
                    ratio_orig = 1.0
                end

                if ratio_orig < 1.0 # Positive Energy < min_ref_energy (and not approx equal)
                    # This is unexpected if min_ref_energy is correctly found.
                    @error "Logic error or data inconsistency: Positive original layer Energy $energy_orig_layer (layer index $l_orig) is less than min_ref_energy $min_ref_energy. Assigning step 1 for this layer's binning."
                    energy_step_map_for_orig_layers[l_orig] = 1
                else
                    energy_step_map_for_orig_layers[l_orig] = convert(Int, floor(10 * log10(ratio_orig)) + 1)
                end
            end
        end

        max_final_step = 0
        if !isempty(energy_step_map_for_orig_layers)
            max_final_step = maximum(energy_step_map_for_orig_layers)
        elseif !isempty(irrad_cond_original_layers) # Map is empty, but original layers exist (e.g. all non-positive E)
            max_final_step = 1 # Default to 1 if map is empty but layers existed
        end
        if max_final_step == 0 # Should only happen if irrad_cond_original_layers was empty
            @warn "max_final_step was calculated as 0. This implies irrad_cond_original_layers was empty. Setting to 1 for safety, but irrad_cond might be problematic."
            max_final_step = 1
        end

        # Create the new, condensed irrad_cond.
        irrad_cond_final_temp = Array{AT}(undef, max_final_step)
        filled_steps = falses(max_final_step) # To track which steps in irrad_cond_final_temp are filled.

        # Populate irrad_cond_final_temp: for each step k, take the AT from the first original layer that maps to step k.
        for l_orig in eachindex(irrad_cond_original_layers)
            current_mapped_step = energy_step_map_for_orig_layers[l_orig]
            # Ensure the mapped step is valid and within bounds for irrad_cond_final_temp
            if current_mapped_step > 0 && current_mapped_step <= max_final_step
                if !filled_steps[current_mapped_step]
                    irrad_cond_final_temp[current_mapped_step] = irrad_cond_original_layers[l_orig]
                    filled_steps[current_mapped_step] = true
                end
            else
                # This case should ideally not be reached if max_final_step is correctly derived from energy_step_map_for_orig_layers.
                @warn "Original layer $l_orig (Energy: $(layer_actual_energies[l_orig])) mapped to an invalid step $current_mapped_step. Max step is $max_final_step. This entry will be skipped for irrad_cond_final."
            end
        end

        # Gap-filling pass
        if max_final_step > 0
            # Ensure step 1 is filled if it's missing
            if !filled_steps[1]
                # Find the original layer that corresponds to min_ref_energy.
                idx_min_ref_energy_layer = findfirst(e -> e == min_ref_energy, layer_actual_energies)
                if idx_min_ref_energy_layer !== nothing
                    @warn "Step 1 in irrad_cond_final_temp was not filled by initial pass. Filling with AT from layer of min_ref_energy (original layer index: $idx_min_ref_energy_layer)."
                    irrad_cond_final_temp[1] = irrad_cond_original_layers[idx_min_ref_energy_layer]
                    filled_steps[1] = true
                else
                    # This case (min_ref_energy exists but its layer not found) should ideally not happen if irrad_cond_original_layers is non-empty.
                    # Or if all positive energy layers mapped to a step > 1 due to a very large min_ref_energy relative to other positive energies (unlikely with floor+1).
                    # Fallback: if irrad_cond_original_layers is not empty, use its first element.
                    if !isempty(irrad_cond_original_layers)
                        @warn "Step 1 in irrad_cond_final_temp is not filled, AND layer for min_ref_energy could not be found. Falling back to first element of irrad_cond_original_layers."
                        irrad_cond_final_temp[1] = irrad_cond_original_layers[1]
                        filled_steps[1] = true
                    else
                        @error "Step 1 in irrad_cond_final_temp is not filled, layer for min_ref_energy not found, AND irrad_cond_original_layers is empty. irrad_cond[1] may remain #undef."
                    end
                end
            end

            # Fill subsequent steps if they are missing, using the AT object from the previous filled step
            for k in 2:max_final_step
                if !filled_steps[k]
                    if filled_steps[k-1] # If previous step (k-1) is filled
                        @warn "Step $k in irrad_cond_final_temp was not filled. Copying AT object from step $(k-1)."
                        irrad_cond_final_temp[k] = irrad_cond_final_temp[k-1]
                        filled_steps[k] = true
                    else
                        # This implies step k-1 was also not filled. This could cascade if step 1 wasn't fillable.
                        @error "Step $k in irrad_cond_final_temp was not filled, and previous step $(k-1) was also not filled. Step $k may remain #undef."
                    end
                end
            end
        end

        # Warn if any steps in the new irrad_cond were not filled.
        if !all(filled_steps) && max_final_step > 0 # Only warn if there were steps expected to be filled.
            unfilled_indices = findall(.!filled_steps)
            @warn "Not all steps in the new irrad_cond_final were filled. Max step was $max_final_step. Unfilled step indices: $unfilled_indices. These steps won't have defined AT objects and could cause errors if accessed."
        end
        global irrad_cond = irrad_cond_final_temp # Replace the global irrad_cond with the new condensed version.
    end
end


function distribute_dose_domain_fast(x0::Float64, y0::Float64, radius::Float64, track::Track, irrad_cond::AT, impact_AT::DataFrame)

    x_track, y_track = track.x, track.y
    x_track = (x_track - x0) #* 1e3  # mm -> um ??
    y_track = (y_track - y0) #* 1e3  # mm -> um
    b = sqrt(x_track^2 + y_track^2)
    dose = 0.

    if b <= 0.5 * (radius + irrad_cond.Rc)
        dose = impact_AT.dose[1]
    elseif b <= irrad_cond.Rc + irrad_cond.Rp
        idx = findfirst(x -> x > b, impact_AT.impact)
        if idx == nothing || idx == 1
            error("b is out of the range of impact values")
        end

        x1, x2 = impact_AT.impact[idx-1], impact_AT.impact[idx]
        y1, y2 = impact_AT.dose[idx-1], impact_AT.dose[idx]

        dose = y1 + (y2 - y1) * (b - x1) / (x2 - x1)
    end

    return dose
end


#function to create OER correction
function calculate_OER(ion_::Ion, cell_::Cell)

    M0 = 3.4
    b = 0.41
    a = 8.27 * 10^5
    g = 3.0

    OER = (b .* (a * M0 .+ ion_.LET^g) ./ (a .+ ion_.LET^g) .+ cell_.O) ./ (b .+ cell_.O)

    return OER
end

#function to calculate the average yield of damage per unit Gy
function calculate_kappa(ion_::Ion, cell_::Cell, OER_bool::Bool)

    if ion_.ion == "12C"
        p1 = 6.8
        p2 = 0.156
        p3 = 0.9214
        p4 = 0.005245
        p5 = 1.395
    elseif ion_.ion == "4He"
        p1 = 6.8
        p2 = 0.1471
        p3 = 1.038
        p4 = 0.006239
        p5 = 1.582
    elseif ion_.ion == "3He"
        p1 = 6.8
        p2 = 0.1471
        p3 = 1.038
        p4 = 0.006239
        p5 = 1.582
    elseif ion_.ion == "1H"
        p1 = 6.8
        p2 = 0.1773
        p3 = 0.9314
        p4 = 0.
        p5 = 1.
    elseif ion_.ion == "2H"
        p1 = 6.8
        p2 = 0.1773
        p3 = 0.9314
        p4 = 0.
        p5 = 1.
    elseif ion_.ion == "16O"
        p1 = 6.8
        p2 = 0.1749
        p3 = 0.8722
        p4 = 0.004987
        p5 = 1.347
    else
        println("Unknown ion specie")
        return -1
    end

    yield = (p1 + (p2 * ion_.LET)^p3) / (1 + (p4 * ion_.LET)^p5)
    if OER_bool
        OER = calculate_OER(ion_, cell_)
    else
        OER = 1.
    end
    yield /= OER

    return yield
end

function calculate_damage(ion_::Ion, cell_::Cell, track_::Track, integral_, theta_, Gyr_, radius_)

    x_ = track_.x - cell_.x
    y_ = track_.y - cell_.y
    b = sqrt((x_)^2 + (y_)^2)

    kappa_DSB = 9 * calculate_kappa(ion_, cell_, false)
    lambda_DSB = kappa_DSB * 10^-3

    x0d = rand(Poisson(kappa_DSB * Gyr_))
    y0d = rand(Poisson(lambda_DSB * Gyr_))

    if (x0d == 0) & (y0d == 0)
        X_CD = Array{Float64}(undef, 0, Nd)
        Y_CD = Array{Float64}(undef, 0, Nd)

        return X_CD, Y_CD
    end

    if x0d > 0
        X_CD = Array{Float64}(undef, 0, Nd)
        #local X_CD = zeros(x0d, Nd);

        radius__xP = rand(Categorical((integral_ / sum(integral_))), x0d)
        radius__x = (radius_[radius__xP.+1] - radius_[radius__xP]) .* rand(Uniform(0, 1), x0d) .+ radius_[radius__xP]
        if (x_ >= 0)
            local theta__x = 3 * π / 2 .- acos.(y_ / b) .+ theta_[radius__xP] .* rand(Uniform(0, 1), x0d) .* [-1, 1][rand(Bernoulli(), x0d).+1]
        elseif (x_ < 0)
            theta__x = 3 * π / 2 .+ acos.(y_ / b) .+ theta_[radius__xP] .* rand(Uniform(0, 1), x0d) .* [-1, 1][rand(Bernoulli(), x0d).+1]
        end
        Xx = radius__x .* cos.(theta__x) .+ x_
        Xy = radius__x .* sin.(theta__x) .+ y_
        for i in 1:x0d
            X_CD = vcat(X_CD, reshape([Xx[i] + cell_.x, Xy[i] + cell_.y, cell_.z - cell_.r / 2 + cell_.r * rand(Uniform(0, 1), 1)[1]], 1, :))
        end
    else
        X_CD = Array{Float64}(undef, 0, Nd)
    end

    if y0d > 0
        Y_CD = Array{Float64}(undef, 0, Nd)

        radius__yP = rand(Categorical((integral_ / sum(integral_))), y0d)
        radius__y = (radius_[radius__yP.+1] - radius_[radius__yP]) .* rand(Uniform(0, 1), y0d) .+ radius_[radius__yP]
        if (x_ >= 0)
            theta__y = 3 * π / 2 .- acos.(y_ / b) .+ theta_[radius__yP] .* rand(Uniform(0, 1), y0d) .* [-1, 1][rand(Bernoulli(), y0d).+1]
        elseif (x_ < 0)
            theta__y = 3 * π / 2 .+ acos.(y_ / b) .+ theta_[radius__yP] .* rand(Uniform(0, 1), y0d) .* [-1, 1][rand(Bernoulli(), y0d).+1]
        end
        Yx = radius__y .* cos.(theta__y) .+ x_
        Yy = radius__y .* sin.(theta__y) .+ y_
        for i in 1:y0d
            #global Y_CD = vcat(Y_CD,reshape([Yx[i], Yy[i], cell_.r*rand(Uniform(0,1),1)[1]], 1, :));
            Y_CD = vcat(Y_CD, reshape([Yx[i] + cell_.x, Yy[i] + cell_.y, cell_.z - cell_.r / 2 + cell_.r * rand(Uniform(0, 1), 1)[1]], 1, :))
        end
    else
        Y_CD = Array{Float64}(undef, 0, Nd)
    end

    return X_CD, Y_CD
end


######################################################################
"""
Generate positions for cells in a 2D squared lattice.

X_box: Float64 - length of the box side.
R_cell: Float64 - radius of the cells.

Returns the number of cells and their positions (x, y) in a vector of tuples.
"""
function generate_cells_positions_squaredlattice(X_box::Float64, R_cell::Float64)
    # Calculate the number of cells on each side of the lattice
    N_CellsSide = convert(Int64, ceil(X_box / (2 * R_cell)))

    # Create a vector to hold the positions of the cells
    nodes_positions = Vector{Tuple{Float64,Float64}}(undef, N_CellsSide^2)

    # Loop over the lattice and calculate the positions of the cells
    @inbounds for i in 1:N_CellsSide, j in 1:N_CellsSide
        idx = (i - 1) * N_CellsSide + j
        # Calculate the position of the cell in 2D space
        nodes_positions[idx] = (R_cell + (i - 1) * 2 * R_cell, R_cell + (j - 1) * 2 * R_cell)
    end

    # Return the number of cells and the positions of the cells
    return N_CellsSide^2, nodes_positions
end




"""
Generate a triangular lattice of cells in 2D.

X_box is the length of the box side.
R_cell is the radius of the cells.

Returns the number of cells and their positions (x, y) in a vector of tuples.
"""
function generate_cells_positions_triangularlattice(X_box::Float64, R_cell::Float64)
    N_CellsSide = convert(Int64, floor(X_box / (2 * R_cell)))
    N_CellsSide2 = convert(Int64, floor((X_box) / (R_cell * sqrt(3))))

    println("Number of cells on each side of the lattice: ", N_CellsSide)
    println("Number of cells on each side of the lattice, accounting for the triangular lattice: ", N_CellsSide2)

    nodes_positions = Vector{Tuple{Float64,Float64}}()
    for i in 1:N_CellsSide
        for j in 1:N_CellsSide2
            # Alternate between two positions to create a triangular lattice
            if rem(j, 2) == 1
                push!(nodes_positions, (R_cell + (i - 1) * 2 * R_cell, R_cell + (j - 1) * R_cell * sqrt(3)))
            else
                push!(nodes_positions, ((i) * 2 * R_cell, R_cell + (j - 1) * R_cell * sqrt(3)))
            end
        end
    end
    local N = N_CellsSide * N_CellsSide2
    return N, nodes_positions
end

"""
Generate the positions of cells in a 3D triangular lattice.

X_box: Float64 - length of the box side.
R_cell: Float64 - radius of the cells.

Returns the number of cells and their positions (x, y, z) in a vector of tuples.
"""
function generate_cells_positions_triangularlattice_3D(X_box::Float64, R_cell::Float64)
    # Calculate the number of cells on each side of the lattice
    N_CellsSide = convert(Int64, floor(X_box / (2 * R_cell)))
    N_CellsSide2 = convert(Int64, floor((X_box) / (R_cell * sqrt(3 / 2))))
    # Print the number of cells on each side of the lattice
    println("Number of cells on each side of the lattice: ", N_CellsSide)
    println("Number of cells on each side of the lattice, considering the triangular lattice: ", N_CellsSide2)

    # Initialize the vector to store the positions of the cells
    nodes_positions = Vector{Tuple{Float64,Float64,Float64}}()

    # Loop to generate cell positions in the triangular lattice
    for i in 1:N_CellsSide
        for j in 1:N_CellsSide2
            # Alternate cell positions to create a triangular lattice
            if rem(j, 2) == 1
                push!(nodes_positions, (R_cell + (i - 1) * 2 * R_cell, R_cell + (j - 1) * R_cell * sqrt(3), 0.0))
            else
                push!(nodes_positions, (i * 2 * R_cell, R_cell + (j - 1) * R_cell * sqrt(3), 0.0))
            end
        end
    end

    # Calculate the total number of cells
    local N = N_CellsSide * N_CellsSide2
    return N, nodes_positions
end


# Function to create 3D cells based on input parameters

"""
Function to create 3D cells based on input parameters

Parameters:
- N: total number of cells
- nodes_positions: 3D positions of the cells
- R_cell: radius of the cell
- O2_: oxygenation level
- SP_: survival probability
- gsm2: GSM2 parameters
- cell_df: Data Frame of cell parameters
- domain: domain type (1 = spatial, 2 = domain, 3 = fast)
- tumor_radius: radius of the tumor
"""
function create_cells_3D_voxel!(N::Int64, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, R_cell::Float64, SP_::Float64, gsm2::GSM2, arrayOfCell::Array{Cell}, cell_df::DataFrame, domain::Int64, tumor_radius::Float64, full_cycle::Bool, geometry::String, type::String)
    # Preallocate array of Cells    
    # Compute neighbors for each cell
    Ncell = round(Int64, N^(1 / 3))
    neighbors = compute_neighbors_3d(Ncell, Ncell, Ncell)

    # Loop to create cells
    for i in 1:N
        x, y, z = nodes_positions[i]

        # Randomly assign cell line and cell cycle
        cell_line = rand() < 0.8 ? 1 : 0
        if full_cycle
            ru = rand() * 24
            if ru <= 1
                cell_cycle = "M"
            elseif ru <= 20
                cell_cycle = "S"
            elseif ru <= 24
                cell_cycle = "G2"
            else
                println("Error")
            end
        else
            cell_cycle = (rand() < 0.25) ? "M" : "I"
        end

        proliferation = 15

        N_sideVox = convert(Int64, floor(2 * X_box / (X_voxel)))
        x_range = y_range = z_range = AbstractRange{Float64}
        #is_inside_tumor = false
        i_voxel_x = 0
        i_voxel_y = 0
        i_voxel_z = 0
        #### Fill the array of Voxel with cells within the x,y,z range ####
        for i in 1:N_sideVox
            x_range = range(start=-X_box + (i - 1) * X_voxel, stop=-X_box + i * X_voxel)
            for j in 1:N_sideVox
                y_range = range(start=-X_box + (j - 1) * X_voxel, stop=-X_box + j * X_voxel)
                for k in 1:N_sideVox
                    z_range = range(start=-X_box + (k - 1) * X_voxel, stop=-X_box + k * X_voxel)
                    if (x in x_range) && (y in y_range) && (z in z_range)
                        #is_inside_tumor = true
                        i_voxel_x = i
                        i_voxel_y = j
                        i_voxel_z = k
                    end
                end
            end
        end

        # Check if the cell is inside the tumor
        is_inside_tumor = sqrt(x^2 + y^2 + z^2) < tumor_radius
        O2_TNT = 5

        # Extract GSM2 parameters
        a_cell, b_cell, r_cell, rd_cell, Rn_cell = gsm2.a, gsm2.b, gsm2.r, gsm2.rd, gsm2.Rn
        if geometry == "square"
            is_cell = 1
        elseif geometry == "circle"
            is_cell = is_inside_tumor ? 1 : 0
        end

        # Calculate centers based on domain type
        center_x, center_y = Vector{Float64}(), Vector{Float64}()
        if type == "domain"
            center_x, center_y = calculate_centers(x, y, gsm2.rd, gsm2.Rn)
        end

        # Create the Cell object and assign values
        cell = Cell(x, y, z, center_x, center_y, neighbors[i], Rn_cell, R_cell, a_cell, b_cell, r_cell, rd_cell,
            cell_line, proliferation, cell_cycle, Array{Float64}(undef, 0, Nd), Array{Float64}(undef, 0, Nd),
            fill(0, domain * floor(Int64, gsm2.Rn / gsm2.rd)), fill(0, domain * floor(Int64, gsm2.Rn / gsm2.rd)), O2_TNT, SP_,
            fill(0, domain), 0.0, is_cell, is_cell, is_cell, 0.0, 0.0, 0.0, 0, 0, i_voxel_x, i_voxel_y, i_voxel_z)

        # Add the created cell to the array
        arrayOfCell[i] = cell
    end

    cell_df.nei = [arrayOfCell[i].nei for i in eachindex(nodes_positions)]
    cell_df.is_cell = [arrayOfCell[i].is_cell for i in eachindex(nodes_positions)]
    cell_df.is_stem = [arrayOfCell[i].is_stem for i in eachindex(nodes_positions)]
    cell_df.proliferation = [arrayOfCell[i].proliferation for i in eachindex(nodes_positions)]
    cell_df.cell_cycle = [arrayOfCell[i].cell_cycle for i in eachindex(nodes_positions)]
    cell_df.O = [arrayOfCell[i].O for i in eachindex(nodes_positions)]
    cell_df.i_voxel_x = [arrayOfCell[i].i_voxel_x for i in eachindex(nodes_positions)]
    cell_df.i_voxel_y = [arrayOfCell[i].i_voxel_y for i in eachindex(nodes_positions)]
    cell_df.i_voxel_z = [arrayOfCell[i].i_voxel_z for i in eachindex(nodes_positions)]
end

function create_cells_3D_voxel_1layervoxel!(N::Int64, nodes_positions::Vector{Tuple{Float64,Float64,Float64}}, R_cell::Float64, SP_::Float64, gsm2::GSM2, arrayOfCell::Array{Cell}, cell_df::DataFrame, domain::Int64, tumor_radius::Float64, full_cycle::Bool, geometry::String, type::String, N_sideVox::Int64, N_CellsSide::Int64)
    # Preallocate array of Cells    
    # Compute neighbors for each cell
    #Ncell = round(Int64, N^(1/3))
    Ncell = N_CellsSide
    neighbors = compute_neighbors_3d(Ncell, Ncell, Int64(Ncell / N_sideVox))

    # Loop to create cells
    for i in 1:N
        x, y, z = nodes_positions[i]

        # Randomly assign cell line and cell cycle
        cell_line = rand() < 0.8 ? 1 : 0
        if full_cycle
            ru = rand() * 24
            if ru <= 1
                cell_cycle = "M"
            elseif ru <= 20
                cell_cycle = "S"
            elseif ru <= 24
                cell_cycle = "G2"
            else
                println("Error")
            end
        else
            cell_cycle = (rand() < 0.25) ? "M" : "I"
        end

        proliferation = 15

        N_sideVox = convert(Int64, floor(2 * X_box / (X_voxel)))
        x_range = y_range = z_range = AbstractRange{Float64}
        #is_inside_tumor = false
        i_voxel_x = 0
        i_voxel_y = 0
        i_voxel_z = 0
        #### Fill the array of Voxel with cells within the x,y,z range ####
        for i in 1:N_sideVox
            x_range = range(start=-X_box + (i - 1) * X_voxel, stop=-X_box + i * X_voxel)
            for j in 1:N_sideVox
                y_range = range(start=-X_box + (j - 1) * X_voxel, stop=-X_box + j * X_voxel)
                for k in 1:N_sideVox
                    z_range = range(start=-X_box + (k - 1) * X_voxel, stop=-X_box + k * X_voxel)
                    if (x in x_range) && (y in y_range) && (z in z_range)
                        #is_inside_tumor = true
                        i_voxel_x = i
                        i_voxel_y = j
                        i_voxel_z = k
                    end
                end
            end
        end

        # Check if the cell is inside the tumor
        is_inside_tumor = sqrt(x^2 + y^2 + z^2) < tumor_radius
        O2_TNT = 5

        # Extract GSM2 parameters
        a_cell, b_cell, r_cell, rd_cell, Rn_cell = gsm2.a, gsm2.b, gsm2.r, gsm2.rd, gsm2.Rn
        if geometry == "square"
            is_cell = 1
        elseif geometry == "circle"
            is_cell = is_inside_tumor ? 1 : 0
        end

        # Calculate centers based on domain type
        center_x, center_y = Vector{Float64}(), Vector{Float64}()
        if type == "domain"
            center_x, center_y = calculate_centers(x, y, gsm2.rd, gsm2.Rn)
        end

        # Create the Cell object and assign values
        cell = Cell(x, y, z, center_x, center_y, neighbors[i], Rn_cell, R_cell, a_cell, b_cell, r_cell, rd_cell,
            cell_line, proliferation, cell_cycle, Array{Float64}(undef, 0, Nd), Array{Float64}(undef, 0, Nd),
            fill(0, domain * floor(Int64, gsm2.Rn / gsm2.rd)), fill(0, domain * floor(Int64, gsm2.Rn / gsm2.rd)), O2_TNT, SP_,
            fill(0, domain), 0.0, is_cell, is_cell, is_cell, 0.0, 0.0, 0.0, 0, 0, i_voxel_x, i_voxel_y, i_voxel_z)

        # Add the created cell to the array
        arrayOfCell[i] = cell
    end

    cell_df.nei = [arrayOfCell[i].nei for i in eachindex(nodes_positions)]
    cell_df.is_cell = [arrayOfCell[i].is_cell for i in eachindex(nodes_positions)]
    cell_df.is_stem = [arrayOfCell[i].is_stem for i in eachindex(nodes_positions)]
    cell_df.proliferation = [arrayOfCell[i].proliferation for i in eachindex(nodes_positions)]
    cell_df.cell_cycle = [arrayOfCell[i].cell_cycle for i in eachindex(nodes_positions)]
    cell_df.O = [arrayOfCell[i].O for i in eachindex(nodes_positions)]
    cell_df.i_voxel_x = [arrayOfCell[i].i_voxel_x for i in eachindex(nodes_positions)]
    cell_df.i_voxel_y = [arrayOfCell[i].i_voxel_y for i in eachindex(nodes_positions)]
    cell_df.i_voxel_z = [arrayOfCell[i].i_voxel_z for i in eachindex(nodes_positions)]
end

function create_cells_3D_voxel_new!(N::Int64, nodes_positions::Vector{Tuple{Float64,Float64,Float64}},
    R_cell::Float64, SP_::Float64, gsm2::GSM2,
    arrayOfCell::Array{Cell}, cell_df::DataFrame,
    domain::Int64, tumor_radius::Float64, full_cycle::Bool,
    geometry::String, type::String)

    # Use actual number of nodes (safer than trusting N)
    Nnodes = length(nodes_positions)

    # Compute neighbors: infer Ncell from actual node count (integer cube root)
    # We assume positions form a cubic lattice; use round and check
    Ncell = round(Int, Nnodes^(1 / 3))
    if Ncell^3 != Nnodes
        @warn "Number of nodes $Nnodes is not a perfect cube; computed Ncell = $Ncell (Ncell^3 = $(Ncell^3)). Proceeding but check lattice shape."
    end
    neighbors = compute_neighbors_3d(Ncell, Ncell, Ncell)

    # Loop to create cells — iterate over indices derived from nodes_positions
    for idx in 1:Nnodes
        x, y, z = nodes_positions[idx]

        # Randomly assign cell line and cycle
        cell_line = rand() < 0.8 ? 1 : 0
        if full_cycle
            ru = rand() * 24
            if ru <= 1
                cell_cycle = "M"
            elseif ru <= 20
                cell_cycle = "S"
            else
                cell_cycle = "G2"
            end
        else
            cell_cycle = (rand() < 0.25) ? "M" : "I"
        end

        proliferation = 15

        # Voxelization: do NOT reuse 'idx' (use distinct names)
        N_sideVox = convert(Int64, floor(2 * X_box / X_voxel))
        i_voxel_x = 0
        i_voxel_y = 0
        i_voxel_z = 0

        # iterate with distinct loop variable names to avoid shadowing
        for ix in 1:N_sideVox
            x_range = range(start=-X_box + (ix - 1) * X_voxel, stop=-X_box + ix * X_voxel)
            for jy in 1:N_sideVox
                y_range = range(start=-X_box + (jy - 1) * X_voxel, stop=-X_box + jy * X_voxel)
                for kz in 1:N_sideVox
                    z_range = range(start=-X_box + (kz - 1) * X_voxel, stop=-X_box + kz * X_voxel)
                    if (x in x_range) && (y in y_range) && (z in z_range)
                        i_voxel_x = ix
                        i_voxel_y = jy
                        i_voxel_z = kz
                    end
                end
            end
        end

        # Tumor check
        is_inside_tumor = sqrt(x^2 + y^2 + z^2) < tumor_radius
        O2_TNT = 5

        # Extract GSM2 parameters
        a_cell, b_cell, r_cell, rd_cell, Rn_cell = gsm2.a, gsm2.b, gsm2.r, gsm2.rd, gsm2.Rn

        is_cell = geometry == "square" ? 1 : (geometry == "circle" ? (is_inside_tumor ? 1 : 0) : 0)

        # Centers (only if needed)
        center_x, center_y = Vector{Float64}(), Vector{Float64}()
        if type == "domain"
            center_x, center_y = calculate_centers(x, y, gsm2.rd, gsm2.Rn)
        end

        # Safety: ensure neighbors vector has entry for this idx
        neigh = (idx <= length(neighbors)) ? neighbors[idx] : Int[]  # fallback empty neighbors

        # Create Cell
        cell = Cell(x, y, z, center_x, center_y, neigh, Rn_cell, R_cell, a_cell, b_cell, r_cell, rd_cell,
            cell_line, proliferation, cell_cycle, Array{Float64}(undef, 0, Nd), Array{Float64}(undef, 0, Nd),
            fill(0, domain * floor(Int, gsm2.Rn / gsm2.rd)), fill(0, domain * floor(Int, gsm2.Rn / gsm2.rd)),
            O2_TNT, SP_, fill(0, domain), 0.0, is_cell, is_cell, is_cell, 0.0, 0.0, 0.0,
            0, 0, i_voxel_x, i_voxel_y, i_voxel_z)

        # Store
        arrayOfCell[idx] = cell
    end

    # Update DataFrame using the correct array variable name
    cell_df.nei = [arrayOfCell[i].nei for i in 1:Nnodes]
    cell_df.is_cell = [arrayOfCell[i].is_cell for i in 1:Nnodes]
    cell_df.is_stem = [arrayOfCell[i].is_stem for i in 1:Nnodes]
    cell_df.proliferation = [arrayOfCell[i].proliferation for i in 1:Nnodes]
    cell_df.cell_cycle = [arrayOfCell[i].cell_cycle for i in 1:Nnodes]
    cell_df.O = [arrayOfCell[i].O for i in 1:Nnodes]
    cell_df.i_voxel_x = [arrayOfCell[i].i_voxel_x for i in 1:Nnodes]
    cell_df.i_voxel_y = [arrayOfCell[i].i_voxel_y for i in 1:Nnodes]
    cell_df.i_voxel_z = [arrayOfCell[i].i_voxel_z for i in 1:Nnodes]

    return nothing
end


function set_oxygenation_old(arrayOfCell::Array{Cell}, cell_df::DataFrame)
    max_distance = maximum(sqrt.(cell_df.x[cell_df.is_cell.==1] .^ 2 + cell_df.y[cell_df.is_cell.==1] .^ 2 + cell_df.z[cell_df.is_cell.==1] .^ 2))
    max_distance_box = maximum(sqrt.(cell_df.x .^ 2 + cell_df.y .^ 2 + cell_df.z .^ 2))

    for i in cell_df.index[cell_df.is_cell.==1]
        arrayOfCell[i].O = rescale(max_distance - sqrt(cell_df.x[i]^2 + cell_df.y[i]^2 + cell_df.z[i]^2), max_distance_box)
    end

    cell_df.O = [arrayOfCell[i].O for i in eachindex(nodes_positions)]
end

function rescale(x::Float64, max::Float64)
    scaled = 8. + (0.5 - 8.) / max * x
    return scaled
end


"""
Function to set oxygenation levels for active cells based on their distance from the origin.
Oxygenation is scaled linearly between min_O2 (for closest cells) and max_O2 (for furthest cells).

Parameters:
    - `cell_df`: DataFrame containing the information of all cells. Must have columns :x, :y, :z, :is_cell, and :O (which will be updated).
    - `min_O2`: The minimum oxygenation level (assigned to active cells closest to the origin).
    - `max_O2`: The maximum oxygenation level (assigned to active cells furthest from the origin).
"""
function set_oxygenation(cell_df_::DataFrame, min_O2::Float64, max_O2::Float64, grad::String)
    # Filter for active cells
    active_cell_indices = cell_df_.index[cell_df_.is_cell.==1]

    if isempty(active_cell_indices)
        @warn "No active cells found (is_cell == 1). Cannot set oxygenation based on distance."
        return # Exit if no active cells
    end

    # Ensure .O column exists. It should be initialized by create_cells_3D!
    if !hasproperty(cell_df_, :O)
        @warn "cell_df_.O column does not exist. Creating it and initializing with 0.0."
        cell_df_[!, :O] = zeros(Float64, nrow(cell_df_))
    elseif !(eltype(cell_df_.O) <: AbstractFloat)
        @warn "cell_df_.O column is not of AbstractFloat type. Results may be unpredictable."
    end

    if grad == "false"
        for i in active_cell_indices # i is the actual DataFrame row index
            cell_df_.O[i] = max_O2
        end
    elseif grad == "true"

        # Calculate distances from origin for active cells
        distances = [sqrt(cell_df_.x[i]^2 + cell_df_.y[i]^2 + cell_df_.z[i]^2) for i in active_cell_indices]

        # Find min and max distances among active cells
        d_min_active = minimum(distances)
        d_max_active = maximum(distances)

        # Iterate over the indices of active cells
        for i in active_cell_indices # i is the actual DataFrame row index
            d = sqrt(cell_df_.x[i]^2 + cell_df_.y[i]^2 + cell_df_.z[i]^2)

            local O_val # Declare local variable for calculated oxygenation

            if d_min_active == d_max_active
                # All active cells are at the same distance.
                # They are simultaneously at the "minimum observed distance" and "maximum observed distance".
                # Assign min_O2 as they are at the minimum distance. If min_O2 and max_O2 are different, this is one interpretation.
                # Alternatively, could assign (min_O2 + max_O2) / 2 or handle as an error/warning if min_O2 != max_O2.
                # For "minimum distance to the minimum", this implies min_O2.
                O_val = min_O
            else
                # Linear scaling: closest (d_min_active) gets min_O2, furthest (d_max_active) gets max_O2
                # The scaling maps distance d in [d_min_active, d_max_active] to a value in [0, 1]
                scaled_dist_0_to_1 = (d - d_min_active) / (d_max_active - d_min_active)
                # Then map [0, 1] to [min_O2, max_O2]
                O_val = min_O2 + scaled_dist_0_to_1 * (max_O2 - min_O2)
            end

            # Update the oxygenation directly in the DataFrame for the active cell
            cell_df_.O[i] = O_val
        end

    end

end

"""
Function to compute if a cell can divide or not

This function goes through all cells in the simulation and checks if a cell can
divide or not. A cell can divide if it is a stem cell, if it is in the M phase
of the cell cycle, and if it has at least one empty neighbor.

Parameters:
    - `arrayOfCell`: Array of Cell objects containing the information of all cells
"""
function compute_possible_division!(arrayOfCell::Array{Cell}, cell_df::DataFrame)
    N = size(arrayOfCell)[1]

    # Loop through all cells
    for i in 1:N
        cell = arrayOfCell[i]

        # Initialize variables
        cell.can_divide = 0
        cell.number_nei = 0

        # Check if the cell has any empty neighbors
        for j in cell.nei
            cell.number_nei += 1 - arrayOfCell[j].is_cell
        end

        # Check if the cell can divide
        if (cell.is_cell == 1) & (cell.cell_cycle == "M")
            if cell.number_nei > 0
                cell.can_divide = 1
            end
        end

        # Update the cell
        arrayOfCell[i] = cell
    end

    cell_df.can_divide = [arrayOfCell[i].can_divide for i in eachindex(nodes_positions)]
    cell_df.number_nei = [arrayOfCell[i].number_nei for i in eachindex(nodes_positions)]
end

function compute_possible_division_new!(arrayOfCell::Array{Cell}, cell_df::DataFrame)
    N = size(arrayOfCell)[1]

    # Loop through all cells
    for i in 1:N
        cell = arrayOfCell[i]

        # Initialize variables
        cell.can_divide = 0
        cell.number_nei = 0

        # Check if the cell has any empty neighbors
        for j in cell.nei
            cell.number_nei += 1 - arrayOfCell[j].is_cell
        end

        # Check if the cell can divide
        if (cell.is_cell == 1) & (cell.cell_cycle == "M")
            if cell.number_nei > 0
                cell.can_divide = 1
            end
        end

        # Update the cell
        arrayOfCell[i] = cell
    end

    cell_df.can_divide = [arrayOfCell[i].can_divide for i in 1:Int64(length(arrayOfCell) / N_sideVox)]
    cell_df.number_nei = [arrayOfCell[i].number_nei for i in 1:Int64(length(arrayOfCell) / N_sideVox)]
end



"""
    compute_possible_division_df!(cell_df::DataFrame)

Updates `can_divide` and `number_nei` in `cell_df` without using `arrayOfCell`.
"""
function compute_possible_division_df!(cell_df::DataFrame)
    N = nrow(cell_df)

    can_divide_vec = zeros(Int64, N)
    number_nei_vec = zeros(Int64, N)

    # Extract columns to avoid repeated DataFrame indexing
    nei_col = cell_df.nei
    is_cell_col = cell_df.is_cell
    cell_cycle_col = cell_df.cell_cycle

    Threads.@threads for i in 1:N
        # Calculate number of empty neighbors
        # nei_col[i] is a vector of neighbor indices
        current_nei_indices = nei_col[i]

        # Count empty neighbors (where is_cell == 0)
        # Note: is_cell is 1 if occupied, 0 if empty. 
        # So we sum (1 - is_cell[j]) for j in neighbors

        num_empty = 0
        for j in current_nei_indices
            num_empty += (1 - is_cell_col[j])
        end
        number_nei_vec[i] = num_empty

        # Check if can divide
        if (is_cell_col[i] == 1) && (cell_cycle_col[i] == "M")
            if num_empty > 0
                can_divide_vec[i] = 1
            end
        end
    end

    cell_df.can_divide = can_divide_vec
    cell_df.number_nei = number_nei_vec
end



"""
    create_domain_dataframes(cell_df::DataFrame, rel_center_x::Vector{Float64}, rel_center_y::Vector{Float64})

Creates DataFrames for domain centers and initial AT dataframe.
"""
function create_domain_dataframes(cell_df::DataFrame, rel_center_x::Vector{Float64}, rel_center_y::Vector{Float64})
    println("... Creating domain dataframes ...")
    num_cols = length(rel_center_x)

    # Using Matrix comprehension or broadcasting for clarity and speed
    mat_center_x = cell_df.x .+ transpose(rel_center_x)
    mat_center_y = cell_df.y .+ transpose(rel_center_y)

    df_center_x = DataFrame(mat_center_x, Symbol.("center_$i" for i in 1:num_cols))
    df_center_x.index = cell_df.index

    df_center_y = DataFrame(mat_center_y, Symbol.("center_$i" for i in 1:num_cols))
    df_center_y.index = cell_df.index

    at = DataFrame(zeros(size(df_center_y, 1), (size(df_center_y, 2) - 1)), :auto)
    rename!(at, Symbol.("center_$i" for i in 1:(size(df_center_y, 2)-1)))
    at.index = df_center_y.index

    return df_center_x, df_center_y, at
end

"""
    configure_dose_fractions(target_geom::String)

Configures dose arrays, fractions, and oxygenation levels based on target geometry.
"""
function configure_dose_fractions(target_geom::String)
    println("... Configuring dose and fractions ...")
    if target_geom == "square"
        max_total_dose_array = 90.0
        dose_step = 150
        #dose_array = collect(range(start=0.02,stop=3.0,step=0.02));
        dose_array = collect(range(start=max_total_dose_array / dose_step, stop=max_total_dose_array, length=dose_step))
        #NFraction = vcat(1, round.(Int64, collect(range(start=10,stop=30,step=10))));
        NFraction = [5, 15, 30]

        #Oxygenation array
        O2_array = vcat(0.5, collect(range(start=1.0, stop=7.0, step=2.0)))
    else
        max_total_dose_array = 90.0
        dose_step = 60
        #dose_array = collect(range(start=0.02,stop=3.0,step=0.02));
        dose_array = collect(range(start=max_total_dose_array / dose_step, stop=max_total_dose_array, length=dose_step))
        #NFraction = vcat(1, round.(Int64, collect(range(start=10,stop=30,step=10))));
        NFraction = [15, 30]

        O2_array = [0.0, 0.0]
    end

    return dose_array, NFraction, O2_array, max_total_dose_array, dose_step
end


"""
    configure_volume_effect(ParIrr::String, N_sideVox::Int64, target_geom::String)

Configures volume effect parameters.
"""
function configure_volume_effect(ParIrr::String, N_sideVox::Int64, target_geom::String)
    println("... Configuring volume effect ...")
    if ParIrr == "true"
        m_FSU_array = positive_integer_divisors(N_sideVox * N_sideVox)
        m_FSU_array = filter(x -> x % 3 == 0, m_FSU_array)
        n_FSU_array = floor.(Int, (N_sideVox * N_sideVox) ./ m_FSU_array)
        s = 1.0 ./ n_FSU_array
        ParIrr_array = [1, 2, 3] / 3 #collect(range(start = 1, stop = (N_sideVox*N_sideVox), step = 1))
    else
        #s = collect(range(start = 0.01, stop = 1.0, step = 0.01))
        if target_geom == "square"
            s = [0.08333333333333333, 0.16666666666666666, 0.25, 0.3333333333333333, 0.5, 1.0]
        else
            # target_geom == "circle" or fallback
            s = [0.0]
        end
        ParIrr_array = Float64[] # Empty array when not ParIrr
    end

    return s, ParIrr_array
end

"""
    initialize_tcp_ntcp_arrays(ParIrr::String, dose_array::Vector{Float64}, NFraction::Vector{Int64}, s::Vector{Float64}, ParIrr_array::Vector{Float64}, O2_array::Vector{Float64}, target_geom::String)

Initializes TCP, NTCP, and Dose Voxel arrays.
"""
function initialize_tcp_ntcp_arrays(ParIrr::String, dose_array::Vector{Float64}, NFraction::Vector{Int64}, s::Vector{Float64}, ParIrr_array::Vector{Float64}, O2_array::Vector{Float64}, target_geom::String)
    println("... Initializing TCP/NTCP arrays ...")
    if ParIrr == "true"
        #TCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(ParIrr_array))
        # Note: TCP_all was commented out in original code for ParIrr=true. Returning empty/undef.
        TCP_all = Array{Float64}(undef, 0, 0, 0)

        D_voxel_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(s), length(ParIrr_array))
        NTCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(s), length(ParIrr_array), length(O2_array))
    else
        if target_geom == "square"
            TCP_all = Array{Float64}(undef, length(dose_array), length(NFraction))
            D_voxel_all = Array{Float64}(undef, length(dose_array), length(NFraction))
            NTCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(s))
        else
            # target_geom == "circle"
            TCP_all = Array{Float64}(undef, length(dose_array), length(NFraction), length(O2_array))
            D_voxel_all = Array{Float64}(undef, length(dose_array), length(NFraction))

            # NTCP_all not defined in original circle branch? 
            # Original code:
            # target_geom == "circle"
            # TCP_all = ...
            # D_voxel_all = ...
            # No NTCP_all assignment seen in the snippet provided by user for circle case!
            # I should return something for NTCP_all to maintain Type stability/API.
            NTCP_all = Array{Float64}(undef, 0, 0, 0)
        end
    end

    return TCP_all, D_voxel_all, NTCP_all
end







"""
Function to compute the cycle time for each cell in the simulation

This function is used to compute the cycle time for each cell in the simulation by
drawing a random number from a gamma distribution. The rate of the gamma distribution
is 1 for cells in mitosis and 23 for cells in interphase.

Parameters:
    - `arrayOfCell`: Array of Cell objects containing the information of all cells
"""
function compute_cycle_AB!(arrayOfCell::Array{Cell})
    N = size(arrayOfCell)[1]
    for i in 1:N
        cell = arrayOfCell[i]

        # Set the cycle time to a random value from a gamma distribution
        if cell.cell_cycle == "M"
            # For cells in mitosis, the rate of the gamma distribution is 1
            cell.cycle_time = rand(Gamma(1))
        elseif cell.cell_cycle == "I"
            # For cells in interphase, the rate of the gamma distribution is 23
            cell.cycle_time = rand(Gamma(30))
        end
    end
end

"""
Function to update times of all cells in the simulation

This function is used to update the times of all cells in the simulation by
subtracting the time to the next event from each time.

Parameters:
    - `cell_df`: DataFrame containing the information of all cells
    - `next_time`: time to the next event
"""
function update_time!(cell_df::DataFrame, next_time::Float64)
    cell_df.death_time[cell_df.is_cell.==1] .-= next_time
    cell_df.recover_time[cell_df.is_cell.==1] .-= next_time
    cell_df.cycle_time[cell_df.is_cell.==1] .-= next_time
end

"""
Function to check if any cell has a time of 0 and update its time

This function is used to check if any cell has a time of 0 and update its time
accordingly. This is necessary because the time to the next event is computed
using the minimum of all times, which could be 0. In this case, the time to the
next event is set to the natural apoptosis rate.

Parameters:
    - `cell_df`: DataFrame containing the information of all cells
    - `nat_apo`: natural apoptosis rate
"""
function check_time!(cell_df::DataFrame, nat_apo::Float64)
    for i in cell_df.index[cell_df.is_cell.==1]
        # check if the cell has a time of death of 0
        if cell_df.death_time[i] == 0
            println("Enter check time death $i")
            # if yes, set the time to the natural apoptosis rate
            cell_df.is_cell[i] = 0
            cell_df.apo_time[i] = Inf
            cell_df.death_time[i] = Inf
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i] = Inf

            # increment the number of neighbors for all neighbors of the dead cell
            cell_df.number_nei[cell_df.nei[i]] .+= 1

            # for all neighbors of the dead cell
            for j in cell_df.nei[i]

                # if the cell is a stem cell and has 1 free neighbor (it means it couldn't divide earlier)
                if cell_df.number_nei[j] == 1
                    cell_df.cell_cycle[j] = (rand() < 1 / 24) ? "M" : "I"

                    if cell_df.cell_cycle[j] == "M"
                        cycle_time = rand(Gamma(1))
                        cell_df.cycle_time[j] = cycle_time
                    else
                        cycle_time = rand(Gamma(30))
                        cell_df.cycle_time[j] = cycle_time
                    end
                end
            end
        end

        # check if the cell has a time of recovery of 0
        if cell_df.recover_time[i] == 0
            println("Enter check time recover $i")

            # if yes, set the time to infinity
            cell_df.recover_time[i] = Inf
        end

        # check if the cell has a time of cycle of 0
        if cell_df.cycle_time[i] == 0
            println("Enter check time cycle $i")

            # if yes, set the time to infinity
            cell_df.cycle_time[i] = Inf
        end
    end
end

"""
Helper function to handle cell removal (death/apoptosis).
Resets cell state, updates neighbor counts, and potentially triggers neighbor cycle changes.
"""
function _handle_cell_removal!(cell_df::DataFrame, removed_idx::Int64, is_natural_apoptosis_event::Bool)
    if cell_df.is_cell[removed_idx] == 0
        # Cell is already marked as removed, nothing to do.
        return
    end

    cell_df.is_cell[removed_idx] = 0
    cell_df.apo_time[removed_idx] = Inf
    cell_df.death_time[removed_idx] = Inf
    cell_df.recover_time[removed_idx] = Inf
    cell_df.cycle_time[removed_idx] = Inf
    cell_df.can_divide[removed_idx] = 0 # Cannot divide if not a cell

    if is_natural_apoptosis_event # This flag was from the original apo_time event
        cell_df.is_death_rad[removed_idx] = 0
    end

    # Update neighbors of the removed cell
    neighbor_indices = cell_df.nei[removed_idx]
    if isempty(neighbor_indices)
        return
    end

    # Filter for valid neighbor indices that are within DataFrame bounds
    valid_neighbor_indices = filter(n_idx -> (1 <= n_idx <= nrow(cell_df)), neighbor_indices)

    if !isempty(valid_neighbor_indices)
        cell_df.number_nei[valid_neighbor_indices] .+= 1 # They gain an empty neighbor spot

        for neighbor_j_idx in valid_neighbor_indices
            if cell_df.is_cell[neighbor_j_idx] == 1 && cell_df.number_nei[neighbor_j_idx] == 1
                # This active neighbor now has exactly one free spot (it was previously blocked).
                # Re-evaluate its cycle state and time.
                if rand() < (1 / 24) # Chance to enter M phase
                    cell_df.cell_cycle[neighbor_j_idx] = "M"
                    cell_df.cycle_time[neighbor_j_idx] = rand(Gamma(1)) # M phase duration
                    cell_df.can_divide[neighbor_j_idx] = 1 # Can divide as it's in M and has space
                else # Stays/enters I phase
                    cell_df.cell_cycle[neighbor_j_idx] = "I"
                    cell_df.cycle_time[neighbor_j_idx] = rand(Gamma(30)) # I phase duration
                    cell_df.can_divide[neighbor_j_idx] = 0
                end
            elseif cell_df.is_cell[neighbor_j_idx] == 1 && cell_df.cell_cycle[neighbor_j_idx] == "M" && cell_df.number_nei[neighbor_j_idx] > 0
                # If it was already in M and blocked, and now has space, ensure can_divide is set
                cell_df.can_divide[neighbor_j_idx] = 1
                if isinf(cell_df.cycle_time[neighbor_j_idx]) # If it was stalled
                    cell_df.cycle_time[neighbor_j_idx] = rand(Gamma(1)) # Give it a new M-phase time
                end
            end
        end
    end
end

"""
Function to update the cell dataframe after an event

Parameters:
    - `cell_df`: DataFrame containing the information of all cells
    - `next_time`: time until the next event
    - `event`: type of event (either "death_time", "apo_time", "cycle_time" or "recover_time")
    - `idx`: index of the cell that experienced the event
    - `nat_apo`: natural apoptosis rate
"""
function update_AB_fast!(cell_df::DataFrame, next_time::Float64, event::String, idx::Int64, nat_apo::Float64)

    # update time for all cells
    update_time!(cell_df, next_time) # Subtract next_time from all relevant time columns

    if event == "death_time" || event == "apo_time"
        _handle_cell_removal!(cell_df, idx, event == "apo_time")
    elseif event == "cycle_time"
        phase_that_just_ended = cell_df.cell_cycle[idx]
        num_empty_neighbors_of_idx = cell_df.number_nei[idx]

        if phase_that_just_ended == "M"
            if num_empty_neighbors_of_idx > 0
                _perform_division!(cell_df, idx, nat_apo)
            else # Mitosis completed (or M-time expired), but no space to divide
                cell_df.cell_cycle[idx] = "I" # Original forced to I, could be "M_blocked"
                cell_df.cycle_time[idx] = Inf
                cell_df.can_divide[idx] = 0
            end
        elseif phase_that_just_ended == "I"
            # Interphase completed, transition to M
            cell_df.cell_cycle[idx] = "M"
            if num_empty_neighbors_of_idx > 0
                cell_df.cycle_time[idx] = rand(Gamma(1)) # M phase duration
                cell_df.can_divide[idx] = 1
            else
                cell_df.cycle_time[idx] = Inf # M phase, but blocked by no space
                cell_df.can_divide[idx] = 0
            end
        else # Should not happen if cell_cycle is only "M" or "I"
            @warn "Unexpected cell cycle state '$(phase_that_just_ended)' for cell $idx during cycle_time event."
            cell_df.cycle_time[idx] = Inf # Default to stalled
            cell_df.can_divide[idx] = 0
        end
    elseif event == "recover_time"
        cell_df.recover_time[idx] = Inf
    end

    # check_time! handles cases where times became exactly 0 after update_time!
    # and might further modify states (e.g., a cell dying in check_time! if its death_time became 0).
    check_time!(cell_df, nat_apo)
end

"""
Helper function to manage cell division.
"""
function _perform_division!(cell_df::DataFrame, parent_idx::Int64, nat_apo::Float64)
    # 1. Find an empty neighbor spot for the new daughter cell
    # Assumes cell_df.nei[parent_idx] contains valid indices for cell_df
    empty_neighbor_indices = filter(n_idx -> (1 <= n_idx <= nrow(cell_df) && cell_df.is_cell[n_idx] == 0), cell_df.nei[parent_idx])

    if isempty(empty_neighbor_indices)
        @warn "Division called for cell $parent_idx (M-phase, num_nei > 0 initially), but no empty neighbors found now. Stalling parent."
        cell_df.cell_cycle[parent_idx] = "I" # Or "M_blocked"
        cell_df.cycle_time[parent_idx] = Inf
        cell_df.can_divide[parent_idx] = 0
        return
    end
    new_cell_idx = sample(empty_neighbor_indices)

    # 2. Determine daughter cell's stemness
    parent_is_stem = cell_df.is_stem[parent_idx]
    parent_original_proliferation = cell_df.proliferation[parent_idx]

    daughter_is_stem = 0
    if parent_is_stem == 1
        daughter_is_stem = rand() < 0.8 ? 1 : 0
    end

    # 3. Parent cell transitions to Interphase (if not removed)
    parent_removed_due_to_proliferation_limit = false
    if (parent_is_stem == 0) && (parent_original_proliferation == 1)
        # Non-stem parent reaches proliferation limit and is removed
        _handle_cell_removal!(cell_df, parent_idx, false) # false: not natural apoptosis
        parent_removed_due_to_proliferation_limit = true
    else
        cell_df.cell_cycle[parent_idx] = "I"
        if parent_is_stem == 0
            cell_df.proliferation[parent_idx] -= 1
        end
        # Parent's cycle time will be set after daughter cell placement affects its number_nei
    end

    # 4. Setup the new daughter cell
    cell_df.is_cell[new_cell_idx] = 1
    cell_df.is_stem[new_cell_idx] = daughter_is_stem
    cell_df.cell_cycle[new_cell_idx] = "I" # Daughter starts in Interphase

    # Proliferation count for daughter
    if daughter_is_stem == 1
        cell_df.proliferation[new_cell_idx] = 15 # Stem cells have high proliferation potential
    else # Daughter is non-stem
        if parent_is_stem == 1
            cell_df.proliferation[new_cell_idx] = 15 # First gen non-stem from stem parent
        else
            # Non-stem from non-stem parent. Inherits remaining proliferation.
            # If parent was removed, it had 1. Daughter gets 1 (terminal).
            # Otherwise, parent_original_proliferation was > 1. Daughter gets that.
            cell_df.proliferation[new_cell_idx] = parent_original_proliferation
        end
    end

    # Initialize times for the new daughter cell
    cell_df.death_time[new_cell_idx] = Inf
    cell_df.recover_time[new_cell_idx] = Inf
    l_apo_rate = -log(nat_apo) / 24.0 # Natural apoptosis rate per hour
    cell_df.apo_time[new_cell_idx] = log(1.0 - rand()) / (-l_apo_rate) # Time to natural apoptosis

    # Update number_nei for the new daughter cell and its cycle time
    new_cell_total_neighbors = length(cell_df.nei[new_cell_idx])
    new_cell_occupied_neighbors = new_cell_total_neighbors > 0 ? sum(cell_df.is_cell[filter(n_idx -> (1 <= n_idx <= nrow(cell_df)), cell_df.nei[new_cell_idx])]) : 0
    cell_df.number_nei[new_cell_idx] = new_cell_total_neighbors - new_cell_occupied_neighbors

    if cell_df.number_nei[new_cell_idx] > 0
        cell_df.cycle_time[new_cell_idx] = rand(Gamma(30)) # I-phase duration
    else
        cell_df.cycle_time[new_cell_idx] = Inf # No space for daughter to cycle
    end
    cell_df.can_divide[new_cell_idx] = 0 # Starts in I

    # Update neighbor counts for neighbors of the new_cell_idx (they lose an empty spot)
    new_daughter_neighbors = filter(n_idx -> (1 <= n_idx <= nrow(cell_df)), cell_df.nei[new_cell_idx])
    if !isempty(new_daughter_neighbors)
        cell_df.number_nei[new_daughter_neighbors] .-= 1
        for neighbor_of_daughter_idx in new_daughter_neighbors
            if cell_df.is_cell[neighbor_of_daughter_idx] == 1 && cell_df.number_nei[neighbor_of_daughter_idx] == 0
                # This neighbor of the daughter cell now has no empty spots
                cell_df.cycle_time[neighbor_of_daughter_idx] = Inf # Stall its cycle
                cell_df.can_divide[neighbor_of_daughter_idx] = 0
            end
        end
    end

    # 5. Set parent cell's new cycle time (if it wasn't removed)
    if !parent_removed_due_to_proliferation_limit
        parent_total_neighbors = length(cell_df.nei[parent_idx])
        parent_occupied_neighbors = parent_total_neighbors > 0 ? sum(cell_df.is_cell[filter(n_idx -> (1 <= n_idx <= nrow(cell_df)), cell_df.nei[parent_idx])]) : 0
        cell_df.number_nei[parent_idx] = parent_total_neighbors - parent_occupied_neighbors

        if cell_df.number_nei[parent_idx] > 0
            cell_df.cycle_time[parent_idx] = rand(Gamma(30)) # New I-phase duration
        else
            cell_df.cycle_time[parent_idx] = Inf # No space for parent to continue cycle
        end
        cell_df.can_divide[parent_idx] = 0 # Parent is now in I phase
    end
end


"""
Function to update the cell dataframe after an event

Parameters:
    - `cell_df`: DataFrame containing the information of all cells
    - `next_time`: time until the next event
    - `event`: type of event (either "death_time", "apo_time", "cycle_time" or "recover_time")
    - `idx`: index of the cell that experienced the event
    - `nat_apo`: natural apoptosis rate
"""
function update_AB!(cell_df::DataFrame, next_time::Float64, event::String, idx::Int64, nat_apo::Float64)

    # update time for all cells
    update_time!(cell_df, next_time)
    #check_time!(cell_df, nat_apo)

    # variables to store the index of the new cell and its cell line
    new_cell = 0
    cell_line = 1

    # variables to store the death time and cycle time of the new cell
    death_time = Inf
    cycle_time = Inf

    # if the event is a death
    if event == "death_time"

        # remove the cell from the dataframe
        cell_df.is_cell[idx] = 0
        cell_df.apo_time[idx] = Inf
        cell_df.death_time[idx] = Inf
        cell_df.recover_time[idx] = Inf
        cell_df.cycle_time[idx] = Inf

        # increment the number of neighbors for all neighbors of the dead cell
        cell_df.number_nei[cell_df.nei[idx]] .+= 1

        # for all neighbors of the dead cell
        for j in cell_df.nei[idx]

            # if the cell is a stem cell and has 1 free neighbor (it means it couldn't divide earlier)
            if cell_df.number_nei[j] == 1
                cell_df.cell_cycle[j] = (rand() < 1 / 24) ? "M" : "I"

                if cell_df.cell_cycle[j] == "M"
                    cycle_time = rand(Gamma(1))
                    cell_df.cycle_time[j] = cycle_time
                else
                    cycle_time = rand(Gamma(30))
                    cell_df.cycle_time[j] = cycle_time
                end
            end
        end
        # if the event is an apoptosis
    elseif event == "apo_time"

        # remove the cell from the dataframe
        cell_df.is_cell[idx] = 0
        cell_df.apo_time[idx] = Inf
        cell_df.death_time[idx] = Inf
        cell_df.recover_time[idx] = Inf
        cell_df.cycle_time[idx] = Inf
        cell_df.is_death_rad[idx] = 0

        # increment the number of neighbors for all neighbors of the dead cell
        cell_df.number_nei[cell_df.nei[idx]] .+= 1

        # for all neighbors of the dead cell
        for j in cell_df.nei[idx]

            # if the cell is a stem cell and has 1 free neighbor (it means it couldn't divide earlier)
            if cell_df.number_nei[j] == 1
                cell_df.cell_cycle[j] = (rand() < 1 / 24) ? "M" : "I"

                if cell_df.cell_cycle[j] == "M"
                    cycle_time = rand(Gamma(1))
                    cell_df.cycle_time[j] = cycle_time
                else
                    cycle_time = rand(Gamma(30))
                    cell_df.cycle_time[j] = cycle_time
                end
            end
        end

        # if the event is a cell cycle
    elseif event == "cycle_time"

        # if the cell is a stem cell and has more than one neighbor
        if (cell_df.cell_cycle[idx] == "M") & (cell_df.number_nei[idx] > 0)

            # sample a new cell from the neighbors of the cell
            new_cell = sample(cell_df.nei[idx][cell_df.is_cell[cell_df.nei[idx]].==0])

            # if the new cell is a stem cell, set its cell line to 1
            if cell_df.is_stem[idx] == 1
                if rand() < 0.8
                    cell_line = 1
                else
                    cell_line = 0
                end
            else
                cell_line = 0
            end

            if cell_df.number_nei[idx] > 0
                # set the cycle time to a random value
                cycle_time = rand(Gamma(30))
                cell_df.cycle_time[idx] = cycle_time
            else
                # set the cycle time to infinity
                cell_df.cycle_time[new_cell] = Inf
            end

            # set the cell cycle of the cell to "I"
            cell_df.cell_cycle[idx] = "I"
            if cell_df.is_stem[idx] == 0

                if cell_df.proliferation[idx] == 1

                    # remove the cell from the dataframe
                    cell_df.is_cell[idx] = 0
                    cell_df.apo_time[idx] = Inf
                    cell_df.death_time[idx] = Inf
                    cell_df.recover_time[idx] = Inf
                    cell_df.cycle_time[idx] = Inf
                    cell_df.is_death_rad[idx] = 0

                    # increment the number of neighbors for all neighbors of the dead cell
                    cell_df.number_nei[cell_df.nei[idx]] .+= 1

                    # for all neighbors of the dead cell
                    for j in cell_df.nei[idx]

                        # if the cell is a stem cell and has 1 free neighbor (it means it couldn't divide earlier)
                        if (cell_df.is_stem[j] == 1) & (cell_df.number_nei[j] == 1)
                            cell_df.cell_cycle[j] = (rand() < 1 / 24) ? "M" : "I"

                            if cell_df.cell_cycle[j] == "M"
                                cycle_time = rand(Gamma(1))
                                cell_df.cycle_time[j] = cycle_time
                            else
                                cycle_time = rand(Gamma(30))
                                cell_df.cycle_time[j] = cycle_time
                            end
                        end
                    end
                else
                    cell_df.proliferation[idx] -= 1
                end
            end

            # set the cell cycle of the new cell to "I"
            cell_df.cell_cycle[new_cell] = "I"
            cell_df.recover_time[new_cell] = Inf
            cell_df.is_cell[new_cell] = 1
            cell_df.is_stem[new_cell] = cell_line

            if (cell_line == 1) | (cell_df.is_stem[idx] == 1)
                cell_df.proliferation[new_cell] = 15
            else
                cell_df.proliferation[new_cell] = cell_df.proliferation[idx] + 1
            end

            l = -log(nat_apo) / 24
            apo_time = log(1 - rand()) / (-l)
            cell_df.apo_time[new_cell] = apo_time
            cell_df.number_nei[new_cell] = size(cell_df.nei[new_cell], 1) - sum(cell_df.is_cell[cell_df.nei[new_cell]])

            # if the new cell is a stem cell and has more than one neighbor
            if cell_df.number_nei[new_cell] > 0
                # set the cycle time to a random value
                cycle_time = rand(Gamma(30))
                cell_df.cycle_time[new_cell] = cycle_time
            else
                # set the cycle time to infinity
                cell_df.cycle_time[new_cell] = Inf
            end

            # decrement the number of neighbors for all neighbors of the new cell
            cell_df.number_nei[cell_df.nei[new_cell]] .-= 1
            for cc in cell_df.nei[new_cell]
                if cell_df.number_nei[cc] == 0
                    cell_df.cycle_time[cc] = Inf
                end
            end
        elseif (cell_df.cell_cycle[idx] == "I") & (cell_df.number_nei[idx] > 0)
            cycle_time = rand(Gamma(30))
            cell_df.cycle_time[idx] = cycle_time
            cell_df.cell_cycle[idx] = "M"
            cell_df.can_divide[idx] = 0
        else
            cell_df.cycle_time[idx] = Inf
            cell_df.cell_cycle[idx] = "I"
            cell_df.can_divide[idx] = 0
        end

        # if the event is a recovery
    elseif event == "recover_time"

        # set the recovery time to infinity
        cell_df.recover_time[idx] = Inf
    end

    # check if the cells are still alive
    check_time!(cell_df, nat_apo)
end



function sigmoid(x, C, K)
    return C .+ (K .- C) / (1 + exp(-0.05 * (x - 250)))
end

"""
    compute_repair_domain_p53_optimized(X_input::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2; return_history::Bool = false, j_nc_param::Float64 = 3.0, time_factor::Float64 = 720.0)

Optimized function to compute cell fate (survival/death time and type) based on initial DNA damage (X, Y),
GSM2 parameters, and p53 network dynamics. Uses Gillespie simulation coupled with ODE solving.

Arguments:
- `X_input::Vector{Int64}`: Vector representing initial counts of reparable damage lesions in different domains.
- `Y::Vector{Int64}`: Vector representing initial counts of irreparable damage lesions.
- `gsm2::GSM2`: Struct containing GSM2 model parameters (a, b, r, rd, Rn).
- `return_history::Bool`: If true, returns a DataFrame containing the time evolution of the p53 network state. Defaults to false.
- `j_nc_param::Float64`: Parameter 'j_nc' for the p53_network ODE. Defaults to 3.0.
- `time_factor::Float64`: Factor to convert Gillespie time step (tau) to ODE time units (e.g., 720 for days to minutes). Defaults to 720.0.

Returns:
- Tuple `(death_time, recover_time, death_type)` if `return_history` is false.
- Tuple `(death_time, recover_time, death_type, history_df)` if `return_history` is true.
- `death_time`: Time of cell death (Inf if survives).
- `recover_time`: Time when all damage is repaired (Inf if dies).
- `death_type`: Integer code for fate:
    - 0: Survived (damage repaired)
    - 1: Death (mitotic catastrophe / irreparable / p21-low)
    - 2: Senescence (p21-high)
    - 3: Apoptosis (Caspase-3 high)
    - -1: Error during simulation
- `history_df`: DataFrame with p53 network state history (only if `return_history=true` and cell survives or error occurs). Returns `nothing` otherwise.
"""
function compute_repair_domain_p53_optimized_history(X_input::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2; return_history::Bool=false, j_nc_param::Float64=3.0, time_factor::Float64=360.0)

    # --- Initial Checks ---
    if sum(Y) > 0
        # Death type 1: Irreparable damage (Y > 0)
        return return_history ? (0.0, Inf, 1, nothing) : (0.0, Inf, 1)
    end

    # Make a copy to avoid modifying the input vector directly
    X = copy(X_input)
    sum_X = sum(X) # Initial sum

    if sum_X == 0
        # Death type 0: No damage initially, survived
        return return_history ? (Inf, 0.0, 0, nothing) : (Inf, 0.0, 0)
    end

    # --- Constants and Initial State Setup (Define ONCE) ---
    ATM_tot = 5.0
    Akt_tot = 1.0
    Pip_tot = 1.0
    CytoC_tot = 3.0
    Casp3_tot = 3.0
    # Use j_nc_param argument
    j_nc = j_nc_param

    # Initial concentrations for the ODE system
    ATM2_0 = 2.17
    ATMs_0 = 0.01
    p53s_0 = 0.0
    p53_0 = 0.8
    Mdm2c_0 = 0.01
    Mdm2cp_0 = 0.4
    Mdm2n_0 = 0.26
    Akts_0 = 0.94
    Pip3_0 = 0.89
    PTEN_0 = 0.1
    p53killer_0 = 0.0
    p21_0 = 0.1
    Wip1_0 = 0.2
    p53DINP1_0 = 0.0
    p53AIP1_0 = 0.1
    CytoC_0 = 0.06
    Casp3_0 = 0.05

    initial_state = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0]
    state = copy(initial_state) # Current ODE state

    # Indices for accessing elements in the state vector
    # net = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3"]
    p53s_idx = 3
    p21_idx = 14
    casp3_idx = 17

    # GSM2 parameters
    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    # --- History Storage (Optimized) ---
    history_states = Vector{Vector{Float64}}()
    history_times = Vector{Float64}()
    df_mol_history = nothing # Initialize history DataFrame placeholder
    if return_history
        push!(history_states, initial_state)
        push!(history_times, 0.0)
    end

    # --- Initial Rate Calculation ---
    p53s_val = initial_state[p53s_idx]
    p53s_term = p53s_val / (1.0 + p53s_val)
    # Ensure sigmoid function is available in the scope
    aC = a * sigmoid(sum_X, 1.0 - p53s_term, 1.0 + p53s_term)
    rC = r * sigmoid(sum_X, 1.0 + p53s_term, 1.0 - p53s_term)

    # --- Main Simulation Loop ---
    current_time = 0.0
    n_domains = length(X)

    # Preallocate propensity vectors
    aX = Vector{Float64}(undef, n_domains)
    bX = Vector{Float64}(undef, n_domains)
    rX = Vector{Float64}(undef, n_domains)

    local death_time_result = Inf
    local recover_time_result = Inf
    local death_type_result = -1 # Default to error state

    try # Wrap simulation in try-catch for robustness
        while sum_X > 0
            # --- Calculate Propensities ---
            @inbounds for i in 1:n_domains
                xi = X[i]
                aX[i] = aC * xi
                # Use max(0.0, xi - 1.0) for clarity and potential Float safety
                bX[i] = b * xi * max(0.0, xi - 1.0)
                rX[i] = rC * xi
            end

            sum_rX = sum(rX)
            sum_aX = sum(aX)
            sum_bX = sum(bX)
            a0 = sum_rX + sum_aX + sum_bX

            if a0 <= 0.0 # Safety check
                @warn "Total propensity a0 <= 0 while sum_X = $sum_X > 0 at time $current_time. Assuming survival."
                death_time_result = Inf
                recover_time_result = current_time
                death_type_result = 0
                break # Exit loop
            end

            # --- Time Step ---
            r1 = rand() # Uses default RNG
            tau = (1.0 / a0) * log(1.0 / r1)
            current_time += tau

            # --- Select Reaction ---
            r2 = rand()
            target_prop = r2 * a0
            vec_index = 0
            dom = 0

            # Find reaction without creating concatenated vector/cumsum
            if target_prop <= sum_rX # Reaction type 1 (repair)
                vec_index = 1
                cumulative = 0.0
                @inbounds for i in 1:n_domains
                    cumulative += rX[i]
                    if target_prop <= cumulative
                        dom = i
                        break
                    end
                end
            elseif target_prop <= sum_rX + sum_aX # Reaction type 2 (death a)
                vec_index = 2
                cumulative = sum_rX
                @inbounds for i in 1:n_domains
                    cumulative += aX[i]
                    if target_prop <= cumulative
                        dom = i
                        break
                    end
                end
            else # Reaction type 3 (death b)
                vec_index = 3
                cumulative = sum_rX + sum_aX
                @inbounds for i in 1:n_domains
                    cumulative += bX[i]
                    if target_prop <= cumulative
                        dom = i
                        break
                    end
                end
            end

            if dom == 0 # Safety check
                @error "Reaction selection failed. target_prop=$target_prop, a0=$a0, sums=($sum_rX, $sum_aX, $sum_bX). Returning error state."
                death_time_result = current_time
                recover_time_result = Inf
                death_type_result = -1
                break # Exit loop
            end

            # --- Solve ODE ---
            # Use time_factor argument for scaling
            t_ode_end = tau * time_factor
            times = (0.0, t_ode_end)
            # Pass current sum_X as Float64 if ODE expects it
            p_ode = (ATM_tot, Akt_tot, Pip_tot, Float64(sum_X), CytoC_tot, Casp3_tot, j_nc)

            # Ensure p53_network function is available in the scope
            prob = ODEProblem(p53_network, state, times, p_ode)
            # Use Tsit5() solver instead of RK4()
            # Add tolerance options if needed (e.g., abstol=1e-6, reltol=1e-3)
            sol = solve(prob, Tsit5(), isoutofdomain=(u, p, t) -> any(x -> x < 0, u),
                save_everystep=false, save_start=false, saveat=t_ode_end / 10.0, dense=false)
            # save_start=false because we only need evolution *during* this step

            # --- Check for ODE Solver Failure ---
            if sol.retcode != :Success && sol.retcode != :Terminated
                @warn "ODE solver failed with retcode $(sol.retcode) at time $current_time. Assuming p53-induced death (type 3)."
                death_time_result = current_time
                recover_time_result = Inf
                death_type_result = 3
                break # Exit loop
            end

            # --- Check for Apoptosis (Casp3 threshold) ---
            max_casp3_interval = 0.0
            if !isempty(sol.u)
                # Check the saved points and the final point
                max_casp3_interval = maximum(u[casp3_idx] for u in sol.u)
                max_casp3_interval = max(max_casp3_interval, sol.u[end][casp3_idx])
            else
                # If solver took zero steps, check the state *before* this step
                max_casp3_interval = state[casp3_idx]
            end

            if max_casp3_interval > 1.0
                # Death type 3: p53/Caspase-induced apoptosis
                death_time_result = current_time
                recover_time_result = Inf
                death_type_result = 3
                break # Exit loop
            end

            # --- Update State and Rates ---
            state = sol.u[end] # Update ODE state to the end of the interval

            # --- Store History (Optional) ---
            if return_history
                for k in 1:length(sol.t)
                    # Convert ODE time back to simulation time scale using time_factor
                    push!(history_times, current_time - tau + sol.t[k] / time_factor)
                    push!(history_states, sol.u[k])
                end
                # Ensure final state at current_time is captured if not by saveat
                if isempty(sol.t) || (sol.t[end] / time_factor) < tau
                    if isempty(history_times) || history_times[end] < current_time
                        push!(history_times, current_time)
                        push!(history_states, state)
                    end
                end
            end
            # --- End History ---

            # Recalculate p53-dependent rates based on new state
            p53s_val = state[p53s_idx]
            p53s_term = p53s_val / (1.0 + p53s_val)
            # Note: sum_X hasn't changed *yet* within this iteration for rate calculation
            aC = a * sigmoid(sum_X, 1.0 - p53s_term, 1.0 + p53s_term)
            rC = r * sigmoid(sum_X, 1.0 + p53s_term, 1.0 - p53s_term)

            # --- Process Selected Reaction ---
            if vec_index == 2 || vec_index == 3 # Death reaction (a or b)
                th = 0.1
                p21_val = state[p21_idx]
                prob_senescence = p21_val / (th + p21_val)

                death_time_result = current_time
                recover_time_result = Inf
                death_type_result = rand() < prob_senescence ? 2 : 1 # Type 2 if senescence, else Type 1
                break # Exit loop

            elseif vec_index == 1 # Repair reaction
                X[dom] -= 1
                sum_X -= 1 # Update the sum efficiently
                if sum_X == 0
                    # Death type 0: Survived (all damage repaired)
                    death_time_result = Inf
                    recover_time_result = current_time
                    death_type_result = 0
                    # Loop terminates naturally
                end
            end
        end # End while loop

    catch e
        @error "Error during simulation: $e"
        death_time_result = current_time # Mark time of error
        recover_time_result = Inf
        death_type_result = -1 # Indicate error
        # Optionally rethrow(e) if needed
    end # End try-catch

    # --- Finalize History DataFrame (if requested and not immediate death) ---
    if return_history && death_type_result != 1 && death_type_result != 2 && death_type_result != 3
        # Ensure final state/time is included
        final_t = (death_type_result == 0) ? recover_time_result : death_time_result
        if isempty(history_times) || history_times[end] < final_t
            push!(history_times, final_t)
            push!(history_states, state) # Use the last known state
        end

        if !isempty(history_states)
            net_ = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3"]
            try
                state_matrix = hcat(history_states...)'
                df_mol_history = DataFrame(state_matrix, Symbol.(net_))
                df_mol_history[!, :time] = history_times
            catch err_df
                @error "Error creating history DataFrame: $err_df. History may be incomplete."
                df_mol_history = nothing # Ensure it's nothing on error
            end
        else
            df_mol_history = nothing # No history collected
        end
    else
        df_mol_history = nothing # Not requested or died type 1/2/3
    end

    # --- Return Results ---
    if return_history
        return (death_time_result, recover_time_result, death_type_result, df_mol_history)
    else
        return (death_time_result, recover_time_result, death_type_result)
    end
end

# --- Helper Functions (assumed defined elsewhere) ---
# function sigmoid(x, C, K) ... end
# function p53_network(dydt, y, p, t) ... end
# --- GSM2 Struct (assumed defined elsewhere) ---
# struct GSM2 ... end


function compute_repair_domain_p53_optimized(X_input::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2; return_history::Bool=false)

    # --- Initial Checks ---
    if sum(Y) > 0
        # Death type 1: Irreparable damage (Y > 0)
        return (0.0, Inf, 1)
    end

    # Make a copy to avoid modifying the input vector directly
    X = copy(X_input)
    sum_X = sum(X) # Initial sum

    if sum_X == 0
        # Death type 0: No damage initially, survived
        return (Inf, 0.0, 0)
    end

    # --- Constants and Initial State Setup (Define ONCE) ---
    ATM_tot = 5.0
    Akt_tot = 1.0
    Pip_tot = 1.0
    CytoC_tot = 3.0
    Casp3_tot = 3.0
    j_nc = 3.0 # Parameter for p53 network ODE

    # Initial concentrations for the ODE system
    ATM2_0 = 2.17
    ATMs_0 = 0.01
    p53s_0 = 0.0
    p53_0 = 0.8
    Mdm2c_0 = 0.01
    Mdm2cp_0 = 0.4
    Mdm2n_0 = 0.26
    Akts_0 = 0.94
    Pip3_0 = 0.89
    PTEN_0 = 0.1
    p53killer_0 = 0.0
    p21_0 = 0.1
    Wip1_0 = 0.2
    p53DINP1_0 = 0.0
    p53AIP1_0 = 0.1
    CytoC_0 = 0.06
    Casp3_0 = 0.05

    initial_state = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0]
    state = copy(initial_state) # Current ODE state

    # Indices for accessing elements in the state vector (based on 'net' in original code)
    # net = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3"]
    p53s_idx = 3
    p21_idx = 14
    casp3_idx = 17

    # GSM2 parameters
    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    # --- Initial Rate Calculation ---
    # Calculate initial p53-dependent rates aC and rC
    p53s_val = initial_state[p53s_idx]
    p53s_term = p53s_val / (1.0 + p53s_val)
    aC = a * sigmoid(sum_X, 1.0 - p53s_term, 1.0 + p53s_term)
    rC = r * sigmoid(sum_X, 1.0 + p53s_term, 1.0 - p53s_term)

    # --- Main Simulation Loop ---
    current_time = 0.0
    n_domains = length(X)

    # Preallocate propensity vectors
    aX = Vector{Float64}(undef, n_domains)
    bX = Vector{Float64}(undef, n_domains)
    rX = Vector{Float64}(undef, n_domains)

    while sum_X > 0
        # --- Calculate Propensities ---
        @inbounds for i in 1:n_domains
            xi = X[i]
            aX[i] = aC * xi
            bX[i] = b * xi * max(0.0, xi - 1.0) # Ensure non-negative
            rX[i] = rC * xi
        end

        sum_rX = sum(rX)
        sum_aX = sum(aX)
        sum_bX = sum(bX)
        a0 = sum_rX + sum_aX + sum_bX

        if a0 <= 0.0 # Should not happen if sum_X > 0, but safety check
            @warn "Total propensity a0 is zero or negative while sum_X = $sum_X > 0. Breaking loop."
            # This might indicate an issue with rates becoming zero unexpectedly.
            # Decide how to handle: maybe return current state or an error indicator.
            # For now, let's assume survival if damage remains but no reactions occur.
            return (Inf, current_time, 0) # Treat as survival
        end

        # --- Time Step ---
        r1 = rand() # Uses default RNG
        tau = (1.0 / a0) * log(1.0 / r1)
        current_time += tau

        # --- Select Reaction ---
        r2 = rand()
        target_prop = r2 * a0
        vec_index = 0
        dom = 0

        # Find reaction without creating concatenated vector/cumsum
        if target_prop <= sum_rX # Reaction type 1 (repair)
            vec_index = 1
            cumulative = 0.0
            @inbounds for i in 1:n_domains
                cumulative += rX[i]
                if target_prop <= cumulative
                    dom = i
                    break
                end
            end
        elseif target_prop <= sum_rX + sum_aX # Reaction type 2 (death a)
            vec_index = 2
            cumulative = sum_rX
            @inbounds for i in 1:n_domains
                cumulative += aX[i]
                if target_prop <= cumulative
                    dom = i
                    break
                end
            end
        else # Reaction type 3 (death b)
            vec_index = 3
            cumulative = sum_rX + sum_aX
            @inbounds for i in 1:n_domains
                cumulative += bX[i]
                if target_prop <= cumulative
                    dom = i
                    break
                end
            end
        end

        if dom == 0 # Safety check if reaction wasn't found (shouldn't happen)
            @error "Reaction selection failed. target_prop=$target_prop, a0=$a0, sums=($sum_rX, $sum_aX, $sum_bX)"
            # Handle error appropriately, e.g., return an error state
            return (NaN, NaN, -1) # Indicate error
        end

        # --- Solve ODE ---
        # Time interval for ODE solver (convert tau from days? to minutes?)
        # Original code used tmin = tau * 720. Assuming tau is in days.
        t_ode_end = tau * 360.0
        times = (0.0, t_ode_end)
        p_ode = (ATM_tot, Akt_tot, Pip_tot, sum_X, CytoC_tot, Casp3_tot, j_nc)

        prob = ODEProblem(p53_network, state, times, p_ode)
        # Use Tsit5() solver (generally good default) instead of RK4()
        # Keep saveat for checking max Casp3 during the interval
        sol = solve(prob, Tsit5(), isoutofdomain=(u, p, t) -> any(x -> x < 0, u),
            save_everystep=false, save_start=false, saveat=t_ode_end / 10.0, dense=false)
        # Note: save_start=false because we only need the evolution *during* this step

        # --- Check for Apoptosis (Casp3 threshold) ---
        max_casp3_interval = 0.0
        if !isempty(sol.u)
            # Check the saved points and the final point
            max_casp3_interval = maximum(u[casp3_idx] for u in sol.u)
            # Also include the very last state if saveat didn't capture it exactly
            max_casp3_interval = max(max_casp3_interval, sol.u[end][casp3_idx])
        else
            # If solver failed or took zero steps, check initial state for this step
            max_casp3_interval = state[casp3_idx]
        end

        if max_casp3_interval > 1.0
            # Death type 3: p53/Caspase-induced apoptosis
            return (current_time, Inf, 3)
        end

        # --- Update State and Rates ---
        state = sol.u[end] # Update ODE state to the end of the interval

        # Recalculate p53-dependent rates based on new state
        p53s_val = state[p53s_idx]
        p53s_term = p53s_val / (1.0 + p53s_val)
        # Note: sum_X hasn't changed *yet* within this iteration for rate calculation
        aC = a * sigmoid(sum_X, 1.0 - p53s_term, 1.0 + p53s_term)
        rC = r * sigmoid(sum_X, 1.0 + p53s_term, 1.0 - p53s_term)

        # --- Process Selected Reaction ---
        if vec_index == 2 || vec_index == 3 # Death reaction (a or b)
            # Check p21 level for senescence vs immediate death
            th = 0.1
            p21_val = state[p21_idx]
            prob_senescence = p21_val / (th + p21_val)

            if rand() < prob_senescence
                # Death type 2: Senescence (p21 dependent)
                return (current_time, Inf, 2)
            else
                # Death type 1: Mitotic catastrophe / other death (p21 dependent)
                return (current_time, Inf, 1)
            end
        elseif vec_index == 1 # Repair reaction
            X[dom] -= 1
            sum_X -= 1 # Update the sum efficiently
            if sum_X == 0
                # Return survival, repair time, and optionally the history
                return (Inf, current_time, 0) # Modify return signature if history is returned
            end
        end
    end # End while loop

    # Should technically not be reached if sum_X starts > 0, but as a fallback:
    # If loop terminates unexpectedly (e.g., a0 <= 0), assume survival at current time.
    return (Inf, current_time, 0) # Modify return signature if history is returned
end

function compute_repair_domain_p53(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2)

    if sum(Y) > 0
        return (0., Inf, 1)
    end

    if sum(X) == 0
        return (Inf, 0., 0)
    end

    ATM_tot = 5.
    Akt_tot = 1.
    Pip_tot = 1.
    CytoC_tot = 3.
    Casp3_tot = 3.

    ATM2_0 = 2.17
    ATMs_0 = 0.01
    p53_0 = 0.8
    p53s_0 = 0.
    Mdm2c_0 = 0.01
    Mdm2cp_0 = 0.4
    Mdm2n_0 = 0.26
    Akts_0 = 0.94
    Pip3_0 = 0.89
    PTEN_0 = 0.1
    p53killer_0 = 0.
    p21_0 = 0.1
    Wip1_0 = 0.2
    p53DINP1_0 = 0.
    p53AIP1_0 = 0.1
    CytoC_0 = 0.06
    Casp3_0 = 0.05

    ATM_tot = 5.
    Akt_tot = 1.
    Pip_tot = 1.
    CytoC_tot = 3.
    Casp3_tot = 3.

    net = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3"]
    net_ = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3", "time"]

    state = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0]
    state_ = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0, 0.]

    j_nc = 3
    p = (ATM_tot, Akt_tot, Pip_tot, sum(X), CytoC_tot, Casp3_tot, j_nc)

    state = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0]

    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    df_mol = DataFrame()
    for (i, name) in enumerate(net_)
        df_mol[!, name] = [state_[i]]  # Create a column with a single-element vector
    end
    aC = a * sigmoid.(sum(X), 1 - df_mol.p53s[1] / (1 + df_mol.p53s[1]), 1 + df_mol.p53s[1] / (1 + df_mol.p53s[1]))
    rC = r * sigmoid.(sum(X), 1 + df_mol.p53s[1] / (1 + df_mol.p53s[1]), 1 - df_mol.p53s[1] / (1 + df_mol.p53s[1]))

    current_time = 0.
    while sum(X) > 0
        #println(current_time )

        r1 = rand(Uniform(0, 1))
        r2 = rand(Uniform(0, 1))

        aX = aC .* X
        bX = max.(b .* X .* (X .- 1), 0)
        rX = rC .* X

        a0 = sum(vcat(rX, aX, bX))
        fire = cumsum(vcat(rX, aX, bX)) .- r2 * a0
        reac_idx = findfirst(x -> x >= 0, fire)
        current_time += (1 / a0) * log(1 / r1)

        if reac_idx <= size(X, 1)
            vec_index = 1
            dom = reac_idx
        elseif reac_idx <= 2 * size(X, 1)
            vec_index = 2
            dom = reac_idx - size(X, 1)
        elseif reac_idx <= 3 * size(X, 1)
            vec_index = 3
            dom = reac_idx - 2 * size(X, 1)
        end

        tmin = (1 / a0) * log(1 / r1) * 720
        times = (0.0, tmin)
        p = (ATM_tot, Akt_tot, Pip_tot, sum(X), CytoC_tot, Casp3_tot, j_nc)

        prob = ODEProblem(p53_network, state, times, p)
        sol = solve(prob, RK4(), isoutofdomain=(u, p, t) -> any(x -> x < 0, u), save_everystep=false, save_start=true, saveat=tmin / 10)

        sol_mat = hcat(sol.u...)
        df_sol = DataFrame(sol_mat', :auto)
        rename!(df_sol, Symbol.(names(df_sol)) .=> Symbol.(net))

        state = Vector(df_sol[end, net])
        df_sol[!, :time] = sol.t ./ 720. .+ current_time

        append!(df_mol, df_sol)

        aC = a * sigmoid.(sum(X), 1 - df_mol.p53s[end] / (1 + df_mol.p53s[end]), 1 + df_mol.p53s[end] / (1 + df_mol.p53s[end]))
        rC = r * sigmoid.(sum(X), 1 + df_mol.p53s[end] / (1 + df_mol.p53s[end]), 1 - df_mol.p53s[end] / (1 + df_mol.p53s[end]))

        if maximum(df_sol.Casp3) > 1
            #println("p53 induced apoptosis")
            return (current_time, Inf, 3)
        end

        if (vec_index == 2) | (vec_index == 3)
            #println("death a or b")
            th = 0.1
            p = df_sol.p21[end] / (th + df_sol.p21[end])

            if rand() < p
                return (current_time, Inf, 2)
            else
                return (current_time, Inf, 1)
            end
        elseif vec_index == 1
            X[dom] -= 1
            if sum(X) == 0
                return (Inf, current_time, 0)
            end
        end
    end
end

function compute_repair_domain(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2)

    if sum(Y) > 0
        return (0., Inf, 1)
    end

    if sum(X) == 0
        return (Inf, 0., 0)
    end

    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    current_time = 0.
    while sum(X) > 0
        #println(current_time )

        r1 = rand(Uniform(0, 1))
        r2 = rand(Uniform(0, 1))

        aX = a .* X
        bX = max.(b .* X .* (X .- 1), 0)
        rX = r .* X

        a0 = sum(vcat(rX, aX, bX))
        fire = cumsum(vcat(rX, aX, bX)) .- r2 * a0
        reac_idx = findfirst(x -> x >= 0, fire)
        current_time += (1 / a0) * log(1 / r1)

        if reac_idx <= size(X, 1)
            vec_index = 1
            dom = reac_idx
        elseif reac_idx <= 2 * size(X, 1)
            vec_index = 2
            dom = reac_idx - size(X, 1)
        elseif reac_idx <= 3 * size(X, 1)
            vec_index = 3
            dom = reac_idx - 2 * size(X, 1)
        end

        if (vec_index == 2) | (vec_index == 3)
            return (current_time * 6, Inf, 1)
        elseif vec_index == 1
            X[dom] -= 1
            if sum(X) == 0
                return (Inf, current_time * 6, 0)
            end
        end
    end
end

function compute_induction_domain_doserate(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2, irrad::Irrad)

    if sum(Y) > 0
        return (0., Inf, 1)
    end

    if sum(X) == 0
        return (Inf, 0., 0)
    end

    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    current_time = 0.
    while sum(X) > 0
        #println(current_time )

        r1 = rand(Uniform(0, 1))
        r2 = rand(Uniform(0, 1))

        aX = a .* X
        bX = max.(b .* X .* (X .- 1), 0)
        rX = r .* X

        a0 = sum(vcat(rX, aX, bX))
        fire = cumsum(vcat(rX, aX, bX)) .- r2 * a0
        reac_idx = findfirst(x -> x >= 0, fire)
        current_time += (1 / a0) * log(1 / r1)

        if reac_idx <= size(X, 1)
            vec_index = 1
            dom = reac_idx
        elseif reac_idx <= 2 * size(X, 1)
            vec_index = 2
            dom = reac_idx - size(X, 1)
        elseif reac_idx <= 3 * size(X, 1)
            vec_index = 3
            dom = reac_idx - 2 * size(X, 1)
        end

        if (vec_index == 2) | (vec_index == 3)
            return (current_time * 6, Inf, 1)
        elseif vec_index == 1
            X[dom] -= 1
            if sum(X) == 0
                return (Inf, current_time * 6, 0)
            end
        end
    end
end


function compute_repair_domain_p53_OLD(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2, NC::Int64)

    if sum(Y) > 0
        return (0., Inf, 1)
    end

    if sum(X) == 0
        return (Inf, 0., 0)
    end

    X_vec = Vector{Vector{Float64}}()
    XC_vec = Vector{Vector{Float64}}()

    push!(X_vec, X)
    XC = zeros(Int64, size(X, 1))

    for _ in 1:min(NC, sum(X))
        (_, dmx) = findmax(X)
        X[dmx] -= 1
        XC[dmx] += 1
    end

    push!(XC_vec, XC)

    ATM_tot = 5.
    Akt_tot = 1.
    Pip_tot = 1.
    CytoC_tot = 3.
    Casp3_tot = 3.

    ATM2_0 = 2.17
    ATMs_0 = 0.01
    p53_0 = 0.8
    p53s_0 = 0.
    Mdm2c_0 = 0.01
    Mdm2cp_0 = 0.4
    Mdm2n_0 = 0.26
    Akts_0 = 0.94
    Pip3_0 = 0.89
    PTEN_0 = 0.1
    p53killer_0 = 0.
    p21_0 = 0.1
    Wip1_0 = 0.2
    p53DINP1_0 = 0.
    p53AIP1_0 = 0.1
    CytoC_0 = 0.06
    Casp3_0 = 0.05

    ATM_tot = 5.
    Akt_tot = 1.
    Pip_tot = 1.
    CytoC_tot = 3.
    Casp3_tot = 3.

    net = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3"]
    net_ = ["ATM2", "ATMs", "p53s", "p53", "Mdm2c", "Mdm2cp", "Mdm2n", "Akts", "Pip3", "p53killer", "Wip1", "p53DINP1", "PTEN", "p21", "p53AIP1", "CytoC", "Casp3", "time"]

    state = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0]
    state_ = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0, 0.]

    j_nc = 5
    p = (ATM_tot, Akt_tot, Pip_tot, sum(X + XC), CytoC_tot, Casp3_tot, j_nc)

    state = [ATM2_0, ATMs_0, p53s_0, p53_0, Mdm2c_0, Mdm2cp_0, Mdm2n_0, Akts_0, Pip3_0, p53killer_0, Wip1_0, p53DINP1_0, PTEN_0, p21_0, p53AIP1_0, CytoC_0, Casp3_0]

    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    # Set a seed for reproducibility (optional)
    uniform_dist = Uniform(0, 1)  # Uniform distribution between 0 and 1
    ub = rand(uniform_dist, size(X, 1))
    tb = zeros(Float64, size(X, 1))
    sb = log.(1 ./ ub)
    taub = (sb .- tb) ./ (max.(b * (X .+ XC) .* (X .+ XC .- 1), 0.0))

    ur = rand(uniform_dist, size(X, 1))
    tr = zeros(Float64, size(X, 1))
    sr = log.(1 ./ ur)
    taur = (sr .- tr) ./ (r .* X)

    ua = rand(uniform_dist, size(X, 1))
    ta = zeros(Float64, size(X, 1))
    sa = log.(1 ./ ua)
    taua = (sa .- ta) ./ (a .* X)

    df_mol = DataFrame()
    for (i, name) in enumerate(net_)
        df_mol[!, name] = [state_[i]]  # Create a column with a single-element vector
    end

    ur_C = rand(uniform_dist, size(XC, 1))
    tr_C = zeros(Float64, size(XC, 1))
    sr_C = log.(1 ./ ur_C)
    rC = r * sigmoid.(sum(X .+ XC), 1 + df_mol.p53s[1] / (1 + df_mol.p53s[1]), 1 - df_mol.p53s[1] / (1 + df_mol.p53s[1]))
    taur_C = (sr_C .- tr_C) ./ (rC .* XC)

    ua_C = rand(uniform_dist, size(XC, 1))
    ta_C = zeros(Float64, size(XC, 1))
    sa_C = log.(1 ./ ua_C)
    aC = a * sigmoid.(sum(X .+ XC), 1 - df_mol.p53s[1] / (1 + df_mol.p53s[1]), 1 + df_mol.p53s[1] / (1 + df_mol.p53s[1]))
    taua_C = (sa_C .- ta_C) ./ (aC .* XC)

    current_time = 0.
    while sum(X .+ XC) > 0
        #println(current_time )

        tmin, reac_idx = findmin(vcat(taub, taua, taur, taua_C, taur_C))
        if reac_idx ≤ size(X, 1)
            vec_index = 1
            dom = reac_idx
        elseif reac_idx ≤ 2 * size(X, 1)
            vec_index = 2
            dom = reac_idx - size(X, 1)
        elseif reac_idx ≤ 3 * size(X, 1)
            vec_index = 3
            dom = reac_idx - 2 * size(X, 1)
        elseif reac_idx ≤ 4 * size(X, 1)
            vec_index = 4
            dom = reac_idx - 3 * size(X, 1)
        elseif reac_idx ≤ 5 * size(X, 1)
            vec_index = 5
            dom = reac_idx - 4 * size(X, 1)
        end

        times = (0.0, tmin)
        p = (ATM_tot, Akt_tot, Pip_tot, sum(X .+ XC), CytoC_tot, Casp3_tot, j_nc)

        prob = ODEProblem(p53_network, state, times, p)
        sol = solve(prob, RK4(), isoutofdomain=(u, p, t) -> any(x -> x < 0, u), save_everystep=false, save_start=true, saveat=tmin / 10)

        sol_mat = hcat(sol.u...)
        df_sol = DataFrame(sol_mat', :auto)
        rename!(df_sol, Symbol.(names(df_sol)) .=> Symbol.(net))

        state = Vector(df_sol[end, net])
        df_sol[!, :time] = sol.t ./ 60. .+ current_time

        append!(df_mol, df_sol)

        if maximum(df_sol.Casp3) > 1
            #println("p53 induced apoptosis")
            return (current_time, Inf, 3)
        end

        if (vec_index == 1) | (vec_index == 2) | (vec_index == 4)
            #println("death a or b")
            p = df_sol.p21[end] / (df_sol.p21[end])

            if rand() < p
                return (current_time, Inf, 2)
            else
                return (current_time, Inf, 1)
            end

        elseif vec_index == 3
            #println("repair")
            ta .+= a .* X .* tmin
            tb .+= b .* (X .+ XC) .* (X .+ XC .- 1) .* tmin
            tr .+= r .* X .* tmin
            ta_C .+= aC .* XC .* tmin
            tr_C .+= rC .* XC .* tmin
            X[dom] -= 1

            push!(X_vec, X)
            push!(XC_vec, XC)

            current_time += tmin
            rsim = rand(uniform_dist, 1)
            sa .+= log(1 / rsim[1])
            sb .+= log(1 / rsim[1])
            sr .+= log(1 / rsim[1])
            sa_C .+= log(1 / rsim[1])
            sr_C .+= log(1 / rsim[1])

            taua = (sa .- ta) ./ (a .* X)
            taub = (sb .- tb) ./ (max.(b * (X .+ XC) .* (X .+ XC .- 1), 0.0))
            taur = (sr .- tr) ./ (r .* X)

            aC = a * sigmoid.(sum(X .+ XC), 1 - df_mol.p53s[end] / (1 + df_mol.p53s[end]), 1 + df_mol.p53s[end] / (1 + df_mol.p53s[end]))
            rC = r * sigmoid.(sum(X .+ XC), 1 + df_mol.p53s[end] / (1 + df_mol.p53s[end]), 1 - df_mol.p53s[end] / (1 + df_mol.p53s[end]))

            taua_C = (sa_C .- ta_C) ./ (aC .* XC)
            taur_C = (sr_C .- tr_C) ./ (rC .* XC)

            if sum(X .+ XC) == 0
                return (Inf, current_time, 0)
            end
        elseif vec_index == 5
            #println("repair")
            ta .+= a .* X .* tmin
            tb .+= b .* (X .+ XC) .* (X .+ XC .- 1) .* tmin
            tr .+= r .* X .* tmin
            ta_C .+= aC .* XC .* tmin
            tr_C .+= rC .* XC .* tmin
            XC[dom] -= 1

            if sum(X) > 0
                (_, dmx) = findmax(X)
                X[dmx] -= 1
                XC[dmx] += 1
            end

            push!(X_vec, X)
            push!(XC_vec, XC)

            if sum(X .+ XC) <= 0
                return (Inf, current_time, 0)
            end

            current_time += tmin
            rsim = rand(uniform_dist, 1)
            sa .+= log(1 / rsim[1])
            sb .+= log(1 / rsim[1])
            sr .+= log(1 / rsim[1])
            sa_C .+= log(1 / rsim[1])
            sr_C .+= log(1 / rsim[1])

            taua = (sa .- ta) ./ (a .* X)
            taub = (sb .- tb) ./ (max.(b * (X .+ XC) .* (X .+ XC .- 1), 0.0))
            taur = (sr .- tr) ./ (r .* X)

            aC = a * sigmoid.(sum(X .+ XC), 1 - df_mol.p53s[end] / (1 + df_mol.p53s[end]), 1 + df_mol.p53s[end] / (1 + df_mol.p53s[end]))
            rC = r * sigmoid.(sum(X .+ XC), 1 + df_mol.p53s[end] / (1 + df_mol.p53s[end]), 1 - df_mol.p53s[end] / (1 + df_mol.p53s[end]))

            taua_C = (sa_C .- ta_C) ./ (aC .* XC)
            taur_C = (sr_C .- tr_C) ./ (rC .* XC)
        end
    end
end

function p53_network(dydt, y, p, t)

    (ATM_tot, Akt_tot, Pip_tot, DBS_c, CytoC_tot, Casp3_tot, j_nc) = p

    k_dim = 10.
    k_undim = 1.
    k_acatm = 2.
    j_acatm = 1.
    k_deatm = 1.3
    j_deatm = 2.3

    k_dmdm2n0 = 0.003
    k_dmdm2n1 = 0.05
    j_atm = 1.0
    k_acp531 = 0.2
    k_dep53 = 0.1
    k_dp53s = 0.01
    k_sp53 = 0.2
    k_dp53n = 0.05
    k_dp53 = 0.7
    j_1p53n = 0.1
    k_smdm20 = 0.002
    k_smdm2 = 0.024
    j_smdm2 = 1.0
    k_dmdm2c = 0.003
    k_1mdm2s = 0.3
    j_1mdm2s = 0.1
    k_mdm2s = 8.0
    j_mdm2s = 0.3
    k_i = 0.06
    k_0 = 0.09
    k_acakt = 0.25
    j_acakt = 0.1
    k_deakt = 0.1
    j_deakt = 0.2
    k_p2 = 0.1
    j_p2 = 0.2
    k_p3 = 0.5
    j_p3 = 0.4
    k_p46 = 0.6
    j_p46 = 0.5
    k_dp46 = 0.3
    j_dp46 = 0.2
    k_swip10 = 0.01
    k_swip1 = 0.09
    j_swip1 = 0.5
    k_dwip1 = 0.05
    k_sdinp10 = 0.001
    k_sdinp11 = 0.01
    j_sdinp11 = 0.7
    k_sdinp12 = 0.07
    j_sdinp12 = 0.3
    k_ddinp1 = 0.01
    k_spten0 = 0.01
    k_spten = 0.5
    j_spten = 1.0
    k_dpten = 0.1
    k_sp210 = 0.01
    k_sp21 = 0.2
    j_sp21 = 0.6
    k_dp21 = 0.1
    k_saip10 = 0.01
    k_saip1 = 0.32
    j_saip1 = 1.5
    k_daip1 = 0.1
    k_accytoc0 = 0.001
    k_accytoc1 = 0.9
    j_casp3 = 0.5
    k_decytoc = 0.05
    k_accasp30 = 0.001
    k_accasp31 = 0.9
    j_cytoc = 0.5
    k_decasp3 = 0.07

    dydt[1] = 0.5 * k_dim * (ATM_tot - 2 * y[1] - y[2])^2 - k_undim * y[1]
    dydt[2] = k_acatm * ((DBS_c) / (DBS_c + j_nc)) * y[2] * ((ATM_tot - 2 * y[1] - y[2]) / (ATM_tot - 2 * y[1] - y[2] + j_acatm)) - k_deatm * ((y[2]) / (y[2] + j_deatm)) * (1 + y[11])
    dydt[3] = k_acp531 * ((y[2]) / (y[2] + j_atm)) * y[4] - k_dep53 * y[3] - k_dp53s * y[7] * ((y[3]) / (y[3] + j_1p53n))
    dydt[4] = k_sp53 - k_dp53n * y[4] - k_dp53 * y[7] * ((y[4]) / (y[4] + j_1p53n)) - k_acp531 * ((y[2]) / (y[2] + j_atm)) * y[4] + k_dep53 * y[3]
    dydt[5] = k_smdm20 + k_smdm2 * ((y[3]^4) / (y[3]^4 + j_smdm2^4)) + k_1mdm2s * ((y[6]) / (y[6] + j_1mdm2s)) - k_dmdm2c * y[5] - k_mdm2s * (y[8]) * ((y[5]) / (y[5] + j_mdm2s))
    dydt[6] = k_mdm2s * (y[8]) * ((y[5]) / (y[5] + j_mdm2s)) - k_1mdm2s * ((y[6]) / (y[6] + j_1mdm2s)) - k_i * y[6] + k_0 * y[7] - k_dmdm2c * y[6]
    dydt[7] = k_i * y[6] - k_0 * y[7] - y[7] * (k_dmdm2n0 + k_dmdm2n1 * ((y[2]) / (y[2] + j_atm)))
    dydt[8] = k_acakt * y[9] * ((Akt_tot - y[8]) / (Akt_tot - y[8] + j_acakt)) - k_deakt * ((y[8]) / (y[8] + j_deakt))
    dydt[9] = k_p2 * ((Pip_tot - y[9]) / (Pip_tot - y[9] + j_p2)) - k_p3 * y[13] * ((y[9]) / (y[9] + j_p3))
    dydt[10] = k_p46 * y[12] * ((y[3] - y[10]) / (y[3] - y[10] + j_p46)) - k_dp46 * y[11] * ((y[10]) / (y[10] + j_dp46))
    dydt[11] = k_swip10 + k_swip1 * (((y[3] - y[10])^3) / ((y[3] - y[10])^3 + j_swip1^3)) - k_dwip1 * y[11]
    dydt[12] = k_sdinp10 + k_sdinp11 * (((y[3] - y[10])^3) / ((y[3] - y[10])^3 + j_sdinp11^3)) + k_sdinp12 * (((y[10])^3) / ((y[10])^3 + j_sdinp12^3)) - k_ddinp1 * y[12]
    dydt[13] = k_spten0 + k_spten * (((y[10])^3) / ((y[10])^3 + j_spten^3)) - k_dpten * y[13]
    dydt[14] = k_sp210 + k_sp21 * (((y[3] - y[10])^3) / ((y[3] - y[10])^3 + j_sp21^3)) - k_dp21 * y[14]
    dydt[15] = k_saip10 + k_saip1 * (((y[10])^3) / ((y[10])^3 + j_saip1^3)) - k_daip1 * y[15]
    dydt[16] = (k_accytoc0 + k_accytoc1 * y[15] * ((y[17]^4) / (y[17]^4 + j_casp3^4))) * (CytoC_tot - y[16]) - k_decytoc * y[16]
    dydt[17] = (k_accasp30 + k_accasp31 * ((y[16]^4) / (y[16]^4 + j_cytoc^4))) * (Casp3_tot - y[17]) - k_decasp3 * y[17]
end

function compute_repair_domain_OLD(X::Vector{Int64}, Y::Vector{Int64}, gsm2::GSM2)

    a = gsm2.a
    b = gsm2.b
    r = gsm2.r

    dom_num = size(X, 1)

    if sum(Y) > 0
        return (0., Inf, 1)
    end

    if sum(X) == 0
        return (Inf, 0., 0)
    end

    uniform_dist = Uniform(0, 1)  # Uniform distribution between 0 and 1
    ub = rand(uniform_dist, size(X, 1))
    tb = zeros(Float64, size(X, 1))
    sb = log.(1 ./ ub)
    taub = (sb .- tb) ./ (max.(b * (X) .* (X .- 1), 0.0))

    ur = rand(uniform_dist, size(X, 1))
    tr = zeros(Float64, size(X, 1))
    sr = log.(1 ./ ur)
    taur = (sr .- tr) ./ (r .* X)

    ua = rand(uniform_dist, size(X, 1))
    ta = zeros(Float64, size(X, 1))
    sa = log.(1 ./ ua)
    taua = (sa .- ta) ./ (a .* X)

    current_time = 0.
    while sum(X) > 0
        tmin, reac_idx = findmin(vcat(taub, taua, taur))
        if reac_idx ≤ dom_num
            vec_index = 1
            dom = reac_idx
        elseif reac_idx ≤ 2 * dom_num
            vec_index = 2
            dom = reac_idx - dom_num
        elseif reac_idx ≤ 3 * dom_num
            vec_index = 3
            dom = reac_idx - 2 * dom_num
        end

        if (vec_index == 1) | (vec_index == 2)
            current_time += tmin
            return (current_time, Inf, 1)
        elseif vec_index == 3
            ta .+= a .* X .* tmin
            tb .+= b .* (X) .* (X .- 1) .* tmin
            tr .+= r .* X .* tmin
            X[dom] -= 1

            current_time += tmin
            rsim = rand(uniform_dist, 1)
            sa .+= log(1 / rsim[1])
            sb .+= log(1 / rsim[1])
            sr .+= log(1 / rsim[1])

            taua = (sa .- ta) ./ (a .* X)
            taub = (sb .- tb) ./ (max.(b * (X) .* (X .- 1), 0.0))
            taur = (sr .- tr) ./ (r .* X)
            if sum(X) <= 0
                return (Inf, current_time, 0)
            end
        end
    end
end


function compute_times!(arrayOfCell::Array{Cell}, cell_df::DataFrame, gsm2::GSM2, nat_apo::Float64, type::String, alpha=0., beta=0.)
    """
    Compute times of death, recovery, division and apoptosis for each cell.
    """

    cell_df.sp .= 1.
    cell_df.apo_time .= Inf
    cell_df.death_time .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time .= Inf
    cell_df.is_death_rad .= 0

    apo_time = Inf
    death_time = Inf
    division_time = 0.
    recover_time_sample = 0.
    l = 0

    for i in cell_df.index[cell_df.is_cell.==1]
        """
        Compute Survival Probability
        """
        cell = arrayOfCell[i]

        if type == "spatial"
            SP_cell = spatial_GSM2_fast_OLD(cell.dam_X, cell.dam_Y, gsm2)
        elseif type == "domain"
            SP_cell = domain_GSM2(cell.dam_X_dom, cell.dam_Y_dom, gsm2)
        elseif type == "fast"
            SP_cell = exp(-alpha * cell.dose[1] - beta * cell.dose[1] * cell.dose[1])
        else
            println("Error: type of GSM2 must be 'spatial', 'domain' or 'fast'")
            return 0
        end

        #save SP in cell and cell_df
        cell.SP = SP_cell
        cell_df.sp[i] = SP_cell

        """
        Compute death time by radiation
        """
        if SP_cell < 1
            if type != "domain"
                l = -log(SP_cell) / 120
                death_time = log(1 - rand()) / (-l)
            else
                (death_time, recover_time_sample) = compute_repair_domain(cell.dam_X_dom, cell.dam_Y_dom, gsm2)
            end
            if death_time == 0
                cell_df.is_cell[i] = 0
            end
        elseif SP_cell == 1
            death_time = Inf
        end

        """
        Compute times if cell dies
        """
        if (death_time < 120) & (type != "domain")
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i] = Inf
            cell_df.is_death_rad[i] = 1
            if (cell_df.cell_cycle[i] == "M") & (cell_df.is_stem == 1)
                division_time = rand(Gamma(1))
                cell_df.death_time[i] = min(death_time, division_time)
            else
                cell_df.death_time[i] = death_time
            end
        elseif (death_time > 0) & (type == "domain")
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i] = Inf
            cell_df.is_death_rad[i] = 1
            if (cell_df.cell_cycle[i] == "M") & (cell_df.is_stem == 1)
                division_time = rand(Gamma(1))
                cell_df.death_time[i] = min(death_time, division_time)
            else
                cell_df.death_time[i] = death_time
            end
        else
            """
            Compute recover time if cell survives radiation
            """
            cell_df.death_time[i] = Inf
            if (size(cell.dam_X)[1] > 0) & (type != "domain")
                for i in 1:size(cell.dam_X)[1]
                    l = cell.r_gsm2 * i
                    recover_time_sample += log(1 - rand()) / (-l)
                end
                cell_df.recover_time[i] = recover_time_sample
            elseif (type == "domain") & (recover_time_sample > 0)
                cell_df.recover_time[i] = recover_time_sample
            else
                cell_df.recover_time[i] = Inf
            end

            """
            Compute division time if cell can divide
            """
            if (cell_df.cell_cycle[i] == "M") & (cell_df.is_stem[i] == 1) & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(1))

                if cycle_time < cell_df.recover_time[i]
                    cell_df.recover_time[i] = Inf
                    cell_df.cycle_time[i] = Inf
                    cell_df.is_death_rad[i] = 1
                    cell_df.death_time[i] = cycle_time
                else
                    cell_df.cycle_time[i] = cycle_time
                end

            elseif (cell_df.cell_cycle[i] == "I") & (cell_df.is_stem[i] == 1) & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(30))
                if cell_df.recover_time[i] > 0
                    cell_df.cycle_time[i] = min(cycle_time, cell_df.recover_time[i])
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            else
                cell_df.cycle_time[i] = Inf
            end
        end

        """
        Compute apoptosis
        """
        l = -log(nat_apo) / 24
        apo_time = log(1 - rand()) / (-l)
        cell_df.apo_time[i] = apo_time

        if cell_df.is_cell[i] == 0
            cell_df.death_time[i] = Inf
            cell_df.apo_time[i] = Inf
            cell_df.cycle_time[i] = Inf
            cell_df.recover_time[i] = Inf
            cell_df.is_death_rad[i] = 1
        end
    end
end



function compute_times_AB_cycle!(cell_df::DataFrame, gsm2::Vector{GSM2})

    cell_df.sp .= 1.
    cell_df.apo_time .= Inf
    cell_df.death_time .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type .= -1

    apo_time = Inf
    death_time = Inf
    division_time = 0.
    recover_time_sample = 0.
    l = 0
    death_type = 0

    for i in cell_df.index[cell_df.is_cell.==1]

        #! compute cell survival
        SP_cell = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

        #save SP in cell and cell_df
        cell_df.sp[i] = SP_cell

        #! compute death time        

        if SP_cell < 1
            X = copy(cell_df.dam_X_dom[i])
            Y = copy(cell_df.dam_Y_dom[i])
            death_time, recover_time_sample, death_type = compute_repair_domain(X, Y, gsm2)

            if death_time == 0
                cell_df.is_cell[i] = 0
                cell_df.number_nei[cell_df.nei[i]] .+= 1
                cell_df.recover_time[i] = Inf
                cell_df.death_time[i] = Inf
                cell_df.cycle_time[i] = Inf
                cell_df.apo_time[i] = Inf
                cell_df.death_type[i] = death_type
            end
        else
            death_time = Inf
        end

        #if cell dies
        if death_time != Inf
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i] = Inf
            cell_df.death_type[i] = death_type
            if (cell_df.cell_cycle[i] == "M") & (cell_df.is_stem == 1) & (cell_df.number_nei[i] > 0)
                division_time = rand(Gamma(1))
                cell_df.death_time[i] = min(death_time, division_time)
            else
                cell_df.death_time[i] = death_time
                cell_df.death_type[i] = death_type
            end
        else #! if the cell survives radiation
            cell_df.death_time[i] = Inf
            cell_df.recover_time[i] = recover_time_sample
            cell_df.death_type[i] = death_type

            #! if cell survies radiation, compute the division time if possible

            if (cell_df.cell_cycle[i] == "M") & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(1))

                if cell_df.recover_time[i] < Inf
                    if cycle_time < cell_df.recover_time[i]
                        cell_df.recover_time[i] = Inf
                        cell_df.cycle_time[i] = Inf
                        cell_df.death_time[i] = cycle_time
                    end
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            elseif (cell_df.cell_cycle[i] == "I") & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(30))
                if cell_df.recover_time[i] < Inf
                    cell_df.cycle_time[i] = max(cycle_time, cell_df.recover_time[i])
                    cell_df.recover_time[i] = Inf
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            else
                cell_df.cycle_time[i] = Inf
            end
        end

        #! compute apoptosis
        l = -log(1 - 10^-5)
        apo_time = log(1 - rand()) / (-l)
        cell_df.apo_time[i] = apo_time
    end
end

function compute_times_AB_cellcycle!(cell_df::DataFrame, gsm2::GSM2)

    cell_df.sp .= 1.
    cell_df.apo_time .= Inf
    cell_df.death_time .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type .= -1

    apo_time = Inf
    death_time = Inf
    division_time = 0.
    recover_time_sample = 0.
    l = 0
    death_type = 0

    for i in cell_df.index[cell_df.is_cell.==1]

        #! compute cell survival
        SP_cell = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

        #save SP in cell and cell_df
        cell_df.sp[i] = SP_cell

        #! compute death time        

        if SP_cell < 1
            X = copy(cell_df.dam_X_dom[i])
            Y = copy(cell_df.dam_Y_dom[i])
            death_time, recover_time_sample, death_type = compute_repair_domain(X, Y, gsm2)

            if death_time == 0
                cell_df.is_cell[i] = 0
                cell_df.number_nei[cell_df.nei[i]] .+= 1
                cell_df.recover_time[i] = Inf
                cell_df.death_time[i] = Inf
                cell_df.cycle_time[i] = Inf
                cell_df.apo_time[i] = Inf
                cell_df.death_type[i] = death_type
            end
        else
            death_time = Inf
        end

        #if cell dies
        if death_time != Inf
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i] = Inf
            cell_df.death_type[i] = death_type
            if (cell_df.cell_cycle[i] == "M") & (cell_df.is_stem == 1) & (cell_df.number_nei[i] > 0)
                division_time = rand(Gamma(1))
                cell_df.death_time[i] = min(death_time, division_time)
            else
                cell_df.death_time[i] = death_time
                cell_df.death_type[i] = death_type
            end
        else #! if the cell survives radiation
            cell_df.death_time[i] = Inf
            cell_df.recover_time[i] = recover_time_sample
            cell_df.death_type[i] = death_type

            #! if cell survies radiation, compute the division time if possible

            if (cell_df.cell_cycle[i] == "M") & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(1))

                if cell_df.recover_time[i] < Inf
                    if cycle_time < cell_df.recover_time[i]
                        cell_df.recover_time[i] = Inf
                        cell_df.cycle_time[i] = Inf
                        cell_df.death_time[i] = cycle_time
                    end
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            elseif (cell_df.cell_cycle[i] == "I") & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(30))
                if cell_df.recover_time[i] < Inf
                    cell_df.cycle_time[i] = max(cycle_time, cell_df.recover_time[i])
                    cell_df.recover_time[i] = Inf
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            else
                cell_df.cycle_time[i] = Inf
            end
        end

        #! compute apoptosis
        l = -log(1 - 10^-5)
        apo_time = log(1 - rand()) / (-l)
        cell_df.apo_time[i] = apo_time
    end
end


function compute_times_domain!(cell_df::DataFrame, gsm2::GSM2, nat_apo::Float64, p53::Bool)

    cell_df.sp .= 1.
    cell_df.apo_time .= Inf
    cell_df.death_time .= Inf
    cell_df.recover_time .= Inf
    cell_df.cycle_time .= Inf
    cell_df.is_death_rad .= 0
    cell_df.death_type .= -1

    apo_time = Inf
    death_time = Inf
    division_time = 0.
    recover_time_sample = 0.
    l = 0
    death_type = 0

    for i in cell_df.index[cell_df.is_cell.==1]

        #! compute cell survival
        SP_cell = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], gsm2)

        #save SP in cell and cell_df
        cell_df.sp[i] = SP_cell

        #! compute death time        

        if SP_cell < 1
            X = copy(cell_df.dam_X_dom[i])
            Y = copy(cell_df.dam_Y_dom[i])
            if p53
                death_time, recover_time_sample, death_type = compute_repair_domain_p53_optimized(X, Y, gsm2)
            else
                death_time, recover_time_sample, death_type = compute_repair_domain(X, Y, gsm2)
            end
            if death_time == 0
                cell_df.is_cell[i] = 0
                cell_df.number_nei[cell_df.nei[i]] .+= 1
                cell_df.recover_time[i] = Inf
                cell_df.death_time[i] = Inf
                cell_df.cycle_time[i] = Inf
                cell_df.apo_time[i] = Inf
                cell_df.death_type[i] = death_type
            end
        else
            death_time = Inf
        end

        #if cell dies
        if death_time != Inf
            cell_df.recover_time[i] = Inf
            cell_df.cycle_time[i] = Inf
            cell_df.death_type[i] = death_type
            if (cell_df.cell_cycle[i] == "M") & (cell_df.is_stem == 1) & (cell_df.number_nei[i] > 0)
                division_time = rand(Gamma(1))
                cell_df.death_time[i] = min(death_time, division_time)
            else
                cell_df.death_time[i] = death_time
                cell_df.death_type[i] = death_type
            end
        else #! if the cell survives radiation
            cell_df.death_time[i] = Inf
            cell_df.recover_time[i] = recover_time_sample
            cell_df.death_type[i] = death_type

            #! if cell survies radiation, compute the division time if possible

            if (cell_df.cell_cycle[i] == "M") & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(1))

                if cell_df.recover_time[i] < Inf
                    if cycle_time < cell_df.recover_time[i]
                        cell_df.recover_time[i] = Inf
                        cell_df.cycle_time[i] = Inf
                        cell_df.death_time[i] = cycle_time
                    end
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            elseif (cell_df.cell_cycle[i] == "I") & (cell_df.number_nei[i] > 0)
                cycle_time = rand(Gamma(30))
                if cell_df.recover_time[i] < Inf
                    cell_df.cycle_time[i] = max(cycle_time, cell_df.recover_time[i])
                    cell_df.recover_time[i] = Inf
                else
                    cell_df.cycle_time[i] = cycle_time
                end
            else
                cell_df.cycle_time[i] = Inf
            end
        end

        #! compute apoptosis
        l = -log(nat_apo)
        apo_time = log(1 - rand()) / (-l)
        cell_df.apo_time[i] = apo_time
    end
end

"""
Function to compute the next event in the simulation.

Arguments:
- `cell_df`: DataFrame containing the cell data

Returns:
- A tuple containing the time of the next event, the index of the cell associated with the event, and the type of event (apoptosis, death, recovery, or division).

"""
function compute_next_event(cell_df::DataFrame)
    # Create a view of the DataFrame, filtering out cells that are not alive
    cell_df_sub = cell_df[cell_df.is_cell.==1, [:apo_time, :death_time, :recover_time, :cycle_time]]

    # Initialize variables to keep track of the minimum time and the associated event
    min_time = Inf
    min_row = 0
    min_event = ""

    # Iterate over columns and find the minimum in each column
    for (col_idx, col) in enumerate(eachcol(cell_df_sub))
        # Find the index and value of the minimum in the current column
        local_min_idx = argmin(col)
        local_min_time = col[local_min_idx]

        # If the minimum in this column is smaller than the current overall minimum, update
        if local_min_time < min_time
            min_time = local_min_time
            min_row = local_min_idx
            min_event = ["apo_time", "death_time", "recover_time", "cycle_time"][col_idx]
        end
    end

    return (min_time, min_row, min_event)
end


"""
    MC_loop_ions_domain_fast!(Npar::Int, x_cb::Float64, y_cb::Float64, irrad_cond::Vector{AT}, gsm2::GSM2,
                                cell_df::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
                                R_beam::Float64, type_AT::String, ion::Ion)

Optimized calculation of dose deposition for active cells (is_cell == 1),
considering layer-dependent irradiation conditions (`track_seg = false`).
Loops over particles first, then domains, using precomputed layer-specific data.

# Arguments
- ... (same as previous version) ...
"""
function MC_loop_ions_domain_fast!(Npar::Int, x_cb::Float64, y_cb::Float64, irrad_cond::Vector{AT}, gsm2::GSM2,
    cell_df::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    R_beam::Float64, type_AT::String, ion::Ion)



    println("Finished Optimized Monte Carlo Loop (track_seg=false)")
end


"""
    MC_dose_fast!(ion::Ion, Npar::Int64, x_cb::Float64, y_cb::Float64, R_beam::Float64, irrad_cond::Vector{AT},
                cell_df::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
                gsm2::GSM2, type_AT::String, track_seg::Bool)

Main function to calculate dose distribution using fast methods. Dispatches based on `track_seg`.

# Arguments
- ... (see MC_loop_ions_domain_tsc_fast! and MC_loop_ions_domain_fast!)
- `gsm2::GSM2`: GSM2 parameters (needed by both branches).
- `track_seg::Bool`: If true, calculates dose for one layer and copies. If false, calculates layer-dependently.
"""


function MC_dose_fast_sim1voxel!(ion::Ion, Npar::Int64, x_cb::Float64, y_cb::Float64, R_beam::Float64, irrad_cond::Vector{AT},
    cell_df_first_voxel_copy::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    gsm2::GSM2, type_AT::String, track_seg::Bool, cell_df_copy::DataFrame, ParIrr::String)

    # Ensure 'at' is initialized (e.g., with zeros) before calling loops
    # Example: at[:, Not(:index)] .= 0.0

    if track_seg
        println("Track Segment Enabled (track_seg=true)")
        # --- Filter for representative layer ---
        # Find cells that are marked as 'is_cell'
        cell_df_is = filter(row -> row.is_cell == 1, cell_df_first_voxel_copy)
        if nrow(cell_df_is) == 0
            @warn "No cells marked with is_cell=1 found. Cannot perform track segmentat calculation."
            return # Or handle appropriately
        end
        # Group by x,y and get the index of the first cell in each group
        # Using combine with first assumes the first cell encountered is representative
        grouped_df = combine(groupby(cell_df_is, [:x, :y]), :index => first => :representative_index)

        # Filter the domain center DataFrames and the target 'at' DataFrame
        # Use 'in' with Ref() for efficient filtering based on the representative indices
        rep_indices_set = Set(grouped_df.representative_index)
        cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
        cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
        at_single = filter(row -> row.index in rep_indices_set, at)

        if nrow(at_single) == 0
            @warn "No representative cells found after filtering 'at' DataFrame. Check index matching."
            return
        end

        # --- Run calculation for representative layer ---
        # Pass only the first irradiation condition (assuming it's representative for tsc)
        MC_loop_ions_domain_tsc_fast!(Npar, x_cb, y_cb, [irrad_cond[1]], gsm2, cell_df_single_x, cell_df_single_y, at_single, R_beam, type_AT, ion)

        # --- Copy results and calculate damage ---
        MC_loop_copy_dose_domain_fast!(cell_df_first_voxel_copy, at_single, at) # Copies from at_single to at AND populates cell_df.dose/dose_cell

        row_cell_df = nrow(cell_df_first_voxel_copy)
        row_cell_df_copy = nrow(cell_df_copy) / row_cell_df

        # ripete cell_df.dose row_cell_df_copy volte (una per ogni blocco)
        #dose_pattern = [cell_df.dose[i] for _ in 1:row_cell_df_copy, i in 1:length(cell_df.dose)]
        #dose_pattern = repeat(cell_df.dose, row_cell_df_copy)
        #dose_pattern_cell = repeat(cell_df.dose_cell, row_cell_df_copy)
        dose_pattern = vcat([cell_df_first_voxel_copy.dose for _ in 1:row_cell_df_copy]...)
        dose_pattern_cell = vcat([cell_df_first_voxel_copy.dose_cell for _ in 1:row_cell_df_copy]...)

        if ParIrr == "true"
            cell_df_copy[cell_df_copy.i_voxel_z.>1, :is_cell] .= 0
        end

        # assegna la colonna in cell_df_copy
        cell_df_copy.dose = ifelse.(cell_df_copy.is_cell .== 1, dose_pattern, dose_pattern .* 0.0)
        cell_df_copy.dose_cell = ifelse.(cell_df_copy.is_cell .== 1, dose_pattern_cell, dose_pattern_cell .* 0.0)

        # Remove the following # for the original version
        #MC_loop_damage_domain_fast_NTCP!(ion, cell_df_copy, gsm2) # Calculates damage based on cell_df.dose

    else
        println("Track Segment Disabled (track_seg=false)")

        Np = rand(Poisson(Npar))
        x_list = Array{Float64}(undef, Np)
        y_list = Array{Float64}(undef, Np)
        Threads.@threads for ip in 1:Np
            x_list[ip], y_list[ip] = GenerateHit_Circle(x_cb, y_cb, R_beam)
        end

        for id in unique(cell_df_first_voxel_copy.energy_step)
            cell_df_is = filter(row -> (row.is_cell == 1) & (row.energy_step == id), cell_df_first_voxel_copy)
            if (nrow(cell_df_is) != 0)
                grouped_df = combine(groupby(cell_df_is, [:x, :y]), :index => first => :representative_index)

                rep_indices_set = Set(grouped_df.representative_index)
                cell_df_single_x = filter(row -> row.index in rep_indices_set, df_center_x)
                cell_df_single_y = filter(row -> row.index in rep_indices_set, df_center_y)
                at_single = filter(row -> row.index in rep_indices_set, at)

                MC_loop_ions_domain_fast!(x_list, y_list, [irrad_cond[id]], gsm2, cell_df_single_x, cell_df_single_y, at_single, type_AT, ion)
                MC_loop_copy_dose_domain_layer_fast_notsc!(cell_df_first_voxel_copy, at_single, at, id)
            end
        end

        row_cell_df = nrow(cell_df_first_voxel_copy)
        row_cell_df_copy = nrow(cell_df_copy) / row_cell_df

        # ripete cell_df.dose row_cell_df_copy volte (una per ogni blocco)
        #dose_pattern = [cell_df.dose[i] for _ in 1:row_cell_df_copy, i in 1:length(cell_df.dose)]
        #dose_pattern = repeat(cell_df.dose, row_cell_df_copy)
        #dose_pattern_cell = repeat(cell_df.dose_cell, row_cell_df_copy)
        dose_pattern = vcat([cell_df_first_voxel_copy.dose for _ in 1:row_cell_df_copy]...)
        dose_pattern_cell = vcat([cell_df_first_voxel_copy.dose_cell for _ in 1:row_cell_df_copy]...)

        if ParIrr == "true"
            cell_df_copy[cell_df_copy.i_voxel_z.>1, :is_cell] .= 0
        end

        # assegna la colonna in cell_df_copy
        cell_df_copy.dose = ifelse.(cell_df_copy.is_cell .== 1, dose_pattern, dose_pattern .* 0.0)
        cell_df_copy.dose_cell = ifelse.(cell_df_copy.is_cell .== 1, dose_pattern_cell, dose_pattern_cell .* 0.0)

        # Remove the following # for the original version
        #MC_loop_damage_domain_fast_NTCP!(ion, cell_df_copy, gsm2)
    end
    println("MC_dose_fast! finished.")
end




"""
    MC_loop_copy_dose_domain_layer_fast!(cell_df::DataFrame, at_single::DataFrame, at::DataFrame, id::Int64)

Copies the calculated domain doses from representative cells in `at_single` (which
should contain doses for representative cells of a specific layer `id`) to all
active cells (`is_cell == 1`) in the main `cell_df` and `at` DataFrames that
belong to the specified layer `id`.

If `:dose` or `:dose_cell` columns do not exist in `cell_df`, they will be created
and initialized to zeros/empty. Only cells matching `layer == id` and `is_cell == 1`
will have their dose information updated. Cells in other layers or inactive cells
in this layer will retain their existing dose values (presumably zero if initialized
correctly before the layer loop).

# Arguments
- `cell_df::DataFrame`: DataFrame containing information for ALL cells, including `index`, `x`, `y`, `layer`, `is_cell`. Will be modified in-place.
- `at_single::DataFrame`: DataFrame containing calculated domain doses for REPRESENTATIVE cells *of layer `id`*. Must have an `index` column linking to `cell_df`. Domain columns (e.g., `center_1`, ...) hold the doses.
- `at::DataFrame`: Target DataFrame, structured like `at_single` but for ALL cells. This DataFrame will be modified in-place for rows matching layer `id`.
- `id::Int64`: The layer ID for which doses are being copied.
"""
function MC_loop_copy_dose_domain_layer_fast!(cell_df::DataFrame, at_single::DataFrame, at::DataFrame, id::Int64)

    println("Copying doses for layer $id...")

    # --- Input Validation & Column Initialization ---
    if !hasproperty(cell_df, :index) || !hasproperty(at_single, :index) || !hasproperty(at, :index)
        error("One or more DataFrames are missing the required 'index' column.")
    end
    if !hasproperty(cell_df, :x) || !hasproperty(cell_df, :y) || !hasproperty(cell_df, :layer) || !hasproperty(cell_df, :is_cell)
        error("cell_df is missing required columns (:x, :y, :layer, :is_cell).")
    end

    num_cells = nrow(cell_df)
    # Determine expected number of domains from at_single
    domain_cols = names(at_single, Not(:index))
    num_domains = length(domain_cols)

    if num_domains == 0 && nrow(at_single) > 0
        @warn "No domain columns found in 'at_single' (excluding :index) for layer $id."
        # Proceed, but dose vectors will be empty
    elseif nrow(at_single) == 0
        @warn "'at_single' is empty for layer $id. No doses to copy. Skipping copy for this layer."
        # Initialize columns if they don't exist, but they will be empty/zero
        # This is handled below, but we can exit early if nothing to copy
        # return # Or continue to ensure columns are created if needed
    end

    # Check and initialize :dose column
    if !hasproperty(cell_df, :dose)
        println("Initializing cell_df.dose column")
        # Initialize with vectors of the correct size (or empty if num_domains is 0)
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    elseif !(eltype(cell_df.dose) <: AbstractVector) # Check if it holds vectors
        @warn "cell_df.dose column exists but does not hold Vectors (type: $(eltype(cell_df.dose))). Reinitializing."
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
        # Optional: Check if existing vectors have the right size - might be tricky if num_domains varies
        # Let's assume num_domains is consistent across cells/layers for now, or handle resizing below.
    end

    # Check and initialize :dose_cell column
    if !hasproperty(cell_df, :dose_cell)
        println("Initializing cell_df.dose_cell column")
        cell_df.dose_cell = zeros(Float64, num_cells)
    elseif !(eltype(cell_df.dose_cell) <: AbstractFloat) # Check if it holds Floats
        @warn "cell_df.dose_cell column exists but does not hold Floats (type: $(eltype(cell_df.dose_cell))). Reinitializing."
        cell_df.dose_cell = zeros(Float64, num_cells)
    end
    # --- End Initialization ---

    if nrow(at_single) == 0
        println("Finished copying doses for layer $id (at_single was empty).")
        return # Nothing to copy for this layer
    end

    # --- Create mapping from (x, y) to dose data from at_single ---
    # This map holds the dose data for the representative cells of the current layer `id`
    xy_to_dose_data = Dict{Tuple{Float64,Float64},Tuple{Vector{Float64},Float64}}()
    sizehint!(xy_to_dose_data, nrow(at_single))

    # Need to get x, y from cell_df using the index from at_single
    # Create a temporary map from cell_df index to its row index for quick lookup
    cell_df_index_to_row = Dict(idx => r for (r, idx) in enumerate(cell_df.index))

    for row_single in eachrow(at_single)
        representative_index = row_single.index
        cell_df_row_idx = get(cell_df_index_to_row, representative_index, 0)

        if cell_df_row_idx == 0
            @warn "Representative index $representative_index from at_single (layer $id) not found in cell_df. Skipping."
            continue
        end

        # Ensure this representative cell is indeed in the current layer
        if cell_df.layer[cell_df_row_idx] != id
            @warn "Representative index $representative_index from at_single (layer $id) points to cell in layer $(cell_df.layer[cell_df_row_idx]) in cell_df. Skipping."
            continue
        end

        rep_x = Float64(cell_df.x[cell_df_row_idx])
        rep_y = Float64(cell_df.y[cell_df_row_idx])
        coords = (rep_x, rep_y)

        # Extract the vector of domain doses and calculate scalar mean
        domain_doses_vector = Vector(row_single[domain_cols])
        scalar_dose_cell = isempty(domain_doses_vector) ? 0.0 : mean(domain_doses_vector)

        xy_to_dose_data[coords] = (domain_doses_vector, scalar_dose_cell)
    end
    # --- End mapping creation ---

    # --- Create mapping from cell index to row index in the main 'at' DataFrame ---
    at_index_to_row = Dict(idx => r for (r, idx) in enumerate(at.index))
    # --- End mapping creation ---


    # --- Iterate through all rows of the main cell_df and copy doses for the target layer ---
    # Use @Threads.threads for potential parallelization if cell_df is large
    Threads.@threads for i in 1:num_cells
        # Check if the cell is in the target layer AND is active
        if cell_df.layer[i] == id && cell_df.is_cell[i] == 1
            coords = (Float64(cell_df.x[i]), Float64(cell_df.y[i]))

            # Look up the dose data for these coordinates (from the representative cell)
            dose_data = get(xy_to_dose_data, coords, nothing)

            if dose_data === nothing
                # This cell's (x,y) didn't have a representative in at_single for this layer.
                # This might indicate an issue in representative cell selection or filtering.
                # For now, assign zero dose for this cell in this layer.
                @warn "Coordinates $coords (cell index $(cell_df.index[i]), layer $id) not found in xy_to_dose_data map. Assigning zero dose."
                domain_doses_vector = zeros(Float64, num_domains)
                scalar_dose_cell = 0.0
            else
                domain_doses_vector, scalar_dose_cell = dose_data
            end

            # 1. Assign the vector of doses to the 'cell_df.dose' column
            # Ensure the vector has the correct size before assignment
            if length(cell_df.dose[i]) != num_domains
                cell_df.dose[i] = zeros(Float64, num_domains) # Reinitialize if size mismatch
                @warn "Resizing cell_df.dose vector for cell index $(cell_df.index[i]). Expected $num_domains, found $(length(cell_df.dose[i]))."
            end
            cell_df.dose[i] .= domain_doses_vector # Assign element-wise

            # 2. Assign the calculated scalar average dose to 'cell_df.dose_cell'
            cell_df.dose_cell[i] = scalar_dose_cell

            # 3. Update the corresponding row in the main 'at' DataFrame
            cell_idx = cell_df.index[i]
            at_row_idx = get(at_index_to_row, cell_idx, 0)

            if at_row_idx == 0
                @warn "Cell index $cell_idx (layer $id) not found in main 'at' DataFrame. Cannot update 'at'."
            else
                # Ensure 'at' row has the correct number of domain columns
                if length(names(at, Not(:index))) != num_domains
                    @error "Number of domain columns in main 'at' DataFrame ($(length(names(at, Not(:index))))) does not match num_domains from at_single ($num_domains) for layer $id. Cannot update 'at'."
                elseif !isempty(domain_cols)
                    at[at_row_idx, domain_cols] .= domain_doses_vector
                end
            end
        end
        # If cell_df.layer[i] != id or cell_df.is_cell[i] == 0, do nothing.
        # Their dose columns are assumed to be initialized to zero/empty beforehand.
    end # End of loop over cell_df rows

    println("Finished copying doses for layer $id.")
    # No return value needed as 'at' and 'cell_df' are modified in-place
end


"""
    populate_cell_df_from_at!(cell_df::DataFrame, at::DataFrame)

Helper function to populate `cell_df.dose` and `cell_df.dose_cell` based on the
final calculated doses in the `at` DataFrame. Assumes `at` contains the dose
for each domain for every cell. Creates columns if they don't exist.
"""
function populate_cell_df_from_at!(cell_df::DataFrame, at::DataFrame)
    println("Populating cell_df.dose and cell_df.dose_cell from 'at'...")

    num_cells = nrow(cell_df)
    domain_cols = names(at, Not(:index))
    num_domains = length(domain_cols)

    # --- Initialize Columns if necessary ---
    if !hasproperty(cell_df, :dose)
        println("Initializing cell_df.dose column")
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    elseif !(eltype(cell_df.dose) <: AbstractVector)
        @warn "cell_df.dose column exists but does not hold Vectors. Reinitializing."
        cell_df.dose = [zeros(Float64, num_domains) for _ in 1:num_cells]
    end

    if !hasproperty(cell_df, :dose_cell)
        println("Initializing cell_df.dose_cell column")
        cell_df.dose_cell = zeros(Float64, num_cells)
    elseif !(eltype(cell_df.dose_cell) <: AbstractFloat)
        @warn "cell_df.dose_cell column exists but does not hold Floats. Reinitializing."
        cell_df.dose_cell = zeros(Float64, num_cells)
    end
    # --- End Initialization ---

    # Create a mapping from cell index to row index in 'at' for faster lookup
    at_index_map = Dict(row.index => i for (i, row) in enumerate(eachrow(at)))

    for i in 1:num_cells
        cell_idx = cell_df.index[i]
        at_row_idx = get(at_index_map, cell_idx, 0)

        if at_row_idx == 0
            #@warn "Cell index $cell_idx not found in 'at' DataFrame. Skipping dose population for this cell."
            # Ensure vectors/values are zeroed or empty
            if length(cell_df.dose[i]) != num_domains
                cell_df.dose[i] = zeros(Float64, num_domains)
            else
                cell_df.dose[i] .= 0.0
            end
            cell_df.dose_cell[i] = 0.0
            continue
        end

        # Extract dose vector from the corresponding row in 'at'
        domain_doses_vector = Vector(at[at_row_idx, domain_cols])

        # Assign vector and mean to cell_df
        if length(cell_df.dose[i]) != num_domains
            cell_df.dose[i] = copy(domain_doses_vector) # Resize and copy
        else
            cell_df.dose[i] .= domain_doses_vector # Assign element-wise
        end
        cell_df.dose_cell[i] = isempty(domain_doses_vector) ? 0.0 : mean(domain_doses_vector)
    end
    println("Finished populating cell_df.")
end


#
#function MC_dose_fast!(ion::Ion, Npar::Int64, x_cb::Float64, y_cb::Float64, R_beam::Float64, irrad_cond::Array{AT},
#    cell_df::DataFrame, df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
#    type_AT::String, track_seg::Bool)
#    if track_seg
#        cell_df_is = cell_df[cell_df.is_cell .== 1, :]
#        grouped_df = combine(groupby(cell_df_is, [:x, :y]), :index => first => :representative_index)
#        cell_df_single_x = df_center_x[in.(df_center_x.index, Ref(grouped_df.representative_index)), :]
#        cell_df_single_y = df_center_y[in.(df_center_y.index, Ref(grouped_df.representative_index)), :]
#        at_single = at[in.(at.index, Ref(grouped_df.representative_index)), :]
#        MC_loop_ions_domain_tsc_fast!(Npar, x_cb, y_cb, irrad_cond, gsm2, cell_df_single_x, cell_df_single_y, at_single, R_beam, type_AT, ion);
#        MC_loop_copy_dose_domain_fast!(cell_df, at_single, at)
#        MC_loop_damage_domain_fast!(ion, cell_df, gsm2)
#    else
#        MC_loop_ions_domain_fast!(Npar, x_cb, y_cb, irrad_cond, gsm2, df_center_x, df_center_y, at, R_beam, type_AT, ion);
#        MC_loop_damage_domain_fast!(ion, cell_df, gsm2)
#    end
#end

function MC_dose!(ion::Ion, Npar::Int64, x_cb::Float64, y_cb::Float64, R_beam::Float64, irrad_cond::Array{AT}, Rn::Float64, arrayOfCell::Array{Cell}, cell_df::DataFrame, type_AT::String, track_seg::Bool)
    if track_seg
        cell_df_is = cell_df[cell_df.is_cell.==1, :]
        grouped_df = combine(groupby(cell_df_is, [:x, :y]), :index => first => :representative_index)
        cell_df_single = cell_df[in.(cell_df.index, Ref(grouped_df.representative_index)), :]
        arrayOfCell_single = arrayOfCell[grouped_df.representative_index]
        MC_loop_ions_domain_tsc!(ion, Npar, x_cb, y_cb, R_beam, Rn, irrad_cond, arrayOfCell_single, cell_df_single, type_AT)
        MC_loop_copy_dose_domain!(arrayOfCell, cell_df, arrayOfCell_single, cell_df_single)
        MC_loop_damage_domain!(ion, arrayOfCell, cell_df)
    else
        MC_loop_ions_domain!(ion, Npar, x_cb, y_cb, R_beam, Rn, arrayOfCell, cell_df, irrad_cond, type_AT)
        MC_loop_damage_domain!(ion, arrayOfCell, cell_df)
    end

    cell_df.dose .= getfield.(arrayOfCell, :dose)
    cell_df.dose_cell .= getfield.(arrayOfCell, :dose_cell)
end

function MC_loop_ions_domain_tsc!(
    ion::Ion, Npar::Int64, x_cb::Float64, y_cb::Float64,
    R_beam::Float64, Rn::Float64, irrad_cond::Vector{AT},
    arrayOfCell::Vector{Cell}, cell_df::DataFrame, type_AT::String)

    N = length(arrayOfCell)
    Np = rand(Poisson(Npar))

    Rapprox = min(irrad_cond[1].Rapprox, irrad_cond[1].Rp)
    Rp = irrad_cond[1].Rp

    println("Monte Carlo Loop ", Np, " particles of ", ion.ion)

    Threads.@threads for i in ProgressBar(1:Np)
        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)
        track = Track(x, y, Rn)

        # Compute distances in a vectorized manner
        distance_xy = @. sqrt((cell_df.x - x)^2 + (cell_df.y - y)^2)

        # Identify relevant cells
        loop_j = findall((distance_xy .<= Rapprox + Rn) .& (cell_df.is_cell .== 1))
        loop_j_app = findall((distance_xy .> Rapprox + Rn) .& (distance_xy .<= Rp + Rn) .& (cell_df.is_cell .== 1))

        if !isempty(loop_j)
            for j in loop_j
                cell = arrayOfCell[j]
                nucleus, radius = cell.r, cell.rd_gsm2

                ndom = floor(Int, nucleus / radius)
                dose_circle = zeros(Float64, length(cell.center_y))

                for cc in eachindex(cell.center_y)
                    dose, _, dose_circle[cc] = distribute_dose_domain(
                        cell.center_x[cc], cell.center_y[cc], radius, track, irrad_cond[1], type_AT
                    )
                end

                #dose_circle_ = repeat(dose_circle, floor(Int, nucleus / radius))
                @inbounds arrayOfCell[j].dose .+= dose_circle
                arrayOfCell[j].dose_cell += sum(dose_circle) / length(cell.center_y)
            end
        end

        Dm = GetRadialLinearDose(0.5 * (Rapprox + Rp), irrad_cond[1], type_AT)

        if !isempty(loop_j_app)
            for j in loop_j_app
                cell = arrayOfCell[j]
                nucleus, radius = cell.r, cell.rd_gsm2
                ndom = floor(Int, nucleus / radius)

                dose_circle = fill(Dm, length(cell.center_y))
                #dose_circle_ = repeat(dose_circle, ndom)

                @inbounds arrayOfCell[j].dose .+= dose_circle
                arrayOfCell[j].dose_cell += sum(dose_circle) / length(cell.center_y)
            end
        end
    end

    # Update dose in DataFrame efficiently
    cell_df.dose .= getfield.(arrayOfCell, :dose)
    cell_df.dose_cell .= getfield.(arrayOfCell, :dose_cell)
end

function MC_loop_ions_domain!(ion::Ion, Npar::Int64, x_cb::Float64, y_cb::Float64, R_beam::Float64, Rn::Float64, arrayOfCell::Array{Cell}, cell_df::DataFrame, irrad_cond::Array{AT}, type_AT::String)
    # Initialize variables
    # Initialize variables
    N = size(arrayOfCell)[1]
    Np = rand(Poisson(Npar))

    println("Monte Carlo Loop ", Np, " particles of ", ion.ion)

    # Parallelize the loop using threads
    Threads.@threads for i in ProgressBar(1:Np)

        # Generate a random hit position within the beam circle
        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)
        track = Track(x, y, Rk)

        for zz in unique(cell_df.layer)
            cell_df_ = cell_df[cell_df.layer.==zz, :]
            distance_xy = @. sqrt((cell_df.x - x)^2 + (cell_df.y - y)^2)

            Rapprox = min(irrad_cond[zz].Rapprox, irrad_cond[zz].Rp)
            Rp = irrad_cond[zz].Rp

            loop_j = findall((distance_xy .<= Rapprox + Rn) .& (cell_df.is_cell .== 1))
            loop_j_app = findall((distance_xy .> Rapprox + Rn) .& (distance_xy .<= Rp + Rn) .& (cell_df.is_cell .== 1))
            if size(loop_j)[1] != 0
                for j in loop_j
                    cell = arrayOfCell[j]
                    nucleus, radius = cell.r, cell.rd_gsm2

                    ndom = floor(Int, nucleus / radius)
                    dose_circle = zeros(Float64, length(cell.center_y))

                    for cc in eachindex(cell.center_y)
                        dose, _, dose_circle[cc] = distribute_dose_domain(
                            cell.center_x[cc], cell.center_y[cc], radius, track, irrad_cond[zz], type_AT
                        )
                    end

                    #dose_circle_ = repeat(dose_circle, ndom)
                    @inbounds arrayOfCell[j].dose .+= dose_circle
                    arrayOfCell[j].dose_cell += sum(dose_circle) / length(cell.center_y)
                end
            end

            Dm = GetRadialLinearDose(0.5 * (Rapprox + Rp), irrad_cond[zz], type_AT)

            if size(loop_j_app)[1] != 0
                for j in loop_j_app
                    for j in loop_j_app
                        cell = arrayOfCell[j]
                        nucleus, radius = cell.r, cell.rd_gsm2
                        ndom = floor(Int, nucleus / radius)

                        dose_circle = fill(Dm, length(cell.center_y))
                        #dose_circle_ = repeat(dose_circle, ndom)

                        @inbounds arrayOfCell[j].dose .+= dose_circle
                        arrayOfCell[j].dose_cell += sum(dose_circle) / length(cell.center_y)
                    end
                end
            end
        end
    end

    cell_df.dose .= getfield.(arrayOfCell, :dose)
    cell_df.dose_cell .= getfield.(arrayOfCell, :dose_cell)
end

# Assume these are defined elsewhere and passed in:
# Npar, irrad_cond, gsm2, df_center_x, df_center_y, at, R_beam, Rn, type_AT, ion
# Also assume distribute_dose_domain and GenerateHit_Circle are defined








function MC_loop_ions_singlecell_fast!(Npar::Int, x_cb::Float64, y_cb::Float64, irrad_cond::AT, gsm2::GSM2,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    R_beam::Float64, type_AT::String, ion::Ion)

    Rp = irrad_cond.Rp
    Rc = irrad_cond.Rc
    Kp = irrad_cond.Kp
    Rk = Rp # Assuming Rk = Rp as in the original snippet

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2 # Squared core boundary for distance check
    mid_radius_sq = (gsm2.rd + 150 * Rc)^2 # Squared mid boundary for distance check
    penumbra_radius_sq = Rp^2             # Squared penumbra boundary

    sim_ = 1000
    if lower_bound_log <= 0
        error("Lower bound for impact_p calculation is non-positive.")
    end
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)
    dose_cell_lookup = zeros(Float64, sim_)

    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _dose, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond, type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_cell_lookup = sum(dose_cell_lookup_threads)

    impact_vec = impact_p
    dose_vec = dose_cell_lookup
    core_dose = dose_vec[1] # Dose for distances <= core_radius

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    at_row_accumulators = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    Np = rand(Poisson(Npar))
    println("Monte Carlo Loop ", Np, " particles of ", ion.ion)

    Threads.@threads for _ in 1:Np
        tid = Threads.threadid()
        local_at_row = at_row_accumulators[tid] # Get thread-local accumulator

        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        for k in 1:total_domains
            # Calculate squared distance first
            dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

            if dist_sq <= core_radius_sq
                local_at_row[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq) # Calculate sqrt only when needed

                idx_lookup = searchsortedfirst(impact_vec, dist)

                if idx_lookup == 1
                    local_at_row[k] += core_dose # Or dose_vec[1]
                elseif idx_lookup > length(impact_vec)
                    local_at_row[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_lookup-1], impact_vec[idx_lookup]
                    y1, y2 = dose_vec[idx_lookup-1], dose_vec[idx_lookup]
                    interpolated_dose = y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                    local_at_row[k] += interpolated_dose
                end
            elseif dist_sq < penumbra_radius_sq # Penumbra region (use < Rp^2)
                local_at_row[k] += Kp / dist_sq
            end
        end
    end

    final_at_row = sum(at_row_accumulators)

    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            at[r, c] += final_at_row[idx] # Or += if 'at' had initial values
            idx += 1
        end
    end

    println("Finished Monte Carlo Loop")
end

function MC_loop_ions_singlecell_dr_fast!(x_cb::Float64, y_cb::Float64, irrad_cond::AT, gsm2::GSM2,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    R_beam::Float64, type_AT::String, ion::Ion)

    Rp = irrad_cond.Rp
    Rc = irrad_cond.Rc
    Kp = irrad_cond.Kp
    Rk = Rp # Assuming Rk = Rp as in the original snippet

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2 # Squared core boundary for distance check
    mid_radius_sq = (gsm2.rd + 150 * Rc)^2 # Squared mid boundary for distance check
    penumbra_radius_sq = Rp^2             # Squared penumbra boundary

    sim_ = 1000
    if lower_bound_log <= 0
        error("Lower bound for impact_p calculation is non-positive.")
    end
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)
    dose_cell_lookup = zeros(Float64, sim_)

    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _dose, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond, type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_cell_lookup = sum(dose_cell_lookup_threads)

    impact_vec = impact_p
    dose_vec = dose_cell_lookup
    core_dose = dose_vec[1] # Dose for distances <= core_radius

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    at_row_accumulators = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    Np = 1
    Threads.@threads for _ in 1:Np
        tid = Threads.threadid()
        local_at_row = at_row_accumulators[tid] # Get thread-local accumulator

        x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

        for k in 1:total_domains
            # Calculate squared distance first
            dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

            if dist_sq <= core_radius_sq
                local_at_row[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq) # Calculate sqrt only when needed

                idx_lookup = searchsortedfirst(impact_vec, dist)

                if idx_lookup == 1
                    local_at_row[k] += core_dose # Or dose_vec[1]
                elseif idx_lookup > length(impact_vec)
                    local_at_row[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_lookup-1], impact_vec[idx_lookup]
                    y1, y2 = dose_vec[idx_lookup-1], dose_vec[idx_lookup]
                    interpolated_dose = y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                    local_at_row[k] += interpolated_dose
                end
            elseif dist_sq < penumbra_radius_sq # Penumbra region (use < Rp^2)
                local_at_row[k] += Kp / dist_sq
            end
        end
    end

    final_at_row = sum(at_row_accumulators)

    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            at[r, c] += final_at_row[idx] # Or += if 'at' had initial values
            idx += 1
        end
    end
end


function MC_loop_ions_singlecell_dr_lut_fast!(x_cb::Float64, y_cb::Float64, irrad_cond::AT, gsm2::GSM2,
    df_center_x::DataFrame, df_center_y::DataFrame,
    R_beam::Float64, type_AT::String)

    Rp = irrad_cond.Rp
    Rc = irrad_cond.Rc
    Kp = irrad_cond.Kp
    Rk = Rp # Assuming Rk = Rp as in the original snippet

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2 # Squared core boundary for distance check
    mid_radius_sq = (gsm2.rd + 150 * Rc)^2 # Squared mid boundary for distance check
    penumbra_radius_sq = Rp^2             # Squared penumbra boundary

    sim_ = 1000
    if lower_bound_log <= 0
        error("Lower bound for impact_p calculation is non-positive.")
    end
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)
    dose_cell_lookup = zeros(Float64, sim_)

    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _dose, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond, type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_cell_lookup = sum(dose_cell_lookup_threads)

    impact_vec = impact_p
    dose_vec = dose_cell_lookup
    core_dose = dose_vec[1] # Dose for distances <= core_radius

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    return (dom_x_row, dom_y_row, impact_vec, core_dose, dose_vec)
end

function MC_loop_ions_singlecell_dr_at_fast!(x_cb::Float64, y_cb::Float64, irrad_cond::AT, gsm2::GSM2,
    R_beam::Float64, dom_x_row::Vector{Float64}, dom_y_row::Vector{Float64},
    impact_vec::Vector{Float64}, core_dose::Float64, dose_vec::Vector{Float64})

    Rp = irrad_cond.Rp
    Rc = irrad_cond.Rc
    Kp = irrad_cond.Kp
    Rk = Rp # Assuming Rk = Rp as in the original snippet

    total_domains = size(dom_x_row, 1)

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2 # Squared core boundary for distance check
    mid_radius_sq = (gsm2.rd + 150 * Rc)^2 # Squared mid boundary for distance check
    penumbra_radius_sq = Rp^2

    local_at_row = zeros(Float64, total_domains) # Get thread-local accumulator
    x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

    for k in 1:total_domains
        # Calculate squared distance first
        dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2
        if dist_sq <= core_radius_sq
            local_at_row[k] += core_dose
        elseif dist_sq <= mid_radius_sq
            dist = sqrt(dist_sq) # Calculate sqrt only when needed
            idx_lookup = searchsortedfirst(impact_vec, dist)
            if idx_lookup == 1
                local_at_row[k] += core_dose # Or dose_vec[1]
            elseif idx_lookup > length(impact_vec)
                local_at_row[k] += dose_vec[end]
            else
                x1, x2 = impact_vec[idx_lookup-1], impact_vec[idx_lookup]
                y1, y2 = dose_vec[idx_lookup-1], dose_vec[idx_lookup]
                interpolated_dose = y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                local_at_row[k] += interpolated_dose
            end
        elseif dist_sq < penumbra_radius_sq # Penumbra region (use < Rp^2)
            local_at_row[k] += Kp / dist_sq
        end
    end

    return local_at_row
end


"""
    setup_dose_rate_simulation_data(irrad_cond::AT, gsm2::GSM2, df_center_x::DataFrame, df_center_y::DataFrame, type_AT::String)

Performs all one-time pre-computations for a dose-rate simulation.

This function calculates the dose look-up table (LUT), flattens the domain coordinates,
and pre-calculates physical constants like squared radii and Kp. It packages all
results into a `NamedTuple` for efficient use in the simulation loop.

# Arguments
- `irrad_cond::AT`: Struct with irradiation conditions (Rp, Rc, Kp).
- `gsm2::GSM2`: Struct with GSM2 model parameters (rd).
- `df_center_x::DataFrame`: DataFrame of domain center x-coordinates.
- `df_center_y::DataFrame`: DataFrame of domain center y-coordinates.
- `type_AT::String`: The amorphous track model type (e.g., "KC").

# Returns
- A `NamedTuple` containing all pre-computed data.
"""
function setup_dose_rate_simulation_data(irrad_cond::AT, gsm2::GSM2, df_center_x::DataFrame, df_center_y::DataFrame, type_AT::String)
    # Extract physical constants from input structs
    Rp = irrad_cond.Rp
    Rc = irrad_cond.Rc
    Kp = irrad_cond.Kp
    Rk = Rp # Assuming Rk = Rp

    # --- Pre-calculate radii and other constants ---
    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2
    mid_radius_sq = (gsm2.rd + 150 * Rc)^2
    penumbra_radius_sq = Rp^2

    # --- Generate the Dose Look-Up Table (LUT) ---
    sim_ = 1000
    if lower_bound_log <= 0
        error("Lower bound for impact_p calculation is non-positive.")
    end
    impact_vec = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)

    # Thread-safe LUT generation
    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]
    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_vec[i], 0.0, Rk)
        _dose, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond, type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_vec = sum(dose_cell_lookup_threads)
    core_dose = dose_vec[1]

    # --- Flatten Domain Coordinates ---
    # More concise way to flatten the DataFrames, excluding the :index column
    dom_x_row = vec(Matrix(df_center_x))
    dom_y_row = vec(Matrix(df_center_y))

    # --- Package all pre-computed data into a NamedTuple for clarity ---
    return (
        dom_x_row=dom_x_row,
        dom_y_row=dom_y_row,
        impact_vec=impact_vec,
        dose_vec=dose_vec,
        core_dose=core_dose,
        core_radius_sq=core_radius_sq,
        mid_radius_sq=mid_radius_sq,
        penumbra_radius_sq=penumbra_radius_sq,
        Kp=Kp
    )
end

"""
    calculate_dose_for_hit(x_cb::Float64, y_cb::Float64, R_beam::Float64, sim_data::NamedTuple)

Calculates the dose deposited in each domain from a single particle hit.

This is a performance-critical function designed to be called many times. It uses
pre-computed data from `setup_dose_rate_simulation_data` to avoid redundant calculations.

# Arguments
- `x_cb::Float64`, `y_cb::Float64`: Center coordinates of the particle beam.
- `R_beam::Float64`: Radius of the particle beam.
- `sim_data::NamedTuple`: The pre-computed data object from `setup_dose_rate_simulation_data`.

# Returns
- A `Vector{Float64}` where each element is the dose deposited in the corresponding domain.
"""
function calculate_dose_for_hit(x_cb::Float64, y_cb::Float64, R_beam::Float64, sim_data)
    # Unpack the pre-computed simulation data for clarity and performance
    (; dom_x_row, dom_y_row, impact_vec, dose_vec, core_dose,
        core_radius_sq, mid_radius_sq, penumbra_radius_sq, Kp) = sim_data

    total_domains = length(dom_x_row)
    local_at_row = zeros(Float64, total_domains)

    # Generate a single particle hit
    x, y = GenerateHit_Circle(x_cb, y_cb, R_beam)

    # Loop over all domains and calculate dose from this single hit
    @inbounds for k in 1:total_domains
        # Calculate squared distance to avoid sqrt
        dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

        if dist_sq <= core_radius_sq
            local_at_row[k] += core_dose
        elseif dist_sq <= mid_radius_sq
            dist = sqrt(dist_sq) # Calculate sqrt only when needed

            idx_lookup = searchsortedfirst(impact_vec, dist)

            if idx_lookup == 1
                local_at_row[k] += core_dose
            elseif idx_lookup > length(impact_vec)
                local_at_row[k] += dose_vec[end]
            else
                # Linear interpolation
                x1, x2 = impact_vec[idx_lookup-1], impact_vec[idx_lookup]
                y1, y2 = dose_vec[idx_lookup-1], dose_vec[idx_lookup]
                interpolated_dose = y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                local_at_row[k] += interpolated_dose
            end
        elseif dist_sq < penumbra_radius_sq
            local_at_row[k] += Kp / dist_sq
        end
    end

    return local_at_row
end


"""
    run_dose_rate_simulation(gsm2::GSM2, irrad::Irrad, ion::Ion, at_start::AT, center_x::Vector{Float64}, center_y::Vector{Float64}, domain::Int, Rn::Float64, type_AT::String)

Runs a complete dose-rate simulation using a Gillespie-like algorithm.

This function encapsulates the setup, pre-computation, and the main simulation loop.
It tracks the evolution of reparable (X) and irreparable (Y) DNA damage over time,
considering repair, lethal events, and damage induction from particle hits.

# Arguments
- `gsm2::GSM2`: GSM2 model parameters (repair rates, domain sizes).
- `irrad::Irrad`: Irradiation parameters (dose, dose rate).
- `ion::Ion`: Ion properties (LET, particle type).
- `at_start::AT`: Amorphous track model parameters.
- `center_x`, `center_y`: Vectors of domain center coordinates.
- `domain::Int`: The number of domains.
- `Rn::Float64`: The radius of the cell nucleus.
- `type_AT::String`: The amorphous track model type (e.g., "KC").

# Returns
- A tuple `(X, Y, status)` where:
  - `X::Vector{Int}`: Final state of the reparable damage vector.
  - `Y::Vector{Int}`: Final state of the irreparable damage vector.
  - `status::Int`: Simulation outcome code:
    - `0`: Finished normally (time limit reached).
    - `1`: Terminated due to a lethal event.
    - `-1`: Terminated due to an error.
"""
function run_dose_rate_simulation(gsm2::GSM2, irrad::Irrad, ion::Ion, at_start::AT, center_x::Vector{Float64}, center_y::Vector{Float64}, domain::Int, Rn::Float64, type_AT::String)

    # --- Initial Setup & Parameter Calculation ---
    n_repeat = floor(Int64, gsm2.Rn / gsm2.rd)

    # Create temporary DataFrames for the setup function
    df_center_x = DataFrame(reshape(center_x, 1, :), :auto)
    df_center_y = DataFrame(reshape(center_y, 1, :), :auto)

    # Calculate damage induction constants
    temp_cell = Cell(0.0, 0.0, 0.0, [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "", Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), [], [], 21., 0.0, [], 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
    kappa_base = 9.0 * calculate_kappa(ion, temp_cell, false) / (n_repeat * domain)
    lambda_base = kappa_base * 1e-3

    # Calculate irradiation time and particle rate
    R_beam = at_start.Rp + Rn
    x_beam, y_beam = 0.0, 0.0
    F = irrad.dose / (1.602e-9 * ion.LET)
    Npar = round(Int, F * (pi * (R_beam)^2 * 1e-8))
    zF = irrad.dose / Npar
    dr = irrad.doserate / zF
    T = irrad.dose / (zF * dr)

    # --- Pre-computation Step ---
    sim_data = setup_dose_rate_simulation_data(at_start, gsm2, df_center_x, df_center_y, type_AT)

    # --- Simulation Initialization ---
    X = vec(zeros(Int64, n_repeat, domain))
    Y = vec(zeros(Int64, n_repeat, domain)) # Correctly initialize Y independently
    current_time = 0.0

    # Pre-allocate work arrays to avoid re-allocation inside the loop
    X_dam = zeros(Int64, n_repeat, domain)
    Y_dam = zeros(Int64, n_repeat, domain)
    len_X = length(X)
    propensities = Vector{Float64}(undef, 3 * len_X + 1)
    prop_view_rX = view(propensities, 1:len_X)
    prop_view_aX = view(propensities, (len_X+1):(2*len_X))
    prop_view_bX = view(propensities, (2*len_X+1):(3*len_X))

    # --- Main Simulation Loop (Gillespie-like algorithm) ---
    while current_time < T
        # --- 1. Calculate Propensities (in-place) ---
        prop_view_rX .= gsm2.r .* X
        prop_view_aX .= gsm2.a .* X
        @inbounds for i in 1:len_X
            prop_view_bX[i] = max(0.0, gsm2.b * X[i] * (X[i] - 1.0))
        end
        propensities[end] = dr
        a0 = sum(propensities)

        if a0 <= 0.0
            @warn "Total propensity is zero. Simulation might be stuck. Exiting."
            break
        end

        # --- 2. Calculate time to next

        r1 = rand()
        current_time += (1 / a0) * log(1 / r1)

        if current_time > T
            # Simulation time has exceeded the total irradiation time.
            # The last event did not fully occur. Return the state before this last step.
            return (X, Y, 0) # Status 0: Finished normally
        end

        # Select which reaction "fires" using an efficient cumulative sum method
        r2 = rand()
        fire_target = r2 * a0

        reac_idx = 0
        cumulative_prop = 0.0
        # This loop is more memory-efficient than creating a full cumsum array
        for i in 1:length(propensities)
            cumulative_prop += propensities[i]
            if cumulative_prop >= fire_target
                reac_idx = i
                break
            end
        end

        if reac_idx == 0
            @error "Could not select a reaction. This should not happen if a0 > 0."
            return (X, Y, -1) # Status -1: Error
        end

        # --- 3. Map reaction index to reaction type ---
        vec_index = 0
        dom = 0

        if reac_idx <= len_X
            vec_index = 1 # Repair reaction
            dom = reac_idx
        elseif reac_idx <= 2 * len_X
            vec_index = 2 # Lethal reaction 'a'
            dom = reac_idx - len_X
        elseif reac_idx <= 3 * len_X
            vec_index = 3 # Lethal reaction 'b'
            dom = reac_idx - 2 * len_X
        else # reac_idx == 3 * len_X + 1
            vec_index = 4 # Particle hit
        end

        # --- 4. Execute the selected reaction ---
        if vec_index == 1
            # Repair reaction (r * X)
            X[dom] -= 1
        elseif vec_index == 2 || vec_index == 3
            # Lethal reaction (a * X or b * X * (X-1))
            return (X, Y, 1) # Return status 1 (cell death)
        elseif vec_index == 4
            # Particle hit reaction (dose rate)

            # Calculate dose from this single particle hit using the lean "application" function
            local_at_row = calculate_dose_for_hit(x_beam, y_beam, R_beam, sim_data)

            # Calculate new damage based on the dose from the hit
            for i in 1:domain
                lambda_X = max(0.0, kappa_base * local_at_row[i])
                lambda_Y = max(0.0, lambda_base * local_at_row[i])
                X_dam[:, i] = rand(Poisson(lambda_X), n_repeat)
                Y_dam[:, i] = rand(Poisson(lambda_Y), n_repeat)
            end

            # Add the new damage to the existing damage vectors
            X .+= vec(X_dam)
            Y .+= vec(Y_dam)
        end
    end # End of while loop

    # If the loop finishes because current_time >= T, it's a normal completion.
    return (X, Y, 0) # Status 0: Finished normally
end

function MC_loop_ions_domain_fast!(x_list::Array{Float64}, y_list::Array{Float64}, irrad_cond::Vector{AT}, gsm2::GSM2,
    df_center_x::DataFrame, df_center_y::DataFrame, at::DataFrame,
    type_AT::String, ion::Ion)

    Rp = irrad_cond[1].Rp
    Rc = irrad_cond[1].Rc
    Kp = irrad_cond[1].Kp
    Rk = Rp # Assuming Rk = Rp as in the original snippet

    lower_bound_log = max(1e-9, gsm2.rd - 10 * Rc)
    core_radius_sq = (gsm2.rd - 10 * Rc)^2 # Squared core boundary for distance check
    mid_radius_sq = (gsm2.rd + 150 * Rc)^2 # Squared mid boundary for distance check
    penumbra_radius_sq = Rp^2             # Squared penumbra boundary

    sim_ = 1000
    if lower_bound_log <= 0
        error("Lower bound for impact_p calculation is non-positive.")
    end
    impact_p = 10 .^ range(log10(lower_bound_log), stop=log10(gsm2.rd + 150 * Rc), length=sim_)
    dose_cell_lookup = zeros(Float64, sim_)

    dose_cell_lookup_threads = [zeros(Float64, sim_) for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:sim_
        tid = Threads.threadid()
        track = Track(impact_p[i], 0.0, Rk)
        _dose, _, Gyr = distribute_dose_domain(0.0, 0.0, gsm2.rd, track, irrad_cond[1], type_AT)
        dose_cell_lookup_threads[tid][i] = Gyr
    end
    dose_cell_lookup = sum(dose_cell_lookup_threads)

    impact_vec = impact_p
    dose_vec = dose_cell_lookup
    core_dose = dose_vec[1] # Dose for distances <= core_radius

    num_domains_per_cell = size(df_center_x, 2) - 1
    num_cells = size(df_center_x, 1)
    total_domains = num_cells * num_domains_per_cell

    dom_x_row = Vector{Float64}(undef, total_domains)
    dom_y_row = Vector{Float64}(undef, total_domains)
    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            dom_x_row[idx] = df_center_x[r, c]
            dom_y_row[idx] = df_center_y[r, c]
            idx += 1
        end
    end

    at_row_accumulators = [zeros(Float64, total_domains) for _ in 1:Threads.maxthreadid()]

    Np = size(x_list, 1)
    println("Monte Carlo Loop ", Np, " particles of ", ion.ion)

    Threads.@threads for ip in 1:Np
        tid = Threads.threadid()
        local_at_row = at_row_accumulators[tid] # Get thread-local accumulator

        x = x_list[ip]
        y = y_list[ip]

        for k in 1:total_domains
            # Calculate squared distance first
            dist_sq = (dom_x_row[k] - x)^2 + (dom_y_row[k] - y)^2

            if dist_sq <= core_radius_sq
                local_at_row[k] += core_dose
            elseif dist_sq <= mid_radius_sq
                dist = sqrt(dist_sq) # Calculate sqrt only when needed

                idx_lookup = searchsortedfirst(impact_vec, dist)

                if idx_lookup == 1
                    local_at_row[k] += core_dose # Or dose_vec[1]
                elseif idx_lookup > length(impact_vec)
                    local_at_row[k] += dose_vec[end]
                else
                    x1, x2 = impact_vec[idx_lookup-1], impact_vec[idx_lookup]
                    y1, y2 = dose_vec[idx_lookup-1], dose_vec[idx_lookup]
                    interpolated_dose = y1 + (y2 - y1) * (dist - x1) / (x2 - x1)
                    local_at_row[k] += interpolated_dose
                end
            elseif dist_sq < penumbra_radius_sq # Penumbra region (use < Rp^2)
                local_at_row[k] += Kp / dist_sq
            end
        end
    end

    final_at_row = sum(at_row_accumulators)

    idx = 1
    for r in 1:num_cells
        for c in 1:num_domains_per_cell
            at[r, c] = final_at_row[idx] # Or += if 'at' had initial values
            idx += 1
        end
    end

    println("Finished Monte Carlo Loop")
end

function MC_loop_copy_dose_domain!(arrayOfCell::Array{Cell}, cell_df::DataFrame, arrayOfCell_single::Array{Cell}, cell_df_single::DataFrame)
    cell_df.dose = [arrayOfCell[i].dose for i in eachindex(arrayOfCell)]
    cell_df.dose_cell = [arrayOfCell[i].dose_cell for i in eachindex(arrayOfCell)]
    # Copy the dose from the single plane to all the other planes
    for i in 1:size(cell_df_single, 1)
        if cell_df_single[i, "dose_cell"] > 0
            idx = findall((cell_df.x .== arrayOfCell_single[i].x) .& (cell_df.y .== arrayOfCell_single[i].y) .& (cell_df.is_cell .== 1))
            for j in idx
                cell_df.dose[j] = cell_df_single.dose[i]
                cell_df.dose_cell[j] = cell_df_single.dose_cell[i]
                arrayOfCell[j].dose = arrayOfCell_single[i].dose
                arrayOfCell[j].dose_cell = arrayOfCell_single[i].dose_cell
            end
        end
    end
end

using DataFrames

# Assume AT, Cell, GSM2 etc. structs and other necessary functions are defined elsewhere.
# Assume cell_df, at_single, at are pre-populated DataFrames as described.

"""
    MC_loop_copy_dose_domain_fast!(cell_df::DataFrame, at_single::DataFrame, at::DataFrame)

Copies the calculated domain doses from representative cells (in `at_single`, one per unique x,y)
to all cells sharing the same (x, y) coordinates in the target `at` DataFrame AND
updates the `cell_df.dose` (vector) and `cell_df.dose_cell` (scalar mean) columns.
If `dose` or `dose_cell` columns do not exist in `cell_df`, they will be created.

# Arguments
- `cell_df::DataFrame`: DataFrame containing information for ALL cells, including `index`, `x`, `y`. Will be modified in-place.
- `at_single::DataFrame`: DataFrame containing calculated domain doses for REPRESENTATIVE cells (one per unique x,y). Must have an `index` column linking to `cell_df`. Domain columns (e.g., `center_1`, ...) hold the doses.
- `at::DataFrame`: Target DataFrame, structured like `at_single` but for ALL cells. This DataFrame will be modified in-place.
"""




"""
    MC_loop_damage_domain!(ion::Ion, arrayOfCell::Array{Cell}, cell_df::DataFrame)

    This function computes the damage distribution in a cell array for a given ion, taking into account the cell properties and dose.

    Arguments:
    - `ion::Ion`: Ion object containing the ion properties
    - `arrayOfCell::Array{Cell}`: Array of Cell objects
    - `cell_df::DataFrame`: DataFrame containing the cell data

    Returns:
    - Nothing
"""
function MC_loop_damage_domain!(ion::Ion, arrayOfCell::Array{Cell}, cell_df::DataFrame)

    cell_df.dam_X_dom .= getfield.(arrayOfCell, :dam_X_dom)
    cell_df.dam_Y_dom .= getfield.(arrayOfCell, :dam_Y_dom)

    @inbounds for i in 1:size(cell_df, 1)
        # Check if the cell has received any dose
        if cell_df[i, "dose_cell"] > 0
            # Calculate the kappa and lambda values for the cell
            kappa_DSB = (9 * calculate_kappa(ion, arrayOfCell[i], false)) / size(arrayOfCell[i].dam_X_dom, 1)
            kappa_p = kappa_DSB .* repeat(arrayOfCell[i].dose, floor(Int, arrayOfCell[i].r / arrayOfCell[i].rd_gsm2))
            lambda_p = kappa_p * 10^-3

            # Generate Poisson distributions for x0d and y0d
            x0d = rand.(Poisson.(kappa_p))
            y0d = rand.(Poisson.(lambda_p))

            # Update the damage distribution in the cell
            arrayOfCell[i].dam_X_dom .= x0d
            arrayOfCell[i].dam_Y_dom .= y0d

            # Update the DataFrame with the damage distribution
            cell_df[i, :dam_X_dom] .= x0d
            cell_df[i, :dam_Y_dom] .= y0d
        end
    end
end

"""
    MC_loop_damage_domain_fast!(ion::Ion, cell_df::DataFrame, gsm2::GSM2)

Calculates simulated damage vectors for each cell based on its domain dose vector.
For each element `d` in the input `cell_df.dose` vector, it generates
`n_repeat = floor(Int64, gsm2.Rn / gsm2.rd)` Poisson random numbers twice:
1. Using mean `kappa * d` (stored in `:dam_X_dom`)
2. Using mean `lambda * d` (where lambda = kappa * 1e-3, stored in `:dam_Y_dom`)
The results for each type are concatenated across dose elements.
Creates `:dam_X_dom` and `:dam_Y_dom` columns in `cell_df` if they don't exist.

# Arguments
- `ion::Ion`: Ion object containing ion properties (used for kappa calculation).
- `cell_df::DataFrame`: DataFrame containing cell data, must include `.dose` (Vector{Float64}), `.O` (Float64 for kappa), and ideally `.dose_cell` (Float64 for checking if dose > 0). Will be modified in-place.
- `gsm2::GSM2`: GSM2 parameters, used for `Rn` and `rd`.
"""
function MC_loop_damage_domain_fast!(ion::Ion, cell_df::DataFrame, gsm2::GSM2)
    println("Calculating fast domain damage (X and Y)...")

    num_cells = nrow(cell_df)
    n_repeat = floor(Int64, gsm2.Rn / gsm2.rd)

    # --- Input Validation & Column Initialization ---
    if !hasproperty(cell_df, :dose)
        error("cell_df is missing the required 'dose' column (Vector{Float64}).")
    end
    if !hasproperty(cell_df, :O) # Assuming calculate_kappa needs Oxygen
        error("cell_df is missing the required 'O' column (Float64).")
    end
    # Check if dose column holds vectors
    if num_cells > 0 && !isempty(cell_df.dose) && !(eltype(cell_df.dose) <: AbstractVector)
        error("cell_df.dose column does not hold Vectors.")
    end
    # Check if dose_cell exists for the check below
    has_dose_cell_check = hasproperty(cell_df, :dose_cell)
    if !has_dose_cell_check
        @warn "cell_df does not have :dose_cell column. Damage calculation will proceed for all rows, assuming dose > 0 if dose vector is non-empty."
    end

    # Determine expected length of the output vectors
    expected_len = 0
    first_valid_dose_idx = findfirst(x -> !isempty(x), cell_df.dose)
    if first_valid_dose_idx !== nothing
        expected_len = length(cell_df.dose[first_valid_dose_idx]) * n_repeat
    else
        @warn "Could not determine expected length for damage vectors as all dose vectors seem empty."
    end

    # Initialize the output columns if they don't exist or have wrong type/size
    col_names = [:dam_X_dom, :dam_Y_dom]
    for col_name in col_names
        if !hasproperty(cell_df, col_name)
            println("Initializing cell_df.$col_name column")
            cell_df[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        elseif !(eltype(cell_df[!, col_name]) <: AbstractVector{<:Integer})
            @warn "cell_df.$col_name column exists but has wrong type. Reinitializing."
            cell_df[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        elseif num_cells > 0 && expected_len > 0 && length(cell_df[1, col_name]) != expected_len
            @warn "cell_df.$col_name vectors have incorrect size. Reinitializing."
            cell_df[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        end
    end
    # --- End Initialization ---


    # --- Main Loop ---
    # Consider @Threads.threads if beneficial and safe for DataFrame modification
    for i in ProgressBar(1:num_cells)
        # Check if calculation is needed
        calculate_damage_for_row = true
        if has_dose_cell_check && cell_df.dose_cell[i] <= 0.0
            calculate_damage_for_row = false
        elseif isempty(cell_df.dose[i])
            calculate_damage_for_row = false
        end

        if !calculate_damage_for_row
            # Assign zeros or empty vector if no dose / empty dose vector
            for col_name in col_names
                if hasproperty(cell_df, col_name) # Ensure column exists
                    current_len = expected_len > 0 ? expected_len : (isempty(cell_df[i, col_name]) ? 0 : length(cell_df[i, col_name]))
                    if current_len > 0
                        cell_df[i, col_name] .= 0 # Fill existing vector with zeros
                    else
                        cell_df[i, col_name] = Int64[] # Assign empty if length is 0
                    end
                end
            end
            continue # Skip to next row
        end

        dose_vector = cell_df.dose[i]
        O_value = cell_df.O[i]

        # Create a minimal temporary Cell object for calculate_kappa
        # Adapt if calculate_kappa needs more fields than just .O
        temp_cell = Cell(0.0, 0.0, 0.0, [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "", Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), [], [], O_value, 0.0, [], 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

        # Calculate base kappa and lambda rates
        kappa_base = 0.0
        lambda_base = 0.0
        try
            # Factor of 9 based on original MC_loop_damage_domain!
            # Division by expected_len might be needed if kappa is per *total* dose?
            # Let's follow the structure from MC_loop_damage_domain! more closely:
            # kappa_DSB = (9 * calculate_kappa(ion, temp_cell, false)) / size(arrayOfCell[i].dam_X_dom, 1) # This division seems problematic if size isn't constant or known here.
            # Let's assume calculate_kappa gives yield per Gy, and the dose vector elements are Gy.
            # The division by size might have been intended to normalize kappa_DSB if it represented total yield over all domains.
            # Let's proceed *without* the division first, assuming kappa is per Gy per domain element.
            kappa_base = 9.0 * calculate_kappa(ion, temp_cell, true) / (n_repeat * size(dose_vector, 1))
            lambda_base = kappa_base * 1e-3 # Factor from original MC_loop_damage_domain!
        catch e
            @error "Error calculating kappa for row $i: $e"
            cell_df[i, :dam_X_dom] .= 0
            cell_df[i, :dam_Y_dom] .= 0
            continue
        end

        # Preallocate/get references to the result vectors for the row
        row_expected_len = length(dose_vector) * n_repeat
        # Ensure the target vectors have the correct size for this specific row
        if length(cell_df[i, :dam_X_dom]) != row_expected_len
            cell_df[i, :dam_X_dom] = Vector{Int64}(undef, row_expected_len)
        end
        if length(cell_df[i, :dam_Y_dom]) != row_expected_len
            cell_df[i, :dam_Y_dom] = Vector{Int64}(undef, row_expected_len)
        end
        damage_vector_X_row = cell_df[i, :dam_X_dom] # Reference to DF column's vector
        damage_vector_Y_row = cell_df[i, :dam_Y_dom] # Reference to DF column's vector
        current_idx = 1

        for d in dose_vector
            lambda_X = max(0.0, kappa_base * d) # Mean for X damage
            lambda_Y = max(0.0, lambda_base * d) # Mean for Y damage

            # Generate n_repeat Poisson numbers for X
            poisson_samples_X = zeros(Int64, n_repeat)
            try
                if lambda_X > 0
                    poisson_samples_X = rand(Poisson(lambda_X), n_repeat)
                end
            catch e
                @warn "Error generating Poisson(X) samples for row $i, dose $d, lambda_X $lambda_X: $e. Using zeros."
            end

            # Generate n_repeat Poisson numbers for Y
            poisson_samples_Y = zeros(Int64, n_repeat)
            try
                if lambda_Y > 0
                    poisson_samples_Y = rand(Poisson(lambda_Y), n_repeat)
                end
            catch e
                @warn "Error generating Poisson(Y) samples for row $i, dose $d, lambda_Y $lambda_Y: $e. Using zeros."
            end

            # Fill the preallocated vector segments
            stop_idx = current_idx + n_repeat - 1
            if stop_idx <= length(damage_vector_X_row) # Check bounds once (lengths are same)
                damage_vector_X_row[current_idx:stop_idx] = poisson_samples_X
                damage_vector_Y_row[current_idx:stop_idx] = poisson_samples_Y
                current_idx = stop_idx + 1
            else
                @error "Index out of bounds error for row $i. Expected length $row_expected_len, trying to write up to $stop_idx."
                break # Stop processing this row
            end
        end
        # Vectors in DataFrame are updated directly via the references

    end # End of loop over rows

    println("Finished calculating fast domain damage (X and Y).")
end

function MC_loop_damage_domain_fast_NTCP!(ion::Ion, cell_df_::DataFrame, gsm2::GSM2)
    println("Calculating fast domain damage (X and Y)...")

    num_cells = nrow(cell_df_)
    n_repeat = floor(Int64, gsm2.Rn / gsm2.rd)

    # --- Input Validation & Column Initialization ---
    if !hasproperty(cell_df_, :dose)
        error("cell_df_ is missing the required 'dose' column (Vector{Float64}).")
    end
    if !hasproperty(cell_df_, :O) # Assuming calculate_kappa needs Oxygen
        error("cell_df_ is missing the required 'O' column (Float64).")
    end
    # Check if dose column holds vectors
    if num_cells > 0 && !isempty(cell_df_.dose) && !(eltype(cell_df_.dose) <: AbstractVector)
        error("cell_df_.dose column does not hold Vectors.")
    end
    # Check if dose_cell exists for the check below
    has_dose_cell_check = hasproperty(cell_df_, :dose_cell)
    if !has_dose_cell_check
        @warn "cell_df_ does not have :dose_cell column. Damage calculation will proceed for all rows, assuming dose > 0 if dose vector is non-empty."
    end

    # Determine expected length of the output vectors
    expected_len = 0
    first_valid_dose_idx = findfirst(x -> !isempty(x), cell_df_.dose)
    if first_valid_dose_idx !== nothing
        expected_len = length(cell_df_.dose[first_valid_dose_idx]) * n_repeat
    else
        @warn "Could not determine expected length for damage vectors as all dose vectors seem empty."
    end

    # Initialize the output columns if they don't exist or have wrong type/size
    col_names = [:dam_X_dom, :dam_Y_dom]
    for col_name in col_names
        if !hasproperty(cell_df_, col_name)
            println("Initializing cell_df_.$col_name column")
            cell_df_[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        elseif !(eltype(cell_df_[!, col_name]) <: AbstractVector{<:Integer})
            @warn "cell_df_.$col_name column exists but has wrong type. Reinitializing."
            cell_df_[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        elseif num_cells > 0 && expected_len > 0 && length(cell_df_[1, col_name]) != expected_len
            @warn "cell_df_.$col_name vectors have incorrect size. Reinitializing."
            cell_df_[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        end
    end
    # --- End Initialization ---


    # --- Main Loop ---
    # Consider @Threads.threads if beneficial and safe for DataFrame modification
    for i in ProgressBar(1:num_cells)
        # Check if calculation is needed
        calculate_damage_for_row = true
        if has_dose_cell_check && cell_df_.dose_cell[i] <= 0.0
            calculate_damage_for_row = false
        elseif isempty(cell_df_.dose[i])
            calculate_damage_for_row = false
        end

        if !calculate_damage_for_row
            # Assign zeros or empty vector if no dose / empty dose vector
            for col_name in col_names
                if hasproperty(cell_df_, col_name) # Ensure column exists
                    current_len = expected_len > 0 ? expected_len : (isempty(cell_df_[i, col_name]) ? 0 : length(cell_df_[i, col_name]))
                    if current_len > 0
                        cell_df_[i, col_name] .= 0 # Fill existing vector with zeros
                    else
                        cell_df_[i, col_name] = Int64[] # Assign empty if length is 0
                    end
                end
            end
            continue # Skip to next row
        end

        dose_vector = cell_df_.dose[i]
        O_value = cell_df_.O[i]

        # Create a minimal temporary Cell object for calculate_kappa
        # Adapt if calculate_kappa needs more fields than just .O
        temp_cell = Cell(0.0, 0.0, 0.0, [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "", Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), [], [], O_value, 0.0, [], 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

        # Calculate base kappa and lambda rates
        kappa_base = 0.0
        lambda_base = 0.0
        try
            # Factor of 9 based on original MC_loop_damage_domain!
            # Division by expected_len might be needed if kappa is per *total* dose?
            # Let's follow the structure from MC_loop_damage_domain! more closely:
            # kappa_DSB = (9 * calculate_kappa(ion, temp_cell, false)) / size(arrayOfCell[i].dam_X_dom, 1) # This division seems problematic if size isn't constant or known here.
            # Let's assume calculate_kappa gives yield per Gy, and the dose vector elements are Gy.
            # The division by size might have been intended to normalize kappa_DSB if it represented total yield over all domains.
            # Let's proceed *without* the division first, assuming kappa is per Gy per domain element.
            kappa_base = 9.0 * calculate_kappa(ion, temp_cell, true) / (n_repeat * size(dose_vector, 1))
            lambda_base = kappa_base * 1e-3 # Factor from original MC_loop_damage_domain!
        catch e
            @error "Error calculating kappa for row $i: $e"
            cell_df_[i, :dam_X_dom] .= 0
            cell_df_[i, :dam_Y_dom] .= 0
            continue
        end

        # Preallocate/get references to the result vectors for the row
        row_expected_len = length(dose_vector) * n_repeat
        # Ensure the target vectors have the correct size for this specific row
        if length(cell_df_[i, :dam_X_dom]) != row_expected_len
            cell_df_[i, :dam_X_dom] = Vector{Int64}(undef, row_expected_len)
        end
        if length(cell_df_[i, :dam_Y_dom]) != row_expected_len
            cell_df_[i, :dam_Y_dom] = Vector{Int64}(undef, row_expected_len)
        end
        damage_vector_X_row = cell_df_[i, :dam_X_dom] # Reference to DF column's vector
        damage_vector_Y_row = cell_df_[i, :dam_Y_dom] # Reference to DF column's vector
        current_idx = 1

        for d in dose_vector
            lambda_X = max(0.0, kappa_base * d) # Mean for X damage
            lambda_Y = max(0.0, lambda_base * d) # Mean for Y damage

            # Generate n_repeat Poisson numbers for X
            poisson_samples_X = zeros(Int64, n_repeat)
            try
                if lambda_X > 0
                    poisson_samples_X = rand(Poisson(lambda_X), n_repeat)
                end
            catch e
                @warn "Error generating Poisson(X) samples for row $i, dose $d, lambda_X $lambda_X: $e. Using zeros."
            end

            # Generate n_repeat Poisson numbers for Y
            poisson_samples_Y = zeros(Int64, n_repeat)
            try
                if lambda_Y > 0
                    poisson_samples_Y = rand(Poisson(lambda_Y), n_repeat)
                end
            catch e
                @warn "Error generating Poisson(Y) samples for row $i, dose $d, lambda_Y $lambda_Y: $e. Using zeros."
            end

            # Fill the preallocated vector segments
            stop_idx = current_idx + n_repeat - 1
            if stop_idx <= length(damage_vector_X_row) # Check bounds once (lengths are same)
                damage_vector_X_row[current_idx:stop_idx] = poisson_samples_X
                damage_vector_Y_row[current_idx:stop_idx] = poisson_samples_Y
                current_idx = stop_idx + 1
            else
                @error "Index out of bounds error for row $i. Expected length $row_expected_len, trying to write up to $stop_idx."
                break # Stop processing this row
            end
        end
        # Vectors in DataFrame are updated directly via the references

    end # End of loop over rows

    println("Finished calculating fast domain damage (X and Y).")
end

function MC_loop_damage_singlecell_fast!(ion::Ion, cell_df::DataFrame, gsm2::GSM2)
    println("Calculating fast domain damage (X and Y)...")

    num_cells = nrow(cell_df)
    n_repeat = floor(Int64, gsm2.Rn / gsm2.rd)

    # --- Input Validation & Column Initialization ---
    if !hasproperty(cell_df, :dose)
        error("cell_df is missing the required 'dose' column (Vector{Float64}).")
    end
    if !hasproperty(cell_df, :O) # Assuming calculate_kappa needs Oxygen
        error("cell_df is missing the required 'O' column (Float64).")
    end
    # Check if dose column holds vectors
    if num_cells > 0 && !isempty(cell_df.dose) && !(eltype(cell_df.dose) <: AbstractVector)
        error("cell_df.dose column does not hold Vectors.")
    end
    # Check if dose_cell exists for the check below
    has_dose_cell_check = hasproperty(cell_df, :dose_cell)
    if !has_dose_cell_check
        @warn "cell_df does not have :dose_cell column. Damage calculation will proceed for all rows, assuming dose > 0 if dose vector is non-empty."
    end

    # Determine expected length of the output vectors
    expected_len = 0
    first_valid_dose_idx = findfirst(x -> !isempty(x), cell_df.dose)
    if first_valid_dose_idx !== nothing
        expected_len = length(cell_df.dose[first_valid_dose_idx]) * n_repeat
    else
        @warn "Could not determine expected length for damage vectors as all dose vectors seem empty."
    end

    # Initialize the output columns if they don't exist or have wrong type/size
    col_names = [:dam_X_dom, :dam_Y_dom]
    for col_name in col_names
        if !hasproperty(cell_df, col_name)
            println("Initializing cell_df.$col_name column")
            cell_df[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        elseif !(eltype(cell_df[!, col_name]) <: AbstractVector{<:Integer})
            @warn "cell_df.$col_name column exists but has wrong type. Reinitializing."
            cell_df[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        elseif num_cells > 0 && expected_len > 0 && length(cell_df[1, col_name]) != expected_len
            @warn "cell_df.$col_name vectors have incorrect size. Reinitializing."
            cell_df[!, col_name] = [zeros(Int64, expected_len) for _ in 1:num_cells]
        end
    end
    # --- End Initialization ---


    # --- Main Loop ---
    # Consider @Threads.threads if beneficial and safe for DataFrame modification
    for i in ProgressBar(1:num_cells)
        # Check if calculation is needed
        calculate_damage_for_row = true
        if has_dose_cell_check && cell_df.dose_cell[i] <= 0.0
            calculate_damage_for_row = false
        elseif isempty(cell_df.dose[i])
            calculate_damage_for_row = false
        end

        if !calculate_damage_for_row
            # Assign zeros or empty vector if no dose / empty dose vector
            for col_name in col_names
                if hasproperty(cell_df, col_name) # Ensure column exists
                    current_len = expected_len > 0 ? expected_len : (isempty(cell_df[i, col_name]) ? 0 : length(cell_df[i, col_name]))
                    if current_len > 0
                        cell_df[i, col_name] .= 0 # Fill existing vector with zeros
                    else
                        cell_df[i, col_name] = Int64[] # Assign empty if length is 0
                    end
                end
            end
            continue # Skip to next row
        end

        dose_vector = cell_df.dose[i]
        O_value = cell_df.O[i]

        # Create a minimal temporary Cell object for calculate_kappa
        # Adapt if calculate_kappa needs more fields than just .O
        temp_cell = Cell(0.0, 0.0, 0.0, [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "", Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), [], [], O_value, 0.0, [], 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)

        # Calculate base kappa and lambda rates
        kappa_base = 0.0
        lambda_base = 0.0
        try
            # Factor of 9 based on original MC_loop_damage_domain!
            # Division by expected_len might be needed if kappa is per *total* dose?
            # Let's follow the structure from MC_loop_damage_domain! more closely:
            # kappa_DSB = (9 * calculate_kappa(ion, temp_cell, false)) / size(arrayOfCell[i].dam_X_dom, 1) # This division seems problematic if size isn't constant or known here.
            # Let's assume calculate_kappa gives yield per Gy, and the dose vector elements are Gy.
            # The division by size might have been intended to normalize kappa_DSB if it represented total yield over all domains.
            # Let's proceed *without* the division first, assuming kappa is per Gy per domain element.
            kappa_base = 9.0 * calculate_kappa(ion, temp_cell, true) / (n_repeat * size(dose_vector, 1))
            lambda_base = kappa_base * 1e-3 # Factor from original MC_loop_damage_domain!
        catch e
            @error "Error calculating kappa for row $i: $e"
            cell_df[i, :dam_X_dom] .= 0
            cell_df[i, :dam_Y_dom] .= 0
            continue
        end

        # Preallocate/get references to the result vectors for the row
        row_expected_len = length(dose_vector) * n_repeat
        # Ensure the target vectors have the correct size for this specific row
        if length(cell_df[i, :dam_X_dom]) != row_expected_len
            cell_df[i, :dam_X_dom] = Vector{Int64}(undef, row_expected_len)
        end
        if length(cell_df[i, :dam_Y_dom]) != row_expected_len
            cell_df[i, :dam_Y_dom] = Vector{Int64}(undef, row_expected_len)
        end
        damage_vector_X_row = cell_df[i, :dam_X_dom] # Reference to DF column's vector
        damage_vector_Y_row = cell_df[i, :dam_Y_dom] # Reference to DF column's vector
        current_idx = 1

        for d in dose_vector
            lambda_X = max(0.0, kappa_base * d) # Mean for X damage
            lambda_Y = max(0.0, lambda_base * d) # Mean for Y damage

            # Generate n_repeat Poisson numbers for X
            poisson_samples_X = zeros(Int64, n_repeat)
            try
                if lambda_X > 0
                    poisson_samples_X = rand(Poisson(lambda_X), n_repeat)
                end
            catch e
                @warn "Error generating Poisson(X) samples for row $i, dose $d, lambda_X $lambda_X: $e. Using zeros."
            end

            # Generate n_repeat Poisson numbers for Y
            poisson_samples_Y = zeros(Int64, n_repeat)
            try
                if lambda_Y > 0
                    poisson_samples_Y = rand(Poisson(lambda_Y), n_repeat)
                end
            catch e
                @warn "Error generating Poisson(Y) samples for row $i, dose $d, lambda_Y $lambda_Y: $e. Using zeros."
            end

            # Fill the preallocated vector segments
            stop_idx = current_idx + n_repeat - 1
            if stop_idx <= length(damage_vector_X_row) # Check bounds once (lengths are same)
                damage_vector_X_row[current_idx:stop_idx] = poisson_samples_X
                damage_vector_Y_row[current_idx:stop_idx] = poisson_samples_Y
                current_idx = stop_idx + 1
            else
                @error "Index out of bounds error for row $i. Expected length $row_expected_len, trying to write up to $stop_idx."
                break # Stop processing this row
            end
        end
        # Vectors in DataFrame are updated directly via the references

    end # End of loop over rows

    println("Finished calculating fast domain damage (X and Y).")
end


function residual_energy_after_distance(E_u::Float64, Z::Int, A::Int, x_um::Float64, ion::String, sp::Dict; dx_um=0.1)
    # Constants
    K = 0.307                  # MeV·cm²/mol
    me = 0.511                 # MeV
    c = 3e10                   # cm/s
    I = 75e-6                  # MeV (mean excitation energy of water)
    ρ = 1.0                    # g/cm³ (water)
    Z_med = 7.42               # effective Z of water
    A_med = 18.015             # g/mol

    # Particle rest mass in MeV
    M = A * 931.494

    # Convert distances from micrometers to cm
    total_distance_cm = x_um * 1e-4
    step_cm = dx_um * 1e-4

    # Initialize energy (total kinetic energy in MeV)
    E_total = E_u * A

    # Function to calculate stopping power at current energy
    function stopping_power(E::Float64)
        γ = (E + M) / M
        β2 = 1 - 1 / γ^2
        γ2 = γ^2
        β = sqrt(β2)

        me_over_M = me / M
        Tmax = (2 * me * β2 * γ2) / (1 + 2 * γ * me_over_M + me_over_M^2)

        arg = (2 * me * β2 * γ2 * Tmax) / I^2
        if arg <= 0 || β2 <= 0
            return 0.0
        end

        Zeff = Z * (1 - exp(-125 * β * Z^(-2 / 3)))
        dEdx = K * (Z_med / A_med) * (Zeff^2 / β2) * (log(arg) - 2 * β2) * ρ

        return dEdx
    end

    distance_covered = 0.0
    while distance_covered < total_distance_cm && E_total > 0
        dEdx = stopping_power(E_total)
        dE = dEdx * step_cm

        if dE < 0
            E_total = 0.
            break
        end

        # Update energy (ensure energy does not go below zero)
        E_total = max(E_total - dE, 0.0)
        distance_covered += step_cm
    end

    E = E_total / A

    if E == 0.0
        LET = 0.0
    else
        LET = linear_interpolation(ion, E, sp)
    end

    return E, LET
end






function get_R_approx(ion::AT, type_AT::String, th::Float64)
    lstep = 10^5
    rdd = Array{Float64}(undef, 0)
    steps = 10 .^ collect(range(log10(10^-9), stop=log10(ion.Rp), length=lstep))
    for i in steps
        push!(rdd, GetRadialLinearDose(i, ion, type_AT))
    end
    index = findfirst(x -> x < th, rdd)
    value = index !== nothing ? steps[index] : (ion.Rp)
    return value
end



##################################################################################################
################################# FUNCTIONS voxels ################################################
function CreationArrayVoxels(X_box::Float64, X_voxel::Float64, arrayOfCell::Array{Cell,1})
    N_sideVox = convert(Int64, floor(2 * X_box / (X_voxel)))
    x_range = y_range = z_range = AbstractRange{Float64}
    ### Create and initialize the VoxArray
    VoxArray = Array{Voxel,3}(undef, N_sideVox, N_sideVox, N_sideVox)
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                VoxArray[i, j, k] = Voxel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1., 0., [])
            end
        end
    end
    #### Fill the array of Voxel with cells within the x,y,z range ####
    for cell in arrayOfCell
        for i in 1:N_sideVox
            x_range = range(start=-X_box + (i - 1) * X_voxel, stop=-X_box + i * X_voxel)
            for j in 1:N_sideVox
                y_range = range(start=-X_box + (j - 1) * X_voxel, stop=-X_box + j * X_voxel)
                for k in 1:N_sideVox
                    z_range = range(start=-X_box + (k - 1) * X_voxel, stop=-X_box + k * X_voxel)
                    VoxArray[i, j, k].xmin = -X_box + (i - 1) * X_voxel
                    VoxArray[i, j, k].xmax = -X_box + i * X_voxel
                    VoxArray[i, j, k].ymin = -X_box + (j - 1) * X_voxel
                    VoxArray[i, j, k].ymax = -X_box + j * X_voxel
                    VoxArray[i, j, k].zmin = -X_box + (k - 1) * X_voxel
                    VoxArray[i, j, k].zmax = -X_box + k * X_voxel
                    if (cell.x in x_range) && (cell.y in y_range) && (cell.z in z_range)
                        push!(VoxArray[i, j, k].CellList, cell)
                    end
                end
            end
        end
    end
    return N_sideVox, VoxArray
end

function CreationArrayVoxels_NTCP(X_box::Float64, X_voxel::Float64)
    N_sideVox = convert(Int64, floor(2 * X_box / (X_voxel)))
    x_range = y_range = z_range = AbstractRange{Float64}
    ### Create and initialize the VoxArray
    VoxArray = Array{Voxel,3}(undef, N_sideVox, N_sideVox, N_sideVox)
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                VoxArray[i, j, k] = Voxel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1., 0.)
            end
        end
    end
    #### Fill the array of Voxel with cells within the x,y,z range ####
    #for cell in arrayOfCell 
    for i in 1:N_sideVox
        x_range = range(start=-X_box + (i - 1) * X_voxel, stop=-X_box + i * X_voxel)
        for j in 1:N_sideVox
            y_range = range(start=-X_box + (j - 1) * X_voxel, stop=-X_box + j * X_voxel)
            for k in 1:N_sideVox
                z_range = range(start=-X_box + (k - 1) * X_voxel, stop=-X_box + k * X_voxel)
                VoxArray[i, j, k].xmin = -X_box + (i - 1) * X_voxel
                VoxArray[i, j, k].xmax = -X_box + i * X_voxel
                VoxArray[i, j, k].ymin = -X_box + (j - 1) * X_voxel
                VoxArray[i, j, k].ymax = -X_box + j * X_voxel
                VoxArray[i, j, k].zmin = -X_box + (k - 1) * X_voxel
                VoxArray[i, j, k].zmax = -X_box + k * X_voxel
                #if (cell.x in x_range) && (cell.y in y_range) && (cell.z in z_range)
                #    push!(VoxArray[i,j,k].CellList,cell)
                #end
            end
        end
    end
    #end    
    return N_sideVox, VoxArray
end

function Survival_Voxels_ijk!(i, j, k, VoxArray::Array{Voxel,3}, X_voxel::Float64)
    SPcells = 1.
    SumDose = 0.
    r_nuc = VoxArray[i, j, k].CellList[1].r
    NumbOfCellsinVox = size(VoxArray[i, j, k].CellList)[1]
    for cell in VoxArray[i, j, k].CellList
        #if rand(Uniform(0,1)) < cell.SP
        SPcells *= (1. - cell.SP) #Francesco
        #SPcells *= cell.SP;
        #end
        SumDose += cell.Dose
    end
    ###Probability of survival voxel
    VoxArray[i, j, k].ni = SPcells
    #VoxArray[i,j,k].Dose = ((SumDose*2*π*r_nuc^3)*NumbOfCellsinVox)/(X_voxel^3);
    VoxArray[i, j, k].Dose = SumDose / NumbOfCellsinVox
end

function Survival_Voxels_ijk_NTCP!(i, j, k, VoxArray::Array{Voxel,3}, X_voxel::Float64, cell_df::DataFrame)

    filtered_df = cell_df[(cell_df.i_voxel_x.==i).&(cell_df.i_voxel_y.==j).&(cell_df.i_voxel_z.==k), :]
    SPcells = prod(1.0 .- filtered_df.sp)
    SumDose = sum(filtered_df.dose_cell)
    NumbOfCellsinVox = size(filtered_df.sp)[1]

    ###Probability of survival voxel
    VoxArray[i, j, k].ni = SPcells
    #VoxArray[i,j,k].Dose = ((SumDose*2*π*r_nuc^3)*NumbOfCellsinVox)/(X_voxel^3);
    VoxArray[i, j, k].Dose = SumDose / NumbOfCellsinVox
end

function Survival_Voxels_ijk_old!(i, j, k, VoxArray::Array{Voxel,3}, X_voxel::Float64)
    ProdSPcells = 1.
    SumDose = 0.
    R = VoxArray[i, j, k].CellList[1].R
    NumbOfCellsinVox = size(VoxArray[i, j, k].CellList)[1]
    for cell in VoxArray[i, j, k].CellList
        ProdSPcells *= cell.SP
        SumDose += cell.Dose
    end
    ###Probability of survival voxel
    VoxArray[i, j, k].ni = ProdSPcells^NumbOfCellsinVox###########3TO CHECK        
    VoxArray[i, j, k].Dose = SumDose * 2 * π * R^3 * NumbOfCellsinVox / X_voxel^3
end

############### Fill the voxel array with the SP for each voxel ##############################
function Survival_Voxels!(VoxArray::Array{Voxel,3}, X_voxel::Float64)

    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                Survival_Voxels_ijk!(i, j, k, VoxArray, X_voxel)

                #niS = VoxArray[i,j,k].ni;
                #println(" ", niS)
            end
            #println("\n")
        end
        #println("\n")
    end
    return VoxArray
end
############### Fill the voxel array with the SP for each voxel ##############################
function Survival_Voxels_NTCP!(VoxArray::Array{Voxel,3}, X_voxel::Float64, cell_df::DataFrame)

    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                Survival_Voxels_ijk_NTCP!(i, j, k, VoxArray, X_voxel, cell_df)

                #niS = VoxArray[i,j,k].ni;
                #println(" ", niS)
            end
            #println("\n")
        end
        #println("\n")
    end
    return VoxArray
end
######################################################################################################################
######################################################################################################################
####################################################Functions (Normal) Tumor Control Parameter########################
function TCP_Box(VoxArray::Array{Voxel,3}, X_voxel::Float64, X_box::Float64, N_sideVox::Int64)
    sumProba = 1.
    sumDose = 0.
    Voxel_Volume = X_voxel * X_voxel * X_voxel / (8 * X_box * X_box * X_box) ###in mm^3 
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                sumProba *= VoxArray[i, j, k].ni
                sumDose += VoxArray[i, j, k].Dose
            end
        end
    end
    TCP = sumProba
    DOSE = sumDose / (N_sideVox * N_sideVox * N_sideVox)
    return TCP, DOSE
end
######## There is still some weird stuf about

function TCP_Circle(cell_df::DataFrame)

    filtered_df = cell_df[cell_df.is_cell.==1, :]
    TCP = prod(1.0 .- filtered_df.sp)
    DOSE = mean(filtered_df.dose_cell)

    return TCP, DOSE
end

function NTCP_Box(VoxArray::Array{Voxel,3}, X_voxel::Float64, X_box::Float64, N_sideVox::Int64, sj::Float64)
    Prod = 1.
    Voxel_Volume = X_voxel * X_voxel * X_voxel / (8 * X_box * X_box * X_box) ###in mm^3 
    #Rel_Vol = 1/(N_sideVox*N_sideVox*N_sideVox)
    # println("Voxel_Volume ",Voxel_Volume)
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                sijk = VoxArray[i, j, k].ni
                Prod *= (1 - (sijk)^(sj))
                #Prod *= ((1 - (sijk)^(1/sj))^Rel_Vol)  ##here noyt sure if 1-sijk or simply sijk; sijk is the propbaility of dying, i.e. 1 - SF
            end
        end
    end
    NTCP = (1 - Prod)^(1 / sj)   ##the first 1 - shouldn't be there but I get the opposite values if I don't use it
    return NTCP
end

function NTCP_Box_PI(VoxArray::Array{Voxel,3}, N_sideVox::Int64, sj::Float64, m_FSU::Int64, n_FSU::Int64, PI::Float64)
    Prod = 1.
    sumDose = 0.
    #Voxel_Volume=X_voxel*X_voxel*X_voxel/(8*X_box*X_box*X_box); ###in mm^3 
    #Rel_Vol = 1/(N_sideVox*N_sideVox*N_sideVox)
    # println("Voxel_Volume ",Voxel_Volume)
    FSU_irr = round(Int64, (m_FSU * PI) * n_FSU)
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            if i * j <= FSU_irr
                sijk = VoxArray[i, j, 1].ni
                Prod *= (1.0 - (sijk)^(sj))
                sumDose += VoxArray[i, j, 1].Dose
            else
                sijk = 0.0
                Prod *= (1.0 - (sijk)^(sj))
            end
            #Prod *= ((1 - (sijk)^(1/sj))^Rel_Vol)  ##here noyt sure if 1-sijk or simply sijk; sijk is the propbaility of dying, i.e. 1 - SF
        end
    end
    DOSE = sumDose / FSU_irr
    NTCP = (1.0 - Prod)^(1 / sj)   ##the first 1 - shouldn't be there but I get the opposite values if I don't use it
    return NTCP, DOSE
end

function NTCP_Box_PI_new(VoxArray::Array{Voxel,3}, N_sideVox::Int64, sj::Float64, m_FSU::Int64, n_FSU::Int64, PI::Float64)
    Prod = 1.
    sumDose = 0.
    if PI < 1.0
        Irr = hcat(ones(n_FSU, round(Int, m_FSU * PI)), zeros(n_FSU, round(Int, m_FSU * (1.0 - PI))))
    else
        Irr = ones(n_FSU, m_FSU)
    end
    #SPcells_t = transpose(SPcells)
    #P_FSU = Irr.*VoxArray[i,j,1].ni
    P_FSU = zeros(n_FSU, m_FSU)
    NTCP_ = 1.0
    VoxArray_temp = reshape(VoxArray[:, :, 1], size(Irr, 1), size(Irr, 2), 1)
    for j in 1:n_FSU
        for i in 1:m_FSU
            P_FSU[j, i] = Irr[j, i] .* VoxArray_temp[j, i, 1].ni
            #Prod *= (1.0 - (P_FSU[j,i])^(sj)) #original
            Prod *= ((1.0 - P_FSU[j, i])^(sj)) #OK
            sumDose += Irr[j, i] .* VoxArray_temp[j, i, 1].Dose
        end
    end
    NTCP_ = (1.0 - Prod)^(1 / sj)
    FSU_irr = round(Int64, (m_FSU * PI) * n_FSU)
    DOSE = sumDose / FSU_irr
    return NTCP_, DOSE
end

########### Compute the dose given for a Bos organ j with serial j###
function GEUD_Box(VoxArray, N_sideVox, sj)
    Nvoxels = N_sideVox^3
    iNvoxel = 1 / Nvoxels
    Sum = 0.
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                Sum += VoxArray[i, j, k].Dose^(sj) ##check if here 1/sj should be sj, maybe this is the reason for the inverse NTCP
            end
        end
    end
    Dose_Box = (iNvoxel * Sum)^(1 / sj)
    return Dose_Box
end

#Compute dose in a voxel and TCP, NTCP, gEUD for N voxel   
function TCP_Box_Nvoxel(VoxArray::Array{Voxel,3}, N_voxel_xy::Int64, N_voxel_xy_dose::Int64, N_voxel_z::Int64)
    sumProba = 1.
    if N_voxel_xy_dose < N_voxel_xy
        sumProba = 0.
    else
        for i in 1:N_voxel_z
            for j in 1:N_voxel_xy_dose
                sumProba *= VoxArray[1, 1, 1].ni
            end
        end
    end
    TCP = sumProba
    return TCP
end

function NTCP_Box_Nvoxel(VoxArray::Array{Voxel,3}, sj::Float64, N_voxel_xy::Int64, N_voxel_xy_dose::Int64, N_voxel_z::Int64)
    Prod = 1.
    for i in 1:N_voxel_z
        for j in 1:N_voxel_xy_dose
            Prod *= (1 - (VoxArray[1, 1, 1].ni)^(1 / sj))  ##here noyt sure if 1-sijk or simply sijk; sijk is the propbaility of dying, i.e. 1 - SF
        end
    end
    NTCP = (1 - Prod)^(sj)   ##the first 1 - shouldn't be there but I get the opposite values if I don't use it
    return NTCP
end

function GEUD_Box_Nvoxel(VoxArray::Array{Voxel,3}, sj::Float64, N_voxel_xy::Int64, N_voxel_xy_dose::Int64, N_voxel_z::Int64)
    iNvoxel = 1 / (N_voxel_xy * N_voxel_z)
    Sum = 0.
    for i in 1:N_voxel_z
        for j in 1:N_voxel_xy_dose
            Sum += VoxArray[1, 1, 1].Dose^(sj) ##check if here 1/sj should be sj, maybe this is the reason for the inverse NTCP
        end
    end
    Dose_Box = (iNvoxel * Sum)^(1 / sj)
    return Dose_Box
end

# Compute NTCP for partial irradiation
function NTCP_nxm_PI(VoxArray::Array{Voxel,3}, N_sideVox::Int64, sj::Float64, m_FSU::Int64, n_FSU::Int64, PI::Float64)
    sumDose = 0.
    if PI < 1.0
        Irr = hcat(ones(n_FSU, round(Int, m_FSU * PI)), zeros(n_FSU, round(Int, m_FSU * (1.0 - PI))))
    else
        Irr = ones(n_FSU, m_FSU)
    end
    #SPcells_t = transpose(SPcells)
    #P_FSU = Irr.*VoxArray[i,j,1].ni
    P_FSU = zeros(n_FSU, m_FSU)
    NTCP_m = ones(1, n_FSU)
    NTCP_ = 1.0
    VoxArray_temp = reshape(VoxArray[:, :, 1], size(Irr, 1), size(Irr, 2), 1)
    for j in 1:n_FSU
        for i in 1:m_FSU
            P_FSU[j, i] = Irr[j, i] .* VoxArray_temp[j, i, 1].ni
            NTCP_m[j] *= (1.0 - P_FSU[j, i])
            sumDose += Irr[j, i] .* VoxArray_temp[j, i, 1].Dose
        end
        NTCP_ *= (1.0 - NTCP_m[j])
    end
    FSU_irr = round(Int64, (m_FSU * PI) * n_FSU)
    DOSE = sumDose / FSU_irr
    return NTCP_, DOSE
end

function MC_ions_domain_single!(ion::Ion, Np::Int64, x_cb::Float64, y_cb::Float64, irrad_cond::AT, cell::Cell, dose_domain::Array{Float64}, dose_cell::Array{Float64}, ts::Bool, LET_dom::Array{Float64}, E_dom::Array{Float64})
    # Initialize variables
    #println("Monte Carlo Loop ", Np, " particles of ", ion.ion)

    #@Threads.threads for i in ProgressBar(1:Np)
    Rk = irrad_cond.Rk
    Rapprox = irrad_cond.Rapprox
    Threads.@threads for i in 1:Np
        # Generate a random hit position within the beam circle
        x, y = GenerateHit_Circle(x_cb, y_cb, cell.r + Rapprox)
        track = Track(x, y, Rk)

        # Calculate the distance from the hit to each cell center
        dose_circle = Array{Float64}(undef, 0)

        # Extract the cell properties
        nucleus = cell.r
        radius = cell.rd_gsm2

        # Extract the centers of the cell domains
        center_x = cell.center_x
        center_y = cell.center_y

        # Distribute dose across the cell's domain
        if ts
            for cc in 1:size(center_y)[1]
                # Calculate the dose distribution in the cell
                dose, _, Gyr = distribute_dose_domain(center_x[cc], center_y[cc], radius, track, irrad_cond, type)
                push!(dose_circle, Gyr)
            end
            ndom_ = floor(Int64, nucleus / radius)
            dose_circle_ = repeat(dose_circle, ndom_)
        else
            for ccd in 1:size(E_dom)[1]
                ion = Ion(df.Ion[ir], E_dom[ccd], 1, Z, LET_dom[ccd], 1.0)
                (Rc, Rp, Rk) = ATRadius(ion, irrad)

                for cc in 1:size(center_y)[1]
                    # Calculate the dose distribution in the cell
                    dose, _, Gyr = distribute_dose_domain(center_x[cc], center_y[cc], radius, track, Rk)
                    push!(dose_circle, Gyr)
                end
            end
            dose_circle_ = dose_circle
        end

        dose_domain .+= dose_circle_
        #push!(dose_cell , sum((dose_circle.*(pi * radius^2))/(pi*nucleus^2)))
        dose_cell[i] = sum((dose_circle) / (size(center_y)[1]))
    end
end


function load_phsp!(spectra_domain::Dict, spectra_nucleus::Dict, path_domain::String, path_nucleus::String=nothing)
    files_domain = sort(readdir(path_domain), lt=natural)

    println("Reading spectra domain")
    for i in 1:size(files_domain, 1)
        phsp_data = read_phsp(path_domain * "\\" * files_domain[i])
        spectra_domain[files_domain[i]] = [x[1] for x in phsp_data]
    end
    spectra_domain = OrderedDict(k => spectra_domain[k] for k in files_domain)

    println("Reading spectra nucleus")
    if path_nucleus == nothing
        files_nucleus = files_domain
        path_nucleus = path_domain
        spectra_nucleus = spectra_domain
    else
        files_nucleus = sort(readdir(path_nucleus), lt=natural)
        for i in 1:size(files_nucleus, 1)
            phsp_data = read_phsp(path_nucleus * "\\" * files_nucleus[i])
            spectra_nucleus[files_nucleus[i]] = [x[1] for x in phsp_data]
        end
        spectra_nucleus = OrderedDict(k => spectra_nucleus[k] for k in files_nucleus)
    end
end

function read_phsp(filename)
    data = []
    open(filename, "r") do file
        for line in eachline(file)
            push!(data, parse.(Float64, split(line)))  # Convert values to numbers
        end
    end
    return data
end

# Function to compute histogram for TOPAS y
function compute_histogram_TOPAS_y(x::Vector{Float64}, mkm=nothing)

    bin_start, bin_end, bin_number = 0.1, 1000, 100
    br = 10 .^ range(log10(bin_start), log10(bin_end), length=bin_number)

    h = fit(Histogram, x[(x.>bin_start).&(x.<bin_end)], br)
    y = 0.5 * (h.edges[1][1:end-1] .+ h.edges[1][2:end])
    BinWidth = abs.(diff(h.edges[1]))

    hist = DataFrame(count=h.weights, y=y, xmin=h.edges[1][1:end-1], xmax=h.edges[1][2:end])
    hist[!, :BinWidth] = abs.(hist.xmax .- hist.xmin)

    B = 1 ./ diff(log10.(hist.BinWidth))[1]
    C = log.(10) .* diff(log10.(hist.BinWidth))[1]

    H = hist
    H[!, :fy_bw] = hist.count ./ hist.BinWidth
    H[!, :fy_bw_norm] = H.fy_bw ./ (C * sum(hist.y .* hist.fy_bw))
    H[!, :yfy] = H.fy_bw_norm .* hist.y
    H[!, :yfy_norm] = H.yfy ./ (C * sum(hist.y .* hist.yfy))
    H[!, :ydy] = H.yfy_norm .* hist.y
    H[!, :ydy_norm] = H.ydy ./ (C * sum(hist.y .* hist.ydy))

    yF = sum(H.yfy .* H.y) / sum(H.yfy)
    yD = sum(H.ydy .* H.y) / sum(H.yfy)
    if mkm == nothing
        y0 = 150
    else
        y0 = (pi * mkm.rd * mkm.Rn^2) / (sqrt(mkm.betaX * (mkm.rd^2 + mkm.Rn^2)))
    end
    ystar = (y0 .* y0 .* sum(((1 .- exp.(-(H.y .^ 2) ./ (y0 .* y0))) .* H.BinWidth .* H.fy_bw))) / (yF .* sum(H.BinWidth .* H.fy_bw))

    return Dict("C" => C, "hist" => H, "yF" => yF, "yD" => yD, "ystar" => ystar)
end

##

function positive_integer_divisors(Nvox_tot::Int)
    divisori = Int[]
    for i in 1:floor(Int, sqrt(Nvox_tot))
        if Nvox_tot % i == 0
            push!(divisori, i)
            if i != Nvox_tot ÷ i
                push!(divisori, Nvox_tot ÷ i)
            end
        end
    end
    return sort(divisori)
end




# ===============================
# %O2 profile in a spherical spheroid
# ===============================
# Piecewise model:
#  - Necrotic core clamped at a non-zero minimum (O2_min).
#  - Smooth, monotone rise from core to surface using an exponential with length scale λ.
#
# Main function: o2_percent(r; ...)
#  - r can be a scalar or an array (broadcasting is supported).
#
# Parameters (with reasonable defaults):
#  R       : total spheroid radius [µm]
#  r_core  : necrotic core radius [µm]
#  O2_surf : %O2 at the surface [%]
#  O2_min  : %O2 inside the core [%] (e.g., 0.25)
#  λ       : attenuation length of the gradient [µm]
#
# Analytical form for r_core < r ≤ R:
#  Let Δ = R - r                  (distance from surface)
#  Let Δ_core = R - r_core
#  O2(r) = O2_min +
#          (O2_surf - O2_min) * (1 - exp(-Δ/λ)) / (1 - exp(-Δ_core/λ))
#
# Properties:
#  - O2(r_core) = O2_min  (continuity at core boundary)
#  - O2(R)      = O2_surf (surface boundary condition)
#  - Monotone increasing for r ∈ [r_core, R]
#
# ===============================

function o2_percent(r;
    R::Float64        = 300.0,   # µm
    r_core::Float64   = 80.0,    # µm
    O2_surf::Float64  = 7.0,     # %
    O2_min::Float64   = 0.25,    # %
    λ::Float64        = 60.0     # µm
)
    # Sanity checks
    R > 0 || error("R must be > 0")
    0.0 ≤ r_core < R || error("Require 0 ≤ r_core < R")
    O2_surf > O2_min || error("Need O2_surf > O2_min for an increasing gradient")
    λ > 0 || error("λ must be > 0")

    # Internal scalar evaluator
    function o2_scalar(x::Float64)
        x < 0 && error("Radial distance r cannot be negative")
        if x ≤ r_core
            return O2_min
        elseif x ≤ R
            Δ       = R - x
            Δ_core  = R - r_core
            denom   = 1 - exp(-Δ_core/λ)
            # Guard against numerical issues if Δ_core is tiny
            denom ≈ 0 && return O2_surf
            return O2_min + (O2_surf - O2_min) * (1 - exp(-Δ/λ)) / denom
        else
            # Outside the spheroid: hold the surface value
            return O2_surf
        end
    end

    # Support scalar and array inputs via broadcasting
    return o2_scalar.(float.(r))
end

# ===============================
# Usage example
# ===============================
if abspath(PROGRAM_FILE) == @__FILE__
    using Printf
    R      = 300.0
    rgrid  = range(0, R; length=7)  # demonstration points

    O2vals = o2_percent(rgrid; R=R, r_core=80.0, O2_surf=7.0, O2_min=0.25, λ=60.0)

    println("r [µm]    %O2")
    for (ri, oi) in zip(rgrid, O2vals)
        @printf("%7.1f   %6.3f\n", ri, oi)
    end

    # Plot suggestion (if you have Plots.jl):
    # using Plots
    # rfine  = range(0, R; length=400)
    # O2fine = o2_percent(rfine; R=R, r_core=80.0, O2_surf=7.0, O2_min=0.25, λ=60.0)
    # plot(rfine, O2fine, xlabel="r [µm]", ylabel="%O₂", lw=2, legend=false)
end
