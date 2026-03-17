# Compute the positive integer divisors of Nvoxel
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

# Creation of the array of Voxel
function CreationArrayVoxels_NTCP(X_box::Float64,X_voxel::Float64)
    N_sideVox=convert(Int64, floor(2*X_box/(X_voxel)));
    #x_range = y_range = z_range = AbstractRange{Float64}
    ### Create and initialize the VoxArray
    VoxArray = Array{Voxel, 3}(undef, N_sideVox, N_sideVox, N_sideVox)
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                VoxArray[i, j, k] = Voxel(0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.,0.)
            end
        end
    end
    #### Fill the array of Voxel with cells within the x,y,z range ####
    #for cell in arrayOfCell 
        for i in 1:N_sideVox
            #x_range = range(start=-X_box+(i-1)*X_voxel,stop=-X_box+i*X_voxel);
            for j in 1:N_sideVox
                #y_range = range(start=-X_box+(j-1)*X_voxel,stop=-X_box+j*X_voxel);
                for k in 1:N_sideVox
                    #z_range = range(start=-X_box+(k-1)*X_voxel,stop=-X_box+k*X_voxel);    
                    VoxArray[i,j,k].xmin = -X_box+(i-1)*X_voxel;
                    VoxArray[i,j,k].xmax = -X_box+i*X_voxel;
                    VoxArray[i,j,k].ymin = -X_box+(j-1)*X_voxel;
                    VoxArray[i,j,k].ymax = -X_box+j*X_voxel;
                    VoxArray[i,j,k].zmin = -X_box+(k-1)*X_voxel;
                    VoxArray[i,j,k].zmax = -X_box+k*X_voxel;
                    #if (cell.x in x_range) && (cell.y in y_range) && (cell.z in z_range)
                    #    push!(VoxArray[i,j,k].CellList,cell)
                    #end
                end
            end
        end     
    #end    
    return N_sideVox,VoxArray
end

############### Fill the voxel array with the SP for each voxel ##############################
function Survival_Voxels_NTCP!(VoxArray::Array{Voxel,3},X_voxel::Float64, cell_df::DataFrame)

    N_sideVox_z = size(VoxArray)[3]
    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox_z
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

function Survival_Voxels_ijk_NTCP!(i,j,k,VoxArray::Array{Voxel,3},X_voxel::Float64, cell_df::DataFrame)

    filtered_df = cell_df[(cell_df.i_voxel_x .== i) .& (cell_df.i_voxel_y .== j) .& (cell_df.i_voxel_z .== k), :]
    SPcells = prod(1.0 .- filtered_df.sp)
    SumDose = sum(filtered_df.dose_cell)
    NumbOfCellsinVox=size(filtered_df.sp)[1]

    ###Probability of survival voxel
    VoxArray[i,j,k].ni = SPcells;   
    #VoxArray[i,j,k].Dose = ((SumDose*2*π*r_nuc^3)*NumbOfCellsinVox)/(X_voxel^3);
    VoxArray[i,j,k].Dose = SumDose/NumbOfCellsinVox;
end


# Compute TCP for full irradiation - BOX MODEL xyz, SPHERE MODEL
function compute_TCP_df(cell_df::DataFrame)

    filtered_df = cell_df[cell_df.is_cell .== 1, :]
    TCP = prod(1.0 .- filtered_df.sp)
    #DOSE = mean(filtered_df.dose_cell)

    return TCP #, DOSE
end

# Compute TCP for full irradiation - BOX MODEL xyz, SPHERE MODEL
function compute_TCP(Sp_::Array{Float64,1})

    TCP = prod(1.0 .- Sp_)

    return TCP
end

# Compute NTCP for full irradiation - BOX MODEL xyz
function compute_NTCP_Box_s(VoxArray::Array{Voxel,3}, N_sideVox::Int64, sj::Float64)
    Prod = 1.
    sumDose = 0.

    for i in 1:N_sideVox
        for j in 1:N_sideVox
            for k in 1:N_sideVox
                #FSU response
                P_FSU = VoxArray[i,j,k].ni;
                
                #Prod *= (1.0 - (P_FSU)^(sj)) #original Kallman
                Prod *= ((1.0 - P_FSU)^(sj)) #OK

                #Dose
                sumDose += VoxArray[i,j,k].Dose
            end
        end
    end

    NTCP_ = (1 - Prod)^(1/sj)
    DOSE = sumDose/(N_sideVox^3)

    return NTCP_, DOSE
end

# Compute NTCP for partial irradiation - SINGLE LAYER MODEL
function compute_NTCP_nxm_PI(VoxArray::Array{Voxel,3}, m_FSU::Int64, n_FSU::Int64, PI::Float64)
    sumDose = 0.
    if PI < 1.0 
        Irr = hcat(ones(n_FSU, round(Int, m_FSU*PI)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI))))
    else
        Irr = ones(n_FSU, m_FSU)
    end

    if PI == 0.0
        println("Warning: no irradiation (PI = 0.0) -> NTCP = 0.0 and Dose = 0.0")
        return 0.0, 0.0
    end

    P_FSU = zeros(n_FSU, m_FSU)
    NTCP_m = ones(1, n_FSU)
    NTCP_ = 1.0
    VoxArray_temp = reshape(VoxArray[:, :, 1], size(Irr, 1), size(Irr, 2), 1)

    for j in 1:n_FSU
        for i in 1:m_FSU
            #FSU response
            P_FSU[j,i] = Irr[j,i].*VoxArray_temp[j,i,1].ni

            NTCP_m[1,j] *= (1.0 - P_FSU[j,i])

            #Dose
            sumDose += Irr[j,i].*VoxArray_temp[j,i,1].Dose
        end
        NTCP_ *= (1.0 - NTCP_m[1,j])
    end

    FSU_irr = round(Int64, (m_FSU*PI)*n_FSU)
    DOSE = sumDose/FSU_irr

    return NTCP_, DOSE, VoxArray_temp
end

