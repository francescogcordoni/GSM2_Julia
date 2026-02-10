begin
    mutable struct Cell
        x::Float64
        y::Float64
        z::Float64
        center_x::Vector{Float64}
        center_y::Vector{Float64}
        nei::Array{Int64}
        r::Float64### r_nucleus    
        R::Float64### R_cell
        a_gsm2::Float64
        b_gsm2::Float64
        r_gsm2::Float64 #repear rate
        rd_gsm2::Float64 #interaction range of damage
        is_stem::Int64 #is a stem cell
        proliferation::Int64
        cell_cycle::String #cell cycle for now is M-phase or other
        dam_X::Matrix{Float64}
        dam_Y::Matrix{Float64}
        dam_X_dom::Vector{Int64}
        dam_Y_dom::Vector{Int64}
        O::Float64 #Oxygenation [%]
        SP::Float64 #survival probability
        dose::Array{Float64} #dose recevied
        dose_cell::Float64
        is_cell::Int64 #is node occupied by cell?
        can_divide::Int64 #is node occupied by cell?
        number_nei::Int64 #number of adjacent cell
        apo_time::Float64 #time of death
        death_time::Float64 #time of death
        recover_time::Float64 #time of recovery
        cycle_time::Float64 #time of cycle change (at the end of cycle M it divides)
        is_death_rad::Int64
        i_voxel_x::Int64
        i_voxel_y::Int64
        i_voxel_z::Int64
    end
    
    mutable struct Voxel
        xmin::Float64
        xmax::Float64
        ymin::Float64
        ymax::Float64
        zmin::Float64
        zmax::Float64
        ni::Float64 #1 - Survival probability
        Dose::Float64
        #CellList::Array{Cell,1}#####list of the cells in the voxel
    end
    struct Track
        x::Float64
        y::Float64
        Rk::Float64
    end

    mutable struct Ion
        ion::String
        E::Float64
        A::Int64
        Z::Int64
        LET::Float64
        rho::Float64
    end

    mutable struct Irrad
        dose::Float64
        kR::Float64
        doserate::Float64
    end

    mutable struct AT
        ion::String
        E::Float64
        A::Int64
        Z::Int64
        LET::Float64
        rho::Float64
        Rc::Float64
        Rp::Float64
        Rk::Float64
        Kp::Float64
    end

    mutable struct GSM2
        r::Float64
        a::Float64
        b::Float64
        rd::Float64
        Rn::Float64
    end
end