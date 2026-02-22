#! ============================================================================
#! utilities_structures.jl
#!
#! STRUCTS
#! -------
#~ Cell & Spatial
#?   Cell          — mutable: single cell state (position, cycle, damage, dose, neighbors...)
#?   CellPopulation — mutable: SoA representation of a cell population (optimized memory)
#?   Voxel         — mutable: spatial voxel with dose and survival probability
#?   Track         — immutable: ion track (position + radius)
#
#~ Radiation & Physics
#?   Ion    — mutable: ion properties (type, energy, mass, charge, LET, density)
#?   Irrad  — mutable: irradiation parameters (dose, dose rate, kR)
#?   AT     — mutable: Amorphous Track model parameters
#?   GSM2   — mutable: GSM2 model parameters (r, a, b, rd, Rn)
#
#~ Simulation Tracking
#?   SimulationTimeSeries — mutable: time-series recorder for population dynamics
#                           (total, G0/G1/S/G2/M, stem/non-stem counts over time)
#! ============================================================================

begin
    mutable struct Cell
        x::Float64
        y::Float64
        z::Float64
        center_x::Vector{Float64}
        center_y::Vector{Float64}
        nei::Array{Int64}
        r::Float64          ### r_nucleus    
        R::Float64          ### R_cell
        a_gsm2::Float64
        b_gsm2::Float64
        r_gsm2::Float64     # repair rate
        rd_gsm2::Float64    # interaction range of damage
        is_stem::Int64      # is a stem cell
        proliferation::Int64
        cell_cycle::String  # cell cycle phase
        dam_X::Matrix{Float64}
        dam_Y::Matrix{Float64}
        dam_X_dom::Vector{Int64}
        dam_Y_dom::Vector{Int64}
        O::Float64          # Oxygenation [%]
        SP::Float64         # survival probability
        dose::Array{Float64}
        dose_cell::Float64
        is_cell::Int64      # is node occupied by cell?
        can_divide::Int64
        number_nei::Int64   # number of adjacent cells
        apo_time::Float64
        death_time::Float64
        recover_time::Float64
        cycle_time::Float64 # time of cycle change (at end of M → divides)
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
        ni::Float64     # 1 - Survival probability
        Dose::Float64
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

mutable struct SimulationTimeSeries
    time::Vector{Float64}
    total_cells::Vector{Int32}
    g0_cells::Vector{Int32}
    g1_cells::Vector{Int32}
    s_cells::Vector{Int32}
    g2_cells::Vector{Int32}
    m_cells::Vector{Int32}
    stem_cells::Vector{Int32}
    non_stem_cells::Vector{Int32}

    function SimulationTimeSeries()
        new(Float64[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[])
    end
end

mutable struct CellPopulation
    # Boolean states (Int8: 1 byte vs 8 bytes for Int64)
    is_cell::Vector{Int8}
    can_divide::Vector{Int8}
    is_stem::Union{Vector{Int8}, Nothing}
    is_death_rad::Union{Vector{Int8}, Nothing}

    # Timing information
    death_time::Vector{Float64}
    cycle_time::Vector{Float64}
    recover_time::Vector{Float64}

    # Cell cycle phase (String7: inline string, no heap allocation)
    # Fits "G1", "S", "G2", "M", "G0" with zero allocation
    cell_cycle::Vector{String7}

    # Spatial information
    number_nei::Vector{Int16}       # Typically small (< 256)
    nei::Vector{Vector{Int32}}      # Neighbor indices

    # Optional spatial coordinates
    x::Union{Vector{Int32}, Nothing}
    y::Union{Vector{Int32}, Nothing}

    # Metadata
    n_cells::Int32      # Current number of cell slots
    n_alive::Int32      # Number of alive cells (cached for performance)

    # Original indices (for DataFrame conversion)
    indices::Vector{Int32}
end
