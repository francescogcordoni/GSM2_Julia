# GSM2_Julia

**GSM2_Julia** is a high-performance Julia implementation of the **Generalized Stochastic Microdosimetric Model (GSM2)** coupled with a spatial **Agent-Based Model (ABM)**. 

This framework simulates the biological effects of ion radiation on cell populations, spanning from physical energy deposition (microdosimetry) to biological repair kinetics and tissue-level dynamics.

## Key Features

*   **GSM2 radiobiological model**: Simulates DNA damage formation and repair using the generalized stochastic microdosimetric model. It tracks two types of lesions ($X$ and $Y$) and handles repair ($r$), misrepair ($a$), and lethal interaction ($b$) rates.
*   **Microdosimetry & Track Structure**: 
    *   Supports various ion types (e.g., Protons, Carbon) and energies.
    *   Calculates energy deposition in subcellular domains using Monte Carlo simulations.
    *   Includes both **Track-Segment (TSC)** and **Full Monte Carlo** modes for dose calculation.
*   **Agent-Based Model (ABM)**:
    *   Simulates a 3D population of cells in a lattice (square or circular geometries).
    *   Models cell cycle progression (G1, S, G2, M), division, and contact inhibition.
    *   Handles radiation-induced death (mitotic catastrophe, interphase death) and natural apoptosis.
*   **Performance**: Utilizes Julia's multi-threading (`Base.Threads`) for parallel Monte Carlo dose computation and stochastic repair simulations.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/GSM2_Julia.git
    cd GSM2_Julia
    ```

2.  **Install Dependencies**:
    Open the Julia REPL and install the required packages:
    ```julia
    using Pkg
    Pkg.add(["DataFrames", "Distributions", "CSV", "Plots", "StatsBase", "ProgressMeter", "JLD2", "LsqFit", "GLM"])
    ```

## Project Structure

*   **`src/`**: Core source code.
    *   `utilities_env.jl`: Environment setup and global variable injection.
    *   `utilities_dose_computation.jl`: Monte Carlo algorithms for dose deposition.
    *   `utilities_abm.jl`: Agent-based model logic (time stepping, division, death).
    *   `utilities_structures.jl`: Data structures (`Cell`, `GSM2`, `Ion`, `SimulationTimeSeries`).
    *   `utilities_GSM2.jl`: Core GSM2 kinetic equations.
*   **`scripts/`**: Example scripts.
    *   `sphere.jl`: A complete workflow simulating a spheroid of cells under irradiation.

## Usage Example

Below is a simplified workflow based on `scripts/sphere.jl`.

### 1. Setup Environment and Model
```julia
using GSM2_Julia # Assuming package structure or include files directly

# Define GSM2 parameters (repair, lethal, interaction rates, domain radius, nucleus radius)
r, a, b = 4.3, 0.01, 0.30
rd, Rn  = 0.8, 7.2

# Initialize GSM2 globals
setup_GSM2!(r, a, b, rd, Rn)

# Define Irradiation (Dose in Gy, Energy in MeV/u, Ion type)
setup_IonIrrad!(1.0, 150.0, "1H") # 1 Gy of 150 MeV Protons
```

### 2. Initialize Cell Population
```julia
# Define geometry (Box size, Cell radius, Voxel settings)
X_box, R_cell = 900.0, 15.0
N_sideVox = 6
N_CellsSide = 60

# Create the lattice and populate cells
setup_cell_lattice!("circle", X_box, R_cell, N_sideVox, N_CellsSide)
setup_cell_population!("circle", X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

# Compute irradiation conditions per layer
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, true)
```

### 3. Compute Dose and Damage
```julia
# Run Monte Carlo Dose Calculation
# Npar is the number of particles, R_beam is beam radius
MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df, df_center_x, df_center_y, at, gsm2_cycle, type_AT, true)

# Compute initial DNA damage (Lesions X and Y)
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)
```

### 4. Run Agent-Based Simulation
```julia
# Compute survival probabilities and repair times
nat_apo = 1e-10 # Natural apoptosis rate
compute_times_domain!(cell_df, gsm2_cycle, nat_apo)

# Run the time-dependent simulation
ts, snapshots = run_simulation_abm!(cell_df, nat_apo; terminal_time=48.0)

# Plot results
plot_simulation_results(ts)
```

## Core Functions

### `setup_GSM2!`
Configures the GSM2 kinetic parameters and generates the domain template for the cell nucleus.

### `setup_IonIrrad!`
Sets up the ion species, energy, and LET. It computes the amorphous track structure parameters ($R_c$, $R_p$).

### `MC_dose_fast!`
The main engine for dose calculation.
*   **Track-Segment Mode (`track_seg=true`)**: Simulates a representative layer and propagates dose to all cells, optimizing performance for uniform beams.
*   **Full Mode**: Simulates particle hits explicitly for every layer.

### `run_simulation_abm!`
Evolves the cell population over time.
*   Handles **Cell Cycle**: Transitions between G1, S, G2, M phases.
*   Handles **Division**: Checks for empty neighbors before dividing.
*   Handles **Death**: Removes cells based on radiogenic damage or natural apoptosis.

## License

[Insert License Here]

## Authors

*   **Utente** (GitHub: GSM2_Julia)