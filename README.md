# Radiobiology ABM Simulation Framework

A Julia framework for Monte Carlo dose deposition, DNA damage computation, and agent-based modelling (ABM) of cell population dynamics under ion irradiation. Designed for spheroid geometries but applicable to any 2D/3D cell lattice.

---

## Overview

The framework computes the full chain from physical dose to biological outcome:

```
Ion beam → MC dose → DNA damage (GSM2) → Survival → ABM cell dynamics
```

Each step is handled by a dedicated utility module. The pipeline supports:
- Proton and heavy ion beams (Katz–Chatterjee / LEM track structure)
- Phase-dependent radiosensitivity (G1, S, G2/M)
- Oxygen enhancement ratio (OER)
- Partial irradiation and track segmentation
- GPU-accelerated dose computation (CUDA)
- Agent-based proliferation/death dynamics on a cell lattice

---

## File Structure

```
src/
  utilities_structures.jl          — core structs (Cell, Ion, Irrad, AT, GSM2, ...)
  utilities_general.jl             — general-purpose helpers and shared utilities
  utilities_radiation.jl           — hit generation (box, circle, halo)
  utilities_GSM2.jl                — GSM2 damage model, OER, domain survival
  utilities_biology.jl             — cell biology: cycle, division, apoptosis logic
  utilities_env.jl                 — environment setup (GSM2, ion/irrad, cell lattice)
  utilities_dose_computation.jl    — CPU Monte Carlo dose kernels
  utilities_dose_computation_GPU.jl — GPU-accelerated dose (CUDA), auto CPU/GPU dispatch
  utilities_AT_computation.jl      — amorphous track structure, stopping power, LET
  utilities_plot.jl                — dose, damage, survival, cell-cycle plots
  utilities_abm.jl                 — ABM event loop, repair simulation, CellPopulation
  utilities_plot_abm.jl            — population dynamics, phase plots, dashboards
```

---

## Dependencies

```julia
using Base.Threads, Distributed
using CSV, DataFrames, JLD2, DelimitedFiles
using Distributions, Random, Statistics, StatsBase
using Plots, StatsPlots
using GLM, Optim, LsqFit
using ProgressBars, ProgressMeter
using InlineStrings
using CUDA                  # optional — CPU fallback available
```

---

## Quickstart: Spheroid Irradiation + ABM

### 1. Setup

```julia
# load all utilities (adjust path to match your project layout)
include(joinpath(@__DIR__, "..", "src", "utilities_structures.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_general.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_radiation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_GSM2.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_biology.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_env.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_dose_computation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_dose_computation_GPU.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_AT_computation.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_plot.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_abm.jl"))
include(joinpath(@__DIR__, "..", "src", "utilities_plot_abm.jl"))

# Stopping power table
sp = load_stopping_power()

# GSM2 parameters (phase-dependent)
gsm2_cycle = Vector{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)   # G1
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)   # S
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)   # G2/M
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)    # mixed (fallback)

setup_GSM2!(r, a, b, rd, Rn)
```

### 2. Geometry and Beam

```julia
X_box        = 900.0    # half-side of simulation box (µm)
X_voxel      = 300.0    # voxel size (µm)
R_cell       = 15.0     # cell radius (µm)
tumor_radius = 850.0    # irradiated region radius (µm)

E        = 50.0         # kinetic energy (MeV/u)
particle = "1H"
dose     = 1.0          # Gy

setup_IonIrrad!(dose, E, particle)

R_beam, x_beam, y_beam = calculate_beam_properties(
    "full", "circle", X_box, X_voxel, tumor_radius
)
```

### 3. Cell Lattice

```julia
track_seg = true   # track segmentation mode
setup_cell_lattice!(
    "circle", X_box, R_cell, N_sideVox, N_CellsSide;
    ParIrr="false", track_seg=track_seg
)
setup_cell_population!("circle", X_box, R_cell, N_sideVox, N_CellsSide, gsm2)

setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)

# Optional: oxygen distribution
set_oxygen!(cell_df; plot_oxygen=false)
```

### 4. Monte Carlo Dose

```julia
# Particle fluence from dose
F    = irrad.dose / (1.602e-9 * LET)
Npar = round(Int, F * π * R_beam^2 * 1e-8)

cell_df_copy = deepcopy(cell_df)

# Auto CPU/GPU dispatch (GPU used if CUDA available and Npar ≥ 1_000_000)
MC_dose_fast!(
    ion, Npar, R_beam, irrad_cond,
    cell_df_copy, df_center_x, df_center_y, at,
    gsm2_cycle, type_AT, track_seg
)

plot_scalar_cell(cell_df_copy, :dose_cell)
```

### 5. DNA Damage

```julia
MC_loop_damage!(ion, cell_df_copy, irrad_cond, gsm2_cycle)
plot_damage(cell_df_copy; layer_plot=true)
```

### 6. Cell Survival (GSM2)

```julia
compute_cell_survival_GSM2!(cell_df_copy, gsm2_cycle)
plot_scalar_cell(cell_df_copy, :sp)

mean_sp = mean(cell_df_copy[cell_df_copy.is_cell .== 1, :sp])
println("Mean survival probability: $mean_sp")
```

### 7. ABM: Repair Scheduling

```julia
nat_apo = 1e-10   # natural apoptosis rate
compute_times_domain!(cell_df_copy, gsm2_cycle, nat_apo)

plot_times(cell_df_copy)
plot_initial_distributions(cell_df_copy)
print_phase_distribution(cell_df_copy; label="Post-irradiation")
```

### 8. ABM: Population Dynamics

```julia
cell_df_ = deepcopy(cell_df_copy)

ts, snapshots = run_simulation_abm!(
    cell_df_;
    nat_apo       = nat_apo,
    terminal_time = 48.0,
    snapshot_times = [1, 6, 12, 24, 48]
)

plot_simulation_results(ts)
plot_analysis_dashboard(ts)
print_simulation_summary(ts)
```

### 9. Analysis and Visualization

```julia
# Phase evolution
plot_phase_proportions(ts)
plot_phase_stacked(ts)
plot_cycling_vs_quiescent(ts)
plot_growth_rate(ts; window_size=12)

# Snapshot comparisons
plot_snapshot_comparison(snapshots; times=[0, 6, 12, 24])
plot_phase_comparison_before_after(cell_df_copy, cell_df_)

# Spatial distribution (requires x, y coordinates)
plot_cell_cycle_distribution(cell_df_)
plot_spatial_distribution(snapshots[12]; color_by=:cell_cycle)

# Animation
create_spatial_animation(snapshots; output_file="dynamics.gif", fps=3)
animate_cell_cycle_3d(snapshots; output_file="cc3d.gif")

# Export
export_timeseries_csv(ts, "results.csv")
```

---

## Key Structs

| Struct | Purpose |
|---|---|
| `Ion` | Ion species: Z, A, E, LET |
| `Irrad` | Beam: dose, dose rate, particle |
| `AT` | Amorphous track: Rc, Rp, Kp |
| `GSM2` | Repair model: a, b, r, rd, Rn |
| `Cell` | Single cell: position, radius, phase |
| `CellPopulation` | Structure-of-arrays population (used by ABM) |
| `SimulationTimeSeries` | Time series buffers for ABM recording |
| `Track` | Single ion track: position, halo radius |
| `Voxel` | Spatial voxel for layered dose |

---

## GPU Acceleration

`MC_dose_fast!` automatically selects the backend:

```
CUDA.functional() && Npar ≥ GPU_PARTICLE_THRESHOLD  →  GPU kernel
otherwise                                            →  CPU (zero overhead)
```

Override at runtime:
```julia
GPU_PARTICLE_THRESHOLD = 0          # always GPU (if available)
GPU_PARTICLE_THRESHOLD = typemax(Int)  # always CPU
```

The GPU kernel (`_mc_dose_kernel_fast!`) maps one CUDA thread per domain point and loops over all `Npar` primaries. For `Npar >> n_domains`, a 2D kernel with particle-dimension parallelism and `atomicAdd` gives better GPU saturation.

---

## GSM2 Radiosensitivity Model

Phase-dependent parameters are passed as `gsm2_cycle::Vector{GSM2}`:

| Index | Phase | Typical use |
|---|---|---|
| `[1]` | G1 / G0 | slow repair, moderate sensitivity |
| `[2]` | S | low sensitivity |
| `[3]` | G2 / M | high sensitivity |
| `[4]` | mixed | fallback / unknown phase |

The repair simulation (`compute_repair_domain`) runs a Gillespie algorithm over X-lesions with rates:
- repair: `r · X[j]`
- misrepair: `a · X[j]`
- interaction: `b · X[j] · (X[j]-1)`

Outcome codes: `1` = lethal, `0` = recovered, `-1` = timeout.

---

## ABM Cell Cycle

Phase transitions follow `PHASE_TRANSITION`:
```
G1 → S → G2 → M → (division or G0)
```

Cycle times are Gamma-distributed via `PHASE_DURATIONS`:
```julia
# approximate defaults
G1: Gamma(24, 0.5)   # ~12h mean
S:  Gamma(16, 0.5)   # ~8h mean
G2: Gamma(6,  0.5)   # ~3h mean
M:  Gamma(2,  0.5)   # ~1h mean
```

Cells blocked by contact inhibition (`number_nei == 0`) enter G0. They re-enter the cycle when a neighbor dies and space opens.

---

## Comment Convention

All utility files use a consistent inline color convention for navigation:

```julia
#!  file/section headers
#~  category groupings
#?  public function names / constants
#   descriptions and details
```

---
