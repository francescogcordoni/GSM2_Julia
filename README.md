# GSM2 Julia — Ion Irradiation & ABM Framework

A Julia framework for simulating the full chain from physical ion beam delivery to post-irradiation cell population dynamics. Designed for tumour spheroids but applicable to any 3-D cell lattice.

```
Ion beam → MC dose deposition → DNA damage (GSM2) → Survival → ABM cell dynamics
```

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Pipeline in Detail](#pipeline-in-detail)
   - [Loading Utilities](#1-loading-utilities)
   - [GSM2 Parameters](#2-gsm2-parameters)
   - [Environment Setup](#3-environment-setup)
   - [Monte Carlo Dose](#4-monte-carlo-dose)
   - [DNA Damage](#5-dna-damage)
   - [Survival](#6-survival)
   - [Repair Timers](#7-repair-timers)
   - [ABM Simulation](#8-abm-simulation)
   - [Plotting](#9-plotting)
5. [Key Structs](#key-structs)
6. [Configuration Reference](#configuration-reference)
7. [Supported Ions](#supported-ions)
8. [Physics Models](#physics-models)
9. [GPU Acceleration](#gpu-acceleration)
10. [File Structure](#file-structure)
11. [Comment Convention](#comment-convention)

---

## Overview

The framework is composed of independent utility modules (`src/utilities_*.jl`) covering:

| Stage | Module |
|---|---|
| Geometry, beam, cell lattice | `utilities_env.jl` |
| Stopping power, track radii | `utilities_radiation.jl` |
| Amorphous track dose | `utilities_AT_computation.jl` |
| Monte Carlo dose (CPU + GPU) | `utilities_dose_computation.jl`, `utilities_dose_computation_GPU.jl` |
| DNA damage, OER, κ | `utilities_GSM2.jl` |
| Cell biology (cycle, division) | `utilities_biology.jl` |
| ABM event loop | `utilities_abm.jl` |
| Dose/damage/survival plots | `utilities_plot.jl` |
| Population dynamics plots | `utilities_plot_abm.jl` |

---

## Installation

### Requirements

Julia ≥ 1.9. Install the required packages once:

```julia
using Pkg
Pkg.add([
    "CSV", "DataFrames", "JLD2", "DelimitedFiles",
    "Distributions", "Random", "Statistics", "StatsBase",
    "Plots", "StatsPlots",
    "GLM", "Optim", "LsqFit",
    "ProgressBars", "ProgressMeter",
    "InlineStrings", "Printf",
    "CUDA",           # optional — CPU fallback is automatic
])
```

### Clone

```bash
git clone <repo-url>
cd GSM2_Julia
```

---

## Quickstart

The canonical example is **`example/spheroid.jl`**. It runs the complete pipeline
(setup → dose → damage → survival → ABM → plots) for a spheroid of radius 350 µm
and saves 28 figures to `example/output/spheroid/`.

```bash
julia --threads auto example/spheroid.jl
```

Edit the **PARAMETERS** block at the top of the script to change particle, energy,
dose, or geometry without touching any other code:

```julia
PARTICLE     = "1H"      # "1H", "4He", "12C", "16O"
ENERGY_MEV_U = 100.0     # MeV per nucleon
DOSE_GY      = 2.0       # prescribed dose (Gy)
TUMOR_RADIUS = 350.0     # spheroid radius (µm)
TERMINAL_TIME = 72.0     # post-irradiation ABM window (h)
```

---

## Pipeline in Detail

### 1. Loading Utilities

```julia
include(joinpath(@__DIR__, "src", "load_utilities.jl"))
sp = load_stopping_power()   # loads stopping power tables from data/stoppingpower/
```

`load_utilities.jl` includes all source files in the correct dependency order.
`sp` is a `Dict{String, Matrix{Float64}}` mapping ion labels (`"1H"`, `"12C"`, ...)
to two-column energy/stopping-power tables.

---

### 2. GSM2 Parameters

The **Giant Stochastic Model 2 (GSM2)** describes stochastic DNA repair. Parameters
are cell-cycle-phase dependent. Pass a `Vector{GSM2}` of length 4:

```julia
gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)   # G1 / G0
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)   # S phase
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)   # G2 / M
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)    # average (fallback)
```

| Parameter | Meaning |
|---|---|
| `r` | Correct repair rate (h⁻¹) |
| `a` | Mis-repair rate (h⁻¹) |
| `b` | Binary (pairwise) mis-repair rate (h⁻¹) |
| `rd` | Domain radius (µm) |
| `Rn` | Nucleus radius (µm) |

Typical values fitted to HSG cells:

```julia
# G1/G0
r_G1 = 2.7805;  a_G1 = 0.01287;  b_G1 = 0.04030
# S
r_S  = 5.8401;  a_S  = 0.00589;  b_S  = 0.05794
# G2/M
r_G2 = 1.7720;  a_G2 = 0.02431;  b_G2 = 5.705e-5
# Average
r    = 2.5657;  a    = 0.01481;  b    = 0.01266
rd   = 0.8;     Rn   = 7.2
```

---

### 3. Environment Setup

All setup functions inject global variables into `Main` scope (legacy API for
notebook / REPL use). Call them in order:

#### Option A — individual steps (explicit, inspectable)

```julia
# 1. GSM2 geometry: domain centers, nucleus template
setup_GSM2!(r, a, b, rd, Rn)
# Injects: gsm2, center_x, center_y, domain

# 2. Ion and irradiation descriptors
setup_IonIrrad!(dose, E, particle; type_AT="KC")
# Injects: ion, irrad, LET, Rc, Rp, Rk, Kp, A, Z, type_AT

# 3. Cell lattice (positions, neighbour lists, layer assignment)
N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))
setup_cell_lattice!("circle", X_box, R_cell, N_sideVox, N_CellsSide;
                    ParIrr="false", track_seg=true, full_cycle=true)
# Injects: cell_df, N, nodes_positions, N_CellsSide, track_seg, full_cycle

# Required before setup_cell_population!:
@eval Main begin
    tumor_radius = 350.0
    X_voxel      = 700.0
    X_box        = 350.0
end

# 4. Biological attributes (cell cycle, damage vectors, neighbours)
setup_cell_population!("circle", X_box, R_cell, N_sideVox, N_CellsSide, gsm2)
# Injects: rel_center_x/y, df_center_x/y, at, num_cols

# 5. Per-layer irradiation conditions (AT objects indexed by energy step)
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
# Injects: irrad_cond, lets, energies, num_energy_steps

# 6. Oxygenation profile
set_oxygen!(cell_df; plot_oxygen=false)
```

#### Option B — single high-level wrapper

```julia
out = setup(E, particle, dose, tumor_radius;
            X_box=350.0, X_voxel=700.0, R_cell=15.0,
            target_geom="circle", calc_type="full",
            type_AT="KC", track_seg=true)
# Returns NamedTuple: ion, irrad, cell_df, at_start,
#                     R_beam, x_beam, y_beam,
#                     O2_mean, Npar, zF, D, T
```

`setup()` calls all six functions above in sequence and additionally calls
`set_oxygen!` and computes beam fluence metrics.

#### Key geometry parameters

| Parameter | Description | Typical value |
|---|---|---|
| `X_box` | Half-side of simulation box (µm) | 350 – 900 |
| `X_voxel` | Voxel side for beam-radius calculation (µm) | `2 * X_box` |
| `R_cell` | Cell radius (µm) | 15 |
| `tumor_radius` | Spheroid or irradiated region radius (µm) | 150 – 900 |
| `target_geom` | Geometry of cross-section: `"circle"` or `"square"` | `"circle"` |
| `calc_type` | Beam-radius mode: `"full"` (whole target) or `"fast"` (voxel) | `"full"` |
| `type_AT` | Track structure model: `"KC"` or `"LEM"` | `"KC"` |
| `track_seg` | `true` = fixed LET across depth; `false` = Bragg-peak propagation | `true` |
| `full_cycle` | `true` = cells start distributed across full cycle | `true` |
| `ParIrr` | `"true"` = partial irradiation (beam narrower than box) | `"false"` |

---

### 4. Monte Carlo Dose

Compute the dose deposited by `Npar` primary ions in the cell lattice:

```julia
# Estimate particle count from dose and beam area
F    = irrad.dose / (1.602e-9 * LET)          # fluence (particles/cm²)
Npar = round(Int, F * π * R_beam^2 * 1e-8)   # particles through spheroid

# Compute beam radius
R_beam, x_beam, y_beam = calculate_beam_properties(
    "full", "circle", X_box, X_voxel, tumor_radius)

# CPU dose (always available)
MC_dose_CPU!(ion, Npar, R_beam, irrad_cond,
             cell_df, df_center_x, df_center_y, at,
             gsm2_cycle, type_AT, track_seg)

# Auto CPU/GPU dispatch (prefers GPU when CUDA available and Npar is large)
MC_dose_fast!(ion, Npar, R_beam, irrad_cond,
              cell_df, df_center_x, df_center_y, at,
              gsm2_cycle, type_AT, track_seg)
```

After this call, `cell_df` gains the columns:

| Column | Type | Meaning |
|---|---|---|
| `dose_cell` | `Float64` | Total dose received by the cell (Gy) |
| `dose` | `Vector{Float64}` | Per-domain dose vector (one entry per GSM2 domain) |

---

### 5. DNA Damage

Sample stochastic DNA lesions from the dose using Poisson statistics:

```julia
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)
```

This writes four columns to `cell_df`:

| Column | Type | Meaning |
|---|---|---|
| `dam_X_dom` | `Vector{Int}` | Repairable DSBs per domain (length = `n_domains`) |
| `dam_Y_dom` | `Vector{Int}` | Lethal (complex) lesions per domain |
| `dam_X_total` | `Int` | Sum of `dam_X_dom` over all domains |
| `dam_Y_total` | `Int` | Sum of `dam_Y_dom` over all domains |

The damage rate per domain is:
- `λX = κ · d`  where `κ = 9 · kappa_yield / (n_repeat · N_domains)` (DSBs/Gy)
- `λY = λX · 1e-3` (lethal fraction)

`kappa_yield` is ion- and LET-dependent and includes OER correction:

```julia
κ = calculate_kappa(ion.ion, LET, O; OER_bool=true)
```

---

### 6. Survival

Compute the GSM2 survival probability for each live cell:

```julia
compute_cell_survival_GSM2!(cell_df, gsm2_cycle; NFrac=1)
```

Results are stored in `cell_df.sp` (one Float64 per cell, 0–1).

- Any cell with `sum(dam_Y_dom) > 0` gets `sp = 0.0` immediately.
- For purely X-damaged cells, survival is the product of per-domain repair
  probabilities (Gillespie-like formula).
- `NFrac > 1` implements multi-fraction: `sp = sp_single^NFrac`.

```julia
# Population-level summary with Wilson 95% CI
alive = cell_df.is_cell .== 1
p_hat, ci_lo, ci_hi = survival_ci(cell_df.sp[alive])
println("SF = $(round(p_hat, digits=4))  ($(round(ci_lo,digits=4))–$(round(ci_hi,digits=4)))")
```

---

### 7. Repair Timers

Assign stochastic repair, death, and cycle times to each cell before the ABM:

```julia
compute_times_domain!(cell_df, gsm2_cycle;
                      nat_apo       = 1e-10,   # non-radiation apoptosis rate (h⁻¹) (currently not used)
                      terminal_time = 72.0,    # ABM window (h)
                      verbose       = false,
                      summary       = true)
```

This writes timing columns to `cell_df`:

| Column | Meaning |
|---|---|
| `death_time` | Time at which the cell dies (radiation-induced apoptosis) |
| `recover_time` | Time at which the cell completes repair and survives |
| `cycle_time` | Time at which the current cell-cycle phase ends |
| `apo_time` | Time of natural (background) apoptosis |
| `is_death_rad` | 1 if death is radiation-induced, 0 if natural |

---

### 8. ABM Simulation

Run the agent-based model of post-irradiation cell dynamics:

```julia
ts, snaps = run_simulation_abm!(cell_df;
    nat_apo           = 1e-10,
    terminal_time     = 72.0,
    snapshot_times    = [0, 6, 12, 24, 48, 72],
    print_interval    = 6.0,
    verbose           = true,
    return_dataframes = false,
    update_input      = true)
```

**Returns:**
- `ts::SimulationTimeSeries` — time-series of population counts (total, G0/G1/S/G2/M, stem)
- `snaps::Dict{Int, CellPopulation}` — full population state at each snapshot time

The ABM processes the following events per cell per time step:

| Event | Condition |
|---|---|
| Death (radiation) | `time ≥ death_time` and `is_death_rad == 1` |
| Death (natural) | `time ≥ apo_time` |
| Recovery | `time ≥ recover_time` (cell survives, re-enters cycle) |
| Phase transition | `time ≥ cycle_time` (G1→S→G2→M→divide) |
| Division | Cell in M phase with empty neighbour |
| Contact inhibition → G0 | Cell in G1/G0 with no empty neighbours |

---

### 9. Plotting

#### Dose & Damage (from `utilities_plot.jl`)

```julia
# Two-panel: density + 3-D spatial coloured map
plot_scalar_cell(cell_df, :dose_cell)
plot_scalar_cell(cell_df, :dam_X_total)
plot_scalar_cell(cell_df, :sp; xscale=:log10)   # log scale for survival

# Grouped by energy step (depth layer)
plot_scalar_cell(cell_df, :dose_cell; layer_plot=true)

# X-damage density
plot_damage(cell_df)
plot_damage(cell_df; layer_plot=true)

# Timer distributions (4-panel: death/recovery/cycle/damage)
plot_times(cell_df; show_means=true)

# Initial timer distributions
plot_initial_distributions(cell_df)

# Cell cycle bar chart (alive cells)
plot_phase_proportions_alive(cell_df; title_text="Pre-irradiation")
plot_cell_cycle_distribution(cell_df; phase_plot=false, half_sphere=true)

# Text summary
print_phase_distribution(cell_df; label="t = 0h")
```

#### Population Dynamics (from `utilities_plot_abm.jl`)

```julia
plot_cell_dynamics(ts)              # total cell count vs time
plot_phase_dynamics(ts)             # G0/G1/S/G2/M vs time
plot_phase_proportions(ts)          # phase percentages vs time
plot_phase_stacked(ts)              # stacked area chart
plot_cycling_vs_quiescent(ts)       # cycling vs G0
plot_growth_rate(ts; window_size=10)
plot_stem_dynamics(ts)
plot_simulation_results(ts)         # 3-panel summary
plot_analysis_dashboard(ts)         # 5-panel dashboard

# Snapshot comparisons
plot_snapshot_comparison(snaps; metric=:cell_cycle, times=[0,6,12,24])
plot_phase_comparison_before_after(cell_df_before, cell_df_after)

# Spatial / 3-D
plot_spatial_distribution(snap; color_by=:cell_cycle)
plot_cell_cycle_snapshots(snaps; times=[0, 24, 72])
create_spatial_animation(snaps; output_file="anim.gif", fps=2)
animate_cell_cycle_3d(snaps; output_file="cc3d.gif")

# Export
export_timeseries_csv(ts, "timeseries.csv")
print_simulation_summary(ts)
```

---

## Key Structs

### `Ion`
```julia
Ion(ion, E, A, Z, LET, rho)
```
| Field | Type | Meaning |
|---|---|---|
| `ion` | `String` | Species label (`"1H"`, `"12C"`, …) |
| `E` | `Float64` | Kinetic energy (MeV/u) |
| `A` | `Int` | Mass number |
| `Z` | `Int` | Atomic number |
| `LET` | `Float64` | Linear energy transfer (keV/µm) |
| `rho` | `Float64` | Density (g/cm³, typically 1.0) |

### `Irrad`
```julia
Irrad(dose, kR, doserate)
```
| Field | Meaning |
|---|---|
| `dose` | Prescribed dose (Gy) |
| `kR` | Relative biological effectiveness scaling (default 1.0) |
| `doserate` | Dose rate (Gy/h) |

### `GSM2`
```julia
GSM2(r, a, b, rd, Rn)
```
| Field | Meaning |
|---|---|
| `r` | Correct repair rate (h⁻¹) |
| `a` | Mis-repair rate (h⁻¹) |
| `b` | Pairwise interaction rate (h⁻¹) |
| `rd` | Domain radius (µm) |
| `Rn` | Nucleus radius (µm) |

### `AT` (Amorphous Track)
```julia
AT(ion, E, A, Z, LET, rho, Rc, Rp, Rk, Kp)
```
| Field | Meaning |
|---|---|
| `Rc` | Core radius (µm) |
| `Rp` | Penumbra radius (µm) |
| `Rk` | Halo radius used for hit sampling (= `Rp`) |
| `Kp` | Dose amplitude coefficient |

### `SimulationTimeSeries`

Returned by `run_simulation_abm!`. Fields:

```julia
ts.time          # Vector{Float64} — simulation time (h)
ts.total_cells   # Vector{Int32}
ts.g0_cells      # Vector{Int32}
ts.g1_cells      # ...
ts.s_cells
ts.g2_cells
ts.m_cells
ts.stem_cells
ts.non_stem_cells
```

---

## Configuration Reference

### `setup_IonIrrad!` keywords

| Keyword | Default | Meaning |
|---|---|---|
| `type_AT` | `"KC"` | Track model: `"KC"` (Kiefer-Chatterjee) or `"LEM"` |
| `kR` | `1.0` | RBE scaling factor |
| `dose_rate` | `0.18` | Dose rate (Gy/h) |

### `setup_cell_lattice!` keywords

| Keyword | Default | Meaning |
|---|---|---|
| `ParIrr` | `"false"` | `"true"` = partial irradiation |
| `track_seg` | `true` | Enable track segmentation |
| `full_cycle` | `true` | Distribute cells across full cycle at t=0 |

### `run_simulation_abm!` keywords

| Keyword | Default | Meaning |
|---|---|---|
| `nat_apo` | `1e-10` | Natural apoptosis rate (h⁻¹) |
| `terminal_time` | `72.0` | Simulation end time (h) |
| `snapshot_times` | `[0,12,24,48]` | Times at which to save `CellPopulation` |
| `print_interval` | `6.0` | Console print every N hours |
| `verbose` | `false` | Detailed per-event logging |
| `return_dataframes` | `false` | Return DataFrames instead of CellPopulations in `snaps` |
| `update_input` | `true` | Write final state back into the input `cell_df` |

---

## Supported Ions

| Ion label | Z | A | Notes |
|---|---|---|---|
| `"1H"` | 1 | 1 | Proton |
| `"2H"` | 1 | 2 | Deuteron |
| `"4He"` | 2 | 4 | Alpha particle |
| `"3He"` | 2 | 3 | Helium-3 |
| `"12C"` | 6 | 12 | Carbon ion (hadron therapy) |
| `"16O"` | 8 | 16 | Oxygen ion |

Stopping power tables are pre-computed for H, He, Li, Be, B, C, N, O in water
(`data/stoppingpower/`). Energy range: 0.01 – 1000 MeV/u.

LET at a given energy is obtained by:
```julia
LET = linear_interpolation(particle, E, sp)   # keV/µm
```

---

## Physics Models

### Track Structure

Two models are supported via `type_AT`:

**Kiefer-Chatterjee (`"KC"`)**
- Core radius: `Rc = 0.01116 · β` µm
- Penumbra: `Rp = 0.0616 · (E/A)^1.7` µm
- Amplitude: `Kp = 1.25e-4 · (z_eff/β)²`
- Effective charge: `z_eff = Z · (1 - exp(-125β/Z^(2/3)))`

**Local Effect Model (`"LEM"`)**
- Core: `Rc = 0.01` µm (fixed)
- Penumbra: `Rp = 0.05 · (E/A)^1.7` µm

### Oxygen Enhancement Ratio

```julia
OER = calculate_OER(LET, O)   # O in mmHg
```

Uses the empirical formula:
`OER = (b·(a·M0 + LET^g) / (a + LET^g) + O) / (b + O)`

with `M0 = 3.4, b = 0.41, a = 8.27e5, g = 3.0`.

### Energy Loss Along Track

When `track_seg = false`, energy is propagated layer-by-layer using an explicit
Euler integration of the Bethe stopping power formula:
```julia
E_res, LET_res = residual_energy_after_distance(E, Z, A, step_um, ion, sp)
```

---

## GPU Acceleration

`MC_dose_fast!` dispatches automatically:

```
CUDA.functional() && Npar ≥ GPU_PARTICLE_THRESHOLD  →  GPU kernel
otherwise                                            →  CPU threads
```

Default threshold: `1_000_000` particles. Override:
```julia
# Always use GPU (if available)
global GPU_PARTICLE_THRESHOLD = 0

# Always use CPU
global GPU_PARTICLE_THRESHOLD = typemax(Int)
```

The GPU kernel maps one CUDA thread per domain center and loops over primaries
inside the kernel. No explicit CPU/GPU data migration is needed — the framework
handles it internally.

---

## File Structure

```
GSM2_Julia/
├── example/
│   └── spheroid.jl              ← KEY EXAMPLE (start here)
│
├── src/
│   ├── load_utilities.jl        ← single include entry point
│   ├── utilities_structures.jl  ← all structs (Ion, GSM2, AT, ...)
│   ├── utilities_general.jl     ← random hit sampling (box, circle, halo)
│   ├── utilities_radiation.jl   ← track radii, stopping power, beam geometry
│   ├── utilities_AT_computation.jl  ← amorphous track dose integrals
│   ├── utilities_GSM2.jl        ← damage model, kappa, OER, domain survival
│   ├── utilities_biology.jl     ← cell population creation, cycle, neighbours
│   ├── utilities_env.jl         ← setup_* functions, energy binning
│   ├── utilities_dose_computation.jl     ← CPU Monte Carlo kernels
│   ├── utilities_dose_computation_GPU.jl ← GPU kernel + auto dispatch
│   ├── utilities_abm.jl         ← ABM event loop, repair simulation
│   ├── utilities_plot.jl        ← dose/damage/survival/cycle plots
│   └── utilities_plot_abm.jl    ← population dynamics, spatial, animation
│
├── data/
│   ├── stoppingpower/           ← H,He,Li,Be,B,C,N,O in water (txt)
│   └── <condition>/             ← simulation output CSVs
│
├── scripts/                     ← batch scripts for multiple conditions
└── results/                     ← analysis and publication plots
```

---

## Comment Convention

All source files use a consistent inline annotation scheme:

```julia
#!   section / file headers
#~   category groupings
#?   public function or constant name
#    description text
```

Each utility file begins with a complete **function index** listing all
public functions with their signatures, making it easy to browse without
reading the full implementations.

---

## Tips

- **Start with `example/spheroid.jl`** — it is the canonical reference and
  exercises every function in the framework.
- Use `setup()` for quick interactive experiments; use the individual
  `setup_*!` functions when you need to inspect or modify intermediate state.
- `cell_df` is a standard Julia `DataFrame` — all standard DataFrame
  operations (filtering, grouping, CSV export) work directly on it.
- Snapshots returned by `run_simulation_abm!` are `CellPopulation` structs.
  Convert to DataFrame with `to_dataframe(snap; alive_only=true)`.
- Set `Random.seed!(n)` before any stochastic call (`MC_loop_damage!`,
  `compute_times_domain!`, `run_simulation_abm!`) for reproducible results.
- Threading is enabled automatically when Julia is started with
  `julia --threads auto`. Most inner loops use `Threads.@threads`.
