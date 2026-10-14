# GSM<sup>2</sup> Julia — Ion Irradiation & Agent-Based Model Framework

A Julia framework for simulating the full chain from physical ion-beam delivery to post-irradiation cell-population dynamics: particle-by-particle energy deposition, stochastic DNA damage and repair with the Generalized Stochastic Microdosimetric Model (GSM<sup>2</sup>), and an event-driven agent-based model (ABM) of the cell population with cell-cycle progression, division, contact inhibition, delayed radiation-induced death and migration. Designed for tumour spheroids but applicable to any 3-D cell lattice.

```
Ion beam → MC dose deposition → DNA damage (GSM2) → Fate (repair / commitment) → ABM cell dynamics
             (particle by particle, in time, at a prescribed dose rate)
```

---

## Table of Contents

1. [Overview](#overview)
2. [What's new](#whats-new)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [Pipeline in Detail](#pipeline-in-detail)
   - [Loading Utilities](#1-loading-utilities)
   - [GSM<sup>2</sup> Parameters](#2-gsm2-parameters)
   - [Environment Setup](#3-environment-setup)
   - [Monte Carlo Dose](#4-monte-carlo-dose)
   - [DNA Damage](#5-dna-damage)
   - [Survival](#6-survival)
   - [Fate Assignment and Repair Timers](#7-fate-assignment-and-repair-timers)
   - [ABM Simulation](#8-abm-simulation)
   - [Plotting](#9-plotting)
6. [Biological Rules](#biological-rules)
7. [Fractionated and Continuous Irradiation](#fractionated-and-continuous-irradiation)
8. [Key Structs](#key-structs)
9. [Configuration Reference](#configuration-reference)
10. [Supported Ions](#supported-ions)
11. [Physics Models](#physics-models)
12. [GPU Acceleration](#gpu-acceleration)
13. [Scripts and Reproducing the Paper Figures](#scripts-and-reproducing-the-paper-figures)
14. [File Structure](#file-structure)
15. [Comment Convention](#comment-convention)
16. [Tips and Pitfalls](#tips-and-pitfalls)

---

## Overview

The framework is composed of independent utility modules (`src/utilities_*.jl`):

| Stage | Module |
|---|---|
| Geometry, beam, cell lattice, setup | `utilities_env.jl` |
| Stopping power, track radii | `utilities_radiation.jl` |
| Amorphous-track dose | `utilities_AT_computation.jl` |
| Monte Carlo dose (CPU + GPU) | `utilities_dose_computation.jl`, `utilities_dose_computation_GPU.jl` |
| DNA damage, OER, κ, domain survival | `utilities_GSM2.jl` |
| Oxygenation profiles | `utilities_oxygen.jl` |
| Cell biology (cycle durations, population) | `utilities_biology.jl` |
| Repair kinetics, fate, cell cycle, migration, event engine, continuous irradiation | `utilities_abm.jl` |
| Dose/damage/survival plots | `utilities_plot.jl` |
| Population dynamics plots | `utilities_plot_abm.jl` |

`utilities_abm.jl` is the single ABM file: the former `utilities_abm_migration.jl`, `utilities_abm_fast.jl` and `utilities_checkpoint.jl` are merged into it and must **not** be included anymore.

---

## What's new

| Feature | Function / setting | Default |
|---|---|---|
| Event engine with a binary min-heap and local updates: cost per event independent of population size | `run_simulation_abm!(…; fast = true)` | on |
| Original loops kept for validation | `fast = false`, `compare_engines` | — |
| Two fate models: previous behaviour, or delayed death with incremental fate assignment | `set_fate_model!(:legacy \| :incremental)` | `:legacy` |
| Delayed radiation-induced death (doomed cells stay on the lattice) | `sample_death_delay`, `DEATH_DELAY_MEAN_CYCLES` | with `:incremental` |
| Checkpoint model: arrest in all phases, or G2/M checkpoint only | `set_checkpoint!(:all \| :g2)` | `:all` |
| Contact-inhibition threshold | `set_quiescence_threshold!(k)` | `k = 0` |
| Distance-limited proliferation (growth plateau) | `set_division_limit!` | off |
| Lineage tracking | `set_founder_tracking!`, `founders()` | off |
| Migration (random walk) | `run_simulation_abm_migration!` | — |
| Continuous irradiation driver | `irradiate_continuous_abm!` | — |
| Death pathway returned by the repair SSA | `compute_repair_domain` → death code 2 (a) / 3 (b) | — |

All defaults reproduce the behaviour of the previous version, so existing scripts run unchanged.

---

## Installation

### Requirements

Julia ≥ 1.9 (developed and tested with 1.12). Install the required packages once:

```julia
using Pkg
Pkg.add([
    "CSV", "DataFrames", "JLD2", "DelimitedFiles",
    "Distributions", "Random", "Statistics", "StatsBase",
    "Plots", "StatsPlots",
    "GLM", "Optim", "LsqFit",
    "ProgressBars", "ProgressMeter",
    "InlineStrings", "Printf", "SparseArrays",
    "CUDA",           # optional — CPU fallback is automatic
])
```

### Clone

```bash
git clone https://github.com/francescogcordoni/GSM2_Julia.git
cd GSM2_Julia
```

---

## Quickstart

The canonical example is **`example/spheroid.jl`**. It runs the complete pipeline (setup → dose → damage → survival → fate → ABM → plots) for a spheroid of radius 350 µm and saves the figures to `example/output/spheroid/`.

```bash
julia --threads auto example/spheroid.jl
```

Edit the **PARAMETERS** block at the top of the script to change particle, energy, dose, geometry or fate model without touching any other code:

```julia
PARTICLE      = "1H"          # "1H", "4He", "12C", "16O"
ENERGY_MEV_U  = 100.0         # MeV per nucleon
DOSE_GY       = 2.0           # prescribed dose (Gy)
TUMOR_RADIUS  = 350.0         # spheroid radius (µm)
X_BOX         = 550.0         # box half-size (µm): keep ≥ TUMOR_RADIUS + ~150 µm
TERMINAL_TIME = 72.0          # post-irradiation ABM window (h)
const FATE    = :incremental  # :incremental (delayed death) or :legacy
```

The script prints the mean cell dose, the GSM<sup>2</sup> mean survival probability and the clonogenic survival fraction; the last two should agree.

---

## Pipeline in Detail

### 1. Loading Utilities

```julia
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()   # Dict{String, StoppingTable}, from data/stoppingpower/
```

`load_utilities.jl` includes all source files in dependency order. Check that it includes `utilities_abm.jl` and none of the old ABM files (`utilities_abm_migration.jl`, `utilities_abm_fast.jl`, `utilities_checkpoint.jl`).

---

### 2. GSM<sup>2</sup> Parameters

The **Generalized Stochastic Microdosimetric Model (GSM<sup>2</sup>)** describes the induction, repair, misrepair and pairwise interaction of sublethal (X) and lethal (Y) DNA lesions. Parameters are cell-cycle-phase dependent. Pass a `Vector{GSM2}` of length 4:

```julia
gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)   # G1 / G0
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)   # S phase
gsm2_cycle[3] = GSM2(r_G2, a_G2, b_G2, rd, Rn)   # G2 / M
gsm2_cycle[4] = GSM2(r,    a,    b,    rd, Rn)   # average (fallback)
```

| Parameter | Meaning |
|---|---|
| `r` | Repair rate of sublethal lesions (h⁻¹) |
| `a` | First-order conversion X → Y (misrepair) (h⁻¹) |
| `b` | Pairwise interaction X + X → Y (h⁻¹) |
| `rd` | Domain radius (µm) |
| `Rn` | Nucleus radius (µm) |

Values calibrated to phase-specific LQ data (Sinclair-type synchronised survival curves):

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

**Time scaling.** The repair SSA multiplies every waiting time by `au` (keyword of `compute_repair_domain`; `au = 4` in the paper). Acute survival depends only on `a/r` and `b/r`, so `au` sets the absolute repair time scale: the effective repair half-time of G1 cells is `au·ln2/(r + a) ≈ 1 h`. When converting a dose rate into a particle arrival rate, do **not** multiply by `au` (the clock is in hours).

---

### 3. Environment Setup

The setup functions inject global variables into `Main` with `@eval`. In Julia ≥ 1.12, a global created during a function call is not visible to code that is already running, so **the first setup must be done at top level** of the script (as in `example/spheroid.jl`); later calls inside functions only update existing globals. Inside functions, read the globals with `Base.invokelatest(getfield, Main, :name)`.

#### Option A — individual steps (explicit, inspectable)

```julia
# 1. GSM2 geometry: domain centers, nucleus template
setup_GSM2!(r, a, b, rd, Rn)
# Injects: gsm2, center_x, center_y, domain

# 2. Ion and irradiation descriptors
setup_IonIrrad!(dose, E, particle; type_AT = "KC")
# Injects: ion, irrad, LET, Rc, Rp, Rk, Kp, A, Z, type_AT

# 3. Cell lattice (positions, neighbour lists, layer assignment)
N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))
setup_cell_lattice!("circle", X_box, R_cell, N_sideVox, N_CellsSide;
                    ParIrr = "false", track_seg = true, full_cycle = true)
# Injects: cell_df, N, nodes_positions, N_CellsSide, track_seg, full_cycle

# Required before setup_cell_population!:
@eval Main begin
    tumor_radius = 350.0
    X_voxel      = 1100.0
    X_box        = 550.0
end

# 4. Biological attributes (cell cycle, damage vectors, neighbours)
setup_cell_population!("circle", X_box, R_cell, N_sideVox, N_CellsSide, gsm2_cycle[4];
                       ParIrr = "false")
# Injects: rel_center_x/y, df_center_x/y, at, num_cols

# 5. Per-layer irradiation conditions (AT objects indexed by energy step)
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)
# Injects: irrad_cond, lets, energies, num_energy_steps, cell_df

# 6. Oxygenation profile (analytic diffusion, or uniform)
set_oxygen_diffusion!(cell_df; density = :mean, plot_oxygen = false)
# cell_df.O .= 21.0                  # uniform normoxia instead
```

#### Option B — single high-level wrapper

```julia
out = setup(E, particle, dose, tumor_radius;
            X_box = 550.0, X_voxel = 1100.0, R_cell = 15.0,
            target_geom = "circle", calc_type = "full",
            type_AT = "KC", track_seg = true)
# Returns NamedTuple: ion, irrad, cell_df, at_start, R_beam, x_beam, y_beam,
#                     O2_mean, Npar, zF, D, T
```

#### Key geometry parameters

| Parameter | Description | Typical value |
|---|---|---|
| `X_box` | Half-side of simulation box (µm) | `tumor_radius + 150…250` |
| `X_voxel` | Voxel side for beam-radius calculation (µm) | `2 * X_box` |
| `R_cell` | Cell radius (µm); lattice spacing = `2 R_cell` | 15 |
| `tumor_radius` | Spheroid radius (µm) | 150 – 500 |
| `target_geom` | Cross-section: `"circle"` or `"square"` | `"circle"` |
| `calc_type` | Beam radius: `"full"` (whole target) or `"fast"` (voxel) | `"full"` |
| `type_AT` | Track-structure model: `"KC"` or `"LEM"` | `"KC"` |
| `track_seg` | `true` = constant LET across depth | `true` |
| `full_cycle` | `true` = cells start distributed over the full cycle | `true` |
| `ParIrr` | `"true"` = partial irradiation | `"false"` |

**Box size.** The lattice is a cube with periodic, 26-neighbour (Moore) connectivity. If the spheroid reaches a face of the box, its surface cells see occupied neighbours on the opposite face and are wrongly put in G0. Keep at least 150–200 µm of free lattice around the spheroid for the simulated period.

---

### 4. Monte Carlo Dose

```julia
R_beam, x_beam, y_beam = calculate_beam_properties("full", "circle", X_box, X_voxel, tumor_radius)
F    = irrad.dose / (1.602e-9 * LET)           # fluence (particles/cm²)
Npar = round(Int, F * π * R_beam^2 * 1e-8)     # particles through the beam

MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df,
             df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg;
             seed = 1234)                       # seed optional

# Auto CPU/GPU dispatch
MC_dose_fast!(ion, Npar, R_beam, irrad_cond, cell_df,
              df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg)
```

| Column | Type | Meaning |
|---|---|---|
| `dose_cell` | `Float64` | Mean dose of the cell (Gy) |
| `dose` | `Vector{Float64}` | Dose per domain column |

The beam radius must cover the whole target: pass the **actual** spheroid radius (for a grown spheroid, its current radius). Cells outside the beam receive no dose.

---

### 5. DNA Damage

```julia
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)
```

Lesions are Poisson-sampled from the stored dose (calling it again resamples fresh lesions from the same dose):

| Column | Type | Meaning |
|---|---|---|
| `dam_X_dom` | `Vector{Int}` | Sublethal lesions per sub-domain |
| `dam_Y_dom` | `Vector{Int}` | Lethal lesions per sub-domain |
| `dam_X_total`, `dam_Y_total` | `Int` | Totals per cell |

Damage rates per sub-domain: `λX = κ · d` with `κ = 9 · kappa_yield / (n_repeat · N_domains)`, `λY = 10⁻³ λX`; `kappa_yield = calculate_kappa(ion.ion, LET, O; OER_bool = true)`.

---

### 6. Survival

```julia
compute_cell_survival_GSM2!(cell_df, gsm2_cycle; NFrac = 1)
alive = cell_df.is_cell .== 1
p_hat, ci_lo, ci_hi = survival_ci(cell_df.sp[alive])
```

`cell_df.sp` is the exact GSM<sup>2</sup> probability that the cell completes repair (0 if any Y lesion is present).

> **Warning.** `compute_cell_survival_GSM2!` resets `death_time`, `cycle_time` and `recover_time` of **every** cell. This is harmless with the `:legacy` fate model (which reassigns all timers afterwards) but freezes undamaged cells with `:incremental`. In incremental mode either compute `sp` without resets:
> ```julia
> for i in findall(==(1), cell_df.is_cell)
>     ph = cell_df.cell_cycle[i]
>     g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
>     cell_df.sp[i] = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], g)
> end
> ```
> or restore the clocks from a pre-irradiation copy (see `example/spheroid.jl`).

---

### 7. Fate Assignment and Repair Timers

```julia
set_fate_model!(:incremental)            # or :legacy (default); once, before the run
compute_times_domain!(cell_df, gsm2_cycle;
                      nat_apo = 1e-10, terminal_time = Inf, summary = true)
```

| Model | Behaviour |
|---|---|
| `:legacy` (default) | Resets the timers of all alive cells; cells with direct Y lesions are removed at t = 0 (`is_cell = 0`); other lethal events at the SSA time; survivors get fresh phase times (hard-coded durations). Used by the calibration scripts. |
| `:incremental` | Processes only alive, non-doomed, damaged cells; others keep their timers. Lethally damaged cells become **doomed** (delayed death, below). Required for repeated irradiation and for `set_checkpoint!(:g2)`. |

`terminal_time` is the repair window: with `Inf` all damage is resolved; with a finite window, unrepaired cells keep their residual damage for the next call.

Columns written:

| Column | Meaning |
|---|---|
| `death_time` | Time of removal (finite = doomed) |
| `recover_time` | Time at which repair is completed |
| `cycle_time` | Time at which the current phase ends |
| `death_type` | 1 = direct Y (or mitotic death), 2 = misrepair `a`, 3 = interaction `b` |
| `is_death_rad` | 1 for radiation-induced death |
| `cycle_hold` | Remaining G2 time of cells held at the G2 checkpoint (`:g2` only) |

**Clonogenic survival** (the radiobiological surviving fraction) is the fraction of cells alive and not doomed right after this call:

```julia
SF = count((cell_df.is_cell .== 1) .& .!isfinite.(cell_df.death_time)) / N_init
```

The cell number at later times includes regrowth and (with `:incremental`) doomed cells not yet removed; it is not a survival fraction.

---

### 8. ABM Simulation

```julia
ts, snaps = run_simulation_abm!(cell_df;
    nat_apo           = 1e-10,
    terminal_time     = 72.0,
    snapshot_times    = [0, 6, 12, 24, 48, 72],
    print_interval    = 6.0,
    verbose           = true,
    return_dataframes = false,
    update_input      = true,
    record_dt         = 0.0)   # > 0: record the time series at most every record_dt h
```

**Engine.** The default engine (`fast = true`) stores the next cycle, death and migration time of every cell as absolute times in a binary min-heap; each event updates only the cells within two lattice sites of the event cell (lazy invalidation of outdated heap entries). The cost per event is O(log N) and independent of the population size. `fast = false` runs the original loop (scan of all cells per event). With the same seed both give the same trajectory; check with:

```julia
compare_engines(cell_df; seed = 1, terminal_time = 48.0)                     # no migration
compare_engines(cell_df; seed = 1, terminal_time = 48.0, D = 50.0, h = 30.0)  # with migration
```

**Returns:** `ts::SimulationTimeSeries` (total and per-phase counts; doomed cells are counted in the phase they were arrested in) and `snaps::Dict{Int, CellPopulation}` (alive cells only; a time is captured only if an event falls in that hour).

On return, the timers in `cell_df` are relative to the **last event**, not to `terminal_time`. When chaining ABM segments (fractions, continuous irradiation), advance the clocks by the idle remainder:

```julia
ts, _ = run_simulation_abm!(cell_df; terminal_time = Δ, snapshot_times = Int[], verbose = false,
                            return_dataframes = false, update_input = true)
_advance_idle!(cell_df, Δ - (isempty(ts.time) ? 0.0 : ts.time[end]))
```

**Migration:**

```julia
ts, snaps = run_simulation_abm_migration!(cell_df, D_m, 2R_cell; terminal_time = 72.0)
```

---

### 9. Plotting

#### Dose & Damage (`utilities_plot.jl`)

```julia
plot_scalar_cell(cell_df, :dose_cell)
plot_scalar_cell(cell_df, :dam_X_total)
plot_scalar_cell(cell_df, :sp; xscale = :log10)
plot_scalar_cell(cell_df, :dose_cell; layer_plot = true)
plot_damage(cell_df)
plot_times(cell_df; show_means = true)
plot_initial_distributions(cell_df)
plot_phase_proportions_alive(cell_df; title_text = "Pre-irradiation")
plot_cell_cycle_distribution(cell_df; phase_plot = false, half_sphere = true)
print_phase_distribution(cell_df; label = "t = 0h")
```

#### Population Dynamics (`utilities_plot_abm.jl`)

```julia
plot_cell_dynamics(ts); plot_phase_dynamics(ts); plot_phase_proportions(ts)
plot_phase_stacked(ts); plot_cycling_vs_quiescent(ts); plot_growth_rate(ts; window_size = 10)
plot_simulation_results(ts); plot_analysis_dashboard(ts)
plot_snapshot_comparison(snaps; metric = :cell_cycle)
plot_spatial_distribution(snap; color_by = :cell_cycle)
create_spatial_animation(snaps; output_file = "anim.gif", fps = 2)
export_timeseries_csv(ts, "timeseries.csv"); print_simulation_summary(ts)
```

Publication figures (vector PDF, no titles) are produced by the `plot_*.jl` scripts listed [below](#scripts-and-reproducing-the-paper-figures), which read the saved CSVs without rerunning simulations.

---

## Biological Rules

| Rule | Description | Setting |
|---|---|---|
| Cell cycle | G1 → S → G2 → M → division; Gamma-distributed phase durations from `PHASE_DURATIONS` | `PHASE_DURATIONS` |
| Division | At the end of M, into a random empty neighbour (26-neighbour lattice); both daughters in G1 | — |
| Contact inhibition | G0 when ≤ k of the 26 neighbours are empty; back to G1 when space is freed (unless doomed) | `set_quiescence_threshold!(k; cell_df)`, default k = 0 |
| Growth plateau | Division accepted with p(r) = 1/(1 + exp((r − R50)/w)) | `set_division_limit!(cell_df; R50, w)` |
| Arrest after damage | `:all`: clock held in G1/S/G2 until repair; `:g2`: damaged cells keep cycling through G1/S and are held only in G2 | `set_checkpoint!(:all \| :g2)` |
| Mitotic death | A cell whose mitosis ends before repair is completed is committed to death | — |
| Delayed death (`:incremental`) | Doomed cells keep their site, do not divide, and are removed after `T ~ Gamma(3, m·Tc/3)`, with m = 1 (G2, M), 1.25 (S), 1.5 (G1), 2.5 (G0) and Tc the current mean cycle length | `DEATH_DELAY_MEAN_CYCLES` |
| G0 radiosensitivity | G0 cells use the G1 parameters | — |
| Lineage tracking | Daughters inherit the founder id of their parent; migration carries it | `set_founder_tracking!(cell_df)`, `founders()` |

---

## Fractionated and Continuous Irradiation

**Fractions on a growing spheroid.** For each fraction, repeat the irradiation steps on the **current** `cell_df` with `set_fate_model!(:incremental)`:

```julia
setup_IonIrrad!(dose_fx, E, particle; type_AT = "KC")
R_beam, _, _ = calculate_beam_properties("full", "circle", X_box, X_voxel, current_radius)
setup_irrad_conditions!(ion, irrad, "KC", cell_df, true)          # on the DataFrame being irradiated
MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df, df_center_x, df_center_y, deepcopy(at),
             gsm2_cycle, "KC", true; seed = s)
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)
compute_times_domain!(cell_df, gsm2_cycle; terminal_time = Inf)   # no compute_cell_survival_GSM2!
# … ABM until the next fraction (+ _advance_idle!)
```

See `example/spheroid_fractionation.jl`.

**Continuous low-dose-rate irradiation (high-energy protons):**

```julia
ts, res = irradiate_continuous_abm!(cell_df, gsm2_cycle;
              dose = 4.0, dose_rate_gyh = 0.5, K_tot = K, L_tot = 1e-3K,
              checkpoint = :g2, max_pulse_dose = 0.05, max_pulse_h = 0.1, seed = 1)
res.SF_plated, res.N_rel, res.phase_frac
```

The dose is delivered as short pulses (≤ 0.05 Gy and ≤ 0.1 h); each pulse adds uniform Poisson damage, assigns fates with a repair window equal to the pulse interval, runs the ABM over the same interval and advances the idle time. Uses the incremental model.

**Dose-rate verification against Lea–Catcheside** (static G1 population, any ion): `scripts/continuous_dr_events.jl` generates lesions once as a sparse list of events, reuses them for every dose and dose rate (common random numbers) and compares the simulated β(Ḋ) with the Lea–Catcheside prediction with μ = (r + a)/au.

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
| `A`, `Z` | `Int` | Mass and atomic number |
| `LET` | `Float64` | Linear energy transfer (keV/µm) |
| `rho` | `Float64` | Density (g/cm³) |

### `Irrad`
```julia
Irrad(dose, kR, doserate)
```

### `GSM2`
```julia
GSM2(r, a, b, rd, Rn)
```

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

### `CellPopulation`
Structure-of-arrays representation used by the ABM (`CellPopulation(cell_df)`, `to_dataframe(pop; alive_only = true)`, `update_dataframe!(cell_df, pop)`).

### `SimulationTimeSeries`
```julia
ts.time, ts.total_cells, ts.g0_cells, ts.g1_cells, ts.s_cells, ts.g2_cells, ts.m_cells,
ts.stem_cells, ts.non_stem_cells
```

---

## Configuration Reference

### `run_simulation_abm!` / `run_simulation_abm_migration!` keywords

| Keyword | Default | Meaning |
|---|---|---|
| `nat_apo` | `1e-10` | Natural apoptosis parameter |
| `terminal_time` | `48.0` | Simulation end time (h) |
| `snapshot_times` | `[1,6,12,24]` | Hours at which to save a `CellPopulation` |
| `print_interval` | `1.0` | Console print every N hours |
| `verbose` | `true` | Progress output |
| `fast` | `true` | Heap engine (`false` = original loop) |
| `record_dt` | `0.0` | Time-series resolution (0 = every event) |
| `return_dataframes` | `true` | DataFrame wrapper: snapshots as DataFrames |
| `update_input` | `true` | DataFrame wrapper: write the final state back |

### Global settings (`utilities_abm.jl`)

| Function | Values | Default |
|---|---|---|
| `set_fate_model!(m; death_delay)` | `:legacy`, `:incremental` | `:legacy` |
| `set_checkpoint!(m)` | `:all`, `:g2` | `:all` |
| `set_quiescence_threshold!(k; cell_df)` | integer ≥ 0 | 0 |
| `set_division_limit!(cell_df; R50, w, center, rmax)` / `clear_division_limit!()` | — | off |
| `set_founder_tracking!(cell_df)` / `clear_founder_tracking!()` | — | off |

These are global: set them once, before any threaded loop.

### `setup_IonIrrad!` and `setup_cell_lattice!` keywords

| Function | Keyword | Default | Meaning |
|---|---|---|---|
| `setup_IonIrrad!` | `type_AT` | `"KC"` | Track model |
| `setup_cell_lattice!` | `ParIrr` | `"false"` | Partial irradiation |
| `setup_cell_lattice!` | `track_seg` | `true` | Constant LET across depth |
| `setup_cell_lattice!` | `full_cycle` | `true` | Cells distributed across the full cycle at t = 0 |

---

## Supported Ions

| Ion label | Z | A | Notes |
|---|---|---|---|
| `"1H"` | 1 | 1 | Proton |
| `"2H"` | 1 | 2 | Deuteron |
| `"4He"` | 2 | 4 | Alpha particle |
| `"3He"` | 2 | 3 | Helium-3 |
| `"12C"` | 6 | 12 | Carbon ion |
| `"16O"` | 8 | 16 | Oxygen ion |

Stopping-power tables for H, He, Li, Be, B, C, N, O in water are in `data/stoppingpower/` (0.01 – 1000 MeV/u). LET at a given energy: `LET = linear_interpolation(particle, E, sp)` (keV/µm).

---

## Physics Models

### Track Structure

**Kiefer–Chatterjee (`"KC"`)**: `Rc = 0.01116 · β` µm, `Rp = 0.0616 · (E/A)^1.7` µm, `Kp = 1.25e-4 · (z_eff/β)²`, `z_eff = Z · (1 − exp(−125β/Z^(2/3)))`.

**Local Effect Model (`"LEM"`)**: `Rc = 0.01` µm, `Rp = 0.05 · (E/A)^1.7` µm.

### Oxygen Enhancement Ratio

```julia
OER = calculate_OER(LET, O)
```
`OER = (K·(a·M0 + LET^g)/(a + LET^g) + O) / (K + O)` with `M0 = 3.4`, `K = 0.41`, `a = 8.27e5`, `g = 3.0`; `O` in the units of `cell_df.O`.

### Oxygenation

`set_oxygen_diffusion!` computes the analytic quasi-steady-state profile of a reaction–diffusion equation with zero-order consumption (necrotic radius from the closed-form solution). The profile is static during the ABM; recompute it when the geometry changes (e.g. before each fraction).

### Energy Loss Along Track

With `track_seg = false`, energy is propagated layer by layer: `E_res, LET_res = residual_energy_after_distance(E, Z, A, step_um, ion, sp)`.

---

## GPU Acceleration

`MC_dose_fast!` dispatches automatically:

```
CUDA.functional() && Npar ≥ GPU_PARTICLE_THRESHOLD  →  GPU kernel
otherwise                                            →  CPU threads
```

Default threshold: `1_000_000` particles (`global GPU_PARTICLE_THRESHOLD = 0` forces GPU, `typemax(Int)` forces CPU). The ABM runs on the CPU.

---

## Scripts and Reproducing the Paper Figures

Every simulation script saves its DataFrames as CSV; every `plot_*` script reads them and writes vector PDFs without rerunning the simulation.

| Analysis | Simulation script | Plot script | Output folder |
|---|---|---|---|
| Single irradiation, full pipeline | `example/spheroid.jl` | (built in) | `example/output/spheroid/` |
| Temporal evolution, several ions | `scripts/spheroid_temporal_evolution.jl` | `scripts/plot_spheroid_3d.jl` | `data/spheroid_temporal_evolution/` |
| Two ions + control, PDFs | `example/spheroid_ions.jl` | (built in) | `example/output/spheroid_ions/` |
| Continuous dose rate vs Lea–Catcheside | `scripts/continuous_dr_events.jl` | `scripts/plot_doserate_relative.jl` | `data/continuous_dr_events/` |
| Split dose with cell cycle | `scripts/split_repopulation.jl` | `scripts/plot_split_repopulation.jl` | `data/split_repopulation/` |
| Fractionation (3 × 2 Gy) | `example/spheroid_fractionation.jl` | `example/plot_spheroid_fractionation.jl` | `example/output/spheroid_fractionation/` |
| Regrowth delay and spheroid control | `example/spheroid_regrowth.jl` | (built in) | `example/output/spheroid_regrowth/` |
| Inverse dose-rate test (clones) | `scripts/inverse_doserate_continuous.jl` | (built in) | `data/inverse_doserate_continuous/` |
| Inverse dose-rate test (spheroid, G2 checkpoint) | `scripts/idre_abm_continuous.jl` | (built in) | `data/idre_abm_continuous/` |
| Computational cost | `example/benchmark_computation.jl` | `example/plot_benchmark.jl` | `example/output/benchmark/` |

Run with `julia --threads auto <script>`. All scripts seed their random streams; realisations of different conditions share the same seed up to the irradiation (common random numbers).

---

## File Structure

```
GSM2_Julia/
├── example/
│   ├── spheroid.jl                    ← KEY EXAMPLE (start here)
│   ├── spheroid_ions.jl               ← 1H vs 12C vs control, PDF figures
│   ├── spheroid_fractionation.jl      ← fractionated irradiation, replicates
│   ├── plot_spheroid_fractionation.jl
│   ├── spheroid_regrowth.jl           ← regrowth delay, spheroid control probability
│   ├── benchmark_computation.jl       ← timing vs size, ion and stage
│   ├── plot_benchmark.jl
│   └── output/
│
├── scripts/
│   ├── continuous_dr_events.jl        ← dose-rate verification (Lea–Catcheside)
│   ├── plot_doserate_relative.jl
│   ├── spheroid_temporal_evolution.jl
│   ├── plot_spheroid_3d.jl
│   ├── split_repopulation.jl
│   ├── plot_split_repopulation.jl
│   ├── inverse_doserate_continuous.jl
│   └── idre_abm_continuous.jl
│
├── src/
│   ├── load_utilities.jl              ← single include entry point
│   ├── utilities_structures.jl        ← structs (Ion, GSM2, AT, CellPopulation, …)
│   ├── utilities_general.jl           ← random hit sampling (box, circle, halo)
│   ├── utilities_radiation.jl         ← track radii, stopping power, beam geometry
│   ├── utilities_AT_computation.jl    ← amorphous-track dose integrals
│   ├── utilities_GSM2.jl              ← damage model, κ, OER, domain survival
│   ├── utilities_oxygen.jl            ← oxygenation profiles
│   ├── utilities_biology.jl           ← population creation, cycle durations, neighbours
│   ├── utilities_env.jl               ← setup_* functions, energy binning
│   ├── utilities_dose_computation.jl      ← CPU Monte Carlo kernels
│   ├── utilities_dose_computation_GPU.jl  ← GPU kernel + auto dispatch
│   ├── utilities_abm.jl               ← repair SSA, fate, cell cycle, migration, event engine
│   ├── utilities_plot.jl              ← dose/damage/survival/cycle plots
│   └── utilities_plot_abm.jl          ← population dynamics, spatial, animation
│
└── data/
    ├── stoppingpower/                 ← H … O in water
    └── <analysis>/                    ← simulation output CSVs and figures
```

---

## Comment Convention

```julia
#!   section / file headers
#~   category groupings
#?   public function or constant name
#    description text
```

Each utility file begins with a function index listing all public functions with their signatures.

---

## Tips and Pitfalls

- **Start with `example/spheroid.jl`**: it exercises every stage of the pipeline.
- **Do the first setup at top level** (Julia ≥ 1.12 world age); inside functions read the setup globals with `Base.invokelatest(getfield, Main, :name)`.
- **Restart Julia after replacing a source file.** If behaviour looks unchanged, check which definition is used, e.g. `functionloc(compute_times_domain!, (DataFrame, Vector{GSM2}))`.
- **Survival fraction** = clonogenic cells (alive, not doomed) right after `compute_times_domain!`; the cell count at later times is not a survival fraction.
- **Incremental model**: never call `compute_cell_survival_GSM2!` between fractions without restoring the timers (it resets every cell's clock).
- **Box size**: keep ≥ 150–200 µm of free lattice around the spheroid (periodic lattice).
- **Beam radius**: always from the actual (current) spheroid radius.
- **Dose rate**: `Gy/s × 3600` particles per hour scaling only — do not multiply by `au`.
- **Reproducibility**: call `Random.seed!(n)` before `MC_loop_damage!`, `compute_times_domain!` and `run_simulation_abm!`, and pass `seed` to `MC_dose_CPU!`.
- **Threads**: start Julia with `--threads auto`; global settings (`set_*!`) must be set before threaded loops.
- **Large lattices**: the setup allocates damage and dose vectors on every lattice site; for big boxes, memory grows as `(2 X_box / 2 R_cell)³`.
