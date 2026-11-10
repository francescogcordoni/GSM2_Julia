# GSM<sup>2</sup> Julia — Ion Irradiation & Agent-Based Model Framework

A Julia framework that simulates the full chain from a particle beam to the response of a cell population: particle-by-particle energy deposition, stochastic DNA damage and repair with the Generalized Stochastic Microdosimetric Model (GSM<sup>2</sup>), and an event-driven agent-based model (ABM) with cell-cycle progression, division, contact inhibition, delayed radiation-induced death and migration. It is designed for tumour spheroids but works on any 3-D cell lattice.

```
Ion beam → MC dose deposition → DNA damage (GSM²) → fate (repair / commitment) → ABM cell dynamics
 (mono- or polyenergetic,         (per domain,          (exact survival,            (event-driven,
  energy loss along the path)      Poisson lesions)       delayed death)              heap scheduler)
```

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [Pipeline in Detail](#pipeline-in-detail)
6. [Polyenergetic Beams](#polyenergetic-beams)
7. [Energy Loss Along the Path](#energy-loss-along-the-path)
8. [Biological Rules](#biological-rules)
9. [Fractionated and Continuous Irradiation](#fractionated-and-continuous-irradiation)
10. [Key Structs](#key-structs)
11. [Configuration Reference](#configuration-reference)
12. [Supported Ions](#supported-ions)
13. [Physics Models](#physics-models)
14. [GPU Acceleration](#gpu-acceleration)
15. [Scripts and Reproducing the Paper Figures](#scripts-and-reproducing-the-paper-figures)
16. [File Structure](#file-structure)
17. [Tips and Pitfalls](#tips-and-pitfalls)
18. [Citation](#citation)

---

## Overview

The framework is made of independent utility modules in `src/`:

| Stage | Module |
|---|---|
| Structs (`Ion`, `Irrad`, `GSM2`, `AT`, `CellPopulation`, …) | `utilities_structures.jl` |
| Random hit sampling | `utilities_general.jl` |
| Stopping power, CSDA range, track radii, beam geometry | `utilities_radiation.jl` |
| Geometry, lattice, population and irradiation setup | `utilities_env.jl` |
| Amorphous-track dose integrals | `utilities_AT_computation.jl` |
| Monte Carlo dose (CPU; GPU in a separate file) | `utilities_dose_computation.jl`, `utilities_dose_computation_GPU.jl` |
| DNA damage, OER, lesion yield κ, domain survival | `utilities_GSM2.jl` |
| Oxygenation profiles | `utilities_oxygen.jl` |
| Cell biology (cycle durations, population) | `utilities_biology.jl` |
| Repair kinetics, fate, cell cycle, migration, event engine, continuous irradiation | `utilities_abm.jl` |
| Dose / damage / survival plots, population plots | `utilities_plot.jl`, `utilities_plot_abm.jl` |
| Common style of the paper figures | `plot_style.jl` |

`src/load_utilities.jl` includes everything in dependency order. `utilities_abm.jl` is the single ABM file: `utilities_abm_migration.jl` and `utilities_checkpoint.jl` are older versions kept in the folder but **not** included, and must not be included together with it.

---

## Features

| Feature | Function / setting | Default |
|---|---|---|
| Event engine (binary min-heap, local updates): cost per event independent of population size | `run_simulation_abm!(…; fast = true)` | on |
| Original loops, kept for validation | `fast = false`, `compare_engines` | — |
| Polyenergetic beams (any set of energies or a sampled spectrum, e.g. Gaussian) | vector `setup_IonIrrad!`, `MC_dose_damage_poly!` | — |
| Energy loss along the path (CSDA range inversion layer by layer) | `track_seg = false` | `true` (constant LET) |
| Two fate models: immediate removal, or delayed death with delayed fate assignment | `set_fate_model!(:immediate \| :delayed)` | `:delayed` |
| Delayed radiation-induced death (doomed cells stay on the lattice) | `sample_death_delay`, `DEATH_DELAY_MEAN_CYCLES` | with `:delayed` |
| Checkpoint model: arrest in all phases, or at the G2/M checkpoint only | `set_checkpoint!(:all \| :g2)` | `:all` |
| Contact-inhibition threshold | `set_quiescence_threshold!(k)` | `k = 0` |
| Distance-limited proliferation (growth plateau) | `set_division_limit!` | off |
| Lineage tracking | `set_founder_tracking!`, `founders()` | off |
| Migration (random walk) | `run_simulation_abm_migration!` | — |
| Oxygen profile from the local cell density, recomputed on the current spheroid | `set_oxygen_diffusion!(…; density = :radial)` | `:mean` |
| Continuous irradiation driver | `irradiate_continuous_abm!` | — |
| Calibration against spheroid survival data (V79, Staab et al. 2004) | `scripts/staab_calibration.jl` | — |

All defaults reproduce the behaviour of earlier versions, so existing scripts run unchanged.

---

## Installation

### Requirements

Julia ≥ 1.10 (developed and tested with 1.12). Install the packages once:

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

The notebooks also need [Quarto](https://quarto.org) with the Julia engine.

### Clone

```bash
git clone https://github.com/francescogcordoni/GSM2_Julia.git
cd GSM2_Julia
```

---

## Quickstart

| Start here | What it does |
|---|---|
| `example/spheroid.jl` | Full pipeline for a monoenergetic beam: setup → dose → damage → survival → fate → ABM → plots |
| `example/spheroid_poly.jl` | The same with a beam made of several discrete energies |
| `example/spheroid_walkthrough.qmd` | Notebook version of `spheroid.jl`, step by step |
| `example/features_walkthrough.qmd` | Tour of every feature on a small spheroid: Gaussian energy spectrum, energy loss along the path (`track_seg = false`), two fractions with re-oxygenation, lineage tracking, growth plateau, migration, continuous irradiation, engine check |

```bash
julia --threads auto example/spheroid.jl
quarto render example/features_walkthrough.qmd
```

Edit the **PARAMETERS** block at the top of a script to change particle, energy, dose, geometry or fate model:

```julia
PARTICLE      = "1H"          # "1H", "4He", "12C", "16O"
ENERGY_MEV_U  = 100.0         # MeV per nucleon
DOSE_GY       = 2.0           # prescribed dose (Gy)
TUMOR_RADIUS  = 350.0         # spheroid radius (µm)
X_BOX         = 550.0         # box half-size (µm): keep ≥ TUMOR_RADIUS + ~150 µm
TERMINAL_TIME = 72.0          # post-irradiation ABM window (h)
const FATE    = :delayed  # :delayed (delayed death) or :immediate
```

The script prints the mean cell dose, the GSM<sup>2</sup> mean survival probability and the clonogenic surviving fraction; the last two should agree.

---

## Pipeline in Detail

### 1. Loading Utilities

```julia
include(joinpath(@__DIR__, "..", "src", "load_utilities.jl"))
sp = load_stopping_power()   # Dict{String, StoppingTable}, from data/stoppingpower/
```

### 2. GSM<sup>2</sup> Parameters

GSM<sup>2</sup> describes the induction, repair, misrepair and pairwise interaction of sublethal (X) and lethal (Y) DNA lesions. The rates depend on the cell-cycle phase; pass a `Vector{GSM2}` of length 4:

```julia
gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r_G1, a_G1, b_G1, rd, Rn)   # G1 / G0
gsm2_cycle[2] = GSM2(r_S,  a_S,  b_S,  rd, Rn)   # S
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

Values calibrated on phase-specific LQ data (synchronised V79 cells, Sinclair 1968):

```julia
r_G1 = 2.7805;  a_G1 = 0.01287;  b_G1 = 0.04030   # G1/G0
r_S  = 5.8401;  a_S  = 0.00589;  b_S  = 0.05794   # S
r_G2 = 1.7720;  a_G2 = 0.02431;  b_G2 = 5.705e-5  # G2/M
r    = 2.5657;  a    = 0.01481;  b    = 0.01266   # average
rd   = 0.8;     Rn   = 7.2
```

**Time scaling.** The repair SSA multiplies every waiting time by `au` (keyword of `compute_repair_domain`; `au = 4` in the paper). Acute survival depends only on `a/r` and `b/r`, so `au` sets the absolute repair time scale: the effective repair half-time of G1 cells is `au·ln2/(r + a) ≈ 1 h`. When converting a dose rate into a particle arrival rate, do **not** multiply by `au`.

### 3. Environment Setup

The setup functions write their results as globals of `Main` (`@eval Main`). In Julia ≥ 1.12, a global created during a function call is not visible to code that is already running, so **do the first setup at top level** (as in `example/spheroid.jl`); later calls inside functions only update existing globals. Inside functions, read them with `Base.invokelatest(getfield, Main, :name)`.

#### Option A — individual steps

```julia
setup_GSM2!(r, a, b, rd, Rn)                               # gsm2, center_x, center_y, domain
setup_IonIrrad!(dose, E, particle; type_AT = "KC")          # ion, irrad, LET, Rc, Rp, Rk, Kp, A, Z

N_sideVox   = Int(floor(2 * X_box / X_voxel))
N_CellsSide = 2 * convert(Int64, floor(X_box / (2 * R_cell)))
setup_cell_lattice!("circle", X_box, R_cell, N_sideVox, N_CellsSide;
                    ParIrr = "false", track_seg = true, full_cycle = true)   # cell_df, …

@eval Main begin                                            # required by setup_cell_population!
    tumor_radius = 350.0;  X_voxel = 1100.0;  X_box = 550.0
end
setup_cell_population!("circle", X_box, R_cell, N_sideVox, N_CellsSide, gsm2_cycle[4];
                       ParIrr = "false")                    # df_center_x/y, at, …
setup_irrad_conditions!(ion, irrad, type_AT, cell_df, track_seg)   # irrad_cond, lets, energies

set_oxygen_diffusion!(cell_df; density = :mean)             # or :radial; cell_df.O .= 21.0 for normoxia
```

#### Option B — single wrapper

```julia
out = setup(E, particle, dose, tumor_radius;
            X_box = 550.0, X_voxel = 1100.0, R_cell = 15.0,
            target_geom = "circle", calc_type = "full", type_AT = "KC", track_seg = true)
# out.ion, out.irrad, out.cell_df, out.R_beam, out.Npar, out.zF, …
```

| Parameter | Description | Typical value |
|---|---|---|
| `X_box` | Half-side of the simulation box (µm) | `tumor_radius + 150…250` |
| `X_voxel` | Voxel side for the beam-radius calculation (µm) | `2 * X_box` |
| `R_cell` | Cell radius (µm); lattice spacing = `2 R_cell` | 15 |
| `tumor_radius` | Spheroid radius (µm) | 150 – 500 |
| `target_geom` | Cross-section: `"circle"` or `"square"` | `"circle"` |
| `calc_type` | Beam radius: `"full"` (whole target) or `"fast"` (voxel) | `"full"` |
| `type_AT` | Track model: `"KC"` or `"LEM"` | `"KC"` |
| `track_seg` | `true` = constant LET across the target; `false` = energy loss along the path | `true` |
| `full_cycle` | `true` = cells start distributed over the whole cycle | `true` |
| `ParIrr` | `"true"` = partial irradiation | `"false"` |

**Box size.** The lattice is a cube with periodic, 26-neighbour connectivity. If the spheroid reaches a face of the box, its surface cells see occupied neighbours on the opposite face and are wrongly made quiescent. Keep 150–200 µm of free lattice around the spheroid for the simulated period.

### 4. Monte Carlo Dose

```julia
R_beam, x_beam, y_beam = calculate_beam_properties("full", "circle", X_box, X_voxel, tumor_radius)
F    = irrad.dose / (1.602e-9 * LET)           # fluence (particles/cm²)
Npar = round(Int, F * π * R_beam^2 * 1e-8)

MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df,
             df_center_x, df_center_y, at, gsm2_cycle, type_AT, track_seg; seed = 1234)
```

Writes `cell_df.dose_cell` (mean dose of the cell, Gy) and `cell_df.dose` (dose per domain column). The beam must cover the whole target: pass the **current** spheroid radius.

### 5. DNA Damage

```julia
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)
```

Lesions are Poisson-sampled from the stored dose (`dam_X_dom`, `dam_Y_dom` per domain; `dam_X_total`, `dam_Y_total` per cell). Rates per domain: `λX = κ·d`, `λY = 10⁻³ λX`, with `κ` from `calculate_kappa(ion, LET, O; OER_bool = true)`.

### 6. Survival

```julia
compute_cell_survival_GSM2!(cell_df, gsm2_cycle)
p_hat, ci_lo, ci_hi = survival_ci(cell_df.sp[cell_df.is_cell .== 1])
```

`cell_df.sp` is the exact GSM<sup>2</sup> probability that the cell completes repair (0 if a Y lesion is present).

> **Warning.** `compute_cell_survival_GSM2!` resets the timers of **every** cell. With the `:delayed` fate model, compute `sp` without resets:
> ```julia
> for i in findall(==(1), cell_df.is_cell)
>     ph = cell_df.cell_cycle[i]
>     g  = (ph == "G1" || ph == "G0") ? gsm2_cycle[1] : ph == "S" ? gsm2_cycle[2] : gsm2_cycle[3]
>     cell_df.sp[i] = domain_GSM2(cell_df.dam_X_dom[i], cell_df.dam_Y_dom[i], g)
> end
> ```

### 7. Fate Assignment and Repair Timers

```julia
set_fate_model!(:delayed)            # or :immediate; :delayed is the default; once, before the run
compute_times_domain!(cell_df, gsm2_cycle; nat_apo = 1e-10, terminal_time = Inf, summary = true)
```

| Model | Behaviour |
|---|---|
| `:immediate` | Resets the timers of all alive cells; lethally damaged cells are removed within hours. Used by the original calibration scripts. |
| `:delayed` | Processes only alive, non-doomed, damaged cells. Lethally damaged cells become **doomed**: they keep their site, do not divide, and are removed after a delay. Required for repeated irradiation and for `set_checkpoint!(:g2)`. |

The time of death of a doomed cell is its commitment time (the first lethal event of the GSM<sup>2</sup> kinetics, 0 for a lethal lesion induced directly) plus the delay. `terminal_time` is the repair window: with `Inf` all damage is resolved; with a finite window, unrepaired cells keep their residual damage for the next call.

**Clonogenic survival** (the surviving fraction) = cells alive and not doomed right after this call:

```julia
SF = count((cell_df.is_cell .== 1) .& .!isfinite.(cell_df.death_time)) / N_init
```

### 8. ABM Simulation

```julia
ts, snaps = run_simulation_abm!(cell_df;
    nat_apo = 1e-10, terminal_time = 72.0, snapshot_times = [0, 6, 12, 24, 48, 72],
    verbose = true, return_dataframes = true, update_input = true, record_dt = 0.0)
```

The default engine stores the next cycle, death and migration time of every cell in a binary min-heap; each event updates only the cells within two lattice sites (outdated heap entries are discarded when they reach the top). `compare_engines(cell_df; seed = 1, terminal_time = 48.0)` checks it against the original loop.

On return, the timers are relative to the **last event**. When chaining ABM segments, advance the clocks by the idle remainder:

```julia
ts, _ = run_simulation_abm!(cell_df; terminal_time = Δ, snapshot_times = Int[], verbose = false,
                            return_dataframes = false, update_input = true)
_advance_idle!(cell_df, Δ - (isempty(ts.time) ? 0.0 : ts.time[end]))
```

Migration: `run_simulation_abm_migration!(cell_df, D_m, 2R_cell; terminal_time = 72.0)` (`D_m` in µm²/h).

### 9. Plotting

```julia
plot_scalar_cell(cell_df, :dose_cell); plot_scalar_cell(cell_df, :sp; xscale = :log10)
plot_damage(cell_df); plot_times(cell_df; show_means = true)
plot_cell_cycle_distribution(cell_df; half_sphere = true); print_phase_distribution(cell_df)
plot_cell_dynamics(ts); plot_phase_dynamics(ts); plot_spatial_distribution(snap; color_by = :cell_cycle)
```

Polyenergetic versions: `plot_scalar_cell_poly`, `plot_damage_poly` (`layer_plot = true` groups by depth, `pp = true` per energy component).

---

## Polyenergetic Beams

A beam made of several energies (or species) is described by vectors: one `Ion` and one set of irradiation conditions per component.

```julia
ENERGIES  = [15.0, 20.0, 25.0]           # MeV/u
PARTICLES = fill("12C", length(ENERGIES))
setup_IonIrrad!(dose, ENERGIES, PARTICLES; type_AT = "KC")        # LET, ion, irrad are vectors
setup_irrad_conditions!(ion, irrad, "KC", cell_df, track_seg)     # irrad_cond[i] per component
MC_dose_damage_poly!(ion, Npar_vect, R_beam, irrad_cond, cell_df,
                     df_center_x, df_center_y, at, gsm2_cycle, "KC", track_seg)
```

`MC_dose_damage_poly!` runs dose and damage for each component and sums them per cell (`dose_i`, `dam_X_dom_i`, … are kept per component).

**A Gaussian energy spectrum.** Discretise `E ~ N(Ē, σ²)` into bins with fluence fractions `w_i`, then split the fluence so that the total dose at the entrance is the prescribed one:

```julia
edges    = range(Ē - 2.5σ, Ē + 2.5σ; length = N_BINS + 1)
ENERGIES = collect(0.5 .* (edges[1:end-1] .+ edges[2:end]))
w        = diff(cdf.(Normal(Ē, σ), collect(edges)));  w ./= sum(w)    # fluence fractions
F_tot    = dose / (1.602e-9 * sum(w .* LET))                           # LET from setup_IonIrrad!
Npar_vect = round.(Int, w .* F_tot .* π .* R_beam^2 .* 1e-8)
```

The full example, with plots of the spectrum and of the dose-averaged LET, is in `example/features_walkthrough.qmd`.

---

## Energy Loss Along the Path

With `track_seg = false`, each particle loses energy as it crosses the target, so cells at different depths see different energies, LET and track profiles (as along a Bragg curve):

1. The target is divided into layers perpendicular to the beam, one per lattice plane (thickness `h = 2 R_cell`).
2. For each ion, the CSDA range `R(E) = ∫ dE'/S(E')` is built once from the stopping-power table (`range_table!`).
3. The energy at the next layer is obtained by range inversion, `E_{i+1} = R⁻¹(R(E_i) − h)` (`residual_energy_after_distance`).
4. In each layer, the LET and the track parameters (`Rc`, `Rp`, `Kp`) are recomputed at the local energy.
5. Tracks are straight and parallel (no lateral scattering); a particle below the validity range of the track model stops. Empty layers do not absorb (`absorb = :occupied`), so the depth is measured from the surface of the target.
6. The number of particles is computed from the entrance LET, so the prescribed dose refers to the entrance.

With `track_seg = true` (default), all layers keep the entrance energy. `lets` and `energies` (globals set by `setup_irrad_conditions!`) hold the LET and energy of each layer.

---

## Biological Rules

| Rule | Description | Setting |
|---|---|---|
| Cell cycle | G1 → S → G2 → M → division; Gamma-distributed phase durations (means G1 11 h, S 7 h, G2 5 h, M 1 h; scale 2 h) | `PHASE_DURATIONS` |
| Division | At the end of M, into a random empty neighbour; both daughters in G1 | — |
| Contact inhibition | G0 when ≤ k of the 26 neighbours are empty; back to G1 when space is freed (unless doomed) | `set_quiescence_threshold!(k; cell_df)`, default k = 0 |
| Growth plateau | Division accepted with p(r) = 1/(1 + exp((r − R50)/w)) | `set_division_limit!(cell_df; R50, w)` |
| Arrest after damage | `:all`: clock held in G1/S/G2 until repair; `:g2`: damaged cells keep cycling through G1/S and are held only in G2 | `set_checkpoint!(:all \| :g2)` |
| Mitotic death | A cell whose mitosis ends before repair is complete is committed to death | — |
| Delayed death (`:delayed`) | Removal after `T ~ Gamma(3, m·Tc/3)`, with m = 1 (G2, M), 1.25 (S), 1.5 (G1), 2.5 (G0) | `DEATH_DELAY_MEAN_CYCLES` |
| G0 radiosensitivity | G0 cells use the G1 parameters | — |
| Lineage tracking | Daughters inherit the founder id of their parent | `set_founder_tracking!(cell_df)`, `founders()` |

---

## Fractionated and Continuous Irradiation

**Fractions on a growing spheroid.** With `set_fate_model!(:delayed)`, repeat the irradiation on the **current** `cell_df`:

```julia
set_oxygen_diffusion!(cell_df; density = :radial)                 # oxygen of the current spheroid
setup_IonIrrad!(dose_fx, E, particle; type_AT = "KC")
R_beam, _, _ = calculate_beam_properties("full", "circle", X_box, X_voxel, current_radius)
setup_irrad_conditions!(ion, irrad, "KC", cell_df, track_seg)
MC_dose_CPU!(ion, Npar, R_beam, irrad_cond, cell_df, df_center_x, df_center_y, deepcopy(at),
             gsm2_cycle, "KC", track_seg; seed = s)
MC_loop_damage!(ion, cell_df, irrad_cond, gsm2_cycle)
compute_times_domain!(cell_df, gsm2_cycle; terminal_time = Inf)   # not compute_cell_survival_GSM2!
# … ABM until the next fraction (+ _advance_idle!)
```

Fates are assigned only to cells not already doomed. See `scripts/spheroid_fractionation.jl` and `scripts/spheroid_schedules.jl`.

**Continuous low-dose-rate irradiation:**

```julia
ts, res = irradiate_continuous_abm!(cell_df, gsm2_cycle;
              dose = 4.0, dose_rate_gyh = 0.5, K_tot = K, L_tot = 1e-3K,
              checkpoint = :g2, max_pulse_dose = 0.05, max_pulse_h = 0.1, seed = 1)
res.SF_plated, res.N_rel
```

The dose is delivered as short pulses with uniform (low-LET) damage; between pulses, fates are assigned with a repair window equal to the pulse interval and the ABM runs over the same interval.

**Dose-rate verification** against the Lea–Catcheside factor (static G1 population, any ion): `scripts/continuous_dr_events.jl`.

---

## Key Structs

| Struct | Constructor | Fields |
|---|---|---|
| `Ion` | `Ion(ion, E, A, Z, LET, rho)` | species, energy (MeV/u), mass and atomic number, LET (keV/µm), density |
| `Irrad` | `Irrad(dose, kR, doserate)` | dose (Gy), …, dose rate |
| `GSM2` | `GSM2(r, a, b, rd, Rn)` | rates (h⁻¹), domain and nucleus radii (µm) |
| `AT` | `AT(ion, E, A, Z, LET, rho, Rc, Rp, Rk, Kp)` | amorphous track: core and penumbra radii (µm), amplitude |
| `CellPopulation` | `CellPopulation(cell_df)` | structure of arrays used by the ABM (`to_dataframe`, `update_dataframe!`) |
| `SimulationTimeSeries` | returned by the ABM | `time`, `total_cells`, `g0_cells` … `m_cells` |

---

## Configuration Reference

### `run_simulation_abm!` / `run_simulation_abm_migration!`

| Keyword | Default | Meaning |
|---|---|---|
| `nat_apo` | `1e-10` | Natural apoptosis rate |
| `terminal_time` | `48.0` | End time (h) |
| `snapshot_times` | `[1,6,12,24]` | Hours at which to save the population |
| `print_interval` | `1.0` | Console print every N hours |
| `verbose` | `true` | Progress output |
| `fast` | `true` | Heap engine (`false` = original loop) |
| `record_dt` | `0.0` | Time-series resolution (0 = every event) |
| `return_dataframes` | `true` | Snapshots as DataFrames |
| `update_input` | `true` | Write the final state back to `cell_df` |

### Global settings (`utilities_abm.jl`)

| Function | Values | Default |
|---|---|---|
| `set_fate_model!(m; death_delay)` | `:immediate`, `:delayed` | `:delayed` |
| `set_checkpoint!(m)` | `:all`, `:g2` | `:all` |
| `set_quiescence_threshold!(k; cell_df)` | integer ≥ 0 | 0 |
| `set_division_limit!(cell_df; R50, w, center, rmax, rq)` / `clear_division_limit!()` | — | off |
| `set_founder_tracking!(cell_df)` / `clear_founder_tracking!()` | — | off |

Set them once, before any threaded loop.

---

## Supported Ions

| Label | Z | A |
|---|---|---|
| `"1H"` | 1 | 1 |
| `"2H"` | 1 | 2 |
| `"3He"`, `"4He"` | 2 | 3, 4 |
| `"12C"` | 6 | 12 |
| `"16O"` | 8 | 16 |

Stopping-power tables for H, He, Li, Be, B, C, N, O in water are in `data/stoppingpower/`. LET at a given energy: `let_at(table_for(sp, "12C"), E)` (keV/µm).

---

## Physics Models

**Track structure.** Kiefer–Chatterjee (`"KC"`): `Rc = 0.01116·β` µm, `Rp = 0.0616·(E/A)^1.7` µm, `Kp = 1.25e-4·(z_eff/β)²`. Local Effect Model (`"LEM"`): `Rc = 0.01` µm, `Rp = 0.05·(E/A)^1.7` µm.

**Oxygen enhancement ratio.** `calculate_OER(LET, O)`: `OER = (K·(a·M0 + LET^g)/(a + LET^g) + O)/(K + O)`, with `M0 = 3.4`, `K = 0.41`, `a = 8.27e5`, `g = 3`.

**Oxygenation.** `set_oxygen_diffusion!` computes the quasi-steady-state profile of a reaction–diffusion equation with zero-order consumption; with `density = :radial` the consumption follows the local cell density, so space freed by dead cells is taken into account. The profile is static during the ABM; recompute it before each fraction.

---

## GPU Acceleration

`src/utilities_dose_computation_GPU.jl` provides a CUDA kernel and `MC_dose_fast!`, which dispatches to the GPU when `CUDA.functional()` and `Npar ≥ GPU_PARTICLE_THRESHOLD` (default `1_000_000`). The file is **not** included by `load_utilities.jl`; include it explicitly to use it. The ABM always runs on the CPU.

---

## Scripts and Reproducing the Paper Figures

Simulation scripts are in `scripts/` and save their results as CSV; the plotting scripts in `results/` read those files and write vector PDFs without rerunning anything. Run with `julia --threads auto <script>`.

| Analysis | Simulation | Plot | Data |
|---|---|---|---|
| Dose-rate verification (Lea–Catcheside) | `continuous_dr_events.jl` | `plot_doserate_relative.jl` | `data/continuous_dr_events/` |
| V79 spheroid benchmark (Staab et al.) | `staab_calibration.jl` | (built in) | — |
| Phase-specific LQ calibration | `fit_LQ_data.jl` | `plot_LQ_fit.jl` | `data/fit_LQ/` |
| Two ions + control, 3-D multipanel | `spheroid_ions.jl` | `plot_spheroid_multipanel.jl` | `scripts/output/spheroid_ions/` |
| Temporal evolution | `spheroid_temporal_evolution.jl` | `plot_spheroid_temporal_evolution.jl`, `plot_spheroid_3d.jl` | `data/spheroid_temporal_evolution/` |
| Damage trajectories | `damage_temporal_evolution.jl` | `plot_damage_temporal_evolution.jl` | `data/damage_temporal_evolution/` |
| Split dose | `split_repopulation.jl` | `plot_split_repopulation.jl` | `data/split_repopulation/` |
| Fractionation (3 × 2 Gy) | `spheroid_fractionation.jl` | `plot_spheroid_fractionation.jl` | `scripts/output/spheroid_fractionation/` |
| Single-dose response and equal-dose schedules | `spheroid_schedules.jl` (`STUDY = :dose` / `:fractionation`) | `plot_spheroid_schedules.jl` | `scripts/output/spheroid_schedules/` |
| Fractionation with oxygen | `spheroid_fractionation_oxygen.jl` | `plot_fractionation_oxygen.jl` | `scripts/output/spheroid_fractionation_oxygen/` |
| Regrowth delay and spheroid control | `spheroid_regrowth.jl` | (built in) | `scripts/output/spheroid_regrowth/` |
| Lineage of the surviving clones | `spheroid_lineage.jl` | (built in) | `data/spheroid_lineage/` |
| Stochastic variability | `stochastic_evolution.jl` | `plot_stochastic_evolution.jl` | `data/stochastic_time_ev/` |
| Computational cost | `benchmark_computation.jl` | `plot_benchmark.jl` | `scripts/output/benchmark/` |
| Scaling with spheroid size | `benchmark_spheroid_scaling.jl` | `plot_benchmark_spheroid_scaling.jl` | `data/benchmark_spheroid_scaling/` |
| Continuous dose rate on a spheroid | `continuos_dr_sphere.jl`, `continuous_dr.jl` | `plot_continuous_doserate_sphere.jl`, `plot_continuous_dr.jl` | `data/continuous_doserate_sphere/`, `data/continuous_dr/` |
| Inverse dose-rate effect | `inverse_dose_rate.jl`, `inverse_doserate_continuous.jl`, `idre_abm_continuous.jl` | `plot_inverse_doserate.jl` | `data/inverse_doserate*/`, `data/idre_abm_continuous/` |
| Migration | `sphere_migration.jl` | — | — |

All scripts seed their random streams; within a study, conditions share the same random sequence up to the irradiation (common random numbers).

---

## File Structure

```
GSM2_Julia/
├── example/
│   ├── spheroid.jl                  ← start here: full pipeline, monoenergetic beam
│   ├── spheroid_poly.jl             ← discrete polyenergetic beam
│   ├── spheroid_walkthrough.qmd     ← notebook version of spheroid.jl
│   ├── features_walkthrough.qmd     ← tour of all features (Gaussian beam, track_seg = false, …)
│   ├── splits.jl, splits_async.jl   ← split-dose examples
│   └── plots_domains.jl             ← nucleus and domain geometry
├── scripts/                         ← simulation scripts (see the table above)
├── results/                         ← plotting scripts, vector PDFs
├── src/
│   ├── load_utilities.jl            ← single include entry point
│   ├── utilities_*.jl               ← modules (see Overview)
│   └── plot_style.jl                ← common style of the paper figures
└── data/
    ├── stoppingpower/               ← H … O in water
    └── <analysis>/                  ← simulation outputs
```

Each utility file begins with an index of its public functions. Comment convention: `#!` section headers, `#~` groupings, `#?` public names.

---

## Tips and Pitfalls

- **Do the first setup at top level** (Julia ≥ 1.12 world age); inside functions read the setup globals with `Base.invokelatest(getfield, Main, :name)`.
- **Restart Julia after replacing a source file**; check which definition is used with `functionloc`.
- **Surviving fraction** = clonogenic cells right after `compute_times_domain!`; the cell count at later times includes regrowth and doomed cells.
- **Delayed model**: never call `compute_cell_survival_GSM2!` between fractions without restoring the timers.
- **Box size**: keep ≥ 150–200 µm of free lattice around the spheroid (periodic lattice).
- **Beam radius**: always from the current spheroid radius.
- **Dose rate**: do not multiply the particle arrival rate by `au`.
- **Reproducibility**: `Random.seed!(n)` before `MC_loop_damage!`, `compute_times_domain!` and `run_simulation_abm!`, and pass `seed` to `MC_dose_CPU!`.
- **Threads**: start Julia with `--threads auto`; set global options (`set_*!`) before threaded loops.
- **Memory**: damage and dose vectors are allocated on every lattice site, so memory grows as `(X_box / R_cell)³`.

---

## Citation

If you use this code, please cite:

F. G. Cordoni, M. Battestini, M. Missiaggia, *A stochastic agent-based extension of the GSM² model for particle therapy: cell-cycle dynamics, dose-rate dependence, and fractionation effects*, arXiv:2604.27630.
