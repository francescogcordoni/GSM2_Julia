a = 0.01
b = 0.30
r = 4.30
rd = 0.80
rn = 7.2

gsm2_cycle    = Array{GSM2}(undef, 4)
gsm2_cycle[1] = GSM2(r, a, b, RD, RN)    # G1 / G0
gsm2_cycle[2] = GSM2(r,  a,  b,  RD, RN)    # S
gsm2_cycle[3] = GSM2(r, a, b, RD, RN)    # G2 / M
gsm2_cycle[4] = GSM2(r, a, b, RD, RN) # average (fallback)


# Irradiation
PARTICLE      = "1H"      # ion species: "1H", "4He", "12C", "16O"
ENERGY_MEV_U  = 250.0      # kinetic energy per nucleon (MeV/u)
DOSE_GY       = 1.        # prescribed dose (Gy)

# Spheroid and cell geometry
TUMOR_RADIUS  = 350.0     # spheroid radius (µm)
R_CELL        = 15.0      # cell radius (µm)
X_BOX         = 550.0     # simulation box half-size (µm); match TUMOR_RADIUS
X_VOXEL       = 700.0     # voxel side length for beam-geometry calculation (µm)

# GSM2 domain geometry
RD            = 0.8       # domain radius (µm)
RN            = 7.2       # nucleus radius (µm)

# Phase-specific GSM2 parameters [G1/G0, S, G2/M, average]
# These are fitted to HSG cell line data — replace with your own fits.
const A_G1 = 0.01287;  const B_G1 = 0.04030;  const R_G1 = 2.7805
const A_S  = 0.00589;  const B_S  = 0.05794;  const R_S  = 5.8401
const A_G2 = 0.02431;  const B_G2 = 5.705e-5; const R_G2 = 1.7720
const A_AVG= 0.01481;  const B_AVG= 0.01266;  const R_AVG= 2.5657

# Simulation options
const TYPE_AT        = "KC"         # track structure: "KC" (Kiefer-Chatterjee) or "LEM"
const TRACK_SEG      = false         # true = fixed LET across depth (no Bragg-peak buildup)
const TARGET_GEOM    = "circle"     # spheroid cross-section for beam geometry
const CALC_TYPE      = "full"       # beam radius mode: "full" (whole spheroid) or "fast"
const TERMINAL_TIME  = 72.0         # post-irradiation ABM window (h)
const SNAPSHOT_HOURS = [0, 6, 12, 24, 48, 72]   # save population snapshots at these times (h)
const NAT_APO        = 1e-10        # natural apoptosis rate (h⁻¹, background only)


# Build a clean base: all spheroid cells alive (O > 0 identifies original cells),
# zero prior dose and damage so each sweep iteration starts fresh.
cell_df_base = deepcopy(cell_df)
cell_df_base.is_cell[cell_df_base.O .> 0] .= 1
# Replace entire columns to fix any type corruption from earlier session attempts
n = nrow(cell_df_base)
cell_df_base[!, :dose]        = [Float64[] for _ in 1:n]
cell_df_base[!, :dam_X_dom]   = [Int[]     for _ in 1:n]
cell_df_base[!, :dam_Y_dom]   = [Int[]     for _ in 1:n]
cell_df_base.dose_cell        .= 0.0
cell_df_base.dam_X_total      .= 0
cell_df_base.dam_Y_total      .= 0

doses  = 0.5:0.5:5.0
sp_all = Vector{Float64}(undef, length(doses))

for (i, d) in enumerate(doses)
    cell_df_d = deepcopy(cell_df_base)

    # Assign uniform dose d to every cell (single domain hit)
    cell_df_d[!, :dose] = [cell_df_d.is_cell[j] == 1 ? [d] : Float64[]
                           for j in 1:nrow(cell_df_d)]
    cell_df_d.dose_cell[cell_df_d.is_cell .== 1] .= d

    # MC_loop_damage! computes kappa from each cell's LET and O2, samples Poisson damage
    MC_loop_damage!(ion, cell_df_d, irrad_cond, gsm2_cycle)
    compute_cell_survival_GSM2!(cell_df_d, gsm2_cycle)
    sp_all[i] = mean(cell_df_d.sp[cell_df_d.is_cell .== 1])
    println("d = $d Gy  →  mean SP = $(round(sp_all[i], digits=4))")
end
