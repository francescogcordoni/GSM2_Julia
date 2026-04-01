include("utilities_ABM.jl")
include("migration_ABM.jl")

# Your existing population setup
pop = CellPopulation(cell_df)

# Run with migration — pure diffusive for now
ts, snaps = run_simulation_abm_migration!(pop, D, h;
    terminal_time = 72.0,
    chi = 0.0)        # set chi > 0 once O2 field is ready

# Or use the low-level loop for O2 coupling:
mpop = MigrationCellPopulation(pop)
schedule_all_migrations!(mpop, D, h)
# ... event loop ...
# After each PDE re-solve:
schedule_all_migrations!(mpop, D, h; chi=chi, o2_field=new_o2, lattice_size=(nx,ny))