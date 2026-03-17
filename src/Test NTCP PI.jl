m_FSU = m_FSU_array[1]
n_FSU = n_FSU_array[1]
PI = ParIrr_array[1]
Irr1 = hcat(ones(n_FSU, round(Int, m_FSU*PI)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI))))
PI = ParIrr_array[2]
Irr2 = hcat(ones(n_FSU, round(Int, m_FSU*PI)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI))))
PI = ParIrr_array[3]
Irr3 = hcat(ones(n_FSU, round(Int, m_FSU*PI)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI))))



NTCP_prova1, D_voxel_prova1, VoxArray_temp1 = compute_NTCP_nxm_PI(VoxArray, N_sideVox, s[1], m_FSU_array[1], n_FSU_array[1], ParIrr_array[1])
NTCP_prova2, D_voxel_prova2,  VoxArray_temp2 = compute_NTCP_nxm_PI(VoxArray, N_sideVox, s[1], m_FSU_array[1], n_FSU_array[1], ParIrr_array[2])
NTCP_prova3, D_voxel_prova3, VoxArray_temp3 = compute_NTCP_nxm_PI(VoxArray, N_sideVox, s[1], m_FSU_array[1], n_FSU_array[1], ParIrr_array[3])

