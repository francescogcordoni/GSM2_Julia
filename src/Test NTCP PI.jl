m_FSU = m_FSU_array[2]
n_FSU = n_FSU_array[2]
PI1 = ParIrr_array[1]
Irr1 = hcat(ones(n_FSU, round(Int, m_FSU*PI1)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI1))))
PI2 = ParIrr_array[2]
Irr2 = hcat(ones(n_FSU, round(Int, m_FSU*PI2)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI2))))
PI3 = ParIrr_array[3]
Irr3 = hcat(ones(n_FSU, round(Int, m_FSU*PI3)), zeros(n_FSU, round(Int, m_FSU*(1.0-PI3))))

reshape(VoxArray[:, :, 1], size(Irr1, 1), size(Irr1, 2), 1)


NTCP_prova1, D_voxel_prova1, VoxArray_temp1 = compute_NTCP_nxm_PI(VoxArray, m_FSU, n_FSU, PI1)
NTCP_prova2, D_voxel_prova2, VoxArray_temp2 = compute_NTCP_nxm_PI(VoxArray, m_FSU, n_FSU, PI2)
NTCP_prova3, D_voxel_prova3, VoxArray_temp3 = compute_NTCP_nxm_PI(VoxArray, m_FSU, n_FSU, PI3)

P_FSU1 = zeros(n_FSU, m_FSU)
P_FSU2 = zeros(n_FSU, m_FSU)
for j in 1:n_FSU
    for i in 1:m_FSU
        #FSU response
        P_FSU1[j,i] = Irr1[j,i].*VoxArray_temp1[j,i,1].ni
        P_FSU2[j,i] = Irr2[j,i].*VoxArray_temp2[j,i,1].ni
    end
end