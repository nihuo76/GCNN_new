import matplotlib.pyplot as plt

import scipy.io as sio
from os import listdir
from os.path import join
import numpy as np

k_n = 1
root='data/hamiltonMER'
mat_files = listdir(root)
orb_file = open(join(root, '../orb_list.txt'))
counter = 0
orb_list = dict()
for orb in orb_file:
    orb_list[orb.replace('\n', '')] = counter
    counter += 1

for f_name in mat_files:
    raw_data = sio.loadmat(join(root, f_name))['Hr'][0][0]
    hamiltonian, n_orb, n_cell, cell_pos = raw_data.tolist()[: 4]
    atom_info = raw_data[5][0][0]
    cell_size = np.amax(atom_info[2], axis=0)
    pos_list = []
    for i in range(-k_n, k_n + 1):
        for j in range(-k_n, k_n + 1):
            for k in range(-k_n, k_n + 1):
                pos_list.append(np.array([i, j, k]))
    h_mat = []
    x_feat = []
    for i in range(len(pos_list)):
        h_row = []
        for j in range(len(pos_list)):
            idx = pos_list[j]
            idx_center = pos_list[i]
            idx_relat = idx - idx_center
            if (np.any(np.fabs(idx_relat) > 3)):
                h_row.append(np.zeros_like(hamiltonian[:, :, 0].squeeze()).squeeze())
            else:
                ham_idx = np.where((cell_pos[:, 0] == idx_relat[0]) &
                                   (cell_pos[:, 1] == idx_relat[1]) &
                                   (cell_pos[:, 2] == idx_relat[2]))
                h_row.append(hamiltonian[:, :, ham_idx[0].squeeze()].squeeze())
        h_mat.append(np.concatenate(h_row, axis=1))
    h = np.concatenate(h_mat, axis=0)
    h_real = np.absolute(h)
    h_plot = h_real
    np.fill_diagonal(h_plot, 0)

    plt.contourf(np.arange(x_feat.shape[1]),
                np.arange(x_feat.shape[0]), x_feat, cmap='rainbow')
    plt.colorbar()
    plt.savefig(fname= join('x_visual',f_name.replace('.','')))
    plt.close()