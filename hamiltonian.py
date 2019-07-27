import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.read import read_planetoid_data
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
plt.style.use('seaborn')

import scipy.io as sio
from os import listdir
from os.path import join
import numpy as np


class Hamiltonian(torch.utils.data.Dataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root='data/hamiltonMER', real_thrd=0.0001,
                 transform=None, pre_transform=None, k_n=0):
        self.name = 'Hamiltonian'
        super(Hamiltonian, self).__init__()
        # super(Hamiltonian, self) is equivalent to
        # super() call which calls the initialization of
        # torch.utils.data.Dataset
        self.data, self.slices = None, None
        self.transform = transform
        self.dataset = []
        self.label = []
        mat_files = listdir(root)
        orb_file = open(join(root, '../orb_list.txt'))
        counter = 0
        orb_list = dict()
        for orb in orb_file:
            orb_list[orb.replace('\n', '')] = counter
            counter += 1

        for f_name in mat_files:
            ###############################################
            # for plot:
            # f_name = 'Hr.hh.1100424.Li-Mg-Sb'
            ###############################################
            raw_data = sio.loadmat(join(root, f_name))['Hr'][0][0]

            # sio.loadmat(join(root, f_name)) is a dictionary
            # with keys '__header__', '__version__', '__globals__', 'Hr'
            # sio.loadmat(join(root, f_name))['Hr'] is a numpy
            # array
            # sio.loadmat reads MATLAB structs as numpy array
            # where dtype are of type object and names are the
            # MATLAB struct field names
            # MATLAB structs are a little bit like Python dicts
            # raw_data corresponds to the struct in MATLAB
            hamiltonian, n_orb, n_cell, cell_pos = raw_data.tolist()[: 4]

            # hamiltonian, n_orb, n_cell, cell_pos are all ndarray
            atom_info = raw_data[5][0][0]
            cell_size = np.amax(atom_info[2], axis=0)
            # cell_size is the relative position of atom w.r.t
            # the unit-cell
            pos_list = []
            # pos_list is the atom position w.r.t unit cell
            for i in range(-k_n, k_n + 1):
                for j in range(-k_n, k_n + 1):
                    for k in range(-k_n, k_n + 1):
                        pos_list.append(np.array([i, j, k]))

            h_mat = []
            x_feat = []
            for i in range(len(pos_list)):
                # loop through the pos_list
                h_row = []
                # len(orb_list) + 4 because 3 x,y,z coordinate
                # of atom center in unit cell and the atom center itself
                atom_feat = np.zeros((n_orb.squeeze(), 4), dtype=np.float32)
                # atom_feat's dimension must be ?*20.
                # orb_list is the possible name of the orbital
                for a in range(atom_info[0].squeeze()):
                    for j in range(atom_info[-2][a][0] - 1, atom_info[-2][a][1]):
                        atom_feat[j, -1] = atom_info[3][a][0]
                        # atom number of each atom
                        atom_feat[j, -4: -1] = atom_info[2][a] + np.array(pos_list[i]) * cell_size
                        # -4:-1 is the coordinate of the atom w.r.t. the unit cell
                        # elements_name = atom_info[7][0][a][j - (atom_info[-2][a][0] - 1)][0][0]
                        # elements_name should be orbital name here
                        # since it is the orbitals' names of each atom
                        # atom_feat[j, orb_list[elements_name]] = 1
                        # mark orbitals' names of each atom to be 1
                x_feat.append(atom_feat)

                for j in range(len(pos_list)):
                    idx = pos_list[j]
                    idx_center = pos_list[i]
                    idx_relat = idx - idx_center
                    if (np.any(np.fabs(idx_relat)>3)):
                        h_row.append(np.zeros_like(hamiltonian[:, :, 0].squeeze()).squeeze())
                        # we insert the zero matrix since we ignore the our-of-bounder value
                    else:
                        ham_idx = np.where((cell_pos[:,0] == idx_relat[0]) &
                                           (cell_pos[:,1] == idx_relat[1]) &
                                           (cell_pos[:,2] == idx_relat[2]))
                        h_row.append(hamiltonian[:, :, ham_idx[0].squeeze()].squeeze())

                    # idx = np.where((cell_pos == (pos_list[j] - pos_list[i])).all(axis=1))[0][0]
                    # cell_pos is the cell position matrix in the MATLAB struct
                    #h_row.append(hamiltonian[:, :, idx].squeeze())
                    # the code above are Peng's original code
                    ###########################################################
                    #idx = pos_list[j]
                    #ham_idx = np.where((cell_pos[:,0] == idx[0]) &
                    #                   (cell_pos[:,1] == idx[1]) &
                    #                   (cell_pos[:,2] == idx[2]))
                    #h_row.append(hamiltonian[:,:,ham_idx[0].squeeze()].squeeze())
                    # the code above is the 1st time code I wrote
                    ############################################################
                    # the idx here is the center unit cell in the super cell
                    # thus h_row is a list of the Hamilton matrix in the center

                h_mat.append(np.concatenate(h_row, axis=1))
            h = np.concatenate(h_mat, axis=0)
            x_feat = np.concatenate(x_feat, axis=0)
            # idx = np.where((cell_pos == np.array([0, 0, 0])).all(axis=1))[0][0]
            # h = hamiltonian[:, :, idx].squeeze()

            #h_real = h.real
            # h_real is the center Hamilton matrix with only the real number
            # try take the norm of complex number
            h_real = np.absolute(h)
            ########################################################################
            # h_plot = h_real
            # np.fill_diagonal(h_plot, 0)
            #plt.contourf(np.arange(x_feat.shape[1]),
            #             np.arange(x_feat.shape[0]), x_feat, cmap='rainbow')
            #plt.colorbar()
            #plt.savefig(fname= join('x_visual',f_name.replace('.','')))
            #plt.close()
            # visualize the Hamiton Matrix, should be commented later
            ########################################################################

            edge = np.where(np.abs(h_real)> real_thrd)
            weight = h_real[edge].astype(np.float32)

            edge, weight = remove_self_loops(torch.tensor(edge), torch.tensor(weight))
            edge, weight = coalesce(edge, weight, torch.tensor(n_orb.squeeze()), torch.tensor(n_orb.squeeze()))
            # create the sparse center Hamilton matrix
            # x = torch.ones(((n_orb.squeeze()) * ((2 * k_n + 1)**3), 1))
            # x = torch.tensor(np.tile(atom_feat, ((2 * k_n + 1)**3, 1)).astype(np.float32))
            x = torch.tensor(x_feat)
            y = torch.tensor(float(raw_data.tolist()[9] > 0))
            # y is True/False for material Electrical conductivity

            self.dataset.append(Data(x=x, edge_index=edge, edge_attr=weight, y=y))
            # Transform into the data form of torch_geometric
            self.label.append(int(float(raw_data.tolist()[9] > 0.2)))

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform is not None: data = self.transform(data)
        return data
    def __len__(self):
        return len(self.dataset)
    # @property
    # def raw_file_names(self):
    #     names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    #     return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]
    #
    # @property
    # def processed_file_names(self):
    #     return 'data.pt'
    #
    # def download(self):
    #     for name in self.raw_file_names:
    #         download_url('{}/{}'.format(self.url, name), self.raw_dir)
    #
    # def process(self):
    #     data = read_planetoid_data(self.raw_dir, self.name)
    #     data = data if self.pre_transform is None else self.pre_transform(data)
    #     data, slices = self.collate([data])
    #     torch.save((data, slices), self.processed_paths[0])
    #
    # def __repr__(self):
    #     return '{}()'.format(self.name)
