import os.path as osp
import numpy as np
import random

# seed = 7
# random.seed(seed)
# np.random.seed(seed)

import torch
import torch.nn.functional as F

# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import torch.nn as nn
from hamiltonian import Hamiltonian
from mylayer import MyLayer
import torch_geometric.transforms as T
import torch_geometric.nn as geom_nn

from sklearn.model_selection import StratifiedKFold


# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# go to the parent directory of this file and then go to the
# "data" directory
dataset = Hamiltonian(k_n=1)
#data = dataset[0]

batch_size = 1
n_epoch = 500
res = np.zeros((int(n_epoch * len(dataset) / 5), len(dataset)))



# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         # self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
#         # self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
#         ########################################################
#         self.conv1 = geom_nn.ChebConv(data.num_features, 32, K=1)
#         self.conv2 = geom_nn.ChebConv(32, 64, K=2)
#         self.conv3 = geom_nn.ChebConv(64, 2, K=2)
#         #self.fc = nn.Linear(2, 2)
#         # Peng's original code above
#         ########################################################
#         #self.mlp1 = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
#         #self.mlp2 = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
#         #self.mlp3 = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
#         #self.Nconv1 = NNConv(data.num_features, 20, nn= self.mlp1)
#         #self.Nconv2 = NNConv(20,20,self.mlp2)
#         #self.Nconv3 = NNConv(20,20,self.mlp3)
#         #self.classlayer = nn.Linear(20,2)
#         #
#
#     def forward(self, x, edge_index, attr):
#         # x, edge_index, attr = data.x, data.edge_index, data.edge_attr
#         # x = F.dropout(x, training=self.training)
#         # x = F.dropout(x, training=self.training)
#         # x = self.fc(x.view(-1)).view((-1, 1))
#         ######################################################################
#         x = F.leaky_relu(self.conv1(x, edge_index, attr))
#         x = F.leaky_relu(self.conv2(x, edge_index, attr))
#         x = self.conv3(x, edge_index, attr)
#         x = geom_nn.global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long).cuda())
#         # Original code by Peng
#         ######################################################################
#         #x = self.Nconv1(x, edge_index, attr)
#         #x = self.Nconv2(x, edge_index, attr)
#         #x = self.Nconv3(x, edge_index, attr)
#         #x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long).cuda())
#         #x = self.classlayer(x)
#         return F.log_softmax(x, 1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(iter_count, train_idx=None, val_idx=None, epoch=0):
    model.train()
    epoch_loss = torch.tensor(0.0)
    accs = []
    if train_idx is None:
        train_idx = range(len(dataset))

    random.shuffle(train_idx)

    batch_counter = 0
    loss_accum = torch.tensor(0.)

    for iter, idx in enumerate(train_idx):
        if batch_counter % batch_size == 0:
            optimizer.zero_grad()
        batch_counter = batch_counter + 1
        data = dataset[idx]
        data = data.to(device)
        out_put = model(data.x, data.edge_index, data.edge_attr)
        # loss = F.nll_loss(out_put.view((1, -1)), data.y.view(-1).type(torch.long))
        # Peng's original code
        loss = F.nll_loss(out_put, data.y.view(-1).type(torch.long))
        loss.backward()
        if batch_counter % batch_size == 0:
            optimizer.step()
        epoch_loss += loss.data
        acc = out_put.max(1)[1].eq(data.y.view(-1).type(torch.long))
        accs.append(acc.cpu().numpy())

        # if epoch > 3000:
        #     val_acc = test(val_idx)
        #     res[iter_count][val_idx] = val_acc
        #     iter_count = iter_count + 1

    print(epoch_loss, np.array(accs).mean())


def test(val_idx=None):
    model.eval()
    accs =[]
    if val_idx is None:
        return 0
    for iter in val_idx:
        data = dataset[iter]
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr).max(1)[1]
        acc = pred.eq(data.y.view(-1).type(torch.long))
        accs.append(acc.cpu().numpy().item())
    model.train()
    return np.array(accs)


skf = StratifiedKFold(n_splits=5)
iter = skf.split(dataset, dataset.label)

for train_idx, val_idx in iter:
    model = MyLayer().to(device)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
    # lr should be lr = 0.000001
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.000001,
                                 betas=(0.99, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
    iter_count = 0
    for epoch in range(n_epoch):
        train(iter_count, train_idx, val_idx, epoch)
        val_acc = test(val_idx)
        res[epoch][val_idx] = val_acc
        print(str(epoch) + ' val:' + str(val_acc.mean()))
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))

avg = res.mean(1)
print(avg.max())
