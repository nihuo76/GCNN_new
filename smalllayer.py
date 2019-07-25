import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import global_mean_pool

class MyLayer(nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()

        ###################################################
        # following algorithm achieves 70% accruacy
        self.linear1 = nn.Linear(20, 32)
        self.linear3 = nn.Linear(32, 64)
        self.linear7 = nn.Linear(64, 2)
        # initialization to 0 or identity matrix
        # both achieve at least 60% of accuracy
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear3.weight)
        nn.init.zeros_(self.linear7.weight)


    def forward(self, x, edge, weight):
        row, col = edge

        ###################################################
        # following algorithm achieves 70% accruacy
        out1 = (x[col])*weight[:, None]
        out1 = F.leaky_relu(self.linear1(out1))
        out1 = scatter_mean(out1, row, dim=0)
        out1 = out1 + F.leaky_relu(self.linear1(x))

        out2 = (out1[col])*weight[:, None]
        out2 = F.leaky_relu(self.linear3(out2))
        out2 = scatter_mean(out2, row, dim=0)
        out2 = out2 + F.leaky_relu(self.linear3(out1))

        out2 = global_mean_pool(out2, torch.zeros(out2.shape[0], dtype=torch.long).cuda())
        out2 = self.linear7(out2)

        out2 = F.log_softmax(out2, dim =1)

        return out2
