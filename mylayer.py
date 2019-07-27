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
        self.linear1 = nn.Linear(4, 8)
        # self.linear2 = nn.Linear(20, 64)
        self.linear3 = nn.Linear(8, 16)
        # self.linear4 = nn.Linear(64, 128)
        self.linear5 = nn.Linear(16, 32)
        # self.linear6 = nn.Linear(128, 256)
        self.linear7 = nn.Linear(32, 2)
        # initialization to 0 or identity matrix
        # both achieve at least 60% of accuracy
        # nn.init.zeros_(self.linear1.weight)
        # nn.init.zeros_(self.linear2.weight)
        # nn.init.zeros_(self.linear3.weight)
        # nn.init.zeros_(self.linear4.weight)
        # nn.init.zeros_(self.linear5.weight)
        # nn.init.zeros_(self.linear6.weight)
        # nn.init.zeros_(self.linear7.weight)


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

        out3 = (out2[col]) * weight[:, None]
        out3 = F.leaky_relu(self.linear5(out3))
        out3 = scatter_mean(out3, row, dim=0)
        out3 = out3 + F.leaky_relu(self.linear5(out2))

        out3 = global_mean_pool(out3, torch.zeros(out3.shape[0], dtype=torch.long).cuda())
        out3 = self.linear7(out3)

        out3 = F.log_softmax(out3, dim =1)

        return out3
