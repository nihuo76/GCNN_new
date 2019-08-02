import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        # self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        self.conv1 = ChebConv(20, 32, K=1)
        self.conv2 = ChebConv(32, 64, K=2)
        self.conv3 = ChebConv(64, 2, K=2)
        # self.fc = nn.Linear(16 * data.num_nodes, 1)

    def forward(self, x, edge_index, attr):
        # x, edge_index, attr = data.x, data.edge_index, data.edge_attr
        x = F.leaky_relu(self.conv1(x, edge_index, attr))
        # x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index, attr))
        # x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, attr)
        x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long).cuda())
        # x = self.fc(x.view(-1)).view((-1, 1))
        return F.log_softmax(x, dim=1)



