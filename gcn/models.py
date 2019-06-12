import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layer import GraphConvoluation
# from gcn.train import numSymps, numHerbs

numSymps = 360
numHerbs = 753
class GCN(nn.Module):
    def __init__(self, nfeat, nhid,dimension, dropout):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvoluation(nfeat, nhid)
        self.gcn2 = GraphConvoluation(nhid, dimension)
        self.dropout = dropout
        self.f_sh = nn.Linear(dimension, numHerbs)
        self.f_hc = nn.Linear(dimension, numHerbs)



    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        # x = F.relu(self.gcn2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        sh = torch.sigmoid(self.f_sh(x[:numSymps]))
        hc = torch.sigmoid(self.f_hc(x[numSymps:]))
        # sh = F.softmax(torch.sigmoid(self.f(x[:numSymps])), dim=1)
        return sh, hc