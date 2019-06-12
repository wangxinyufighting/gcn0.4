import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import collections

from gcn.models import GCN
from gcn.utils import *

numSymps = 360
numHerbs = 753

randomseed = 42
epochs = 500
lr = 0.001
weight_decay = 5e-4
hidden = 50
d = 100
dropout = 0.4
lambda_ = 1

useCuda = False
cuda = useCuda and torch.cuda.is_available()
weightAdj = True

np.random.seed(randomseed)
torch.manual_seed(randomseed)
if cuda:
    torch.cuda.manual_seed(randomseed)

filePath = '../data/kg_train_edges_weighted.txt'
label, adj, features, numHerbs, numSymps = load_data(filePath, weighted=weightAdj)
herbPairRule = getHerPairMatrix('../data/herb_pair.txt')

print(label.shape)

model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            dimension=d,
            dropout=dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()

def getoutputHC(output):
    all = []
    outputHC = torch.zeros(numHerbs, numHerbs, dtype=torch.float, requires_grad=True)
    for i in output.detach():
        # print(np.array(i))
        hc = np.where(i.numpy() > 1e-3)[0]
        # print(len(hc))
        all.extend(itertools.product(hc, repeat=2))
    c = collections.Counter(all)
    for k, v in c.items():
        outputHC[k[0]][k[1]] = v/len(all)
        outputHC[k[1]][k[0]] = v/len(all)
    return outputHC

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    outputSH, outputHC = model(features, adj)
    # outputSH= model(features, adj)

    if epoch == epochs-1:
        savedP = outputSH.detach().numpy()
        np.savetxt('../result/P_'+str(randomseed)+'_'+str(epochs)+'_'+str(lr)+'_'+str(weight_decay)\
                   +'_'+str(hidden)+'_'+str(d)+'_'+str(dropout)+'_'+str(weightAdj)+'.txt', savedP)

    hc_weight = torch.tensor([38.83]*numHerbs, dtype=torch.float)
    loss1 = nn.BCEWithLogitsLoss()
    loss2 = nn.BCEWithLogitsLoss(pos_weight=hc_weight)
    # loss2 = nn.KLDivLoss()
    loss_train_sh = loss1(outputSH, label)
    loss_train_hc = loss2(outputHC, herbPairRule)
    # loss_trainKL = loss2(outpucHC, herbPairRule)
    # (lambda_ * loss_train + loss_trainKL).backward()
    (loss_train_sh + loss_train_hc).backward()
    # (lambda_ * loss_train).backward()
    optimizer.step()

    print(
        'Epoch: {:04d}'.format(epoch),
        'loss_train: {:.4f}'.format((loss_train_sh + loss_train_hc).item()),
        'time: {:.4f}'.format(time.time() - t)
    )

def test(topN):
    P = getP('../result/P_'+str(randomseed)+'_'+str(epochs)+'_'+str(lr)+'_'+str(weight_decay)\
                   +'_'+str(hidden)+'_'+str(d)+'_'+str(dropout)+'_'+str(weightAdj)+'.txt')
    testDisct = getTestDataset('../data/kg_test_edges.csv')
    p, r = getPandR(testDisct, P, topN)

    with open('../result/0a_result.txt', 'a', encoding='utf8') as f:
        f.write(str(randomseed)+'\t'+str(epochs)+'\t'+str(lr)+'\t'+str(weight_decay)+\
                   '\t'+str(hidden)+'\t'+str(d)+'\t'+str(dropout)+'\t'+str(weightAdj)+\
                '\tprecision@' + str(topN) + ':' + str(p)+'\trecall@' + str(topN) + ':' + str(r)+'\n')

    print('precision@' + str(topN) + ':' + str(p))
    print('recall@' + str(topN) + ':' + str(r))

time_total = time.time()
print('train start ...')
for epoch in range(epochs):
    train(epoch)

print('Total time elapsd: {:.4f}s'.format(time.time() - time_total))

test(5)



