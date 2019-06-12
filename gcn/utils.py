import numpy as np
import scipy.sparse as sp
import torch

numHerbs = 0
numSymps = 0

def normalize(mx):
    ''' row normalize sparse matrix '''

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def getAdj(adjFile):
    adjTemp = []
    with open(adjFile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            temp = [float(i) for i in line.strip().split()]
            adjTemp.append(temp)
    adj = torch.tensor(adjTemp, dtype=torch.float)

    return adj


def load_data(filePath, weighted=False):
    print('Loading data from '+filePath)
    edges = []
    herbs = []
    symps = []
    global numHerbs
    global numSymps
    with open(filePath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            temp = line.strip().split()
            edges.append((temp[0], temp[1]))
            if temp[0] not in herbs:
                herbs.append(temp[0])
            if temp[1] not in symps:
                symps.append(temp[1])
    #
    # numHerbs = len(herbs)
    # numSymps = len(symps)

    numHerbs = 753
    numSymps = 360

    all = numHerbs + numSymps


    adj = torch.zeros(all, all, dtype=torch.float)
    label = torch.zeros(numSymps, numHerbs, dtype=torch.float)
    # label = torch.zeros(all, numHerbs, dtype=torch.float)

    for i in edges:
        if weighted:
            adj[int(i[1])][int(i[0])] += 1
            adj[int(i[0])][int(i[1])] += 1
            label[int(i[1])][int(i[0])-numSymps] = 1
            # if int(i[0]) > numSymps:
            #     label[int(i[1])][int(i[0])-numSymps] = 1
        else:
            adj[int(i[1])][int(i[0])] = 1
            adj[int(i[0])][int(i[1])] = 1
            # label[int(i[0])][int(i[1])-numSymps] = 1
            label[int(i[1])][int(i[0])-numSymps] = 1
    # adj = adj + adj.t().multiply(adj.t() > adj) - adj.multiply(adj.t() > adj)
    tempAdj = torch.tensor(adj.t() > adj, dtype=torch.float)
    adj = adj + torch.mul(adj.t(), tempAdj) - torch.mul(adj, tempAdj)

    features = torch.eye(numSymps+numHerbs, dtype=torch.float)
    print('Load finished')
    return label, adj, features, numHerbs, numSymps

def getP(pfile):
    P = []
    with open(pfile, 'r', encoding='utf8') as f:
        all = f.readlines()
        numSymps = len(all)
        for line in all:
            temp = [float(i) for i in line.strip().split(' ')]
            sortedT = sorted(temp)
            max = sortedT[-1]
            min = sortedT[0]
            if max != min:
                temp1 = [(temp.index(i)+numSymps, (float(i)-min)/(max - min)) for i in temp]
            else:
                temp1 = [(temp.index(i)+numSymps, float(i)) for i in temp]
            P.append(temp1)
    return P

def getTestDataset(File):
    testDict = {}
    with open(File, 'r', encoding='utf8') as s:
        for line in s.readlines():
            temp = line.strip().split(',')
            tempS = temp[:-1]
            tempH = temp[-1]
            S = ''
            for i in tempS:
                S += str(i)
                if tempS.index(i) != len(tempS) - 1:
                    S += ' '
            testDict[S] = [int(x) for x in tempH.strip().split()]
    return testDict

def getPandR(testDisct, result, topN):
    precision_n = 0
    recall_n = 0
    for k, v in testDisct.items():
        count = 0
        tempSymps = [int(x) for x in k.strip().split()]
        tempResult = []
        for symp in tempSymps:
            for herb in result[symp]:
                tempResult.append(herb)
        tempResult.sort(key=lambda x: x[1], reverse=True)
        for i in tempResult[:topN]:
            herb = i[0]
            if herb in v:
                count += 1

        precision_n += count/topN
        recall_n += count/len(v)
    return precision_n/len(testDisct), recall_n/len(testDisct)



def getDict(sFile, hFile):
    idDict = {}
    id = 0
    with open(sFile, 'r', encoding='utf8') as s:
        for line in s.readlines():
            sSet = line.strip().split()
            for s in sSet:
                if s not in idDict:
                    idDict[s] = id
                    id += 1

    with open(hFile, 'r', encoding='utf8') as h:
        for line in h.readlines():
            hSet = line.strip().split()
            for h in hSet:
                if h not in idDict:
                    idDict[h] = id
                    id += 1
    return idDict

# idDict = getDict(r'D:\GCN_for_TCM\gcn0.2\data\symps_train_og.txt', r'D:\GCN_for_TCM\gcn0.2\data\herbs_train_og.txt')

def getHerbPair(pairFile, idDict):
    with open(pairFile, 'r', encoding='utf8') as f, open(r'../data/herb_pair.txt', 'w', encoding='utf8') as w:
        for line in f.readlines():
            temp = line.strip().split()
            w.write(temp[0] + ' ' + str(idDict[temp[1]])+' '+str(idDict[temp[2]])+'\n')

# getHerbPair(r'../data/herbPair.txt', idDict)

def getHerPairMatrix(pairFile):
    herbPairRule = torch.zeros(numHerbs, numHerbs, dtype=torch.float)
    with open(pairFile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            temp = line.strip().split()
            h1 = int(temp[1])-numSymps
            h2 = int(temp[2])-numSymps
            pn = int(temp[0])
            if pn == 0:
                herbPairRule[h1][h2] = 1
                herbPairRule[h2][h1] = 1
            elif pn == 1:
                herbPairRule[h1][h2] = 0
                herbPairRule[h1][h2] = 0

    return herbPairRule




























