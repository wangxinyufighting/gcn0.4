import torch
import torch.nn.functional as F
import itertools
import numpy as np
import time
import collections
'''
numHerbs = 753
P = []
pfile = r'../result/P_42_110_0.001_0.0005_30_80_0.1_True.txt'
with open(pfile, 'r', encoding='utf8') as f:
    all = f.readlines()
    numSymps = len(all)
    for line in all:
        temp = [float(i) for i in line.strip().split(' ')]
        sortedT = sorted(temp)
        max = sortedT[-1]
        min = sortedT[0]
        if max != min:
            temp1 = [float(float(i)/sum(temp)) for i in temp]
        # else:
        #     print(max)
        #     print(min)
        #     temp1 = [float(i) for i in temp]
        P.append(temp1)
# print(P)

def getoutputHC(output):
    all = []
    outputHC = torch.zeros(numHerbs, numHerbs, dtype=torch.float, requires_grad=True)
    for i in output:
        # print(np.array(i))
        hc = np.where(np.array(i) > 1e-3)[0]
        # print(len(hc))
        all.extend(itertools.product(hc, repeat=2))
    c = collections.Counter(all)
    for k, v in c.items():
        outputHC[k[0]][k[1]] = v/len(all)
        outputHC[k[1]][k[0]] = v/len(all)
    return outputHC

start = time.time()
outputHC = getoutputHC(P)
end = time.time()
print(end-start)
with open('../result/t1.txt', 'w', encoding='utf8') as f:
    for i in outputHC:
        f.write(str(i)+'\n')

'''
# code = ['DB00349','DB04953','DB00849','DB06201','DB05246','DB04841','DB01202','DB00906','DB00230','DB01068','DB00313','DB00273','DB00832 ','DB00532','DB00593','DB00996','DB00555','DB01080','DB06218','DB00252','DB00683','DB09502','DB00829','DB09016','DB01121','DB00776','DB00347','DB00564','DB00909','DB00949','DB09118','DB09061','DB02083','DB05885','DB12131','DB06089','DB12458','DB00837','DB06657','DB05363','DB04982','DB14977','DB00463','DB05821','DB15203','DB06458','DB09001','DB05087','DB00311','DB05919','DB14009','DB11755','DB01553','DB08838','DB09011','DB08831','DB14050']
# file = 'D:\drugbank_all_full_database.xml\drugbank_dump_OG.nt'
# out = 'D:\drugbank_all_full_database.xml\drugbank_epilespy_triples.txt'
# with open(file, 'r', encoding='utf8') as f, open(out, 'w', encoding='utf8') as g:
#     for line in f.readlines():
#         for i in code:
#             if i in line:
#                 g.write(line)

with open('D:\drugbank_all_full_database.xml\hpo.txt', 'r', encoding='utf8') as f, \
    open('D:\drugbank_all_full_database.xml\hpo_.txt', 'w', encoding='utf8') as g:
    # nameFlag = False
    for line in f.readlines():
        if line.strip():
            temp = line.strip().split(': ')
            if 'id: HP' in line:
                name = temp[1]
                # nameFlag = True
            else:
                g.write(name + '\t' + temp[0]+'\t'+temp[1]+'\n')




















