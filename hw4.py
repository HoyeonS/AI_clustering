import csv
from collections import OrderedDict
import numpy as np
import scipy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn import cluster

def load_data(filepath):
    my_list = []
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for line in csv_reader:
            my_list.append(OrderedDict(line))
    return my_list

def calc_feature(row):

    arr = np.array([np.int64(row.get('Attack')), np.int64(row.get('Sp. Atk')), np.int64(row.get('Speed')), np.int64(row.get('Defense')), np.int64(row.get('Sp. Def')), np.int64(row.get('HP'))])
    arr.shape = (6,)
    return arr

def hac(features):
    #return hierarchy.linkage(features)
    n = len(features)
    Z = np.zeros((n-1,4))
    cluster_size = []
    for i in range(n):
        cluster_size.append(1)
    for i in range(n-1):

        Z[i,0],Z[i,1],features,Z[i,2],Z[i,3] = findAndMerge(features)
        # print(features)

    return Z


def findAndMerge(features):
    n = len(features)
    max_dis = -1
    first,second = 0,0
    for x in range(n):
        if(features[x][0] == -1):
            continue
        for tmp in range(n-x - 1):
            y = tmp + x + 1
            if(features[y][0] == -1):
                continue
            dis = getDistance(features[x] , features[y])
            if(max_dis == -1 or max_dis > dis):
                max_dis = dis
                first = x
                second = y
    features.append(concat(features[first], features[second]))
    size = len(concat(features[first], features[second]))/6
    pop(features,first)
    pop(features,second)
    return first, second, features, max_dis, size


def getDistance(v1, v2):
    l1 = []
    l2 = []
    for x in range(int(len(v1)/6)):
        l1.append(v1[x*6:(x+1)*6])
    for y in range(int(len(v2)/6)):
        l2.append(v2[y*6:(y+1)*6])
    list_dis = []
    for p in l1:
        for q in l2:
            list_dis.append(np.linalg.norm(p-q))
    return max(list_dis)

def pop(features, index):
    features[index][0] = -1
    return features
    
def concat(v1, v2):
    arr = np.concatenate((v1,v2))


    return arr


# def find(part, row):
#     point = row[part]
#     clstr = 0
#     distance = -1
#     for count in range(len(row) - part - 1):
#         arr = row[count + part + 1]
#         dis = np.linalg.norm(point, arr)
#         if(distance == -1 | distance > dis):
#             distance = dis
#             clstr = count + part + 1
#     return clstr, distance

        
    

def imshow_hac(Z):
    dn = hierarchy.dendrogram(Z)
    plt.show()