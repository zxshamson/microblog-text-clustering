import sys
import numpy as np
import random


class KmeansMB:

    def __init__(self, k, MB):
        self.maxiteration = 1000
        self.k = k
        self.MB = MB
        self.num = MB.shape[0]
        self.distmatrix = np.zeros((self.num, self.num))
        self.centers = []
        self.clusters = np.zeros(self.num)
        self.clusters_set = {}  # the dict is used to record each cluster contains what points
        self.distmatrix_compute()
        self.center_initialize()
        self.cluster_initialize()
        print(self.centers)
        for key in sorted(self.clusters_set.keys()):
            print(key, self.clusters_set[key])

    '''
    def distance_compute(self, a, b):  # use cosine distance
        n1 = float(np.dot(a, b.T))
        n2 = np.linalg.norm(a) * np.linalg.norm(b)
        if n2 == 0:
            n2 == 1
        cos = n1/n2
        return 1-cos
    '''

    def distance_compute(self, a, b):  # use jaccard similarity
        mul = a * b
        add = a + b
        n1 = float(sum(mul > 0))
        n2 = float(sum(add > 0))
        if n2 == 0:
            return 0
        else:
            return 1 - n1/n2

    def distmatrix_compute(self):  # compute the distance matrix
        for i in range(self.num-1):
            for j in range(i+1, self.num):
                self.distmatrix[i][j] = self.distance_compute(self.MB[i], self.MB[j])
                self.distmatrix[j][i] = self.distmatrix[i][j]

    def center_initialize(self):  # using the method of k-means++ to initialize the centers
        self.centers.append(random.randint(0, self.num-1))  # first center is assigned randomly
        for i in range(1, self.k):
            # the rest centers are selected as having the furthest distance to the existing centers
            distocenter = np.mean(self.distmatrix[self.centers], axis=0)
            self.centers.append(np.argmax(distocenter))

    def cluster_initialize(self):
        self.clusters[:] = -1
        # record each point is assigned to which clusters, all the points are assigned "-1" at the beginning
        for i in range(self.k):  # at the beginning, only the centers are assigned
            self.clusters[self.centers[i]] = i
            self.clusters_set[i] = set([self.centers[i]])
    
    def cal_newcenters(self, newclusters_set):
        newcenters = []
        for i in range(self.k):
            min_dist = float("inf")
            min_index = 0
            cluster_elements = list(newclusters_set[i])
            for j in range(len(cluster_elements)):
                index = cluster_elements[j]
                dist = np.mean(self.distmatrix[index][cluster_elements])
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            newcenters.append(min_index)
        return newcenters

    def iterate(self):
        for it in range(self.maxiteration):
            newclusters = np.zeros(self.num)
            newclusters_set = {}
            for key in sorted(self.clusters_set.keys()):
                newclusters_set[key] = set()
            for i in range(self.num):
                distocenter = self.distmatrix[i][self.centers]
                index = np.argmin(distocenter, axis=0)
                newclusters[i] = index
                newclusters_set[index].add(i)
            self.clusters = newclusters
            self.clusters_set = newclusters_set
            newcenters = self.cal_newcenters(newclusters_set)
            if newcenters == self.centers:
                break
            else:
                self.centers = newcenters

    def print_result(self):
        for key in sorted(self.clusters_set.keys()):
            print("Cluster ", key, ":")
            for t in self.clusters_set[key]:
                print(t, end=' ')
            print("\n")



