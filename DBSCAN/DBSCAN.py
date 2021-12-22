import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import cm

class DBSCAN:
    def __init__(self, D, eps, MinPts):
        self.D = D
        self.eps = eps
        self.MinPts = MinPts
        self.visited = [0]*len(D)
        self.clusters = [0]*len(D)
        self.C = 0

    def performDBSCAN(self):
        for i in range(len(self.D)):
            if self.visited[i] == 0:
                self.visited[i] = 1
                neighborPts = self.regionQuery(i)
                if len(neighborPts) < self.MinPts:
                    self.clusters[i] = -1
                else:
                    self.C += 1
                    self.expandCluster(i, neighborPts)

        return self.clusters


    def expandCluster(self, P, neighborPts):
        if self.clusters[P] == 0:
            self.clusters[P] = self.C

        while len(neighborPts) > 0:
            elem = neighborPts.pop()
            if self.visited[elem] == 0:
                self.visited[elem] = 1
                newNeighborPts = self.regionQuery(elem)
                if len(newNeighborPts) >= self.MinPts:
                    neighborPts = neighborPts + newNeighborPts
                    neighborPts = list(set(neighborPts))

            if self.clusters[elem] == 0 or self.clusters[elem] == -1:
                self.clusters[elem] = self.C

    def regionQuery(self, P):
        neighborPts = []
        for i in range(len(self.D)):
            if (np.linalg.norm(self.D[P]-self.D[i]) <= self.eps):
                neighborPts.append(i)

        return neighborPts

    def plotClusters(self):

        for i in range(1, max(self.clusters) + 1):
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            points = list([self.D[d] for d in range(len(self.D)) if self.clusters[d]==i])
            points = np.array(points)
            x = (points[:,0])
            y = (points[:,1])
            plt.scatter(x, y, color=color, label='Cluster ' + str(i))
        points = np.array([self.D[d] for d in range(len(self.D)) if self.clusters[d] == -1])
        if points != []:
            x = (points[:, 0])
            y = (points[:, 1])
            plt.scatter(x, y, color="black", label='Noise')

        plt.title('Scatter plot of ' + str(max(self.clusters)) + ' different clusters formed')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.legend(fontsize=8)
        plt.show()


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
dataset = np.load('toy_set.npy')
dataset = np.array(dataset)

