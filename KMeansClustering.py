from copy import deepcopy
import numpy as np
import pandas as pd

class KMeans:
    dataset = None
    labels = None
    centroid = None

    def __init__(self, nkluster):
        self.nkluster = nkluster

    def fit(self, dataset):
        self.dataset = dataset
        self.dataset_shape = dataset.shape
        self.labels = np.zeros(self.dataset_shape[0])
        self.centroid = self.__init_centroid()
        self.__KMeans()

    def __init_centroid(self):
        mean = np.mean(self.dataset, axis = 0)
        std = np.std(self.dataset, axis = 0)
        C = np.random.randn(self.nkluster,self.dataset_shape[1])*std + mean
        return C

    def __KMeans(self):
        C_old = np.zeros(self.centroid.shape)
        distances = np.zeros((self.dataset_shape[0],self.nkluster))
        error = np.linalg.norm(self.centroid - C_old)
        while error != 0:
            # memasukkan setiap titik ke cluster terdekatnya
            # Assigning each value to its closest cluster
            for i in range(self.nkluster):
                distances[:,i] = np.linalg.norm(self.dataset - self.centroid[i], axis=1)
                self.labels = np.argmin(distances, axis=1)
            C_old = deepcopy(self.centroid)
            for i in range(self.nkluster):
                #points = [self.dataset[j] for j in range(len(self.dataset)) if self.labels[j] == i]
                self.centroid[i] = np.mean(self.dataset[self.labels==i], axis=0)
            error = np.linalg.norm(self.centroid - C_old)



