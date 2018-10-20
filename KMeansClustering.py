from copy import deepcopy
import numpy as np
import pandas as pd

def jarak(a,b, ax = 1):
    return np.linalg.norm(a-b,axis = ax)

class KMeans:
    def __init__(self, nkluster):
        self.nkluster = nkluster

    def fit(self, dataset):
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.labels = np.zeros(self.dataset_length)
        self.centroid = self.__init_centroid()
        self.__KMeans

    def __init_centroid(self):
        c_x = np.random.randint(0, np.max(self.dataset)-20, size = self.nkluster)
        c_y = np.random.randint(0, np.max(self.dataset)-20, size = self.nkluster)
        C = np.array(list(zip(c_x,c_y)), dtype = np.float32)
        return C

    def __KMeans(self):
        C = self.centroid
        C_old = np.zeros(C.shape)
        self.labels = np.zeros(self.dataset_length)
        error = jarak(C, C_old, None)
        while error != 0:
            # memasukkan setiap titik ke cluster terdekatnya
            # Assigning each value to its closest cluster
            for i in range(len(self.dataset)):
                distances = jarak(self.dataset[i], C)
                cluster = np.argmin(distances)
                self.labels[i] = cluster
            C_old = deepcopy(C)
            for i in range(self.nkluster):
                points = [self.dataset[j] for j in range(len(self.dataset)) if self.labels[j] == i]
                C[i] = np.mean(points, axis=0)
            error = jarak(C, C_old, None)



