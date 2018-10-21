import numpy as np

from sklearn import datasets
import pandas as pd

class AgglomerativeClustering:
    SINGLE_LINKAGE = 'single'
    
    dataset = None
    distance_matrix = None
    data_structure = None
    cluster_key = {}
    
    def __init__(self, linkage=SINGLE_LINKAGE):
        self.linkage = linkage

    def fit(self, dataset):
        self.dataset = dataset
        self.__create_distance_matrix()
        while len(self.distance_matrix) > 1:
            clusters = self.__claim_current_cluster()
            self.__update_distance_matrix(clusters)
        self.data_structure = self.__release_cluster([self.distance_matrix.index[0]])
            
    def __create_distance_matrix(self):
        dataset_size = len(self.dataset)
        distance_matrix = np.full((dataset_size, dataset_size), np.inf)
        for index in range(dataset_size):
            for other_index in range(index + 1, dataset_size):
                distance_matrix[index][other_index] = np.linalg.norm(self.dataset[index] - self.dataset[other_index])
        self.distance_matrix = pd.DataFrame(distance_matrix)

    def __claim_current_cluster(self):
        min_distance = np.min(self.distance_matrix.values)
        indexes = np.where(self.distance_matrix.values == min_distance)
        clusters = []
        for row, col in zip(indexes[0], indexes[1]):
            clusters.append([self.distance_matrix.index[row], self.distance_matrix.columns[col]])
        AgglomerativeClustering.__purify_cluster(clusters)
        addition = 1
        for cluster in clusters:
            self.cluster_key[self.distance_matrix.index[-1] + addition] = cluster
            addition += 1
        return clusters

    def __calculate_distance(self, item_1, item_2):
        row, col = self.__get_cluster_key(item_1), self.__get_cluster_key(item_2)
        item_1_idx = self.distance_matrix.index.get_loc(row)
        item_2_idx = self.distance_matrix.index.get_loc(col)
        if item_2_idx < item_1_idx:
            row, col = col, row
        return self.distance_matrix.loc[row, col]

    def __update_distance_matrix(self, clusters):
        for cluster in clusters:
            new_distances = []
            for row in self.distance_matrix.index:
                member_distances = []
                for col in cluster:
                    distance = self.__calculate_distance(row, col)
                    member_distances.append(distance if distance != np.inf else 0.0)
                new_distances.append(np.min(member_distances))
            self.__inject_cluster_to_matirx(cluster, new_distances)

    def __inject_cluster_to_matirx(self, cluster_label, new_distance):
        self.distance_matrix[self.__get_cluster_key(cluster_label)] = new_distance
        self.distance_matrix.loc[self.__get_cluster_key(cluster_label)] = [np.inf] * len(self.distance_matrix.columns)
        self.distance_matrix.drop(columns=[self.__get_cluster_key(idx) for idx in cluster_label], inplace=True)
        self.distance_matrix.drop(index=[self.__get_cluster_key(idx) for idx in cluster_label], inplace=True)
        
    
    @staticmethod
    def __purify_cluster(clusters):
        idx = 0
        while idx < len(clusters):
            other_idx = idx + 1
            while other_idx < len(clusters):
                intersection = [item for item in clusters[other_idx] if item in clusters[idx]]
                if any(intersection):
                    clusters[idx] += [item for item in clusters[other_idx] if item not in intersection]
                    del clusters[other_idx]
                else:
                    other_idx += 1
            idx += 1

    def __get_cluster_key(self, cluster):
        if type(cluster) != list:
            return cluster
        for key, value in self.cluster_key.items():
            if value == cluster:
                return key
        return None

    def __release_cluster(self, cluster):
        retval = []
        for member in cluster:
            if member not in self.cluster_key.keys():
                retval.append(member)
            else:
                retval.append(self.__release_cluster(self.cluster_key[member]))
        return retval

    @staticmethod
    def dump_newick_format(list_input):
        if type(list_input) != list:
            return str(list_input)
        retval = '('
        list_size = len(list_input)
        for idx, member in enumerate(list_input):
            retval += AgglomerativeClustering.dump_newick_format(member)
            if idx < list_size - 1:
                retval += ','
        retval += ')'
        return retval
