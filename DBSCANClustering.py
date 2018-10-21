BASE_LABEL = -1
OUTLIER_LABEL = -99

# fungsi untuk membentuk union dari 2 set
def set_union(neighbors, neighbors_set, new_neighbors):
    for neighbor in new_neighbors:
        if neighbor not in neighbors_set:
            neighbors.append(neighbor)
            neighbors_set.add(neighbor)

# kelas untuk membandung model DBSCAn
class DBSCAN:
    dataset = None
    prediction = None

    def __init__(self, epsilon, min_points):
        self.epsilon = epsilon
        self.min_points = min_points
        
    def fit(self, dataset):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.prediction = [BASE_LABEL] * self.dataset_size
        self.__dbscan()

    # algoritma utama
    def __dbscan(self):
        current_cluster = BASE_LABEL
        for data_index in range(self.dataset_size):
            if self.prediction[data_index] != BASE_LABEL:
                continue
            neighbors = self.__find_neighbors(data_index)
            if len(neighbors) < self.min_points:
                self.prediction[data_index] = OUTLIER_LABEL
            else:
                current_cluster += 1
                self.__claim_cluster(current_cluster, data_index, neighbors)

    # fungsi untuk mencari tetangga dari suatu titik yang tidak lebih jauh dari epsilon
    def __find_neighbors(self, data_index):
        neighbors = []
        for index in range(self.dataset_size):
            if np.linalg.norm(self.dataset[data_index] - self.dataset[index]) < self.epsilon:
                neighbors.append(index)
        return neighbors

    # fungsi untuk menentukan anggota cluster yang sama dengan seed
    def __claim_cluster(self, current_cluster, data_index, neighbors):
        self.prediction[data_index] = current_cluster
        index = 0
        neighbors_set = set(neighbors)
        while index < len(neighbors):
            current_data = neighbors[index]
            if self.prediction[current_data] == OUTLIER_LABEL:
                self.prediction[current_data] = current_cluster
            elif self.prediction[current_data] == BASE_LABEL:
                self.prediction[current_data] = current_cluster
                current_neighbors = self.__find_neighbors(current_data)
                if len(current_neighbors) >= self.min_points:
                    set_union(neighbors, neighbors_set, current_neighbors)
            index += 1

