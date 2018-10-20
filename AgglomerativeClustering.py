cluster_key = {}

def create_distance_matrix(dataset):
    distance_matrix = np.full((len(df),len(df)), np.inf)
    dataset = iris.data
    dataset_size = len(iris.data)
    for index in range(dataset_size):
        for other_index in range(index + 1, dataset_size):
            distance_matrix[index][other_index] = np.linalg.norm(dataset[index]-dataset[other_index])
    return pd.DataFrame(distance_matrix)

def claim_current_cluster(distance_matrix):
    min_distance = np.min(distance_matrix.values)
    indexes = np.where(distance_matrix.values == min_distance)
    clusters = []
    for row, col in zip(indexes[0], indexes[1]):
        clusters.append([distance_matrix.index[row], distance_matrix.columns[col]])
    purify_cluster(clusters)
    addition = 1
    for cluster in clusters:
        cluster_key[distance_matrix.index[-1] + addition] = cluster
        addition += 1
    return clusters

def purify_cluster(clusters):
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

def calculate_distance(item_1, item_2, distance_matrix):
    row, col = get_cluster_key(item_1), get_cluster_key(item_2)
    item_1_idx = distance_matrix.index.get_loc(row)
    item_2_idx = distance_matrix.index.get_loc(col)
    if item_2_idx < item_1_idx:
        row, col = col, row
    return distance_matrix.loc[row, col]

def update_distance_matrix(distance_matrix, clusters):
    for cluster in clusters:
        new_distances = []
        for row in distance_matrix.index:
            member_distances = []
            for col in cluster:
                distance = calculate_distance(row, col, distance_matrix)
                member_distances.append(distance if distance != np.inf else 0.0)
            new_distances.append(np.min(member_distances))
        inject_cluster_to_matirx(cluster, new_distances, distance_matrix)

def inject_cluster_to_matirx(cluster_label, new_distance, distance_matrix):
    distance_matrix[get_cluster_key(cluster_label)] = new_distance
    distance_matrix.loc[get_cluster_key(cluster_label)] = [np.inf] * len(distance_matrix.columns)
    distance_matrix.drop(columns=[get_cluster_key(idx) for idx in cluster_label], inplace=True)
    distance_matrix.drop(index=[get_cluster_key(idx) for idx in cluster_label], inplace=True)
    
def get_cluster_key(cluster):
    if type(cluster) != list:
        return cluster
    for key, value in cluster_key.items():
        if value == cluster:
            return key
    return None


def release_cluster(cluster):
    retval = []
    
    for member in cluster:
        if member not in cluster_key.keys():
            retval.append(member)
        else:
            retval.append([release_cluster(cluster_key[member])])
    return retval

/* main */
while len(distance_matrix) > 1:
    cluster = claim_current_cluster(distance_matrix)
    update_distance_matrix(distance_matrix, cluster)
    print len(distance_matrix)