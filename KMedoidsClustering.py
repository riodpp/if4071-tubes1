import numpy as np

class kmedoids:

    X = None
    medoids = None
    clusters = None

    def __init__(self, X):
        self.X = X
        self.clusters = []
        self.medoids = []
    
    def make_cluster(self,res):
        cluster = np.zeros(len(self.X))
        for i in range(1,self.k):
            # print res[i]
            for j in res[i]:
                cluster[j] = i
        
        return cluster

    def fit(self,k,max_iter=100):
        # train model
        # input: Matrix of Manhattan distance
        # output: Array of Centroids

        self.k = k
        dist = self.compute_manhattan() # Buat matriks jarak (manhattan distance)
        m, n = dist.shape # hitung dimensi matriks jarak
        M = np.sort(np.random.choice(n, k)) # inisialisasi titik secara random
        Mnew = np.copy(M)
        C = {}

        for t in xrange(max_iter):
            # determine clusters, i.e. arrays of data indices
            J = np.argmin(dist[:,M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J==kappa)[0]
            
            # update cluster medoids
            for kappa in range(k):
                J = np.mean(dist[np.ix_(C[kappa],C[kappa])],axis=1)
                j = np.argmin(J)
                Mnew[kappa] = C[kappa][j]
            np.sort(Mnew)

            # check for convergence
            if np.array_equal(M, Mnew):
                break
            
            M = np.copy(Mnew)
        else:
            # final update of cluster memberships
            J = np.argmin(dist[:,M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J==kappa)[0]
        
        self.medoids = M
        # print C
        # self.clusters = C
        self.clusters = self.make_cluster(C)

        return self


    # def predict(self):
        # generate prediction
        # input: array of points
        # output: array of cluster number

    def compute_manhattan(self):
        # compute manhattan distance
        # input: Array of Points
        # output: matrix of manhattan distance

        size_i, size_j = self.X.shape
        dist = np.zeros((size_i, size_i))
        for i in range(size_i):
            for j in range(i+1, size_i):
                sum = 0
                for k in range(size_j):
                    sum = sum + abs(self.X[i][k]-self.X[j][k])
                dist[i,j] = sum
                dist[j,i] = dist[i,j]
        
        return dist
