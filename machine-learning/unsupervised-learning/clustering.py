import numpy as np 

class KMeans:
    def __init__(self, k: int = 3, max_iters: int = 100, 
                 tol: float = 1e-4):
        self.k = k 
        self.max_iters =  max_iters
        self.tol = tol 
        self.centroids = None 

    def fit(self, X: np.array):
        n_samples, n_features = X.shape 

        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            clusters = self._create_clusters(X)
            old_centroids = self.centroids 

            self.centroids = np.array([X[cluster].mean(axis=0) for cluster in clusters])

            if self._is_converged(old_centroids, self.centroids):
                break 
        
    def predict(self, X: np.array):
        cluster_labels = [self._get_nearest_centroid(x) for x in X]
        return cluster_labels
    
    def _create_clusters(self, X: np.array):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            centroid_idx = self._get_nearest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _get_nearest_centroid(self, sample: np.array):
        distances = [np.linalg.norm(sample - centroid) for centroid in self.centroids]
        nearest_centroid_idx = np.argmin(distances)
        return nearest_centroid_idx
    
    def _is_converged(self, old_centroid: np.array, new_centroids: np.array):
        distances = [np.linalg.norm(old_centroid[i] - new_centroids[i]) for i in range(self.k)]
        return sum(distances) < self.tol 
    
class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps 
        self.min_samples = min_samples
        self.labels = None 

    def fit(self, X: np.array):
        n_samples = X.shape[0]
        self.labels = -np.ones(n_samples)
        cluster_id = 0

        for i in range(n_samples):
            if self.labels[i] != -1:
                continue

            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1 
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
    
    def fit_predict(self, X: np.array):
        self.fit(X)
        return self.labels 
    
    def _expand_cluster(self, X: np.array, i: int, neighbors: list, cluster_id: int):
        self.labels[i] = cluster_id
        queue = list(neighbors)

        while queue:
            point = queue.pop(0)
            if self.labels[point] == -1:
                self.labels[point] = cluster_id
            if self.labels[point] != -1:
                continue

            self.labels[point] = cluster_id
            point_neighbors = self._region_query(X, point)
            if len(point_neighbors) >= self.min_samples:
                queue.extend(point_neighbors)
    
    def _region_query(self, X: np.array, point_idx: int):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[point_idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

class FuzzyCMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 150, 
                 m: float = 2.0, tol: float = 1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m 
        self.tol = tol 
        self.centers = None 
        self.u = None 

    def fit(self, X: np.array):
        n_samples = X.shape[0]
        self.u = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)

        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            
            self.centers = self._compute_centers(X)
            self.u = self._update_u(X)
            if np.linalg.norm(self.u - u_old) < self.tol:
                break
    
    def fit_predict(self, X: np.array):
        self.fit(X)
        return np.argmax(self.u, axis=1)
    
    def _compute_centers(self, X: np.array):
        um = self.u ** self.m 
        return (um.T @ X) / um.sum(axis=0)[:, None]
    
    def _update_u(self, X: np.array):
        power = 2 / (self.m - 1)
        temp = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                temp[:, i] += (np.linalg.norm(X - self.centers[i], axis=1) / np.linalg.norm(X - self.centers[j], axis=1))
        return 1 / temp 
    

if __name__ == "__main__":
    X = np.random.randn(4, 2)
    print("X:\n", X)

    print("-"*5, " KMeans ", "-"*5)
    kmeans = KMeans()
    kmeans.fit(X)
    print("Clusters:\n", kmeans.predict(X))

    print("-"*5, " DBSCAN ", "-"*5)
    dbscan = DBSCAN()
    print("Clusters:\n", dbscan.fit_predict(X))

    print("-"*5, " Fuzzy C-Means ", "-"*5)
    fuzzy_cmeans = FuzzyCMeans()
    print("Clusters:\n", fuzzy_cmeans.fit_predict(X))
