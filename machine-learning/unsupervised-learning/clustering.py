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
    

if __name__ == "__main__":
    X = np.random.randn(4, 2)
    print("X:\n", X)

    print("-"*5, " KMeans ", "-"*5)
    kmeans = KMeans()
    kmeans.fit(X)
    print("Clusters:\n", kmeans.predict(X))
