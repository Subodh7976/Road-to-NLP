import numpy as np 


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None 
        self.mean = None 

    def fit(self, X: np.array):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean 

        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.array):
        X = X - self.mean 
        return np.dot(X, self.components)
    

if __name__ == "__main__":
    X = np.random.randn(4, 3)
    print("X:\n", X)
    print("\n")

    print("-"*5, " PCA ", "-"*5)
    pca = PCA(1)
    pca.fit(X)
    print("Transformed Components: \n", pca.transform(X))
