import numpy as np 
from sklearn.tree import DecisionTreeRegressor


class XGBoost:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, min_samples_split: int = 2, 
                 reg_lambda: float = 1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.trees = []

    def fit(self, X: np.array, y: np.array):
        y_pred = np.full(y.shape, np.mean(y))
        for _ in range(self.n_estimators):
            residuals = y - self._sigmoid(y_pred)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                         min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            self.trees.append(tree)

            update = tree.predict(X)
            y_pred += self.learning_rate * update 
        
    def predict_proba(self, X: np.array):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(y_pred)
    
    def predict(self, X: np.array):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)
    
    def _sigmoid(self, X: np.array):
        return 1 / (1 + np.exp(-X))
    

if __name__ == "__main__":
    X = np.random.randn(4, 2)
    y = np.array([1, 0, 1, 0])
    print("X:\n", X)
    print("y:\n", y)

    print("\n")
    print("-"*5, " XGBoost ", "-"*5)
    xg = XGBoost()
    xg.fit(X, y)
    print("Predictions:\n", xg.predict(X))
