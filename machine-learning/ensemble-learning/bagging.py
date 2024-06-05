import numpy as np 
from collections import Counter 
from sklearn.tree import DecisionTreeClassifier
from typing import Any


class RandomForest:
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, max_features: Any = "sqrt", 
                 bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def _bootstrap_sample(self, X: np.array, y: np.array):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_max_features(self, n_features: int):
        if isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        else:
            return n_features
    
    def fit(self, X: np.array, y: np.array):
        self.trees = []
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          min_samples_split=self.min_samples_split, 
                                          max_features=max_features)
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y 
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def predict(self, X: np.array):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        majority_votes = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(majority_votes)
    
    def predict_proba(self, X: np.array):
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        avg_probs = np.mean(tree_probs, axis=0)
        return avg_probs
    

if __name__ == "__main__":
    X = np.random.randn(4, 2)
    y = np.array([0, 1, 0, 1])
    print("X:\n", X)
    print("y:\n", y)

    print("\n\n")
    print("-"*5, " Random Forest ", "-"*5)
    print("\n")
    rf = RandomForest()
    rf.fit(X, y)
    print("Predictions:\n", rf.predict(X))

