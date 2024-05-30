import numpy as np 
from collections import Counter

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.001, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None 

    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = np.zeros(1)

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias 

            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 

    def predict(self, X: np.array):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def predict_proba(self, X: np.array):
        linear_model = np.dot(X, self.weights) + self.bias 
        y_predicted = self._sigmoid(linear_model)
        return y_predicted
    
    def _sigmoid(self, X: np.array):
        return 1 / (1 + np.exp(-X))
    

class KNN:
    def __init__(self, k: int = 3):
        self.k = k 
        self.X_train = None 
        self.y_train = None 

    def fit(self, X_train: np.array, y_train: np.array):
        self.X_train = X_train 
        self.y_train = y_train 

    def predict(self, X: np.array):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, X: np.array):
        distances = [np.sqrt(np.sum((x_train - X)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class NaiveBayes:
    def __init__(self):
        self.classes = None 
        self.mean = None 
        self.var = None 
        self.priors = None 

    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape 
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y==c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.mean(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X: np.array):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x: np.array):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator 
    

class SVM:
    def __init__(self, 
                 learning_rate: float = 1e-3, 
                 lambda_param: float = 0.01, 
                 n_iterations: int = 1000
                 ):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None 
        self.bias = None 

    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape 
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0 

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - 
                                                          np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]
    
    def predict(self, X: np.array) -> np.array:
        linear_output = np.dot(X, self.weights) - self.bias 
        return np.sign(linear_output)

class Node:
    def __init__(self, feature=None, threshold=None, 
                 left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value 

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None 

    def fit(self, X: np.array, y: np.array):
        self.root = self._grow_tree(X, y)
    
    def predict(self, X: np.array):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X: np.array, y: np.array, depth: int = 0):
        n_samples, n_features = X.shape 
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        best_feat, best_thresh = self._best_split(X, y, n_features)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)
    
    def _best_split(self, X: np.array, y: np.array, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None 
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain 
                    split_idx = feature_idx
                    split_thresh = threshold
        
        return split_idx, split_thresh
    
    def _information_gain(self, y: np.array, X_column: np.array, split_thresh):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r 

        ig = parent_entropy - child_entropy
        return ig 
    
    def _split(self, X_column: np.array, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self, y: np.array):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self, y: np.array):
        hist = np.bincount(y)
        return np.argmax(hist)
    
    def _traverse_tree(self, X: np.array, node):
        if node.value is not None:
            return node.value 
        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)
        return self._traverse_tree(X, node.right)



if __name__ == "__main__":
    X = np.random.rand(4, 2)
    y = np.array([1, 0, 0, 1])

    print("X:\n", X)
    print("y:\n", y)

    print("\n", "-"*5, "Logistic Regression ", "-"*5, "\n")
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X, y)
    predicts = logistic_regression.predict(X)
    print("Predictions:\n", predicts)
    print("Bias:\n", logistic_regression.bias)
    print("Weights:\n", logistic_regression.weights)

    print("\n", "-"*5, "K-Nearest Neighbors  ", "-"*5, "\n")
    knn = KNN()
    knn.fit(X, y)
    predicts = knn.predict(X)
    print("Predictions:\n", predicts)
    print("K:\n", knn.k)

    print("\n", "-"*5, "Naive Bayes  ", "-"*5, "\n")
    naive_bayes = NaiveBayes()
    naive_bayes.fit(X, y)
    predicts = naive_bayes.predict(X)
    print("Predictions:\n", predicts)

    print("\n", "-"*5, "Support Vector Machine  ", "-"*5, "\n")
    svm = SVM()
    svm.fit(X, y)
    predicts = svm.predict(X)
    print("Predictions:\n", predicts)
    print("Weights:\n", svm.weights)
    print("Bias:\n", svm.bias)

    print("\n", "-"*5, "Decision Tree  ", "-"*5, "\n")
    decision_tree = DecisionTree()
    decision_tree.fit(X, y)
    predicts = decision_tree.predict(X)
    print("Predictions:\n", predicts)
