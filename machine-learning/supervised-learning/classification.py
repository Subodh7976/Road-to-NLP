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
