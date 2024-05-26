import numpy as np 

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