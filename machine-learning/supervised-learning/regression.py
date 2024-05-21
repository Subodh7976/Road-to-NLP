import numpy as np 


class LinearRegression:
    def __init__(self, use_intercept: bool = True):
        self.use_intercept = use_intercept
        self.intercept_ = None 
        self.coef_ = None 

    def fit(self, X: np.array, y: np.array):
        if self.use_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        X_transpose = X.T
        theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

        if self.use_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = theta
        
    def predict(self, X: np.array):
        if self.use_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            predictions = X @ np.hstack([self.intercept_, self.coef_])
        else:
            predictions = X @ self.coef_
        
        return predictions
    

if __name__ == "__main__":
    X = np.random.rand(4, 2)
    y = np.dot(X, [2, 3]) + 4

    print("X:\n", X)
    print("y:\n", y)

    print("\n", "-"*5, "Linear Regression ", "-"*5, "\n")
    linear_regression = LinearRegression()
    linear_regression.fit(X, y)
    predicts = linear_regression.predict(X)
    print("Predictions:\n", predicts)
    print("Intercept:\n", linear_regression.intercept_)
    print("Coefficients:\n", linear_regression.coef_)