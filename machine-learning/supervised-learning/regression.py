import numpy as np 


class LinearRegression:
    def __init__(self, use_intercept: bool = True, alpha: float = 0.0):
        self.use_intercept = use_intercept
        self.intercept_ = None 
        self.coef_ = None 
        self.alpha = alpha

    def fit(self, X: np.array, y: np.array):
        if self.use_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        X_transpose = X.T
        theta = np.linalg.inv(X_transpose @ X + self.alpha * np.eye(X_transpose.shape[0])) @ X_transpose @ y

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
    

class RidgeRegression(LinearRegression):
    def __init__(self, alpha: float = 0.01):
        super().__init__(True, alpha)

    def fit(self, X: np.array, y: np.array):
        super().fit(X, y)
        self.coef_ += self.alpha * np.sign(self.coef_)
    

class LassoRegression(LinearRegression):
    def __init__(self, alpha: float = 0.01):
        super().__init__(True, alpha)
    
    def fit(self, X: np.array, y: np.array):
        super().fit(X, y)
        self.coef_ += self.alpha * np.sign(self.coef_) * np.abs(self.coef_)
    

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

    print("\n", "-"*5, "Ridge Regression ", "-"*5, "\n")
    ridge_regression = RidgeRegression()
    ridge_regression.fit(X, y)
    predicts = ridge_regression.predict(X)
    print("Predictions:\n", predicts)
    print("Intercept:\n", ridge_regression.intercept_)
    print("Coefficients:\n", ridge_regression.coef_)

    print("\n", "-"*5, "Lasso Regression ", "-"*5, "\n")
    lasso_regression = LassoRegression()
    lasso_regression.fit(X, y)
    predicts = lasso_regression.predict(X)
    print("Predictions:\n", predicts)
    print("Intercept:\n", lasso_regression.intercept_)
    print("Coefficients:\n", lasso_regression.coef_)