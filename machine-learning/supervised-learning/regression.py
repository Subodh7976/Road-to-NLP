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


class PolynomialRegression:
    def __init__(self, use_intercept: bool = True, degree: int = 2, alpha: float = 0.0):
        self.degree = degree
        self.use_intercept = use_intercept
        self.alpha = alpha

        self.intercept_ = None 
        self.coef_ = None

    def fit(self, X: np.array, y: np.array):
        X = np.hstack([X[:, i][:, np.newaxis]**j for i in range(X.shape[1]) 
                       for j in range(1, self.degree+1)])
        
        if self.use_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        theta = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.T.shape[0])) @ X.T @ y

        if self.use_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = theta 

    def predict(self, X: np.array):
        X = np.hstack([X[:, i][:, np.newaxis]**j for i in range(X.shape[1]) 
                       for j in range(1, self.degree+1)])
        
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

    print("\n", "-"*5, "Polynomial Regression ", "-"*5, "\n")
    poly_regression = PolynomialRegression(degree=2)
    poly_regression.fit(X, y)
    predicts = poly_regression.predict(X)
    print("Degree:\n", poly_regression.degree)
    print("Predictions:\n", predicts)
    print("Intercept:\n", poly_regression.intercept_)
    print("Coefficients:\n", poly_regression.coef_)