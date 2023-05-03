import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.1, max_iter=1000):
        self.max_iter = max_iter
        self.w = None
        self.score_history = []
        self.alpha = alpha
        
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def fit_analytic(self, X, y):
        X_ = self.pad(X)
        self.w = np.linalg.inv(X_.T@X_)@X_.T@y
        
    def fit_gradient(self, X, y, max_iter, alpha):
        X_ = self.pad(X)
        self.w = np.random.rand(X_.shape[1])
        self.score_history = []
        
        P = X_.T@X_
        q = X_.T@y
        for _ in range(max_iter):
            gradient = (P@self.w - q)
            self.w -= alpha * gradient
            self.score_history.append(self.score(X, y))
            
    def fit(self, X, y, method='analytical', max_iter = 1000, alpha = 0.1):
        if method == 'analytical':
            self.fit_analytic(X, y)
        else:
            self.fit_gradient(X, y, max_iter, alpha)
            
    def predict(self, X):
        X_ = self.pad(X)
        return X_@self.w
    
    def score(self, X, y):
        y_bar = y.mean()
        y_hat = self.predict(X)
        return 1 - (sum((y_hat - y)**2) / sum((y_bar - y)**2))