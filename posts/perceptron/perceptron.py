import numpy as np

class Perceptron:
    def __init__(self, alpha=0.1, max_steps=1000):
        self.max_steps = max_steps
        self.w = None
        self.history = []
        self.alpha = alpha
        
    def fit(self, X, y, max_steps):
        n_samples, n_features = X.shape
        self.w = np.random.rand(n_features+1, )
        
        for _ in range(self.max_steps):
            j = np.random.randint(X.shape[0])
            xi = np.append(X[j], 1)
            y_hat = np.dot(xi, self.w)
            yi = 2*y[j] - 1
            self.w += (yi * (np.dot(xi, self.w)) < 0) * yi*xi
                    
            accuracy = self.score(X, y)
            self.history.append(accuracy)
            if self.history[_] == 1:
                break 
            
    def predict(self, X):
        n_samples = X.shape[0]
        ypred = []
        
        for j in range(n_samples):
            xi = np.append(X[j], 1)
            y_hat = np.dot(xi, self.w)
            
            if y_hat >= 0:
                ypred.append(1)
            else:
                ypred.append(0)
                
        return ypred
    
    def score(self, X, y):  
        accuracy = self.predict(X)
        return (np.sign(accuracy) == y).mean() 