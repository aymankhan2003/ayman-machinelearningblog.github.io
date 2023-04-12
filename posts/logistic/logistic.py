import numpy as np

class LogisticRegression:
    def __init__(self, alpha=0.1, max_epochs=1000):
        self.w = None
        self.alpha = alpha
        self.max_epochs = max_epochs
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, y_hat, y):
        return (-y * np.log(y_hat) - (1-y) * np.log(1-y_hat)).mean()
    
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
            
    def fit(self, X, y, alpha=None, max_epochs=None):
        n_samples, n_features = X.shape
        self.w = np.random.rand(n_features+1)
        
        self.loss_history = []
        self.score_history = []
        
        if alpha is not None:
            self.alpha = alpha
        if max_epochs is not None:
            self.max_epochs = max_epochs
        
        for _ in range(self.max_epochs):
            j = np.random.randint(X.shape[0])
            xi = np.append(X[j], 1)
            y_hat = np.dot(xi, self.w)
            yi = self.sigmoid(y_hat)
            
            gradient = np.dot(self.sigmoid(y_hat) - y[j], xi) / n_samples
            self.w -= self.alpha * gradient
            
            accuracy = self.score(X, y)
            self.loss_history.append(self.loss(yi, y))
            self.score_history.append(accuracy)
            
        
    def fit_stochastic(self, X, y, alpha=None, max_epochs=None, batch_size=None, momentum=False):  
        if alpha is not None:
            self.alpha = alpha
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if batch_size is None:
            batch_size = X.shape[0]
        
        n_samples, n_features = X.shape
        self.w = np.random.rand(n_features+1)

    
        self.loss_history = []   #initialize loss_history list
        self.score_history = []  # initialize score_history list 
        beta = np.zeros_like(self.w) if momentum else 0

        for epoch in range(self.max_epochs):
            order = np.arange(n_samples)
            np.random.shuffle(order)
    
            for batch in np.array_split(order, n_samples // batch_size + 1):
                xi = X[batch,:]  
                yi = y[batch]
                y_hat = np.dot(xi, self.w)
        
                if momentum:
                    gradient = np.dot(self.sigmoid(y_hat) - yi, xi) / n_samples
                    beta = 0.8*beta + alpha * gradient
                    self.w -= beta
                else:
                    gradient = np.dot(self.sigmoid(y_hat) - yi, xi) / n_samples
                    self.w -= alpha * gradient
            
            y_hat = self.sigmoid(np.dot(X, self.w))
            accuracy = self.score(X, y)
            self.loss_history.append(self.loss(y_hat, y))
            self.score_history.append(accuracy) 
            
            
    def predict(self, X):
        n_samples = X.shape[0]
        ypred = []

        for j in range(n_samples):
            xi = np.append(X[j], 1) if len(self.w) == X.shape[1]+1 else X[j]
            y_hat = self.sigmoid(np.dot(xi, self.w))

            if y_hat >= 0.5:
                ypred.append(1)
            else:
                ypred.append(0)

        return ypred

    
    def score(self, X, y):  
        accuracy = self.predict(X)
        return (accuracy == y).mean() 