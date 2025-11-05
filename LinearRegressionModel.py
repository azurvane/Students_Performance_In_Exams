import numpy as np
from tqdm import tqdm

class LinearRegression:
    """
    Multiple Linear Regression Model
    
    Model: y = w1*x1 + w2*x2 + ... + wn*xn + b
    
    Parameters:
    -----------
    n_features : int
        Number of input features
    weights : np.ndarray, optional
        Initial weights of shape (n_features,). If None, weights are randomly initialized.
    bias : float, optional
        Initial bias term. If None, bias is randomly initialized. Set to False to disable bias.
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_features, weights = None, bias=None , random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(self.random_state)
        if weights is not None:
            if weights.shape != (n_features,):
                raise ValueError(f"Weights shape must be ({n_features},), got {weights.shape}")
            self.weights = np.array(weights)
        else:
            limit = np.sqrt(1.0 / n_features)
            self.weights = np.random.uniform(-limit, limit, size=n_features)
        
        if bias is not None:
            self.bias = bias
        else:
            self.bias = 0
    
    def set_weights(self, weights, bias=None):
        """
        Set new weights and optionally bias
        
        Parameters:
        -----------
        weights : np.ndarray
            New weights of shape (n_features,)
        bias : float, optional
            New bias value. If None, bias remains unchanged.
        """
        weights = np.array(weights)
        if self.weights.shape != weights.shape:
            raise ValueError(f"Weights shape must be ({self.n_features},), got {weights.shape}")
        
        self.weights = weights.copy()
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.uniform(-0.1, 0.1)
    
    def get_weights(self):
        """
        Get current weights and bias
        
        Returns:
        --------
        tuple
            (weights, bias) where weights is np.ndarray of shape (n_features,)
        """
        return self.weights.copy(), self.bias
    
    def predict(self, X):
        """
        Make prediction for input features
        
        Parameters:
        -----------
        X : np.ndarray
            Input features of shape (m_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Predicted values of shape (n_samples,)
        """

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Feature count mismatch: Expected {self.weights.shape[0]} features (columns), but input X has {X.shape[1]}.")
        
        prediction = np.dot(X, self.weights) + self.bias
        return prediction

    def train(self, X, y, iterations = 10000, learning_rate = 0.1):
        """
        train the model and updates the weights and bias

        Parameters:
        -----------
        X : np.ndarray
            Input features of shape (m_samples, n_features)
        y : np.ndarray
            actual target values of shape (m_samples,)
        iterations : int
            number of iterations for training loop
        learing_rate : float
            by how much should the model learn
        """

        m, n = X.shape
        loss_history = []  # Store losses
        # Create a progress bar object
        pbar = tqdm(range(iterations), desc="Training Linear Regression")

        for epoch in pbar:
            # making the prediction
            prediction = self.predict(X)

            # calculating the loss
            MSE = np.mean((y - prediction) ** 2)

            # calculating the gradient
            error = prediction - y
            dl_db = np.mean(error)
            dl_dw = X.T @ error / m

            # updating the weights and bias
            self.weights = self.weights - learning_rate * dl_dw
            self.bias = self.bias - learning_rate * dl_db

            loss_history.append(MSE)
            # Update the progress bar with the current loss
            pbar.set_postfix({"MSE": f"{MSE:.4f}", "w_norm": f"{np.linalg.norm(self.weights):.2f}"})

        return loss_history

