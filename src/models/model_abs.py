from abc import ABC, abstractmethod

class AbstractModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model using the preprocessed data.

        Parameters:
        - X_train: Input features
        - y_train: Target labels
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
        - X_test: Test input features

        Returns:
        - Predicted values (y_test)
        """
        pass