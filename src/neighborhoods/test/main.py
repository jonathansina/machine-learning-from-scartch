import sys

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))

from src.neighborhoods.neighbors import NearestNeighbor


def run_classification_example():
    print("=== Classification Example ===")
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_classifier = NearestNeighbor(type="classifier")
    knn_classifier.compile(k=3, metrics="euclidean", algorithm="brute-force")
    knn_classifier.train(X_train, y_train)
    
    predictions = knn_classifier.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Classification accuracy: {accuracy:.4f}")
    
    plot_classification_results(X_test, y_test, predictions)


def plot_classification_results(X_test, y_test, predictions):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 2, 2)
    plt.scatter(
        X_test[:, 0], 
        X_test[:, 1], 
        c=y_test, 
        cmap='viridis', 
        alpha=0.7, 
        s=100, 
        edgecolors='k', 
        marker='o', 
        label='True'
    )
    plt.scatter(
        X_test[:, 0], 
        X_test[:, 1], 
        c=predictions, 
        cmap='viridis', 
        alpha=0.3, 
        s=50, 
        marker='s', 
        label='Predicted'
    )
    plt.title('Classification Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_regression_example():
    print("\n=== Regression Example ===")
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_regressor = NearestNeighbor(type="regressor")
    knn_regressor.compile(k=5, metrics="manhattan", algorithm="kd-tree")
    knn_regressor.train(X_train, y_train)
    
    predictions = knn_regressor.predict(X_test)
    mse = ((predictions - y_test) ** 2).mean()
    print(f"Regression MSE: {mse:.4f}")
    
    plot_regression_results(y_test, predictions)


def plot_regression_results(y_test, predictions):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions)
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    residuals = y_test - predictions
    plt.scatter(y_test, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_classification_example()
    run_regression_example()