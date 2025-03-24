# Machine Learning From Scratch

This project implements various machine learning algorithms from scratch in Python. It provides a comprehensive educational toolkit to understand the fundamentals of machine learning algorithms by implementing them with basic numerical libraries like NumPy.

## Table of Contents
- Project Structure
- Installation
- Implemented Algorithms
    - Linear Models
        - Linear Regression
        - Logistic Regression
        - Linear Support Vector Machine (SVM) 
    - Tree-based Models
    - Ensemble Methods
- Nearest Neighbors
- Hyperparameter Tuning
- Contributing
- License

## Project Structure
```
machine-learning-from-scratch/
│
├── src/
│   ├── linear/                # Linear models (regression, classification)
│   │   ├── base.py            # Base linear model class
│   │   ├── models/            # Specific model implementations
│   │   └── components/        # Components (loss, optimizers, regularizers)
│   │
│   ├── tree/                  # Decision tree algorithms
│   │   ├── base.py            # Base decision tree implementation
│   │   ├── factory.py         # Tree factory patterns
│   │   ├── impurity/          # Impurity measures
│   │   └── test/              # Tests for tree models
│   │
│   ├── ensembles/             # Ensemble methods
│   │   ├── bagging/           # Bagging implementations
│   │   └── boosting/          # Boosting implementations
│   │
│   ├── neighborhood/          # Nearest neighbor models
│   │   ├── base.py            # Base KNN implementation
│   │   ├── distances.py       # Distance metrics
│   │   ├── search.py          # Search algorithms
│   │   └── test/              # Tests for KNN models
│   │
│   └── tunning/               # Hyperparameter tuning
│       ├── base.py            # Bayesian optimization
│       └── asquisition/       # Acquisition functions
│
└── README.md            # Project Documentation
```

---

## Installation
To use this project, follow these steps:

```python
# Clone the repository
git clone https://github.com/yourusername/machine-learning-from-scratch.git
cd machine-learning-from-scratch

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib scikit-learn pandas seaborn
```

---

## Implemented Algorithms
### 1. Linear Models

#### Features:

- Various optimization algorithms (SGD, AdaGrad, RMSProp, ADAM, Newton Method)
- Multiple loss functions (MSE, MAE, Huber, Hinge, Binary Crossentropy, Log loss)
- Regularization options (L1, L2, ElasticNet)
- Mini-batch training support

#### 1.1 Linear Regression

Implemented by leveraging optimization techniques to reduce the discrepancy between predicted and actual values, ensuring the model learns effectively from the data.

**Supported Loss Functions**: MSE, MAE, HUBER

#### Example Usage
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.linear.components.optimizer import SGD
from src.linear.models.linear import LinearRegression
from src.linear.components.regularizer import L1Regularizer

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and compile the model
linear_model = LinearRegression()
linear_model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.001),
    loss="mse",
    regularizer=L1Regularizer(_lambda=0.08)
)

# Train the model
linear_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Make predictions
predictions = linear_model.predict(X_test)
mse = ((predictions - y_test) ** 2).mean()
print(f"Mean Squared Error: {mse:.4f}")
```

#### 1.2 Logistic Regression
Implemented similarly to linear regression but designed for classification tasks. Logistic regression uses a sigmoid activation function to map predictions to probabilities and optimizes loss functions such as Log Loss or Binary Crossentropy to improve classification accuracy.

**Supported Loss Functions**: Log Loss, Binary Crossentropy

#### Example Usage
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.linear.components.optimizer import SGD
from src.linear.models.linear import LogisticRegression
from src.linear.components.regularizer import L2Regularizer

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and compile the model
linear_model = LogisticRegression()
linear_model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.001),
    loss="binary_crossentropy",
    regularizer=L2Regularizer(_lambda=0.08)
)

# Train the model
linear_model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Make predictions
predictions = linear_model.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"Logistic regression accuracy: {accuracy:.4f}")
```

#### 1.3 Linear Support Vector Machine (SVM)
