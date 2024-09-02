import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('car_price_dataset.csv', na_values='?')

# Data Preprocessing
df = df.dropna()
df = df.apply(pd.to_numeric, errors='ignore')

# Feature selection and encoding
features = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 
             'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']
target = 'price'

X = df[features]
y = df[target]

# Normalize features
X_scaled = (X - X.mean()) / X.std()

# Split the dataset
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X.iloc[train_indices].values
    X_test = X.iloc[test_indices].values
    y_train = y.iloc[train_indices].values
    y_test = y.iloc[test_indices].values
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000, batch_size=32):
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for epoch in range(epochs):
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            predictions = X_batch.dot(theta)
            errors = predictions - y_batch

            gradient = (1 / batch_size) * X_batch.T.dot(errors)
            theta -= learning_rate * gradient

        # Calculate the loss
        predictions = X.dot(theta)
        mae = np.mean(np.abs(predictions - y))
        loss_history.append(mae)

    return theta, loss_history

# Train the model
theta, loss_history = mini_batch_gradient_descent(X_train, y_train)

# Evaluate the model
y_pred = X_test.dot(theta)
mae = np.mean(np.abs(y_test - y_pred))
print(f"Mean Absolute Error on Test Set: {mae}")

# Plot loss history
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Loss History')

# Plot difference between target and predicted values
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, c='blue', label='Predicted')
plt.scatter(y_test, y_test, c='red', label='Target')
plt.xlabel('Target Value')
plt.ylabel('Predicted Value')
plt.title('Target vs Predicted Values')
plt.legend()

plt.show()