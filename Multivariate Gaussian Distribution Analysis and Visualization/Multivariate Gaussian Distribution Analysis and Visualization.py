import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Section 1: Data Definition
# First dataset
mu1 = np.array([1, 2])
sigma1 = np.array([[1, 0], [0, 3]])

# Second dataset
mu2 = np.array([2, 0])
sigma2 = np.array([[2, 0], [0, 1]])

# Section 2: Computational Functions


def calculate_mean(data):
    """Calculate the mean of the data."""
    return np.mean(data, axis=0)


def calculate_covariance(data):
    """Calculate the covariance matrix of the data."""
    return np.cov(data, rowvar=False)


def calculate_probability(x, mean, cov):
    """Calculate the probability density function for a multivariate normal distribution."""
    return multivariate_normal.pdf(x, mean=mean, cov=cov)


def calculate_error(predicted, actual):
    """Calculate the mean squared error between predicted and actual values."""
    return np.mean((predicted - actual) ** 2)

# Function to generate random data based on Gaussian distribution


def generate_data(mean, cov, num_samples):
    """Generate random samples from a multivariate normal distribution."""
    return np.random.multivariate_normal(mean, cov, num_samples)


# Section 3: Training and Test Data
training_data = np.array([
    [1.2, 2.1],
    [0.9, 1.8],
    [1.5, 2.5],
    [1.0, 2.0]
])

test_data = np.array([
    [1.1, 2.2],
    [0.8, 1.7],
    [1.3, 2.4]
])

# Section 4: Calculate Training Parameters
mu_training = calculate_mean(training_data)
sigma_training = calculate_covariance(training_data)

print("Training data mean:", mu_training)
print("Training data covariance:\n", sigma_training)

# Section 5: Calculate Probability for Test Data
for i, x in enumerate(test_data):
    prob = calculate_probability(x, mu_training, sigma_training)
    print(f"Probability of test data {i + 1}: {prob:.4f}")

# Section 6: Evaluate Error for Predictions
# Predicted values are set to the mean of training data
predicted = np.tile(mu_training, (test_data.shape[0], 1))
error = calculate_error(predicted, test_data)

print("Mean Squared Error (MSE):", error)

# Section 7: Using the Second Dataset
for i, x in enumerate(test_data):
    prob = calculate_probability(x, mu2, sigma2)
    print(f"Probability of test data {i + 1} (second dataset): {prob:.4f}")

# Section 8: Generate New Data from Both Datasets
# Generating new data from the first dataset
new_data1 = generate_data(mu1, sigma1, 100)
# Generating new data from the second dataset
new_data2 = generate_data(mu2, sigma2, 100)

# Section 9: Plotting the Data
plt.figure(figsize=(10, 6))
plt.scatter(new_data1[:, 0], new_data1[:, 1],
            color='blue', label='First Dataset')
plt.scatter(new_data2[:, 0], new_data2[:, 1],
            color='red', label='Second Dataset')
plt.scatter(mu1[0], mu1[1], color='blue', marker='x',
            s=200, label='Mean of First Dataset')
plt.scatter(mu2[0], mu2[1], color='red', marker='x',
            s=200, label='Mean of Second Dataset')
plt.title('Data Distribution')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
