#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:46:10 2023

@author: johnpaulmbagwu
"""

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Define the function to simulate
def f(x):
    return np.sin(x)

# Generate a simulated dataset sampled from f(x)
np.random.seed(42)
X = np.sort(4*np.pi*np.random.rand(100))
Y = f(X) + 0.1*np.random.randn(100)


# Estimate the density using Gaussian Kernel density estimation
kde = gaussian_kde(Y)

# Evaluate the density estimate at 200 points in the interval [-1, 1]
x_grid = np.linspace(-1, 1, 200)
y_grid = kde.evaluate(x_grid)

# Plot the histogram of the dataset
plt.hist(Y, bins=20, density=True, alpha=0.5)

# Plot the density estimate
plt.plot(x_grid, y_grid, color='red')

# Add labels and a legend
plt.xlabel('x')
plt.ylabel('Density')
plt.legend(['KDE', 'Dataset'])
plt.show()

# Try different bandwidth values
bandwidths = [0.01, 0.1, 0.5, 1, 5]

# Plot the density estimates for different bandwidth values
plt.hist(Y, bins=20, density=True, alpha=0.5)
for bw in bandwidths:
    kde = gaussian_kde(Y, bw_method=bw)
    y_grid = kde.evaluate(x_grid)
    plt.plot(x_grid, y_grid, label=f'BW={bw}')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

