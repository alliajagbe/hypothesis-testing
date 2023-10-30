# dimension free measure of skewness

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def b1(n, X):
    numerator = (1/n) * np.sum((X - np.mean(X))**3)
    denominator = (1/n) * np.sum((X - np.mean(X))**2)**(3/2)
    return numerator/denominator

def normalDistribution(n):
    X = np.random.normal(0, 1, n)
    return X

def samplingDistribution(n, replications):
    b1s = []
    for i in range(replications):
        X = normalDistribution(n)
        b1s.append(b1(n, X))
    return b1s

def computeMeanAndStandardDeviation(n, replications):
    b1s = samplingDistribution(n, replications)
    return np.mean(b1s), np.std(b1s)

def theoreticalMeanAndStandardDeviation(n):
    mean = 0
    numerator = 6*(n-2)
    denominator = (n+1)*(n+3)
    std = np.sqrt(numerator/denominator)
    return mean, std


def plotResults(n, replications, ax, colors):
    b1s = samplingDistribution(n, replications)
    mean, std = computeMeanAndStandardDeviation(n, replications)
    t_mean, t_std = theoreticalMeanAndStandardDeviation(n)
    ax.hist(b1s, bins=20, density=True, label=f"n = {n}\nmean = {round(mean, 3)}\nstd = {round(std, 3)}\ntheoretical mean = {round(t_mean, 3)}\ntheoretical std = {round(t_std, 3)}", color = colors[i])
    ax.set_title(f"Sampling Distribution of b1")
    ax.set_xlabel("b1")
    ax.set_ylabel("Density")
    ax.legend(loc='best')



sizes = [10, 20, 100]
replications = 1000
colors = ["red", "green", "blue"]



# creating a canvas of 1x3 subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i in range(len(sizes)):
    plotResults(sizes[i], replications, axs[i], colors)
plt.show()
