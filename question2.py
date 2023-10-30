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

def plotResults(n, replications):
    b1s = samplingDistribution(n, replications)
    plt.hist(b1s, bins=10)
    plt.title(f"Distribution of b1 for n = {n}")
    plt.xlabel("b1")
    plt.ylabel("Frequency")
    plt.show()

n = 20
replications = 1000
plotResults(n, replications)