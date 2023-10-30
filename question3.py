import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import skew


def skewness(X):
    n = len(X)
    numerator = (1/n) * np.sum((X - np.mean(X))**3)
    denominator = ((1/n) * np.sum((X - np.mean(X))**2))**(3/2)
    return numerator/denominator

def powerCurve(n, replications, alpha, true_mu, sigma1, sigma2, epsilons, skewness):

    significant_tests = []

    for e in epsilons:
        rejected_count = 0

        for _ in range(replications):
            if np.random.uniform() < e:
                sample = np.random.normal(true_mu, sigma2, n)
            else:
                sample = np.random.normal(true_mu, sigma1, n)
            
            sk = skewness(sample)
            
            # Checking if the null hypothesis is rejected cdf 
            if sk > 0: 
                p_value = 1 - norm.cdf(sk)
            else:
                p_value = norm.cdf(sk)

            if p_value < alpha:
                rejected_count += 1

        # Calculating the proportion of significant tests
        significant_tests.append(rejected_count / replications)

        print(f"epsilon = {e}, proportion of significant tests = {rejected_count / replications}")

    return significant_tests


def plotResults(epsilons, significant_tests, n):
    plt.plot(epsilons, significant_tests, label=f"n = {n}")
    plt.title(f"Power Curve for Mixture of Normals")
    plt.xlabel("epsilon")
    plt.ylabel("Power")
    plt.show()

epsilons = np.linspace(0, 1, 100)
sigma1 = 1
sigma2 = 10
replications = 1000
alpha = 0.01
true_mu = 0
size = 30

st = powerCurve(size, replications, alpha, true_mu, sigma1, sigma2, epsilons, skewness)
plotResults(epsilons, st, size)





