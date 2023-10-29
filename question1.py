import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def powerCurveNormal(mus, sigma, replications, n, alpha):
    results = []
    significant_tests = []

    for mu in mus:
        for i in range(replications):
            rejected = False


            # generating the jth sample of size 100 from the normal distribution
            sample = np.random.normal(mu, sigma, n)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)
            t_statistic = (sample_mean - mu)/(sample_std/math.sqrt(n))

            # calculating the critical value
            critical_value = stats.t.ppf(1-alpha, n-1)

            # checking if the null hypothesis is rejected
            if t_statistic > critical_value:
                rejected = True

            results.append(rejected)

        # calculating the number of significant tests
        significant_tests.append(np.sum(results)/replications)

    return significant_tests

def plotResults(mus, significant_tests, sigma, n):
    plt.plot(mus, significant_tests, color='blue', marker='o', linestyle='dashed')
    plt.title(f"Power Curve for Normal Distribution with sigma = {sigma} and n = {n}")
    plt.xlabel("mu")
    plt.ylabel("Power")
    plt.show()



mus = np.arange(4.5, 5.5, 0.1)
sigma = 1
replications = 1000
n = 100
alpha = 0.05

st = powerCurveNormal(mus, sigma, replications, n, alpha)
plotResults(mus, st, sigma, n)








