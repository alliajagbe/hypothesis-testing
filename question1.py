import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def powerCurveNormal(mus, sigma, replications, n, alpha):
    significant_tests = []

    df = n - 1
    crit_values = stats.t.interval(1 - alpha, df, loc=0, scale=1)

    for mu in mus:
        rejected_count = 0

        for i in range(replications):
            # generating the jth sample of size 100 from the normal distribution
            sample = np.random.normal(mu, sigma, n)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample)
            t_statistic = (sample_mean - mu)/(sample_std/math.sqrt(n))

            # checking if the null hypothesis is rejected
            if t_statistic > crit_values[1] or t_statistic < crit_values[0]:
                rejected_count += 1

        # calculating the proportion of significant tests
        significant_tests.append(rejected_count / replications)

        print(f"mu = {mu}, proportion of significant tests = {rejected_count / replications} and results = {rejected_count}")

    return significant_tests

def plotResults(mus, significant_tests, sigma, n):
    plt.plot(mus, significant_tests)
    plt.title(f"Power Curve for Normal Distribution with sigma = {sigma} and n = {n}")
    plt.xlabel("mu")
    plt.ylabel("Power")
    plt.show()



mus = np.arange(4.5, 6.5, 0.1)
sigma = 1
replications = 1000
n = 10
alpha = 0.05

st = powerCurveNormal(mus, sigma, replications, n, alpha)
plotResults(mus, st, sigma, n)








