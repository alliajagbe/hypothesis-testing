import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def powerCurveNormal(mus, sigma, replications, n, alpha, true_mu):
    significant_tests = []
    t_tab = stats.t.ppf(1 - alpha, n - 1)

    for mu in mus:
        rejected_count = 0

        for _ in range(replications):

            sample = np.random.normal(mu, sigma, n)

            t, _ = stats.ttest_1samp(sample, true_mu)

            # Checking if the null hypothesis is rejected using the t value
            if t > t_tab:
                rejected_count += 1
            

        # Calculating the proportion of significant tests
        significant_tests.append(rejected_count / replications)

        print(f"mu = {mu}, proportion of significant tests = {rejected_count / replications}")

    return significant_tests

def plotResults(mus, significant_tests, sigma, n):
    plt.plot(mus, significant_tests, label=f"n = {n}")
    plt.title(f"Power Curve for Normal Distribution")
    plt.xlabel("mu")
    plt.ylabel("Power")



mus = np.arange(4.5, 6.5, 0.1)
sigma = 1
replications = 1000
alpha = 0.05
true_mu = 5

sizes = [10, 20, 100]
for i in sizes:
    st = powerCurveNormal(mus, sigma, replications, i, alpha, true_mu)
    plotResults(mus, st, sigma, i)
plt.legend()
plt.show()

#%%
def powerCurveMixtureOfNormals(mus, sigma1, sigma2, replications, n, alpha, true_mu):
    significant_tests = []
    t_tab = stats.t.ppf(1 - alpha, n - 1)

    for mu in mus:
        rejected_count = 0

        for _ in range(replications):
            if np.random.uniform() < 0.9:
                sample = np.random.normal(mu, sigma1, n)
            else:
                sample = np.random.normal(mu, sigma2, n)

            t, _ = stats.ttest_1samp(sample, true_mu)

            # Checking if the null hypothesis is rejected using the t value
            if t > t_tab:
                rejected_count += 1

        # Calculating the proportion of significant tests
        significant_tests.append(rejected_count / replications)

        print(f"mu = {mu}, proportion of significant tests = {rejected_count / replications}")








