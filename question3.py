import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def powerCurve(n, replications, alpha, true_mu, sigma1, sigma2):

    significant_tests = []
    t_tab = stats.t.ppf(1 - alpha, n - 1)

    epsilon = np.arange(0, 1, 0.1)

    for e in epsilon:
        rejected_count = 0

        for _ in range(replications):
            if np.random.uniform() < e:
                sample = np.random.normal(true_mu, sigma2, n)
            else:
                sample = np.random.normal(true_mu, sigma1, n)
            
            t, _ = stats.ttest_1samp(sample, true_mu)

            # Checking if the null hypothesis is rejected using the t value
            if t > t_tab or t < -t_tab:
                rejected_count += 1

        # Calculating the proportion of significant tests
        significant_tests.append(rejected_count / replications)

        print(f"epsilon = {e}, proportion of significant tests = {rejected_count / replications}")

    return significant_tests

powerCurve(30, 1000, 0.01, 0, 1, 10)



