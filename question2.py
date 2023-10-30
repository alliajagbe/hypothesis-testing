# dimension free measure of skewness

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

def b1(n, X):
    numerator = (1/n) * np.sum((X - np.mean(X))**3)
    denominator = ((1/n)*np.sum(X - np.mean(X))**2)**(3/2)
    return numerator/denominator