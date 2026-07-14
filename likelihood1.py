import numpy as np
from scipy.optimize import minimize_scalar
import scipy.stats as stats


heads = 7
tosses = 10


p_mle_direct = heads / tosses
print(f"Direct MLE Probability: {p_mle_direct:.4f}")


def negative_log_likelihood(p):
   
    if p <= 0 or p >= 1:
        return np.inf
    
    return -stats.binom.logpmf(heads, tosses, p)


result = minimize_scalar(negative_log_likelihood, bounds=(0, 1), method='bounded')
print(f"Numerical MLE Probability: {result.x:.4f}")
