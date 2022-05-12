import numpy as np


def reweigth_distribution(original_distribution, temperature=0.1):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


dis = [0.1, 0.4, 0.2, 0.3]
print(reweigth_distribution(dis))
