import numpy as np
import matplotlib.pyplot as plt

samples = np.random.randn(100)
plt.hist(samples, bins=10, density=True, alpha=0.75, rwidth=0.8)
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()