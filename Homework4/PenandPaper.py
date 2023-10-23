import numpy as np
from scipy.stats import multivariate_normal

# definir as observações
data = np.array([
    [1, 0.6, 0.1],
    [0, -0.4, 0.8],
    [0, 0.2, 0.5],
    [1, 0.4, -0.1]
])

# definir parametros iniciais
pi1, pi2 = 0.5, 0.5
p1, p2 = 0.3, 0.7
mu1, mu2 = np.array([1, 1]), np.array([0, 0])
cov1, cov2 = np.array([[2, 0.5], [0.5, 2]]), np.array([[1.5, 1], [1, 1.5]])

# obter as likelihoods
likelihood1 = multivariate_normal.pdf(data[:, 1:], mean=mu1, cov=cov1)
likelihood2 = multivariate_normal.pdf(data[:, 1:], mean=mu2, cov=cov2)

# obter posteriors (responsibilities)
responsibilities = np.empty((4, 2))
responsibilities[:, 0] = pi1 * likelihood1
responsibilities[:, 1] = pi2 * likelihood2
responsibilities /= responsibilities.sum(axis=1, keepdims=True)