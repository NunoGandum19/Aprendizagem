import numpy as np
from scipy.stats import multivariate_normal

# Initial parameters
pi1, pi2 = 0.5, 0.5
p1, p2 = 0.3, 0.7
u1 = np.array([1, 1])
u2 = np.array([0, 0])
sigma1 = np.array([[2, 0.5], [0.5, 2]])
sigma2 = np.array([[1.5, 1], [1, 1.5]])

# observações
observations = np.array([[1, 0.6, 0.1],
                        [0, -0.4, 0.8],
                        [0, 0.2, 0.5],
                        [1, 0.4, -0.1]])

# E-step
responsibilities = np.zeros((observations.shape[0], 2))
# i indica o índice da observação
i = 0
# iteramos pelas observações
for obs in observations:
    y1, y2_y3 = obs[0], obs[1:]

    bernoulli1 = p1**y1 * (1-p1)**(1-y1)
    bernoulli2 = p2**y1 * (1-p2)**(1-y1)

    gaussian1 = multivariate_normal.pdf(y2_y3, mean=u1, cov=sigma1)
    gaussian2 = multivariate_normal.pdf(y2_y3, mean=u2, cov=sigma2)
    
    # responsabilities não normalizadas
    responsibilities[i, 0] = pi1 * bernoulli1 * gaussian1
    responsibilities[i, 1] = pi2 * bernoulli2 * gaussian2

    # normalização das responsabilities
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    i += 1

# M-step
N1 = responsibilities[:, 0].sum()
N2 = responsibilities[:, 1].sum()

pi1 = N1 / observations.shape[0]
pi2 = N2 / observations.shape[0]

p1 = (responsibilities[:, 0] * observations[:, 0]).sum() / N1
p2 = (responsibilities[:, 1] * observations[:, 0]).sum() / N2

u1 = (responsibilities[:, 0][:, np.newaxis] * observations[:, 1:]).sum(axis=0) / N1
u2 = (responsibilities[:, 1][:, np.newaxis] * observations[:, 1:]).sum(axis=0) / N2

diff1 = (observations[:, 1:] - u1)
diff2 = (observations[:, 1:] - u2)

sigma1 = (responsibilities[:, 0][:, np.newaxis, np.newaxis] * diff1[:, np.newaxis, :] * diff1[:, :, np.newaxis]).sum(axis=0) / N1
sigma2 = (responsibilities[:, 1][:, np.newaxis, np.newaxis] * diff2[:, np.newaxis, :] * diff2[:, :, np.newaxis]).sum(axis=0) / N2

print("Updated parameters:")
print("responsibilities:", responsibilities)
print("N1:", N1)
print("N2:", N2)
print("pi1:", pi1)
print("pi2:", pi2)
print("p1:", p1)
print("p2:", p2)
print("u1:", u1)
print("u2:", u2)
print("sigma1:", sigma1)
print("sigma2:", sigma2)
