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

# E-step: Compute responsibilities
responsibilities = np.zeros((observations.shape[0], 2))
for i, obs in enumerate(observations):
y1, y2_y3 = obs[0], obs[1:]

p_y1_given_theta1 = p1**y1 * (1-p1)**(1-y1)
p_y1_given_theta2 = p2**y1 * (1-p2)**(1-y1)

p_y2_y3_given_theta1 = multivariate_normal.pdf(y2_y3, mean=u1, cov=sigma1)
p_y2_y3_given_theta2 = multivariate_normal.pdf(y2_y3, mean=u2, cov=sigma2)

responsibilities[i, 0] = pi1 * p_y1_given_theta1 * p_y2_y3_given_theta1
responsibilities[i, 1] = pi2 * p_y1_given_theta2 * p_y2_y3_given_theta2

responsibilities /= responsibilities.sum(axis=1, keepdims=True)

# M-step: Update parameters
pi1 = responsibilities[:, 0].mean()
pi2 = responsibilities[:, 1].mean()

p1 = (responsibilities[:, 0] * observations[:, 0]).sum() / responsibilities[:, 0].sum()
p2 = (responsibilities[:, 1] * observations[:, 0]).sum() / responsibilities[:, 1].sum()

u1 = (responsibilities[:, 0][:, np.newaxis] * observations[:, 1:]).sum(axis=0) / responsibilities[:, 0].sum()
u2 = (responsibilities[:, 1][:, np.newaxis] * observations[:, 1:]).sum(axis=0) / responsibilities[:, 1].sum()

sigma1 = ((responsibilities[:, 0][:, np.newaxis, np.newaxis] * (observations[:, 1:] - u1).reshape(observations.shape[0], 2, 1) * (observations[:, 1:] - u1).reshape(observations.shape[0], 1, 2)).sum(axis=0)) / responsibilities[:, 0].sum()
sigma2 = ((responsibilities[:, 1][:, np.newaxis, np.newaxis] * (observations[:, 1:] - u2).reshape(observations.shape[0], 2, 1) * (observations[:, 1:] - u2).reshape(observations.shape[0], 1, 2)).sum(axis=0)) / responsibilities[:, 1].sum()

print("Updated parameters:")
print("pi1:", pi1)
print("pi2:", pi2)
print("p1:", p1)
print("p2:", p2)
print("u1:", u1)
print("u2:", u2)
print("sigma1:", sigma1)
print("sigma2:", sigma2)
