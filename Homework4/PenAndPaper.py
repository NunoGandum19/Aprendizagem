import numpy as np
from scipy.stats import multivariate_normal

###
def Ber(x, p):
    return (p**x) * ((1-p)**(1-x))

########## 1. #################################################################################################
'''Perform one epoch of the EM clustering algorithm and determine the new parameters'''
##### Initialization #####
# Observations
x1 = np.array([1, 0.6, 0.1])
x2 = np.array([0, -0.4, 0.8])
x3 = np.array([0, 0.2, 0.5])
x4 = np.array([1, 0.4, -0.1])

pi1 = 0.5
pi2 = 0.5

p1 = 0.3
p2 = 0.7

u1 = np.array([1, 1])
Sigma1 = np.array([[2, 0.5], [0.5, 2]])
u2 = np.array([0, 0])
Sigma2 = np.array([[1.5, 1], [1, 1.5]])

##### Expectation (E-step) #####
"""
- Calcular o gama para cada observação, e cada cluster
"""
# x1
P_x1_c1 = Ber(x1[0], p1) * multivariate_normal.pdf(x1[1:], mean=u1, cov=Sigma1)
P_x1_c2 = Ber(x1[0], p2) * multivariate_normal.pdf(x1[1:], mean=u2, cov=Sigma2)

gamma_1_1 = P_x1_c1 * pi1 / (P_x1_c1 * pi1 + P_x1_c2 * pi2)
gamma_2_1 = P_x1_c2 * pi2 / (P_x1_c1 * pi1 + P_x1_c2 * pi2)

# x2
P_x2_c1 = Ber(x2[0], p1) * multivariate_normal.pdf(x2[1:], mean=u1, cov=Sigma1)
P_x2_c2 = Ber(x2[0], p2) * multivariate_normal.pdf(x2[1:], mean=u2, cov=Sigma2)

gamma_1_2 = P_x2_c1 * pi1 / (P_x2_c1 * pi1 + P_x2_c2 * pi2)
gamma_2_2 = P_x2_c2 * pi2 / (P_x2_c1 * pi1 + P_x2_c2 * pi2)

# x3
P_x3_c1 = Ber(x3[0], p1) * multivariate_normal.pdf(x3[1:], mean=u1, cov=Sigma1)
P_x3_c2 = Ber(x3[0], p2) * multivariate_normal.pdf(x3[1:], mean=u2, cov=Sigma2)

gamma_1_3 = P_x3_c1 * pi1 / (P_x3_c1 * pi1 + P_x3_c2 * pi2)
gamma_2_3 = P_x3_c2 * pi2 / (P_x3_c1 * pi1 + P_x3_c2 * pi2)

# x4
P_x4_c1 = Ber(x4[0], p1) * multivariate_normal.pdf(x4[1:], mean=u1, cov=Sigma1)
P_x4_c2 = Ber(x4[0], p2) * multivariate_normal.pdf(x4[1:], mean=u2, cov=Sigma2)

gamma_1_4 = P_x4_c1 * pi1 / (P_x4_c1 * pi1 + P_x4_c2 * pi2)
gamma_2_4 = P_x4_c2 * pi2 / (P_x4_c1 * pi1 + P_x4_c2 * pi2)

##### Maximization #####
"""
atuliazação dos parametros(u, Sigma, pi, p)
"""
# Cluster 1
N1_new = gamma_1_1 + gamma_1_2 + gamma_1_3 + gamma_1_4
u1_new = (gamma_1_1*x1[1:] + gamma_1_2*x2[1:] + gamma_1_3*x3[1:] + gamma_1_4*x4[1:])/N1_new
#Sigma1_new = (gamma_1_1*np.dot(array1_1, array1_1.T) + gamma_1_2*np.dot(array1_2, array1_2.T) + 
#              gamma_1_3*np.dot(array1_3, array1_3.T) + gamma_1_4*np.dot(array1_4, array1_4.T))/N1_new
Sigma1_new = (gamma_1_1*np.outer(x1[1:] - u1_new, x1[1:] - u1_new) + gamma_1_2*np.outer(x2[1:] - u1_new, x2[1:] - u1_new) + 
              gamma_1_3*np.outer(x3[1:] - u1_new, x3[1:] - u1_new) + gamma_1_4*np.outer(x4[1:] - u1_new, x4[1:] - u1_new))/N1_new
p1_new = (gamma_1_1*x1[0] + gamma_1_2*x2[0] + gamma_1_3*x3[0] + gamma_1_4*x4[0])/(gamma_1_1 + gamma_1_2 + gamma_1_3 + gamma_1_4)
pi1_new = N1_new/4

# Cluster 2
N2_new = gamma_2_1 + gamma_2_2 + gamma_2_3 + gamma_2_4
u2_new = (gamma_2_1*x1[1:] + gamma_2_2*x2[1:] + gamma_2_3*x3[1:] + gamma_2_4*x4[1:])/N2_new
Sigma2_new = (gamma_2_1*np.outer(x1[1:] - u2_new, x1[1:] - u2_new) + gamma_2_2*np.outer(x2[1:] - u2_new, x2[1:] - u2_new) +
              gamma_2_3*np.outer(x3[1:] - u2_new, x3[1:] - u2_new) + gamma_2_4*np.outer(x4[1:] - u2_new, x4[1:] - u2_new))/N2_new
p2_new = (gamma_2_1*x1[0] + gamma_2_2*x2[0] + gamma_2_3*x3[0] + gamma_2_4*x4[0])/(gamma_2_1 + gamma_2_2 + gamma_2_3 + gamma_2_4)
pi2_new = N2_new/4

############ 2. #######################################################################################################
'''
Given the new observation, x_new=(1, 0.3, 0.7), determine the cluster membership probabilities(posteriors)
'''
# -> calcular o gama para o x_new para cada cluster, e normalizar para dar a prob
x_new = np.array([1, 0.3, 0.7])

P_x_new_c1 = Ber(x_new[0], p1_new) * multivariate_normal.pdf(x_new[1:], mean=u1_new, cov=Sigma1_new)
P_x_new_c2 = Ber(x_new[0], p2_new) * multivariate_normal.pdf(x_new[1:], mean=u2_new, cov=Sigma2_new)

gamma_1_new = P_x_new_c1 * pi1_new / (P_x_new_c1 * pi1_new + P_x_new_c2 * pi2_new)
gamma_2_new = P_x_new_c2 * pi2_new / (P_x_new_c1 * pi1_new + P_x_new_c2 * pi2_new)

############ 3. #######################################################################################################
'''
Perform a hard assignment of observations to clusters under ML assumption, identify 
the silhouette of the two clusters under a Manhattan distance.
'''
# -> dizer qual é o cluster de cada observação (está no cluster com maior probabilidade, temos de calcular a prob para os novos cclusters)
# -> calcular a silhueta para os dois clusters (media das silhuetas das observações do cluster)

##### Hard assignment
# x1
P_x1_c1new = Ber(x1[0], p1_new) * multivariate_normal.pdf(x1[1:], mean=u1_new, cov=Sigma1_new)
P_x1_c2new = Ber(x1[0], p2_new) * multivariate_normal.pdf(x1[1:], mean=u2_new, cov=Sigma2_new)

# x2
P_x2_c1new = Ber(x2[0], p1_new) * multivariate_normal.pdf(x2[1:], mean=u1_new, cov=Sigma1_new)
P_x2_c2new = Ber(x2[0], p2_new) * multivariate_normal.pdf(x2[1:], mean=u2_new, cov=Sigma2_new)

# x3
P_x3_c1new = Ber(x3[0], p1_new) * multivariate_normal.pdf(x3[1:], mean=u1_new, cov=Sigma1_new)
P_x3_c2new = Ber(x3[0], p2_new) * multivariate_normal.pdf(x3[1:], mean=u2_new, cov=Sigma2_new)

# x4
P_x4_c1new = Ber(x4[0], p1_new) * multivariate_normal.pdf(x4[1:], mean=u1_new, cov=Sigma1_new)
P_x4_c2new = Ber(x4[0], p2_new) * multivariate_normal.pdf(x4[1:], mean=u2_new, cov=Sigma2_new)

##### Silhueta
def Manhattan_dist(x1, x2):
    return np.sum(np.abs(x1 - x2))

def Silhueta_obs(a, b):
    if a < b:
        return 1 - a/b
    else:
        return b/a - 1

# Cluster 1 -> x2, x3
# Cluster 2 -> x1, x4
### Cluster 1
# x2
a_2 = Manhattan_dist(x2, x3)
b_2 = (Manhattan_dist(x2, x1) + Manhattan_dist(x2, x4))/2
silhueta_x2 = Silhueta_obs(a_2, b_2)

# x3
a_3 = Manhattan_dist(x3, x2)
b_3 = (Manhattan_dist(x3, x1) + Manhattan_dist(x3, x4))/2
silhueta_x3 = Silhueta_obs(a_3, b_3)
silhueta_C1new = (silhueta_x2 + silhueta_x3)/2

### Cluster 2
# x1
a_1 = Manhattan_dist(x1, x4)
b_1 = (Manhattan_dist(x1, x2) + Manhattan_dist(x1, x3))/2
silhueta_x1 = Silhueta_obs(a_1, b_1)

# x4
a_4 = Manhattan_dist(x4, x1)
b_4 = (Manhattan_dist(x4, x2) + Manhattan_dist(x4, x3))/2
silhueta_x4 = Silhueta_obs(a_4, b_4)

silhueta_C2new = (silhueta_x1 + silhueta_x4)/2

print("a_2:", a_2)
print("b_2:", b_2)
print("silhueta_x2:", silhueta_x2)
print("a_3:", a_3)
print("b_3:", b_3)
print("silhueta_x3:", silhueta_x3)
print('Silhueta Cluster 1:', silhueta_C1new)
print("a_1:", a_1)
print("b_1:", b_1)
print("silhueta_x1:", silhueta_x1)
print("a_4:", a_4)
print("b_4:", b_4)
print("silhueta_x4:", silhueta_x4)
print('Silhueta Cluster 2:', silhueta_C2new)

############ 4. #######################################################################################################
'''
Knowing the purity of the clustering solution is 0.75, identify the number of possible classes(ground truth).
'''