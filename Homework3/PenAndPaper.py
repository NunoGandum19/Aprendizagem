import numpy as np

############## 1. ################
### a) ###

# Define the activation function
def phi(x, c):
    return np.exp(-0.5 * (np.linalg.norm(x - c))**2)

# Define the regularization parameter
lambda_ = 0.1

# Define the X matrix
X = np.array([[0.7,-0.3],[0.4,0.5],[-0.2,0.8],[-0.4,0.3]]) 

# Define the z vector
z = np.array([[0.8],[0.6],[0.3], [0.3]])

# Define the c vectors
c1 = np.array([0,0])
c2 = np.array([1,-1])
c3 = np.array([-1,1])

# Define de Phi matrix

Phi = np.array([[1, phi(X[0], c1), phi(X[0], c2), phi(X[0], c3)],
                [1, phi(X[1], c1), phi(X[1], c2), phi(X[1], c3)],
                [1, phi(X[2], c1), phi(X[2], c2), phi(X[2], c3)],
                [1, phi(X[3], c1), phi(X[3], c2), phi(X[3], c3)]])

# Obtain the weights
w = np.dot(np.dot(np.linalg.inv(np.dot(Phi.T, Phi) + lambda_*np.identity(4)), Phi.T), z)
print(f"The weights are {w}.")
### b) ###
def RMSE(z, z_esperado):
    return np.sqrt(np.sum((z - z_esperado)**2)/len(z))

z_esperado = np.dot(Phi, w) 
print(f"The z_esperado is {z_esperado}.")

res = RMSE(z, z_esperado)
print(f"The RMSE is {res}.")
############## 2. ################
# Define the activation function
def f(x):
    return np.tanh(0.5*x-2)

def f_prime(x):
    return 0.5/((np.cosh(0.5*x-2))**2)     
# Define the learning rate
eta = 0.1
# Define the weights matrix
W1 = np.array([[1,1,1,1], [1,1,2,1], [1,1,1,1]])
W2 = np.array([[1,4,1], [1,1,1]])
W3 = np.array([[1,1], [3,1], [1,1]])

# Define the biases matrix
b1 = np.array([[1],[1],[1]])
b2 = np.array([[1],[1]])
b3 = np.array([[1],[1],[1]])

# Define the inputs
x1 = np.array([[1],[1],[1],[1]])
x2 = np.array([[1],[0],[0],[-1]])

# Define the targets
t1 = np.array([[0],[1],[0]]) #y_1 = B
t2 = np.array([[1],[0],[0]]) #y_2 = A

### Propagation
# Para x1
z1_1 = np.dot(W1, x1) + b1
z2_1 = np.dot(W2, f(z1_1)) + b2
z3_1 = np.dot(W3, f(z2_1)) + b3
z_esperado_1 = f(z3_1)
# Para x2
z1_2 = np.dot(W1, x2) + b1
z2_2 = np.dot(W2, f(z1_2)) + b2
z3_2 = np.dot(W3, f(z2_2)) + b3
z_esperado_2 = f(z3_2)

### Backpropagation
## Update W3
# Para x1 
delta3_1 = (z_esperado_1 - t1) * f_prime(z3_1)
dE_dW3_1 = np.dot(delta3_1, f(z2_1).T)
# Para x2
delta3_2 = (z_esperado_2 - t2) * f_prime(z3_2)
dE_dW3_2 = np.dot(delta3_2, f(z2_2).T)

W3_new = W3 - eta * (dE_dW3_1 + dE_dW3_2)

## Update W2
# Para x1
delta2_1 = np.dot(W3.T, delta3_1) * f_prime(z2_1)
dE_dW2_1 = np.dot(delta2_1, f(z1_1).T)
# Para x2
delta2_2 = np.dot(W3.T, delta3_2) * f_prime(z2_2)
dE_dW2_2 = np.dot(delta2_2, f(z1_2).T)

W2_new = W2 - eta * (dE_dW2_1 + dE_dW2_2)

## Update W1
# Para x1
delta1_1 = np.dot(W2.T, delta2_1) * f_prime(z1_1)
dE_dW1_1 = np.dot(delta1_1, x1.T)
# Para x2
delta1_2 = np.dot(W2.T, delta2_2) * f_prime(z1_2)
dE_dW1_2 = np.dot(delta1_2, x2.T)

W1_new = W1 - eta * (dE_dW1_1 + dE_dW1_2)

## Update b3
# Para x1
dE_db3_1 = delta3_1
# Para x2
dE_db3_2 = delta3_2

b3_new = b3 - eta * (dE_db3_1 + dE_db3_2)

## Update b2
# Para x1
dE_db2_1 = delta2_1
# Para x2
dE_db2_2 = delta2_2

b2_new = b2 - eta * (dE_db2_1 + dE_db2_2)

## Update b1
# Para x1
dE_db1_1 = delta1_1
# Para x2
dE_db1_2 = delta1_2

b1_new = b1 - eta * (dE_db1_1 + dE_db1_2)


print(f"W1_new = {W1_new}")
print(f"W2_new = {W2_new}")
print(f"W3_new = {W3_new}")
print(f"b1_new = {b1_new}")
print(f"b2_new = {b2_new}")
print(f"b3_new = {b3_new}")
