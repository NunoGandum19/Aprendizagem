import numpy as np


X = np.array([
    [0.7, -0.3],
    [0.4, 0.5],
    [-0.2, 0.8],
    [-0.4, 0.3]
])

# Valores para a tranformação phi
c1 = np.array([0, 0])
c2 = np.array([1, -1])
c3 = np.array([-1, 1])

# aplicar a função phi (obtemos um array com phi1, phi2, phi3)
# o axis = 1 indica que queremos calcular a distância para cada fila individualmente
phi_X = np.exp(-np.array([
    np.linalg.norm(X - c1, axis=1) ** 2 / 2,   
    np.linalg.norm(X - c2, axis=1) ** 2 / 2,
    np.linalg.norm(X - c3, axis=1) ** 2 / 2
]).T)

# Targets
y = np.array([0.8, 0.6, 0.3, 0.3])

lambda_ = 0.1

# Matriz identidade 3 por 3
I = np.eye(3)

# Fórmula para calcular w
w = np.linalg.inv(phi_X.T @ phi_X + lambda_ * I) @ phi_X.T @ y
# nota: o @ é o operador de multiplicação de matrizes

print('Phi_X = ',phi_X)
print('w =', w)