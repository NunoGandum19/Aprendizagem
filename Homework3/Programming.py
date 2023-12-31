import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import numpy as np
import matplotlib.pyplot as plt

# Ler o ficheiro
data = pd.read_csv("winequality-red.csv", delimiter=';')

X = data.drop('quality', axis=1)
y = data['quality']

# Fazer um train/test split (80 treino e 20 teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### 1 ##########################################

#lista residuos
resid = []

# iteração para cada random state
for i in range(1, 11):
    # Initialize the MLP regressor with random state
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10),
                   activation='relu',
                   early_stopping=True,
                   validation_fraction=0.2,
                   random_state=i)
    
   
    # treinar o modelo
    mlp.fit(X_train, y_train)
    
    # obter y previsto
    y_pred = mlp.predict(X_test)

    # obter os resíduos
    residuos = y_test - y_pred

    # Calcular o módulo dos resíduos
    abs_residuos = abs(residuos)
    
    resid.append(abs_residuos)

# por os resíduos numa lista
resid = np.concatenate(resid)

# fazer o histograma
plt.figure(figsize=(12, 8))
plt.hist(resid, bins='auto', edgecolor='k')
plt.title('Distribution of Absolute Residuals')
plt.xlabel('Absolute Residuals')
plt.ylabel('Frequency')
plt.savefig('residuos.png')
plt.show()

##### 2 #############################

# lista para guardar os MAE
lista_mae_original = []
lista_mae_rounded = []
lista_mae_bounded = []
lista_mae_bounded_rounded = []

# iteração para cada random state
for i in range(1, 11):
    # Initialize the MLP regressor with random state
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10),
                   activation='relu',
                   early_stopping=True,
                   validation_fraction=0.2,
                   random_state=i)
    
   
    # treinar o modelo
    mlp.fit(X_train, y_train)
    # obter y previsto
    y_pred = mlp.predict(X_test)

    # obter máximo e mínimo do target
    min_target = np.min(y_train)
    max_target = np.max(y_train)

    # Obter os valores previstos 
    rounded_predictions = np.round(y_pred)  # arredondar os valores previstos
    bounded_predictions = np.clip(y_pred, min_target, max_target)  # limitar os valores aos limites definidos
    bounded_rounded_predictions = np.clip(rounded_predictions, min_target, max_target)

    # obter o MAE 
    mae_original = mean_absolute_error(y_test, y_pred)
    mae_rounded = mean_absolute_error(y_test, rounded_predictions)
    mae_bounded = mean_absolute_error(y_test, bounded_predictions)
    mae_bounded_rounded = mean_absolute_error(y_test, bounded_rounded_predictions)

    # adicionar às listas
    lista_mae_original.append(mae_original)
    lista_mae_rounded.append(mae_rounded)
    lista_mae_bounded.append(mae_bounded)
    lista_mae_bounded_rounded.append(mae_bounded_rounded)
    

valor_mae_original = sum(lista_mae_original) / len(lista_mae_original)
valor_mae_rounded = sum(lista_mae_rounded) / len(lista_mae_rounded)
valor_mae_bounded = sum(lista_mae_bounded) / len(lista_mae_bounded)
valor_mae_bounded_rounded = sum(lista_mae_bounded_rounded) / len(lista_mae_bounded_rounded)

print("MAE with Original Predictions:", valor_mae_original)
print("MAE with Rounded Predictions:", valor_mae_rounded)
print("MAE with Bounded Predictions:", valor_mae_bounded)
print("MAE with Bounded and Rounded Predictions:", valor_mae_bounded_rounded)

# vê-se que o MAE com os valores arredondados é menor, logo é melhor

##### 3 #############################

# nº de iterações
iterations = [20, 50, 100, 200]

# lista para guardar os RMSE de cada iteração
rmse_values = []

# iteração pela lista de cada nº de iterações máximas
for max_iter in iterations:
    rmse_lista = []
    for i in range(1, 11):
        # Inicializar o MLP regressor com os parâmetros pedidos
        mlp2 = MLPRegressor(hidden_layer_sizes=(10, 10),
                        activation='relu',
                        max_iter=max_iter,
                        random_state=i)

        # treinar o modelo
        mlp2.fit(X_train, y_train)

        # obter y previsto
        y_pred2 = mlp2.predict(X_test)

        # calcular rmse
        rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
        rmse_lista.append(rmse)
    valor_rmse = sum(rmse_lista) / len(rmse_lista)
    rmse_values.append(valor_rmse)

rmse_early_stopping = []
# iteração para cada random state
for i in range(1, 11):
    # Initialize the MLP regressor with random state
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10),
                   activation='relu',
                   early_stopping=True,
                   validation_fraction=0.2,
                   random_state=i)
    
   
    # treinar o modelo
    mlp.fit(X_train, y_train)

    # obter y previsto
    y_pred = mlp.predict(X_test)

    rmse_es = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_early_stopping.append(rmse_es)

valor_rmse_es = sum(rmse_early_stopping) / len(rmse_early_stopping)


for rmse in rmse_values:
        print("RMSE for {} iterations: {}".format(iterations[rmse_values.index(rmse)], rmse))
print("RMSE with Early Stopping:", valor_rmse_es)

### 4 ##########################################


# Array de valores constantes de y
y = [valor_rmse_es for i in range(len(iterations))]

# nº iterações vs early stopping
plt.figure(figsize=(10, 6))
plt.plot(iterations, rmse_values, marker='o', label='Maximum Iterations')
plt.plot(iterations, y, linestyle='--', label='Early Stopping')
plt.xlabel('Maximum Iterations')
plt.ylabel('RMSE')
plt.title('RMSE vs. Maximum Iterations')
plt.xticks(iterations)
plt.legend()
plt.savefig('iterations.png')
plt.show()







