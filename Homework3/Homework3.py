import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
import numpy as np
import matplotlib.pyplot as plt

# Ler o ficheiro
data = pd.read_csv("winequality-red.csv", delimiter=';')

X = data.drop('quality', axis=1)
y = data['quality']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fazer um train/test split (80 treino e 20 teste)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# lista para guardar os coeficientes de correlação (r^2) de cada iteração
r2_scores = []

# iteração para cada random state
for i in range(1, 11):
    # Initialize the MLP regressor with random state
    mlp = MLPRegressor(random_state=i, max_iter=1000)
    # não sei se é suposto por um max_iter, mas se não puser dá warning
    
    # treinar o modelo
    mlp.fit(X_train, y_train)
    
    # obter y previsto
    y_pred = mlp.predict(X_test)
    
    # calcular r^2
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# obtemos a médio dos r^2
average_r2 = sum(r2_scores) / len(r2_scores)
print("Average R2 Score:", average_r2)

##### 1 #############################

# Inicializar o MLP regressor com os parâmetros pedidos
mlp = MLPRegressor(hidden_layer_sizes=(10, 10),
                   activation='relu',
                   early_stopping=True,
                   validation_fraction=0.2,
                   random_state=0)

# treinar o modelo
mlp.fit(X_train, y_train)

# obter y previsto
y_pred = mlp.predict(X_test)

# obter os resíduos
residuos = y_test - y_pred

# Calcular o módulo dos resíduos
abs_residuos = abs(residuos)

# fazer o histograma
plt.hist(abs_residuos, bins=20, edgecolor='k')
plt.title('Distribution of Absolute Residuals')
plt.xlabel('Absolute Residuals')
plt.ylabel('Frequency')
plt.show()

##### 2 #############################

# aproximamos os valores previstos para os valores inteiros mais próximos
rounded_predictions = np.round(y_pred)

# obter máximo e mínimo do target
min_target = np.min(y_train)
max_target = np.max(y_train)

# garantir que os valores previstos estão dentro do intervalo do target
bounded_predictions = np.clip(rounded_predictions, min_target, max_target)

# obter o MAE com os valores originais
mae_original = mean_absolute_error(y_test, y_pred)

# obtet o MAE com os valores arredondados
mae_rounded = mean_absolute_error(y_test, bounded_predictions)

print("MAE with Original Predictions:", mae_original)
print("MAE with Rounded and Bounded Predictions:", mae_rounded)

# vê-se que o MAE com os valores arredondados é menor, logo é melhor

##### 3 #############################

# nº de iterações
iterations = [20, 50, 100, 200]

# lista para guardar os RMSE de cada iteração
rmse_values = []

# iteração pela lista de cada nº de iterações máximas
for max_iter in iterations:
    # Inicializar o MLP regressor com os parâmetros pedidos
    mlp = MLPRegressor(hidden_layer_sizes=(10, 10),
                       activation='relu',
                       max_iter=max_iter,
                       random_state=0)

    # treinar o modelo
    mlp.fit(X_train, y_train)

    # obter y previsto
    y_pred = mlp.predict(X_test)

    # calcular rmse
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append((max_iter, rmse))

for max_iter, rmse in rmse_values:
    print("RMSE for {} iterations: {:.2f}".format(max_iter, rmse))