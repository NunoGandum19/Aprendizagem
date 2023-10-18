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
    # não sei se é suposto por um max_iter, mas se não puser dá warning
    
    # treinar o modelo
    mlp.fit(X_train, y_train)
    
    # obter y previsto
    y_pred = mlp.predict(X_test)

    # obter os resíduos
    residuos = y_test - y_pred

    # Calcular o módulo dos resíduos
    abs_residuos = abs(residuos)
    
    resid.append(abs_residuos)


# obtemos a médio dos resíduos
average_resid = sum(resid) / len(resid)
print("Resíduos:", average_resid)

# fazer o histograma
plt.hist(average_resid, bins=20, edgecolor='k')
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
    rmse_values.append((max_iter, valor_rmse))


# juntar à lista o RMSE com early stopping
rmse_es = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_values.append(('early stopping', rmse_es))


for max_iter, rmse in rmse_values:
        print("RMSE for {} iterations: {:.2f}".format(max_iter, rmse))




