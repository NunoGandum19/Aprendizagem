import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import numpy as np

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



