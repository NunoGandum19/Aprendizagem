import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics, cluster
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt


# Ler o ficheiro
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

X = df.drop('class', axis=1) #variables
y = df['class'] #target


# normalização dos dados
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(X)
# IMPORTANTE AQUI NÃO SEI SE É PARA USAR O FIT_TRANSFORM OU O TRANSFORM

### 1 ##########################################
"""
Using sklearn, apply k-means clustering fully unsupervisedly on the normalized data with
k in {2,3,4,5} (random=0 and remaining parameters default). Assess the silhouette and purity 
of the produced solutions.
"""

# Função para obter a purity
def purity_score(y_true, y_pred):
    # obter confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

k_values = [2, 3, 4, 5]
silhouette_list = []
purity_list = []

for k in k_values:
    # inicializar k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    
    # aprender o modelo com X normalizado
    kmeans.fit(normalized_data)
    
    # obter o y previsto
    labels = kmeans.labels_
    
    # Calculate the silhouette score
    silhouette = silhouette_score(normalized_data, labels)
    silhouette_list.append(silhouette)

    purity = purity_score(y, labels)
    purity_list.append(purity)

    # guardar para o exercício 3
    if k==3: labels_3 = labels

for i in range(len(k_values)):
    print(f'k={k_values[i]}: silhouette={silhouette_list[i]}, purity={purity_list[i]}')

#gráfico silhoutte vs k
plt.plot(k_values, silhouette_list)
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.show()

#gráfico purity vs k
plt.plot(k_values, purity_list)
plt.xlabel('k')
plt.ylabel('Purity score')
plt.show()

### 2 ##########################################
"""
Consider the application of PCA after the data normalization:
i) Identify the variability explained by the top two principal components.
ii) For each one of these two components, sort the input variables by relevance by
inspecting the absolute weights of the linear projection.
"""

pca = PCA(n_components=2)
principal_components = pca.fit_transform(normalized_data)

explained_variance = pca.explained_variance_ratio_
print(f'Variability explained by the first principal component: {explained_variance[0]}')
print(f'Variability explained by the second principal component: {explained_variance[1]}')

xvector = pca.components_[0] * max(principal_components[:, 0])
yvector = pca.components_[1] * max(principal_components[:, 1])

columns = X.columns

# sort das variáveis por ordem de relevância, usando o valor absoluto dos pesos
sorted_features_1 = sorted(zip(columns, xvector), key=lambda x: abs(x[1]), reverse=True)
print("Features sorted by relevance for the first principal component:")
for feature, weight in sorted_features_1:
    print(f'{feature}: {abs(weight)}')


sorted_features_2 = sorted(zip(columns, yvector), key=lambda x: abs(x[1]), reverse=True)
print("\nFeatures sorted by relevance for the second principal component:")
for feature, weight in sorted_features_2:
    print(f'{feature}: {abs(weight)}')

### 3 ##########################################
"""
Visualize side-by-side the data using: i) the ground diagnoses and ii) the previously learned
k = 3 clustering solution. To this end, projected the normalized data onto a 2-dimensional data
space using PCA and the color observations using the reference and clustering annotations.
"""
# Para o k-means foi guardado no ciclo for do exercício anterior as labels_3
# Porque o professor pede as previously learned 
# Não sei se é suposto fazer o k-means outra vez ou não

# Para não dar erro por as classes serem string
le = LabelEncoder()
encoded_classes = le.fit_transform(df['class'])

class_names = le.classes_
labels_list = class_names.tolist()

# fazer os plots
plt.figure(figsize=(12, 5))

# para o ground truth
plt.subplot(1, 2, 1)
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=encoded_classes, cmap='viridis', alpha=0.7)
plt.title("Ground Truth Diagnoses")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(handles=scatter.legend_elements()[0], labels=labels_list)

# para o k-means clustering com k=3
plt.subplot(1, 2, 2)
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels_3, cmap='viridis', alpha=0.7)
plt.title("k=3 Clustering Solution")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

### 4 ##########################################
"""
Considering the results from question 1 and 3, identify two ways on how clustering can 
be used to characterize the population of ill and healthy individuals.
"""

