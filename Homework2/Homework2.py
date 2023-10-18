from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from scipy.stats import ttest_rel
import numpy as np

# Reading the file
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

X = df.drop('class', axis=1) #variables
y = df['class'] #target


##### EXERCÍCIO 1 ##########################################

### a) ###

# Fazer Stratified Cross Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Inicializar os classificadores
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

# Fazer a cross-validation
scores_knn = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
scores_nb = cross_val_score(nb, X, y, cv=cv, scoring='accuracy')

# Por os scores numa lista com 2 arrays (um para cada classificador)
scores_list = [scores_knn, scores_nb]

# Criar um gráfico com boxplots
fig, ax = plt.subplots()

# Plot boxplots
ax.boxplot(scores_list)
ax.set_xticklabels(['kNN (k=5)', 'Naive Bayes'])
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of kNN and Naive Bayes')
plt.savefig('boxplot.png')
plt.show()

### b) ###

# Fazemos um t-test para 'score knn > score nb'
t_stat, p_value = ttest_rel(scores_knn, scores_nb, alternative='greater')

print('t-statistic:', t_stat)
print('p-value:', p_value)

# P-value é menor que 0.05 pelo que a hipótese não é verdadeira

##### EXERCÍCIO 2 ##########################################

knn1 = KNeighborsClassifier(n_neighbors=1, weights='uniform', metric='euclidean')
knn5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')



# Matrizes de confusão para o knn1 e knn5
cm1_total = np.zeros((3, 3))
cm5_total = np.zeros((3, 3))

for train_k, test_k in cv.split(X, y):
    # Obter os X e y de teste e treino  
    X_train, X_test = X.iloc[train_k], X.iloc[test_k]
    y_train, y_test = y.iloc[train_k], y.iloc[test_k]

    # Obter os y previstos para o knn1 e knn5
    knn1.fit(X_train, y_train)
    knn5.fit(X_train, y_train)
    y1_pred = knn1.predict(X_test)
    y5_pred = knn5.predict(X_test)

    # Obter as matrizes de confusão
    cm1 = confusion_matrix(y_test, y1_pred)
    cm5 = confusion_matrix(y_test, y5_pred)

    # Adicionar às matrizes cumulativas
    cm1_total += cm1
    cm5_total += cm5



# Fazer a diferença entre as matrizes de confusão
dif = cm1_total - cm5_total

# Plot de cada matriz de confusão
plt.figure()
plt.imshow(cm1_total, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(cm1_total.shape[0]):
    for j in range(cm1_total.shape[1]):
        plt.text(j, i, str(int(cm1_total[i, j])), ha='center', va='center', color='black')
plt.xticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.yticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.title('Confusion matrix of kNN (k=1)')
plt.savefig('confusion_knn1.png')
plt.show()

plt.figure()
plt.imshow(cm5_total, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(cm5_total.shape[0]):
    for j in range(cm5_total.shape[1]):
        plt.text(j, i, str(int(cm5_total[i, j])), ha='center', va='center', color='black')
plt.xticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.yticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.title('Confusion matrix of kNN (k=5)')
plt.savefig('confusion_knn5.png')
plt.show()

# Plot da diferença
plt.figure()
plt.imshow(dif, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
for i in range(dif.shape[0]):
    for j in range(dif.shape[1]):
        plt.text(j, i, str(int(dif[i, j])), ha='center', va='center', color='black')
plt.xticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.yticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.title('Difference between confusion matrices of kNN (k=1) and kNN (k=5)')
plt.savefig('confusion.png')
plt.show()

##### EXERCÍCIO 3 ##########################################

# Para obter o número de instâncias de cada classe
sns.countplot(x='class', data=df)
plt.savefig('countplot.png')
plt.show()

# Para obter a matriz de correlação
df = df.drop('class', axis=1)
df.corr(method='pearson')
sns.heatmap(df.corr(method='pearson'), annot=True)
plt.savefig('heatmap.png')
plt.show()

# hitograma para cada variável
df.hist(figsize=(12, 8))
plt.savefig('histogram.png')
plt.show()



