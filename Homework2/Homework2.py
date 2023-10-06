from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from scipy.stats import ttest_rel

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

# Obter os y previstos para o knn1 e knn5
knn1.fit(X, y)
knn5.fit(X, y)
y1_pred = knn1.predict(X)
y5_pred = knn5.predict(X)

# Fazer as matrizes de confusão
cm1 = confusion_matrix(y, y1_pred)
cm5 = confusion_matrix(y, y5_pred)

# Fazer a diferença entre as matrizes de confusão
dif = cm1 - cm5

# Plot da diferença
plt.figure()
plt.imshow(dif, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.yticks([0, 1, 2], ['Hernia', 'Normal', 'Spondylolisthesis'])
plt.title('Difference between confusion matrices of kNN (k=1) and kNN (k=5)')
plt.show()

##### EXERCÍCIO 3 ##########################################