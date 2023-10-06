from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
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


# Fazer Stratified Cross Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# não sei se é para por shuffle True ou False

# Inicializar o classificador
#clf = RandomForestClassifier()
clf = DecisionTreeClassifier()

# Fazer a cross-validation
scores = cross_val_score(clf, X, y, cv=cv)

print(scores)

# Inicializar os classificadores
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

# Fazer a cross-validation
scores_knn = cross_val_score(knn, X, y, cv=cv)
scores_nb = cross_val_score(nb, X, y, cv=cv)

# Por os scores numa lista com 2 arrays (um para cada classificador)
scores_list = [scores_knn, scores_nb]

##### EXERCÍCIO 1 ##########################################

### a) ###

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

##### EXERCÍCIO 2 ##########################################



##### EXERCÍCIO 3 ##########################################