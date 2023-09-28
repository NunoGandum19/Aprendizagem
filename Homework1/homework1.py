import pandas as pd
from scipy.io.arff import loadarff
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


# Reading the file
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

X = df.drop('class', axis=1) #variables
y = df['class'] #target


fimportance = f_classif(X, y)

fstat = fimportance[0]  #f-statistic
pval = fimportance[1]   #p-value

print('features', X.columns.values)
print('scores', fimportance[0])
print('pvalues', fimportance[1])

# Create a dataframe with the F-statistic and p-value for each variable
fstat_df = pd.DataFrame({'F-statistic': fstat, 'p-value': pval})
fstat_df.index = X.columns
fstat_df = fstat_df.sort_values(by=['F-statistic'], ascending=False)
print(fstat_df)

# Plot the class-conditional probability density functions of the variables with the highest and lowest F-statistic
plt.figure(figsize=(10, 6))
sns.distplot(df[df['class'] == 'Hernia']['pelvic_radius'], label='Hernia', hist=False)
sns.distplot(df[df['class'] == 'Normal']['pelvic_radius'], label='Normal', hist=False)
sns.distplot(df[df['class'] == 'Spondylolisthesis']['pelvic_radius'], label='Spondylolisthesis', hist=False)
plt.xlabel('pelvic_radius')
plt.ylabel('class-conditional probability density')
plt.title('Class-conditional probability density of pelvic_radius')
plt.legend()
plt.savefig('DensProbPelvicRadius.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.distplot(df[df['class'] == 'Hernia']['degree_spondylolisthesis'], label='Hernia', hist=False)
sns.distplot(df[df['class'] == 'Normal']['degree_spondylolisthesis'], label='Normal', hist=False)
sns.distplot(df[df['class'] == 'Spondylolisthesis']['degree_spondylolisthesis'], label='Spondylolisthesis', hist=False)
plt.xlabel('degree_spondylolisthesis')
plt.ylabel('class-conditional probability density')
plt.title('Class-conditional probability density of degree_spondylolisthesis')
plt.legend()
plt.savefig('DensProbSpondy.png')
plt.show() 


#EXERCÍCIO 2 - QUEREMOS OBTER UM ÚNICO PLOT QUE NOS DÁ A ACCURACY TANTO DO 
# TRAIN COMO DO TEST EM FUNÇÃO DA PROFUNDIDADE DA ÁRVORE

random_seed = 0

train_accuracies = []
test_accuracies = []

depth_limits = [1, 2, 3, 4, 5, 6, 8, 10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed, stratify=y)


# Queremos fazer a accuracy para árvores com profundidade limite de 1 a 10 
for depth_limit in depth_limits:
    clf = DecisionTreeClassifier(max_depth=depth_limit, random_state=random_seed)
    clf.fit(X_train, y_train)
    
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    train_accuracy = metrics.accuracy_score(y_pred_train, y_train)
    train_accuracies.append(train_accuracy)
    
    test_accuracy = metrics.accuracy_score(y_pred_test, y_test)
    test_accuracies.append(test_accuracy)


plt.figure(figsize=(10, 6))
plt.plot(depth_limits, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(depth_limits, test_accuracies, marker='o', label='Testing Accuracy')
plt.xlabel('Max Depth Limit')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracies vs. Max Depth Limit')
plt.xticks(depth_limits)
plt.legend()
plt.show()


############# 4. a) #######################################
"""
- Treinar uma decision tree com todos os dados - random_state=0.
- Fazer o plot de uma decision tree com mínimo de 20 indivíduos por folha, para evitar overfitting.
"""
clf = DecisionTreeClassifier(min_samples_leaf=20, random_state=random_seed)
clf.fit(X, y) #treinamos a árvore com todos os dados

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=df.columns, class_names=np.unique(y).astype(str), fontsize=10)
plt.title("Decision Tree with Minimum Leaf Size of 20")
plt.show()



