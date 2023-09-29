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

#################### PEN AND PAPER QUESTIONS #######################
##############  5.  ###################################
tree_df = pd.DataFrame({'y1':[0.24, 0.06, 0.04, 0.36, 0.32, 0.68, 0.9, 0.76, 0.46, 0.62, 0.44, 0.52], 
                        'y_out' : ['A', 'B','B', 'C', 'C', 'A', 'A', 'A', 'B', 'B', 'C', 'C']})
tree_df_A = tree_df.drop(tree_df[tree_df['y_out'] != 'A'].index)
tree_df_B = tree_df.drop(tree_df[tree_df['y_out'] != 'B'].index)
tree_df_C = tree_df.drop(tree_df[tree_df['y_out'] != 'C'].index)

plt.hist(tree_df_A['y1'], bins=5, range=(0,1), density = True, label = 'A', alpha= 0.4, edgecolor = 'blue')
plt.hist(tree_df_B['y1'], bins=5, range=(0,1), density = True, label = 'B', alpha= 0.4, color = 'orange', edgecolor = 'orange')
plt.hist(tree_df_C['y1'], bins=5, range=(0,1), density = True, label = 'C', alpha= 0.4, color = 'green', edgecolor = 'green')
plt.xlabel('y1')
plt.ylabel('Density Relative Frequency')
plt.legend()
plt.show()

##############  Extra 5 ###################################
"""Para encontrar as root split entre as diferentes classes, é necessário escolher os bins onde cada classe é maior
-> para tentar obter um melhor valor de root split, podia-se aplicar KDE a cada classe e com as KDE estimar os valores
"""
kde_A = stats.gaussian_kde(tree_df_A['y1'])
kde_B = stats.gaussian_kde(tree_df_B['y1'])
kde_C = stats.gaussian_kde(tree_df_C['y1'])
x = np.linspace(0, 1, 100)
kde_values_A = kde_A(x)
kde_values_B = kde_B(x)
kde_values_C = kde_C(x)
plt.plot(x, kde_values_A, 'r', label='Kernel Density Estimation', color = 'blue')
plt.plot(x, kde_values_B, 'b', label='Kernel Density Estimation', color = 'orange')
plt.plot(x, kde_values_C, 'g', label='Kernel Density Estimation', color = 'green')
plt.xlabel('y1')
plt.ylabel('Probability Density')
plt.legend()
#plt.savefig('KDE.png')
plt.show()


#################### PROGRAMMING QUESTIONS #########################
##############  1.  ###################################
# Read in the data
data = loadarff('column_diagnosis.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')

X = df.drop('class', axis=1)       #variables
y = df['class']                    #target

# Calculate the F-statistic and p-value for each variable
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
print("The variables with the highest and lowest F-statistic are degree_spondylolisthesis and pelvic_radius, respectively.")

# Plot the class-conditional probability density functions of the variables with the highest and lowest F-statistic
plt.figure(figsize=(10, 6))
sns.kdeplot(df['pelvic_radius'][df['class'] == 'Hernia'], label='Hernia')
sns.kdeplot(df['pelvic_radius'][df['class'] == 'Spondylolisthesis'], fill=False, label='Spondylolisthesis')
sns.kdeplot(df['pelvic_radius'][df['class'] == 'Normal'], fill=False, label='Normal')
plt.xlabel('pelvic_radius')
plt.ylabel('class-conditional probability density')
plt.title('Class-conditional probability density of pelvic_radius')
plt.legend()
#plt.savefig('DensProbPelvicRadius.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(df['degree_spondylolisthesis'][df['class'] == 'Hernia'], fill = False, label= 'Hernia')
sns.kdeplot(df['degree_spondylolisthesis'][df['class'] == 'Spondylolisthesis'], fill = False, label='Spondylolisthesis')
sns.kdeplot(df['degree_spondylolisthesis'][df['class'] == 'Normal'], fill = False, label='Normal')
plt.xlabel('degree_spondylolisthesis')
plt.ylabel('class-conditional probability density')
plt.title('Class-conditional probability density of degree_spondylolisthesis')
plt.legend()
#plt.savefig('DensProbSpondy.png')
plt.show() 

################## EXERCÍCIO 2 ##################################### 
""" 
Queremos obter um único plot que nos dá a Accuracy tanto do Train como do Test em função da profundidade da árvore
"""

random_seed = 0  # selecionar a seed a usar
depth_limits = [1, 2, 3, 4, 5, 6, 8, 10]  # lista com os valores de profundidade limite que queremos testar

train_accuracies = []                          # lista para guardar as accuracies do train
test_accuracies = []                           # lista para guardar as accuracies do test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed, stratify=y)

# calcular as accuracies para cada profundidade limite 
for depth_limit in depth_limits:
    clf = DecisionTreeClassifier(max_depth=depth_limit, random_state=random_seed)  # criar a árvore com a profundidade limite
    clf.fit(X_train, y_train)                                                      # treinar a árvore com os dados de treino
    
    y_pred_test = clf.predict(X_test)    # prever os dados de teste
    y_pred_train = clf.predict(X_train)  # prever os dados de treino

    train_accuracy = metrics.accuracy_score(y_pred_train, y_train)   # calcular a accuracy do train
    train_accuracies.append(train_accuracy)
    
    test_accuracy = metrics.accuracy_score(y_pred_test, y_test)      # calcular a accuracy do test
    test_accuracies.append(test_accuracy)

# plot das accuracies em função da profundidade limite 
plt.figure(figsize=(10, 6))
plt.plot(depth_limits, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(depth_limits, test_accuracies, marker='o', label='Testing Accuracy')
plt.xlabel('Max Depth Limit')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracies vs. Max Depth Limit')
plt.xticks(depth_limits)
plt.legend()
plt.show()

############# EXERCÍCIO 4. ##########################################
############# 4. a) #######################################
"""
- Treinar uma decision tree com todos os dados - random_state=0.
- Fazer o plot de uma decision tree com mínimo de 20 indivíduos por folha, para evitar overfitting.
"""
clf = DecisionTreeClassifier(min_samples_leaf = 20, random_state = random_seed)     #criar a árvore com o mínimo de 20 indivíduos por folha
clf.fit(X, y)                                                                       #treinar a árvore com todos os dados

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=df.columns, class_names=np.unique(y).astype(str), fontsize=10)
plt.title("Decision Tree with Minimum Leaf Size of 20")
plt.show()

############# 4. b) #######################################
"""
- Characterize a hernia condition by identifying the hernia-conditional associations
"""



