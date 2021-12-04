from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import sklearn.metrics as sm

"""
Authors:
Hubert Korzeniewski s19469
Adrian Szostak s19777

Input:
divorces.csv zawierający zbiór danych dotyczących 150 par z odpowiadającymi im zmiennymi skali predyktorów rozwodu (DPS) na podstawie terapii par Gottmana.

Pary pochodzą z różnych regionów Turcji, w których dane zostały pozyskane z rozmów twarzą w twarz z parami, które były już rozwiedzione lub prowadziły szczęśliwe małżeństwo.
Wszystkie odpowiedzi zebrano w 5-stopniowej skali (0=Nigdy, 1=Rzadko, 2=Średnio, 3=Często, 4=Zawsze).
Pytań jest 54 i znajdują się w pliku marriage-questions.txt

Dane pochodzą ze strony: https://www.kaggle.com/andrewmvd/divorce-prediction?select=divorce_data.csv

Output:
0 - przewidywany rozwód
1 - przewidywane małżeństwo


"""

def dataset(data):
    data = pd.DataFrame(data)
'''
Data loading
'''
input_file = 'divorces.csv'
data = pd.read_csv(input_file)

'''
Do not remove 0, because 0 is also our data!
'''

X, y = data.loc[:,[
    'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34','Q35','Q36','Q37','Q38','Q39','Q40','Q41','Q42','Q43','Q44','Q45','Q46','Q47','Q48','Q49','Q50','Q51','Q52','Q53','Q54'
    ]], data.loc[:,['Divorce']]


'''
Scaling
'''
X = preprocessing.minmax_scale(X)
X = pd.DataFrame(X)

'''
PCA transformation - Merge all columns in new_X to 2 colums.
'''
X_pca = PCA(n_components=2).fit_transform(X)

y = y.astype(int).values
y = y.ravel()

'''
Division into test and training data
'''
num_training = int(0.7 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X_pca[:num_training], y[:num_training]

# Test data
X_test, y_test = X_pca[num_training:], y[num_training:]

svc = svm.SVC(kernel='rbf', C=1, gamma=50).fit(X_pca, y)

# create a mesh to plot in
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

h = sqrt(((x_max / x_min)/100)**2)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))


# Predicted shape
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

y_test_pred = svc.predict(X_test)

# Drawing the plot
plt.contourf(xx, yy, Z, alpha=0.2) #rysuje odpowiednio linie konturowe i wypełnione kontury
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c = y, marker='x')
plt.xlim(X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1)
plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
plt.figure()

plt.contourf(xx, yy, Z, alpha=0.2) #rysuje odpowiednio linie konturowe i wypełnione kontury
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, marker='x')
plt.xlim(X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1)
plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
plt.legend(*scatter.legend_elements(), title='Probability of divorce')

plt.show()
