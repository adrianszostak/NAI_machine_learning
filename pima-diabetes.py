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
pima-indians-diabetes.csv

There are 4,898 observations with 11 input variables and one output variable. The variable names are as follows:
Fixed acidity.
Volatile acidity.
Citric acid.
Residual sugar.
Chlorides.
Free sulfur dioxide.
Total sulfur dioxide.
Density.
pH.
Sulphates.
Alcohol.
Quality (score between 0 and 10).

Output:
The program uses the Support Vector Classificator to classify data into two sets:
1. Not at risk of developing diabetes
2. At risk of developing diabetes

"""

def dataset(data):
    data = pd.DataFrame(data)

input_file = 'pima-indians-diabetes.csv'
data = pd.read_csv(input_file)

# Cleaning dataset with kNN-Imputer
# Replace 0 -> Null
data[['Glucose','BloodPressure','SkinThickness','Insuline','BMI']] = data[
    ['Glucose','BloodPressure','SkinThickness','Insuline','BMI']
    ].replace(0,np.NaN)

X, y = data.loc[:,[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insuline', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]], data.loc[:,['Outcome']]


# Using k-NN imputer replace NaN -> kNNValue
knn = KNNImputer()
knn.fit(X)
new_X = knn.transform(X)
new_X = pd.DataFrame(new_X)


# Scaling
new_X = preprocessing.minmax_scale(new_X)
new_X = pd.DataFrame(new_X)

# PCA transformation - Merge all columns in new_X to 2 colums.
X_pca = PCA(n_components=2).fit_transform(new_X)

y = y.astype(int).values
y = y.ravel()

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X_pca[:num_training], y[:num_training]

# Test data
X_test, y_test = X_pca[num_training:], y[num_training:]


svc = svm.SVC(kernel='rbf', C=1, gamma=50).fit(X_train, y_train)

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
plt.contourf(xx, yy, Z, alpha=0.6)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, marker='x')
plt.xlim(X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1)
plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
plt.legend(*scatter.legend_elements(), title='The probability of diabetes')
plt.show()
