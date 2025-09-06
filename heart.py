# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:45:53 2019

@author: EGC
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

Dataset = pd.read_csv("Stroke.csv")


print (Dataset.head())
print(Dataset.describe())

#Data Preprocessing
#Missing Data Removal

#check missing values
print ('Dataset contain null:\t',Dataset.isnull().values.any())
print ('Describe null:\n',Dataset.isnull().sum())
print ('No of  null:\t',Dataset.isnull().sum().sum())

#Replace values
Dataset['smoking_status'].fillna('Sometimes', inplace=True)
Dataset.fillna(Dataset.mean(), inplace=True)
columns = ['id','ever_married','work_type','Residence_type']
Dataset.drop(columns, inplace=True, axis=1)

x = Dataset.iloc[:, :-1].values
x1=pd.DataFrame(x)
y = Dataset.iloc[:,7].values
y1=pd.DataFrame(y)

#check missing values
print ('Dataset contain null:\t',Dataset.isnull().values.any())
print ('Describe null:\n',Dataset.isnull().sum())
print ('No of  null:\t',Dataset.isnull().sum().sum())

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])

x[:,6]=labelencoder_x.fit_transform(x[:,6])
Y=pd.DataFrame(x[:,1])
onehotencoder=OneHotEncoder(categorical_features=[0])

onehotencoder=OneHotEncoder(categorical_features=[6])
x=onehotencoder.fit_transform(x).toarray()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

# Splitting the dataset into the Training set and Test set for decision
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
model = LogisticRegression()
#traing Dataset
model.fit(X1_train, y1_train)
#testing dataset
predicted_classes = model.predict(X1_test)
accuracy = accuracy_score(y1_test,predicted_classes)
parameters = model.coef_
from sklearn.metrics import classification_report
print (classification_report(y1_test,predicted_classes)) 


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



cm = metrics.confusion_matrix(y1_test, predicted_classes)
print(cm)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size = 10);

#accuracy
print ('ACCURACY:',accuracy)
print ('CO_EFFICIENT',parameters)

#Logistic Regression
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score  
from sklearn.cross_validation import train_test_split
from pandas.tools.plotting import parallel_coordinates
#Advanced optimization
from scipy import optimize as op

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#Regularized cost function
def regCostFunction(theta, X, y, _lambda = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    reg = (_lambda/(2 * m)) * np.sum(theta**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

#Regularized gradient function
def regGradient(theta, X, y, _lambda = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    reg = _lambda * theta /m

    return ((1 / m) * X.T.dot(h - y)) + reg

#Optimal theta 
def logisticRegression(X, y, theta):
    result = op.minimize(fun = regCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = regGradient)
    
    return result.x


all_theta = np.zeros((k, n + 1))
#Predictions
P = sigmoid(X1_test.dot(all_theta.T)) #probability for each flower
p = [y[np.argmax(P[i, :])] for i in range(X1_test.shape[0])]

print("Test Accuracy ", accuracy_score(y1_test, p) * 100 , '%')
#Confusion Matrix
cfm = confusion_matrix(y_test, p, labels = Species)

sb.heatmap(cfm, annot = True, xticklabels = Species, yticklabels = Species);

   