import numpy as np
import pandas as pd

# Load the dataset
heart_data = pd.read_csv("Stroke.csv")

 # print first 5 rows of the data set
print(heart_data.head())

#print last 5 rows of dataset
heart_data.tail()

# number of rows and columns
heart_data.shape

# get some info about data
heart_data.info()

# Data Preprocessing
# Check for missing values
heart_data.isnull().sum()

# Replace non-numeric values in the 'smoking_status' column with 'Sometimes'
heart_data['smoking_status'] = heart_data['smoking_status'].fillna('Sometimes')

# Convert the 'bmi' column to numeric, forcing non-numeric values to NaN
heart_data['bmi'] = pd.to_numeric(heart_data['bmi'], errors='coerce')

# Fill missing values (NaNs) with the mean of the column
# Only include numeric columns in the mean calculation
numeric_cols = heart_data.select_dtypes(include=np.number).columns
heart_data[numeric_cols] = heart_data[numeric_cols].fillna(heart_data[numeric_cols].mean())

# Check the result
heart_data.head()

# stastical measurement about data
heart_data.describe()

import seaborn as sns
sns.catplot(x='gender', data=heart_data, kind='count')

# Continue with the rest of your code...
x = heart_data.iloc[:, :-1].values
y = heart_data.iloc[:, -1].values

 #Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])
x[:, 5] = labelencoder_x.fit_transform(x[:, 5])
x[:, 6] = labelencoder_x.fit_transform(x[:, 6])
x[:, 7] = labelencoder_x.fit_transform(x[:, 7])
x[:, 10] = labelencoder_x.fit_transform(x[:, 10])

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Assuming 'x' is your feature data and 'Dataset' is your DataFrame...

# Create a ColumnTransformer to apply OneHotEncoder to specific columns
ct = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1, 5, 6, 7, 10])], #Specify columns for one-hot encoding
    remainder='passthrough'  # Keep other columns unchanged
)

# Fit and transform the data using the ColumnTransformer
x = ct.fit_transform(x)

# Splitting the dataset into the Training set and Test set for decision
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
# Training Dataset
model.fit(X1_train, y1_train)
# Testing dataset
predicted_classes = model.predict(X1_test)
accuracy = accuracy_score(y1_test, predicted_classes)
parameters = model.coef_

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Select only numeric features for correlation
numeric_features = heart_data.select_dtypes(include=np.number)

# Calculate correlation for numeric features
plt.figure(figsize=(10, 5))
sns.heatmap(numeric_features.corr(), annot=True, fmt='.2f', cbar=True)
plt.show()

cm = metrics.confusion_matrix(y1_test, predicted_classes)
print(cm)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_title, size=10)

# Accuracy
print('ACCURACY:', accuracy)
print('CO_EFFICIENT', parameters)



