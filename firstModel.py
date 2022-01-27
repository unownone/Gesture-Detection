import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('trainingData.csv')
df = df.iloc[: , 3:]
Labels =  df['Label'].unique()
print(Labels)
df['Label'].replace(Labels[df['Label']])
df['Label'] = [i for i in df['Label']]

labelDict = dict()
t = 1
for i in range(196):
    df['Label'][i] = Labels.find(df['Label'][i])

# for i in range(196):
#     if df['Label'][i]=='Hello':
#         df['Label'][i] = 1
#     elif df['Label'][i]=='Stop':
#         df['Label'][i] = 2
#     elif df['Label'][i]=='roundabout':
#         df['Label'][i] = 3
#     elif df['Label'][i]=='No':
#         df['Label'][i] = 4
    

y = df['Label']
y = y.astype('int')
X = df.drop('Label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))

rfc.fit(X,y)

pickle.dump(rfc, open('RFCModel.sav', 'wb'))