import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('trainingData.csv')
df = df.iloc[: , 3:]

for i in range(98):
    if df['Label'][i]=='Come_here':
        df['Label'][i] = 1
    elif df['Label'][i]=='Go_away':
        df['Label'][i] = 2

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