import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('trainingData.csv')
X = pd.read_csv('trainingData.csv',usecols=df.columns[2:-1])
h=dict()
t=1
wor_k=[]
for i in df['index']:
    if i not in h:
        h[i]=t
        wor_k.append(t)
        t+=1
    else:
        wor_k.append(h[i])
df["keys"]=wor_k
y=df["keys"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

predictions = rfc.predict(X_test)