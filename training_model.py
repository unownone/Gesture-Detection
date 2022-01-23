import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
file_location=pd.read_csv("trainingdata.csv",usecols=["sampleLabel_"+str(i)for i in range(10)])
file_location['value']
y = file_location['value']
y = y.astype('int')
X = file_location
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
pickle.dump(rfc, open('firstModel.sav', 'wb'))