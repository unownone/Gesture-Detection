import modules.HandTrackingModule as htm
from modules.GestureMath import *

import cv2
import time
import mediapipe as mp
import numpy as np
import csv
import pandas as pd
import glob
from math import *
import json 
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

class StaticGesture:

    def __init__(self, targetLabel, sampleSize, dataLoc=r"modules\staticTrainingData", trainName="staticData", modelName="RFCModel",show=False, cam=0):
        self.targetLabel = targetLabel
        self.sampleSize = sampleSize
        self.dataLoc = dataLoc
        self.show = show
        self.detector = htm.handDetector()
        self.cam = cam
        self.trainName = trainName
        self.modelName = modelName

        try:
            self.gestures = open(self.dataLoc+'\\gestures.csv', 'r').read().splitlines()
            self.gestures.pop(0)
        except:
            print("Gesture File Not Found. Run the Model method first")

        try:
            self.model = pickle.load(open(self.dataLoc + "\\" + modelName +'.sav','rb'))
        except:
            print("Model File Not Found. Run the Model method first")

    def cameraTest(self):
        cap = cv2.VideoCapture(self.cam)
        while True:
            success,img = cap.read()
            img = self.detector.findhands(img)
            cv2.putText(img, str(random.randint(1,10)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

            cv2.imshow('image1',img)

            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                break

    def staticTrain(self):
        pTime,cTime = 0,0
        cap = cv2.VideoCapture(self.cam)
        countLabel = 0

        p = dict()
        p['index'] = [self.targetLabel+"_" + str(i) for i in range (self.sampleSize)]

        while countLabel < self.sampleSize:
            success,img = cap.read()
            img = self.detector.findhands(img)
            lmlist = self.detector.findPosition(img)
            
            if len(lmlist) != 0:

                try:
                    distFromCOM, angleFromCOM = getVectorFromCenter(lmlist)
                except:
                    continue

                for i in range(0,21):
                    if str(i)+'_dist' in p:
                        p[str(i)+'_dist'].append(distFromCOM[i])
                        p[str(i)+'_angle'].append(angleFromCOM[i])
                    else:
                        p[str(i)+'_dist'] = [distFromCOM[i]]
                        p[str(i)+'_angle'] = [angleFromCOM[i]]

                countLabel=countLabel+1
                
                #print(lmlist)
                #print(distfromCOM)

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
            cv2.putText(img, "Frames taken: "+str(countLabel), (310,30), cv2.FONT_HERSHEY_PLAIN, 2, (150,0,0), 2)
            cv2.imshow('image1',img)
            
            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                break

        # print(p)
        df = pd.DataFrame(p)
        df.insert(43,"Label", [self.targetLabel for i in range(self.sampleSize)])
        # print(df)
        saveLoc = self.dataLoc+'\\'+self.targetLabel+'_data.csv'
        df.to_csv(saveLoc)

    def joinTrainingSets(self):
        path = self.dataLoc # use r in your path
        all_files = glob.glob(path + "/*.csv")
        
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        self.gestureCount = len(li)
        frame = pd.concat(li, axis=0, ignore_index=True)
        saveLoc = self.dataLoc+'\\'+self.trainName+".csv"
        frame.to_csv(saveLoc)

    def modelRFC(self):
        df = pd.read_csv(self.dataLoc+'\\'+"staticData"+".csv")
        df = df.iloc[: , 3:]

        self.gestures = df['Label'].unique()
        pd.DataFrame(self.gestures).to_csv(self.dataLoc+"\\gestures.csv", index=False)

        label_encoder = preprocessing.LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])

        y = df['Label']
        y = y.astype('int')
        X = df.drop('Label', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

        rfc = RandomForestClassifier(n_estimators=600)
        rfc.fit(X_train,y_train)

        predictions = rfc.predict(X_test)
        print(classification_report(y_test,predictions))

        rfc.fit(X.values,y.values)
        self.model = rfc
        pickle.dump(rfc, open(self.dataLoc + "\\" + self.modelName +'.sav', 'wb'))

    def testImage(self, img):
        img = self.detector.findhands(img)
        lmlist = self.detector.findPosition(img)

        if len(lmlist) != 0:
            try:
                distFromCOM, angleFromCOM = getVectorFromCenter(lmlist)
            except:
                return -1
            testList = []
            for i in range(21):
                testList.append(distFromCOM[i])
                testList.append(angleFromCOM[i])
            
            answer = self.model.predict([testList])
            return answer
        else:
            return -1
    
    def staticTest(self):
        pTime,cTime = 0,0
        cap=cv2.VideoCapture(self.cam)

        df = pd.read_csv(self.dataLoc+'\\'+self.trainName+".csv")
        
        while True:
            success,img=cap.read()
            answer = self.testImage(img)
            
            cv2.rectangle(img, (0,0), (650, 40), (0,0,0), -1)
            cv2.rectangle(img, (130,0), (650, 38), (255,255,255), -1)
            cv2.putText(img, "Result:", (140,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

            if answer != -1:
                cv2.putText(img, self.gestures[int(answer)], (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
            else:
                cv2.putText(img, " (No Hands Detected)", (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

            cTime=time.time()
            fps=1/(cTime-pTime)
            pTime=cTime
            
            cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
            cv2.imshow('image1',img)

            keyPressed = cv2.waitKey(5)
            if keyPressed == ord(chr(27)):
                break

# remove the modules parent from initial imports to use the below main method 

# def main():
#     sg = StaticGesture("Flat", 500)
#     # sg.cameraTest()
#     # sg.staticTrain()
#     # sg.joinTrainingSets()
#     # sg.modelRFC()
#     sg.staticTest()

# if __name__=="__main__":
#     main()