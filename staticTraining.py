import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import csv
import pandas as pd
import glob

from modules.GestureMath import *
from math import *
import json 
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pickle

class StaticGesture:

    def __init__(self, targetLabel, sampleSize, trainLoc="trainingDataVector", trainName="trainingData", show=False):
        self.targetLabel = targetLabel
        self.sampleSize = sampleSize
        self.trainLoc = trainLoc
        self.trainName = trainName
        self.show = show

    def staticTrain(self):
        pTime,cTime = 0,0
        cap = cv2.VideoCapture(0)
        detector = htm.handDetector()
        countLabel = 0

        p = dict()
        p['index'] = [self.targetLabel+"_" + str(i) for i in range (self.sampleSize)]

        while countLabel < self.sampleSize:

            success,img = cap.read()
            img = detector.findhands(img)
            lmlist = detector.findPosition(img)
            
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
            # if keyPressed == ord('q'):
            #     break;

        # print(p)
        df = pd.DataFrame(p)
        df.insert(43,"Label", [self.targetLabel for i in range(self.sampleSize)])
        print(df)
        df.to_csv(self.trainLoc+'\\'+self.targetLabel+'_trainingdata.csv')

    def joinTrainingSets(self):
        path = self.trainLoc # use r in your path
        all_files = glob.glob(path + "/*.csv")
        
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        self.gestureCount = len(li)
        frame = pd.concat(li, axis=0, ignore_index=True)
        frame.to_csv(self.trainName+".csv")

    def modelRFC(self):
        df = pd.read_csv(self.trainName+'.csv')
        df = df.iloc[: , 3:]

        self.gestures = df['Label'].unique()
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

        rfc.fit(X,y)

        pickle.dump(rfc, open('RFCModel.sav', 'wb'))

    def staticTest(self):
        pTime,cTime = 0,0
        cap=cv2.VideoCapture(0)
        detector=htm.handDetector()

        # result = dict()
        # result[1]='Victory'
        # result[2]='Victory'
        # result[3]='Good luck'
        # result[4]='Stop'
        # result[5]='You Lose'
        
        loadedModel = pickle.load(open('RFCModel.sav','rb'))
        while True:
            success,img=cap.read()
            img=detector.findhands(img)
            lmlist = detector.findPosition(img)
            
            cv2.rectangle(img, (0,0), (650, 40), (0,0,0), -1)
            cv2.rectangle(img, (130,0), (650, 38), (255,255,255), -1)
            cv2.putText(img, "Result:", (140,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

            if len(lmlist) != 0:
                try:
                    distFromCOM, angleFromCOM = getVectorFromCenter(lmlist)
                except:
                    continue
                testList = []
                for i in range(21):
                    testList.append(distFromCOM[i])
                    testList.append(angleFromCOM[i])
                
                answer = loadedModel.predict([testList])
                # print(result[int(answer)])

                cv2.putText(img, self.gestures[int(answer)], (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

                #print(lmlist)
                #print(distfromCOM)
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



def main():
    sg = StaticGesture("flat", 500)
    # sg.staticTrain()
    # sg.joinTrainingSets()
    sg.modelRFC()
    sg.staticTest()

if __name__=="__main__":
    main()