import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import csv
import pandas as pd

from math import *
import json 
import random

def getFpsColor(fps):
    if fps<=10:
        return (0,0,150)
    if fps>=30:
        return (0,150,0)
    g = min(150,(fps-10)*15)
    r = min(150,150-((fps-20)*15))
    return (0,g,r)
    

def getCenterOfMass(lmList):
    sumX = 0
    for i in range(21):
        sumX = sumX + lmList[i][1]
    sumY = 0
    for i in range(21):
        sumY = sumY + lmList[i][2]

    return sumX/21, sumY/21

def getAngle(comX, comY, x, y):
    angle = atan((y-comY)/(x-comX))
    return angle

def findDistance(x1,y1,x2,y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def getVectorFromCenter(lmList):
    comX, comY = getCenterOfMass(lmList)
    distFromCOM = [0 for i in range(21)]
    angleFromCOM = [0 for i in range(21)]

    for i in range(21):
        distFromCOM[i] = findDistance(comX, comY, lmList[i][1], lmList[i][2])
        angleFromCOM[i] = getAngle(comX, comY, lmList[i][1], lmList[i][2])

    return distFromCOM, angleFromCOM


def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    countLabel = 0
    columnLimit = 10

    p = dict()
    targetLabel = "No"
    sampleSize = 50
    p['index'] = [targetLabel+"_" + str(i) for i in range (sampleSize)]

    counter = 0
    selectedHandPoints = [0,4,8,20]
    handPointSize = len(selectedHandPoints)
    lm = [[0 for i in range(handPointSize)] for j in range(2)]
    dif = [[0 for i in range(handPointSize)] for j in range(2)]

    while countLabel < sampleSize * columnLimit:

        success,img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findPosition(img)

        if len(lmlist):

            for i in range(2):
                for j in range(handPointSize):
                    dif[i][j] = lmlist[selectedHandPoints[j]][i] - lm[i][j]
            
            # put dif X and Y in training data
            if str(counter)+'_0'+'_x' in p:
                for j in range(handPointSize):
                    p[str(counter)+'_'+str(selectedHandPoints[j])+'_x'].append(dif[0][j])
                    p[str(counter)+'_'+str(selectedHandPoints[j])+'_y'].append(dif[1][j])
            else:
                for j in range(handPointSize):
                    p[str(counter)+'_'+str(selectedHandPoints[j])+'_x'] = [dif[0][j]]
                    p[str(counter)+'_'+str(selectedHandPoints[j])+'_y'] = [dif[1][j]]

            counter = (counter+1) % columnLimit

            for i in range(2):
                for j in range(handPointSize):
                    lm[i][j] = lmlist[selectedHandPoints[j]][i]

            countLabel += 1


        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, "FPS:"+str(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, getFpsColor(fps), 2)
        cv2.putText(img, "Frames taken: "+str(countLabel), (310,30), cv2.FONT_HERSHEY_PLAIN, 2, (150,0,0), 2)
        cv2.imshow('image1',img)

        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break


    # print(p)
    
    df = pd.DataFrame(p)
    df.insert((columnLimit*handPointSize*2)+1,"Label", [targetLabel for i in range(sampleSize)])
    df = df.iloc[1: , :]
    print(df)
    df.to_csv('trainingData\\'+targetLabel+'_train.csv')

if __name__=="__main__":
    main()