import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import csv
import pandas as pd
import modules.StaticGesture as sg

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
    gesture = sg.StaticGesture()

    p = dict()
    targetLabel = "Come_here"
    sampleSize = 20
    p['index'] = [targetLabel+"_" + str(i) for i in range (sampleSize)]

    counter = 0
    selectedHandPoints = [i for i in range(21)]
    handPointSize = len(selectedHandPoints)
    lm = [[0 for i in range(handPointSize)] for j in range(2)]
    dif = [[0 for i in range(handPointSize)] for j in range(2)]

    while countLabel < sampleSize * columnLimit:

        success,img = cap.read()
        answer = gesture.testImage(img)

        if str(counter)+'_gesture' in p:
            p[str(counter)+'_gesture'].append(answer)
        else:
            p[str(counter)+'_gesture'] = [answer]

        counter = (counter+1) % columnLimit
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
    df.insert((columnLimit)+1,"Label", [targetLabel for i in range(sampleSize)])
    df = df.iloc[1: , :]
    # print(df)
    df.to_csv('trainingData\\'+targetLabel+'_train.csv')

if __name__=="__main__":
    main()