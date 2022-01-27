import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import csv
import pandas as pd
import pickle

from math import *
import json 
import random

def getFpsColor(fps):
    brightness = 200
    if fps<=10:
        return (0,0,brightness)
    if fps>=30:
        return (0,brightness,0)
    g = min(brightness,(fps-10)*20)
    r = min(brightness,brightness-((fps-20)*(brightness/10)))
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

    result = dict()
    result[1]='Hello'
    result[2]='Stop'
    columnLimit = 10
    
    loadedModel = pickle.load(open('RFCModel.sav','rb'))

    selectedHandPoints = [0,4,8,20]
    handPointSize = len(selectedHandPoints)
    lm = [[0 for i in range(handPointSize)] for j in range(2)]
    dif = [[0 for i in range(handPointSize)] for j in range(2)]

    counter = 0
    testList = []
    answer = 1

    while True:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        
        cv2.rectangle(img, (0,0), (650, 40), (0,0,0), -1)
        cv2.rectangle(img, (130,0), (650, 38), (255,255,255), -1)
        cv2.putText(img, "Result:", (140,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)

        if counter % columnLimit == 0 and len(testList) != 0:
            answer = loadedModel.predict([testList])
            testList = []

        if len(lmlist):

            for i in range(2):
                for j in range(handPointSize):
                    dif[i][j] = lmlist[selectedHandPoints[j]][i] - lm[i][j]

            for i in range(handPointSize):
                for j in range(2):
                    testList.append(dif[j][i])

            for i in range(2):
                for j in range(handPointSize):
                    lm[i][j] = lmlist[selectedHandPoints[j]][i]

            counter = (counter+1) % columnLimit
            cv2.putText(img, result[int(answer)], (260,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

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


if __name__=="__main__":
    main()