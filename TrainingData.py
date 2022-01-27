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
    columnLimit = 3

    p = dict()
    targetLabel = "sampleLabel"
    sampleSize = 50
    p['index'] = [targetLabel+"_" + str(i) for i in range (sampleSize)]

    comX = 0
    comY = 0
    counter = 0

    while countLabel<sampleSize * columnLimit:
        success,img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findPosition(img)

        if len(lmlist):
            try:
                comX_new, comY_new = getCenterOfMass(lmlist)
            except:
                continue

            difX = comX_new - comX
            difY = comY_new - comY
            
            # put dif X and Y in training data
            # for i in range(3):
            if str(counter)+'_x' in p:
                p[str(counter)+'_x'].append(difX)
                p[str(counter)+'_y'].append(difY)
            else:
                p[str(counter)+'_x'] = [difX]
                p[str(counter)+'_y'] = [difY]

            counter = (counter+1) % columnLimit

            comX = comX_new
            comY = comY_new
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
    df.insert((columnLimit*2)+1,"Label", [targetLabel for i in range(sampleSize)])
    df = df.iloc[1: , :]
    print(df)
    # df.to_csv('trainingDataVector\\'+targetLabel+'_trainingdata.csv')
    df.to_csv('trainingdata.csv')

if __name__=="__main__":
    main()