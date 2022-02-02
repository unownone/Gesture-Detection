import cv2
import time
import sys
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
def getCenterOfMass(lmList):
    sumX = 0
    for i in range(21):
        sumX = sumX + lmList[i][1]
    sumY = 0
    for i in range(21):
        sumY = sumY + lmList[i][2]

    return sumX/21, sumY/21
def pro_center_of_mass(val,frame):
    pos_x=0
    pos_y=0
    for i in range(frame):
        pos_x+=val[i][0]
        pos_y+=val[i][1]
    return pos_x/frame,pos_y/frame


def findDistance(x1,y1,x2,y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
def main():
    pTime = 0
    cTime = 0
    frame=20
    training_data=50
    fig_name="come"
    dataval=1
    frame_value=[(0,0) for i in range(frame)]
    cap=cv2.VideoCapture(1)
    detector=htm.handDetector()
    dist_ant=[[(sys.maxsize,sys.maxsize,sys.maxsize) for i in range(21)] for i in range(frame)]
    train_dict={"dist_"+str(i) :[] for i in range(frame*21)}
    framesqu=0
    while framesqu<training_data*frame:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            frame_value.append(getCenterOfMass(lmlist))
            frame_value.pop(0)
            dist_ant.append(lmlist)
            dist_ant.pop(0)
            cx,cy=pro_center_of_mass(frame_value, frame)
            if dist_ant[0][0]!=(sys.maxsize,sys.maxsize,sys.maxsize):
                framesqu+=1
                t=0

                for i in dist_ant:
                    for j in i:
                        train_dict["dist_"+str(t)].append(findDistance(cx, cy, j[1], j[2]))
                        t+=1
        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.putText(img,str(framesqu),(100,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break
    df = pd.DataFrame(train_dict)
    ans=0
    for i in range(1,len(fig_name)+1):
        ans+=((ord(fig_name[i-1])-97)/i)
    df["index"]=[ fig_name for i in range(len(df))]
    df.to_csv('newfolder\\'+fig_name+".csv")
if __name__=="__main__":
    main()