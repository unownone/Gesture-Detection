import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    pTime = 0
    cTime = 0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    graphPlotsX = []
    graphPlotsY = []
    thumbX = []
    thumbY = []

    while True:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            for i in lmlist:
                if i[0]!=4:
                    graphPlotsX.append(-i[1])
                    graphPlotsY.append(-i[2])
                else:
                    thumbX.append(-i[1])
                    thumbY.append(-i[2])
                print(str(i[0]) + ": (" + str(i[1]) + "," + str(i[2]) + ")\t")
            
            chosenIndex = 4
            cv2.circle(img, (lmlist[chosenIndex][1], lmlist[chosenIndex][2]), 14, (255,0,255), cv2.FILLED)
            
            # if lmlist[4][2] > lmlist[3][2]:
            #     cv2.putText(img,"Thumbs Down",(100,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
            # else:
            #     cv2.putText(img,"Thumbs Up",(100,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)

        cTime = time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow('image1',img)
        keyPressed = cv2.waitKey(5)
        if keyPressed == ord(chr(27)):
            break

    plt.scatter(graphPlotsX,graphPlotsY)
    plt.scatter(thumbX,thumbY)
    #print(type(l))
    plt.show()

if __name__=="__main__":
    main()