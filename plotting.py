import cv2
import time
import mediapipe as mp
import modules.HandTrackingModule as htm
import numpy as np
import matplotlib.pyplot as plt

def main():
    pTime = 0
    cTime = 0
    cap=cv2.VideoCapture(0)
    detector=htm.handDetector()
    graphPlotX = [[] for i in range(5)]
    graphPlotY = [[] for i in range(5)]

    while True:
        success,img=cap.read()
        img=detector.findhands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            for i in lmlist:
                if i[0]<=4:
                    graphPlotX[0].append(-i[1])
                    graphPlotY[0].append(-i[2])
                elif i[0]<=8:
                    graphPlotX[1].append(-i[1])
                    graphPlotY[1].append(-i[2])
                elif i[0]<=12:
                    graphPlotX[2].append(-i[1])
                    graphPlotY[2].append(-i[2])
                elif i[0]<=16:
                    graphPlotX[3].append(-i[1])
                    graphPlotY[3].append(-i[2])
                else:
                    graphPlotX[4].append(-i[1])
                    graphPlotY[4].append(-i[2])

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

    plt.scatter(graphPlotX[0],graphPlotY[0])
    plt.scatter(graphPlotX[1],graphPlotY[1])
    plt.scatter(graphPlotX[2],graphPlotY[2])
    plt.scatter(graphPlotX[3],graphPlotY[3])
    plt.scatter(graphPlotX[4],graphPlotY[4])
    plt.show()

if __name__=="__main__":
    main()