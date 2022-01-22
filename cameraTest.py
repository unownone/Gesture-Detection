import cv2
import random

def main():
    cap = cv2.VideoCapture(1)

    while True:
        success,img = cap.read()

        cv2.putText(img, str(random.randint(1,10)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow('image1',img)
        cv2.waitKey(5)


if __name__=="__main__":
    main()