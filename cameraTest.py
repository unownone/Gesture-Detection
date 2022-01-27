import cv2
import random

def main():
    cap = cv2.VideoCapture(1)

    while True:
        success,img = cap.read()
        t_lower = 200  # Lower Threshold
        t_upper = 150  # Upper threshold
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Applying the Canny Edge filter
        edge = cv2.Canny(img, t_lower, t_upper)

        # cv2.putText(edge, str(random.randint(1,10)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow('image1',edge)
        cv2.waitKey(5)


if __name__=="__main__":
    main()