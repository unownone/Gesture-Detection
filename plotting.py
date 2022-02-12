import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import modules.StaticGesture as sg

def main():
    gesture = sg.StaticGesture()
    # gesture.addTrain("point", 500)
    gesture.staticTest()

if __name__=="__main__":
    main()