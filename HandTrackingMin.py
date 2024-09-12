import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

# Parameters are Model, number hands, min det, max det
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    # Captures frame of video
    success, img = cap.read()
    # Conversion neccesary due to hands reading RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for landLms in results.multi_hand_landmarks:
            # Note we are drawing on originl image and not RGB image
            mpDraw.draw_landmarks(img, landLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)