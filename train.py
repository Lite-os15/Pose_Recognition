import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils  # For drawing keypoints
points = mpPose.PoseLandmark  # Landmarks
path = "Dataset/"  # Enter dataset path
data = []

feed= cv2.VideoCapture(0)

for p in points:
    x = str(p)[13:]
    data.append(x + "_x")
    data.append(x + "_y")
    data.append(x + "_z")
    data.append(x + "_vis")

data = pd.DataFrame(columns=data)  # Empty dataset
count = 0

while (True):

    # Capture the video frame
    # by frame
    ret, frame = feed.read()

    temp = []

    imageWidth, imageHeight = frame.shape[:2]
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blackie = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)  # Blank image

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # Draw landmarks on blackie
        landmarks = results.pose_landmarks.landmark

        for i, j in zip(points, landmarks):
            temp.extend([j.x, j.y, j.z, j.visibility])

        data.loc[count] = temp
        count += 1
    cv2.imshow("blackie", blackie)

    # Exit the loop when a key is pressed (e.g., ESC key)
    key = cv2.waitKey(100)
    if key == 27:  # 27 is the ASCII code for the ESC key
        break

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
feed.release()
# Destroy all the windows
cv2.destroyAllWindows()

# Create columns for the DataFrame
# for p in points:
#     x = str(p)[13:]
#     data.append(x + "_x")
#     data.append(x + "_y")
#     data.append(x + "_z")
#     data.append(x + "_vis")
#
# data = pd.DataFrame(columns=data)  # Empty dataset
# count = 0
#
# for img_file in os.listdir(path):
#     temp = []
#
#     img = cv2.imread(os.path.join(path, img_file))
#     imageWidth, imageHeight = img.shape[:2]
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     blackie = np.zeros((imageHeight, imageWidth, 3), dtype=np.uint8)  # Blank image
#
#     results = pose.process(imgRGB)
#
#     if results.pose_landmarks:
#         mpDraw.draw_landmarks(blackie, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # Draw landmarks on blackie
#         landmarks = results.pose_landmarks.landmark
#
#         for i, j in zip(points, landmarks):
#             temp.extend([j.x, j.y, j.z, j.visibility])
#
#         data.loc[count] = temp
#         count += 1
#
#     cv2.imshow("Image", img)
#     cv2.imshow("blackie", blackie)
#
#
#     # Exit the loop when a key is pressed (e.g., ESC key)
#     key = cv2.waitKey(100)
#     if key == 27:  # 27 is the ASCII code for the ESC key
#         break

data.to_csv("dataset3.csv")  # Save the data as a CSV file
cv2.destroyAllWindows()  # Close all OpenCV windows
