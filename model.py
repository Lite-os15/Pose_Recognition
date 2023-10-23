import mediapipe as mp
import cv2
import pandas as pd
import os

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
points = mpPose.PoseLandmark
path = "Dataset/"
data = []
image_names = []  # List to store image names

# Create columns for the DataFrame
for p in points:
    x = str(p)[13:]
    data.append(x + "_x")
    data.append(x + "_y")
    data.append(x + "_z")
    data.append(x + "_vis")

data = pd.DataFrame(columns=data)  # Empty dataset

for img_file in os.listdir(path):
    temp = []
    img = cv2.imread(os.path.join(path, img_file))
    img_name = os.path.splitext(img_file)[0]  # Get the image name without the extension
    image_names.append(img_name)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for j in landmarks:
            temp.extend([j.x, j.y, j.z, j.visibility])

        data.loc[img_name] = temp  # Use the image name as the index

data.to_csv("dataset3.csv")  # Save the data as a CSV file
