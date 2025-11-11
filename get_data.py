import cv2 as cv
import mediapipe as mp
import pandas as pd
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

file_path = "./Images"    

data = {'label' : []}
labels = []


for dir_ in os.listdir(file_path):
    for i in range(100):

        img = cv.imread(f"{file_path}/{dir_}/{i:04d}.png")

        results = hands.process(img)


        if results.multi_hand_landmarks:
            x_ = []
            y_ = []
            z_ = []
            for j in range(21):

                x = results.multi_hand_landmarks[0].landmark[j].x
                y = results.multi_hand_landmarks[0].landmark[j].y
                z = results.multi_hand_landmarks[0].landmark[j].z
                
                x_.append(x)
                y_.append(y)
                z_.append(z)

            for j in range(21):
                if data.get(f'x{j}', []) == []:
                    data[f'x{j}'] = []
                    data[f'y{j}'] = []
                    #data[f'z{j}'] = []

                
                x = results.multi_hand_landmarks[0].landmark[j].x
                y = results.multi_hand_landmarks[0].landmark[j].y
                z = results.multi_hand_landmarks[0].landmark[j].z

                data[f'x{j}'].append(x-min(x_))
                data[f'y{j}'].append(y-min(y_))
                #data[f'z{j}'].append(z-min(z_))
                print(f"{dir_} | {i} | {j}")
            data['label'].append(dir_)



df = pd.DataFrame(data)



print(df)

df.to_csv('train.csv', index=False)
print('data was saved!')

    

cv.waitKey(0)