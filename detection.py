import cv2
import mediapipe as mp
import pickle 
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():

    x_ = []
    y_ = []
    data_aux=[]

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        for hand_landmarks in results.multi_hand_landmarks:
          for i in range(len(hand_landmarks.landmark)):
              x = hand_landmarks.landmark[i].x
              y = hand_landmarks.landmark[i].y

              x_.append(x)
              y_.append(y)

          for i in range(len(hand_landmarks.landmark)):
              x = hand_landmarks.landmark[i].x
              y = hand_landmarks.landmark[i].y
              data_aux.append(x - min(x_))
              data_aux.append(y - min(y_))

      if len(data_aux) == 42:
        prediction = model.predict([np.asarray(data_aux)])
        print(prediction)
        cv2.putText(image, letters[prediction[0]], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
      break

    if key == ord('f'):
      print(results.multi_hand_landmarks)
cap.release()