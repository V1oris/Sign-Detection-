import cv2 as cv
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

camera = cv.VideoCapture(0)

label=0
i = 0
picture = 0
text = 'Press "f" to take images'
os.makedirs(f"Images/{str(label)}", exist_ok=True)
while (True):
    ret, frame = camera.read()

    key = cv.waitKey(5) & 0xFF

    # Start taking images
    if key == ord('f'):
        if i % 1 == 0 and picture < 100:
            cv.imwrite(f'Images/{label}/{picture:04d}.png', frame)
            print(f"image{picture:04d} was saved")
            picture+=1
        elif picture >= 100:
            text = "Press 'g' to reset"

    # Reset taking images
    if key == ord('g'):
        label+=1
        os.makedirs(f"Images/{str(label)}", exist_ok=True)
        picture = 0
        text = 'Press "f" to take images'
        
    # Exit the program
    if key == ord('q'):
        break

    results = hands.process(frame)

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    

    cv.putText(frame, text, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    cv.imshow("Get Images", cv.cvtColor(frame, cv.COLOR_RGB2BGR))
    i+=1