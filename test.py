import sklearn
import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_dict = pickle.load(open("./RandomForest_aug.p", 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data_aux = []
    x_ = []
    y_ = []
    h, w, _ = frame.shape

    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = int(prediction[0])+1
        print(predicted_character)

        # Calculate bounding box only if landmarks are present
        x1 = int(min(x_) * w)
        x2 = int(max(x_) * w)
        y1 = int(min(y_) * h)
        y2 = int(max(y_) * h)

        # Draw rectangle and put text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(predicted_character), (x1 - 5, y1 - 5),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
