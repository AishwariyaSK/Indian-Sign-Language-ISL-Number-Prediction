import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from PIL import Image

# Load trained model
model_dict = pickle.load(open("RandomForest_aug.p", "rb"))
model = model_dict["model"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Streamlit UI Setup
st.title("Indian Sign Language (ISL) Number Prediction")
st.write("Click 'Start Livestream' to begin live hand tracking.")

# Display ISL reference images
st.subheader("ISL Number Reference:")
col1, col2, col3 = st.columns(3)
isl_images = [f"assets/img{i}.jpg" for i in range(1, 10)]

for i in range(3):
    with col1:
        st.image(isl_images[i], width=100, caption=f"Number {i+1}")
    with col2:
        st.image(isl_images[i+3], width=100, caption=f"Number {i+4}")
    with col3:
        st.image(isl_images[i+6], width=100, caption=f"Number {i+7}")

# Initialize session state for livestream
if "live_stream" not in st.session_state:
    st.session_state.live_stream = False

# Button logic
if st.session_state.live_stream:
    if st.button("Stop Livestream", key="stop_btn"):
        st.session_state.live_stream = False
        st.toast("Livestream Ended", icon="✅")
        st.rerun()  # Force UI update
else:
    if st.button("Start Livestream", key="start_btn"):
        st.session_state.live_stream = True
        st.rerun()  # Force UI update

# Start video only when live_stream is True
if st.session_state.live_stream:
    st.write("Livestream started! Show your hand to predict ISL numbers.")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or not st.session_state.live_stream:
            break

        # Convert frame for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data_aux = []
        x_ = []
        y_ = []
        h, w, _ = frame.shape

        # Detect hand landmarks
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

            # Ensure the input data has the correct number of features
            if len(data_aux) == 42:
                # Predict ISL number
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = int(prediction[0]) + 1

                # Draw bounding box & prediction
                x1 = int(min(x_) * w)
                x2 = int(max(x_) * w)
                y1 = int(min(y_) * h)
                y2 = int(max(y_) * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(predicted_character), (x1 - 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Display video in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    # cv2.destroyAllWindows()
