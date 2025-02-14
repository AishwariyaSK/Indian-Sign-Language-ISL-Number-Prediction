import os
import cv2
import pickle
import mediapipe as mp

# Define dataset path
number_dir = "./augmented_data"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data storage
data = []
labels = []

# Process images
for dir in sorted(os.listdir(number_dir)):  # Ensure label order consistency
    dir_path = os.path.join(number_dir, dir)
    if not os.path.isdir(dir_path):  
        continue  # Skip non-directory files

    for img_path in sorted(os.listdir(dir_path)):  # Maintain consistency
        aux = []
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Warning: Unable to read {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    aux.extend([landmark.x, landmark.y])  # Append both x & y

            data.append(aux)
            labels.append(dir)

# Check if data was collected
if data and labels:
    print("Sample Data:", data[0])
    print("Sample Label:", labels[0])
    print("Total Samples:", len(data))

    # Save to pickle safely
    with open('data_aug.pickle', "wb") as f:
        pickle.dump({"data": data, "label": labels}, f)

    print("✅ Data successfully saved to data_aug.pickle")
else:
    print("⚠ No hand landmarks were detected. Check your dataset and mediapipe version.")
