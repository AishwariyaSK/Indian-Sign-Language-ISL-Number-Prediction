# Indian Sign Language (ISL) Number Prediction

📌 ## Overview
This project is a real-time Indian Sign Language (ISL) number recognition system. It uses OpenCV for live video capture, MediaPipe for hand tracking, and a trained RandomForest model to predict hand signs corresponding to numbers. The project is built with Streamlit for an interactive UI.

## Features
- **Live Hand Tracking:** Uses OpenCV and MediaPipe to detect hand landmarks in real time.
- **ISL Number Prediction:** Trained RandomForest model predicts the corresponding ISL number.
- **Dataset Generation & Augmentation:** Scripts to create and augment a dataset for training.
- **Modular Training Pipeline:** Separate scripts for dataset generation, augmentation, landmark extraction, training, and testing.
- **Streamlit Web Interface:** Provides a user-friendly way to interact with the live prediction model.

## Project Structure
```bash
.
├── dataset.py       # Generates a custom dataset using OpenCV
├── augmentation.py  # Augments the generated dataset
├── landmark.py      # Extracts hand landmarks using MediaPipe
├── train.py         # Trains the RandomForest model
├── test.py          # Tests the trained model
├── app.py           # Streamlit application for real-time prediction
├── requirements.txt # Required dependencies
├── assets/          # Contains ISL reference images
```

⚙️## Installation
### Prerequisites
Ensure you have Python 3.8 or later installed.

### Setup
1. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Generating the Dataset
To generate a dataset using OpenCV:
```sh
python dataset.py
```
This will start capturing frames from the webcam and store labeled images for training.

## Data Augmentation
To increase dataset size using transformations:
```sh
python augmentation.py
```
This will generate augmented versions of collected images.

## Landmark Extraction
Extracts hand landmarks from images and converts them into structured data for training:
```sh
python landmark.py
```

## Training the Model
Train a RandomForest classifier using the extracted landmarks:
```sh
python train.py
```
This will save a trained model as `RandomForest_aug.p`.

## Testing the Model
To evaluate the trained model:
```sh
python test.py
```

## Running the Streamlit Application
To start the real-time ISL number prediction app:
```sh
streamlit run app.py
```

## How It Works
### Hand Landmark Detection
- Uses MediaPipe Hands to detect hand landmarks.
- Extracts (x, y) coordinates and normalizes them.
- Creates a feature vector for model inference.

### Training with RandomForest
- Extracted landmarks are used to train a RandomForest classifier.
- The model predicts numbers based on hand poses.

### Live Video Capture
- OpenCV captures video frames from the webcam.
- MediaPipe detects landmarks in each frame.
- The model predicts the ISL number in real time.

## Deployment
To deploy the Streamlit app on platforms like Hugging Face Spaces or Streamlit Cloud:
```sh
pip install streamlit cloudpickle opencv-python mediapipe
streamlit run app.py
```

🚀 ## Future Enhancements
- Improve model accuracy with more training data.
- Optimize video streaming for web-based deployment.
- Add multilingual support and gesture extensions.

---
👩‍💻 Author
AishwariyaSK 🚀

Feel free to ⭐ star the repository if you find this useful! 😊

