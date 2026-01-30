# Face Recognition Based Attendance System

This project is a real-time face recognition based attendance system developed using
Computer Vision and Deep Learning techniques.

## Features
- Real-time face detection using OpenCV Haar Cascade
- Face recognition using CNN trained with TensorFlow/Keras
- Confidence thresholding and margin-based filtering for unknown faces
- Temporal smoothing to prevent label flickering
- Automatic attendance logging with duplicate prevention
- Export attendance data to Excel format

## Technologies Used
- Python
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas
- Haar Cascade Classifier

## Project Workflow
1. Capture face images using webcam
2. Train CNN model on collected dataset
3. Perform real-time face recognition
4. Mark attendance automatically
5. Export attendance to Excel

## How to Run
1. Install dependencies:
pip install -r requirements.txt


2. Collect dataset:
python collect_dataset.py


3. Train the model:
python train_model.py


4. Run attendance system:
python main.py


5. Export attendance to Excel:
python export_to_excel.py


## Note
Dataset images are captured dynamically using webcam and are not uploaded
to the repository for privacy reasons.

## Author
Tarun Kumar
