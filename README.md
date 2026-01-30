# Face Recognition Attendance System

A real-time face recognition–based attendance system built using Computer Vision and Deep Learning.  
The system detects faces through a webcam, recognizes known individuals using a trained CNN model, and automatically records attendance with safeguards against false positives.

---

##  Key Features

- Real-time face detection using OpenCV (Haar Cascade)
- Face recognition using a CNN trained with TensorFlow/Keras
- Confidence thresholding to reject low-confidence predictions
- Margin-based filtering to prevent misclassification between similar faces
- Temporal smoothing across frames to reduce prediction flickering
- Automatic attendance logging with duplicate prevention
- Export of attendance records to Excel format

---

##  System Architecture

1. **Dataset Collection**
   - Face images are captured in real time using a webcam
   - Images are stored in grayscale format for faster processing
   - Each individual has a separate directory for labeled data

2. **Model Training**
   - A Convolutional Neural Network (CNN) is trained on the collected dataset
   - The model learns facial features and maps them to known identities
   - Trained model is saved and reused for real-time inference

3. **Real-Time Recognition**
   - Webcam feed is processed frame by frame
   - Faces are detected using Haar Cascade
   - Each detected face is classified by the CNN model

4. **Decision Logic**
   - Predictions are accepted only if confidence exceeds a defined threshold
   - Margin between top predictions is checked to handle similar faces
   - Temporal smoothing is applied to avoid label flickering

5. **Attendance Management**
   - Attendance is marked only once per person per day
   - Data is stored in CSV format and can be exported to Excel

---

##  Technologies Used

- **Programming Language:** Python  
- **Computer Vision:** OpenCV (Haar Cascade)  
- **Deep Learning:** TensorFlow, Keras  
- **Data Handling:** NumPy, Pandas  
- **Storage & Reporting:** CSV, Excel (OpenPyXL)

---

##  How to Run the Project

### 1. Install Dependencies
pip install -r requirements.txt

2. Collect Dataset
python collect_faces.py

3. Train the Model
python train_model.py

4. Run the Attendance System
python main.py

5. Export Attendance to Excel
python export_to_excel.py

⚠️ Limitations

1.Performance depends on lighting conditions and camera quality

2.Haar Cascade struggles with extreme face angles

3.Model accuracy is limited by dataset size and diversity

4.System is designed for controlled environments, not large-scale deployment

📌 Notes

1.Face image datasets are generated dynamically and are not uploaded to this repository for privacy reasons

2.Attendance files are created automatically during runtime

👤 Author

Tarun Kumar

B.Tech Computer Science & Engineering

Central University of Haryana

