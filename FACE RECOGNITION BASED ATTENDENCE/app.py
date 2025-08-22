import streamlit as st
import cv2
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

st.title("Face Recognition Attendance System")

# Load existing face data and names
if os.path.exists('data/faces_data.pkl') and os.path.exists('data/names.pkl'):
    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
else:
    st.warning("No face data found. Add faces first!")
    st.stop()

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, names)

if not os.path.exists('Attendance'):
    os.mkdir('Attendance')

run = st.button("Start Camera", key="start_cam")
stop = st.button("Stop Camera", key="stop_cam")
FRAME_WINDOW = st.image([])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

date = datetime.now().strftime("%d-%m-%Y")
att_file = f"Attendance/Attendance_{date}.csv"

if os.path.exists(att_file):
    df = pd.read_csv(att_file)
else:
    df = pd.DataFrame(columns=["Name", "Time"])

threshold = 5000  # Adjust this threshold for your data

if run:
    video = cv2.VideoCapture(0)
    st.write("Camera started...")
    while True:
        ret, frame = video.read()
        if not ret:
            st.error("Failed to capture camera feed")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            resized_img_flat = resized_img.flatten().reshape(1, -1)

            distances, indices = knn.kneighbors(resized_img_flat)
            min_distance = float(distances[0][0])

            if min_distance < threshold:
                person_name = names[indices[0][0]]
            else:
                person_name = "Unknown"

            if person_name != "Unknown" and person_name not in df["Name"].values:
                df = pd.concat([df, pd.DataFrame({"Name": [person_name], "Time": [datetime.now().strftime("%H:%M:%S")]})], ignore_index=True)
                df.to_csv(att_file, index=False)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if stop:
            break

    video.release()
    cv2.destroyAllWindows()

st.subheader("Today's Attendance")
st.dataframe(df)
