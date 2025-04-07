import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Load known faces from CSV file
csv_file="E:/blind_app/faces_database1.csv"
def load_known_faces(csv_file):
    data = pd.read_csv(csv_file)
    known_encodings = []
    known_names = []
    known_branches = []

    for index, row in data.iterrows():
        image = face_recognition.load_image_file(row['image_path'])
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(row['name'])
        known_branches.append(row['branch'])

    return known_encodings, known_names, known_branches

# Load data
known_encodings, known_names, known_branches = load_known_faces("faces_database.csv")

# Streamlit UI
st.title("Face Recognition System")
st.write("Upload an image to identify the person.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    # Detect faces in the image
    rgb_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Corrected color conversion
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown Person"
        branch = ""

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            branch = known_branches[match_index]

        st.write(f"Identified: {name} - {branch}")

    st.image(img, caption="Uploaded Image", use_column_width=True)