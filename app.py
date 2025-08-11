import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime

# -------------------- File Paths --------------------
ENCODING_FILE = 'face_encodings.pkl'
ATTENDANCE_FILE = 'attendance.csv'
IMAGE_FOLDER = 'images'

# -------------------- Utility Functions --------------------
def load_encodings():
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, 'rb') as f:
            return pickle.load(f)
    return [], []

def save_encodings(encodings, names):
    with open(ENCODING_FILE, 'wb') as f:
        pickle.dump((encodings, names), f)

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if "Name" not in df.columns or "Time" not in df.columns:
                raise ValueError("Missing required columns.")
            return df
        except Exception:
            pass  # Fall through to return an empty frame
    return pd.DataFrame(columns=["Name", "Time"])


def mark_attendance(name, df):
    if name not in df["Name"].values:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame([[name, now]], columns=["Name", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
    return df

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Facial Attendance", layout="centered")
st.title("🧠 Facial Recognition Attendance System")

# -------------------- Session State --------------------
if "known_encodings" not in st.session_state:
    st.session_state.known_encodings, st.session_state.known_names = load_encodings()

if "attendance_df" not in st.session_state:
    st.session_state.attendance_df = load_attendance()

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["📸 Take Attendance", "➕ Add New Face"])

# -------------------- Tab 1: Take Attendance --------------------
with tab1:
    st.subheader("📤 Upload Photo for Attendance")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_img = cv2.imdecode(file_bytes, 1)  # OpenCV format
        rgb = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Uploaded Image", use_column_width=True)

        faces = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, faces)

        if not encodings:
            st.warning("No face detected in the uploaded image.")
        else:
            result_img = uploaded_img.copy()

            for encoding, location in zip(encodings, faces):
                matches = face_recognition.compare_faces(st.session_state.known_encodings, encoding)
                distances = face_recognition.face_distance(st.session_state.known_encodings, encoding)
                best_match = np.argmin(distances)

                if matches and matches[best_match]:
                    name = st.session_state.known_names[best_match]
                    st.success(f"✅ Recognized: {name}")
                    st.session_state.attendance_df = mark_attendance(name, st.session_state.attendance_df)

                    # Draw box
                    top, right, bottom, left = location
                    cv2.rectangle(result_img, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(result_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    st.warning("😕 Face not recognized.")

            st.image(result_img, channels="BGR", caption="Result with Annotations")

    st.subheader("📋 Attendance Log")
    st.dataframe(st.session_state.attendance_df)

# -------------------- Tab 2: Add New Face --------------------
with tab2:
    st.subheader("👤 Add New Person")
    new_name = st.text_input("Enter Name")

    new_image = st.file_uploader("Upload image of the person", type=["jpg", "jpeg", "png"], key="new_face")

    if st.button("💾 Save New Face"):
        if new_name.strip() == "":
            st.error("❗ Please enter a name.")
        elif new_image is None:
            st.error("❗ Please upload an image.")
        else:
            file_bytes = np.asarray(bytearray(new_image.read()), dtype=np.uint8)
            new_face_img = cv2.imdecode(file_bytes, 1)
            rgb = cv2.cvtColor(new_face_img, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                os.makedirs(IMAGE_FOLDER, exist_ok=True)
                path = os.path.join(IMAGE_FOLDER, f"{new_name}.jpg")
                cv2.imwrite(path, new_face_img)

                st.session_state.known_encodings.append(encodings[0])
                st.session_state.known_names.append(new_name)
                save_encodings(st.session_state.known_encodings, st.session_state.known_names)

                st.success(f"✅ {new_name} added successfully!")
            else:
                st.error("😕 No face detected in the uploaded image.")
