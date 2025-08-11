import face_recognition
import cv2
import os
import pickle

path = 'images'
known_encodings = []
known_names = []

for filename in os.listdir(path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(f"{path}/{filename}")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb_img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

# Save encodings to a pickle file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump((known_encodings, known_names), f)

print("Encoding complete and saved!")

# import face_recognition
# import cv2
# import os
# import pickle

# path = 'images'
# known_encodings = []
# known_names = []

# for filename in os.listdir(path):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         image_path = os.path.join(path, filename)
#         img = cv2.imread(image_path)

#         if img is None:
#             print(f"[WARNING] Could not read image: {filename}")
#             continue

#         try:
#             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         except Exception as e:
#             print(f"[WARNING] Failed to convert image to RGB: {filename} | Error: {e}")
#             continue

#         encodings = face_recognition.face_encodings(rgb_img)
#         if encodings:
#             known_encodings.append(encodings[0])
#             known_names.append(os.path.splitext(filename)[0])
#             print(f"[INFO] Encoded: {filename}")
#         else:
#             print(f"[WARNING] No face found in: {filename}")

# # Save encodings to a pickle file
# with open('face_encodings.pkl', 'wb') as f:
#     pickle.dump((known_encodings, known_names), f)

# print("✅ Encoding complete and saved!")
