# 🧠 Facial Recognition Attendance System (with Streamlit)

This is a simple, clean facial recognition-based attendance system built using **Python**, **face_recognition**, and **Streamlit**.  
It allows users to:
- ✅ Take attendance by uploading a photo
- ➕ Register new people by uploading their image
- 🗃️ View attendance logs saved in a CSV file

---

## ⚙️ Setup Instructions

1. **Clone the repository** (or download the folder):
   ```bash
   git clone <your_repo_url>
   cd face_attendance

2. **Install the libraries**:
   ```bash
   pip install -r requirements.txt

3. **Encoding the faces**:
Before using the app, you need to generate facial encodings:
Place one or more clear face images inside the images/ folder.
The file name (without extension) will be used as the person’s name.
Run the following script to generate face_encodings.pkl:

4. **🚀 Running the App**:
   ```bash
   streamlit run app.py

