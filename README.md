
# 🌾 AI Sugarcane Disease Prediction (Ring Spot & Yellow Leaf)

## 📌 Overview
This project is a **Flask-based AI web application** developed during the **VIT Agrithon Hackathon 2025** to detect two major sugarcane diseases:

- **Ring Spot**
- **Yellow Leaf**

The app integrates:
- **YOLOv8** for image-based disease detection.
- **TabNet** for tabular data prediction.

It is designed to help farmers and researchers in **early diagnosis** and **improved crop health monitoring**.

⚠️ *Note*: This is a **prototype** with ~65% accuracy, created in under 48 hours during the hackathon. Future improvements are planned.

---

## 🚀 Features
- AI-powered **image recognition** for sugarcane disease detection.
- **Tabular data analysis** to support prediction.
- **User-friendly web GUI** built with Flask.
- Cross-platform — runs on Windows, Linux, and Mac.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Flask**
- **YOLOv8** – Computer vision model
- **TabNet** – Tabular deep learning model
- **HTML, CSS, JavaScript**

---

## 📷 Screenshots
*(Add images of your UI and prediction results here)*

---

## 📂 Project Structure
```

app.py               # Flask app entry point
requirements.txt     # Dependencies
models/              # YOLO & TabNet weights
static/              # CSS, JS, and images
templates/           # HTML templates
uploads/             # Uploaded images
outputs/             # Prediction results

````

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/North-Abyss/AI-Sugarcane-Disease-Prediction.git
cd AI-Sugarcane-Disease-Prediction
````

### 2️⃣ Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
python app.py
```

Visit **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

---

## 🎯 Achievements

* Built in under **48 hours** for the VIT Agrithon Hackathon.
* Combined **two AI models** (YOLOv8 + TabNet) into a single web app.
* Functional GUI for non-technical users.

---

## 🙌 Credits

* **App Development**: Yuvanesh KS ([GitHub](https://github.com/North-Abyss) | [LinkedIn](https://www.linkedin.com/in/yuvaneshks/)),**Venkatesh R**([GitHub](https://github.com/Venkatesh-107))
* **Dataset Annotation & AI Model Training**: VIT Agrithon Hackathon Team & Friends — **Sanjai R**, **Thiyagarajan S**, **Venkatesh R**([GitHub](https://github.com/Venkatesh-107))

---

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.
You may not use this work for commercial purposes without permission.

Full license text: [LICENSE](LICENSE)

---

