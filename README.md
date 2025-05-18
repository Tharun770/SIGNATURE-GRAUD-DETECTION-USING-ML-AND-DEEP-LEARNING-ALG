# SIGNATURE-GRAUD-DETECTION-USING-ML-AND-DEEP-LEARNING-ALG
Signature Fraud Detection using Deep Learning and Machine Learning PROJECT

Here's a well-structured and professional `README.md` file tailored for your **Signature Fraud Detection** GitHub project:

```markdown
# ✍️ Signature Fraud Detection

This project aims to detect fraudulent signatures using **Deep Learning (Neural Networks)** and **Machine Learning (feature-based classification)**. It distinguishes genuine signatures from forgeries and is applicable in **banking, legal, and administrative** sectors. A **Flask web interface** enables users to upload signatures, which are then processed, features extracted, and classified for authenticity.

---

## 🚀 Features

- 🔍 Image preprocessing with **OpenCV** and **SciPy** (normalization, binarization)
- 🧠 Feature extraction: ratio, centroid, eccentricity, solidity, skew, kurtosis
- 🤖 Neural network (TensorFlow) for classification
- 🌐 User-friendly Flask web app for signature uploads and predictions
- 📊 CSV-based dataset creation for per-user training and testing

---

## 🧰 Prerequisites

- Python 3.8+
- Git
- Access to signature dataset (e.g., [CEDAR](http://www.cedar.buffalo.edu/), SigComp, or custom)
- (Optional) GPU for faster training

---

## 📁 Project Structure

```

signature-fraud-detection/
├── data/
│   ├── real/                # Genuine signatures
│   └── forged/              # Forged signatures
├── static/
│   └── Uploads/             # Temporary upload storage
├── templates/
│   ├── index.html           # Upload form page
│   └── result.html          # Prediction result page
├── Features/
│   ├── Training/            # Training CSVs
│   └── Testing/             # Testing CSVs
├── TestFeatures/            # Temporary CSV for live tests
├── requirements.txt         # Dependencies
├── signature\_fraud\_detection.py  # Main script
└── README.md                # Project documentation

````

---

## 🛠️ Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/signature-fraud-detection.git
cd signature-fraud-detection
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure the following versions are used:

```
numpy==1.23.5
matplotlib==3.6.2
scipy==1.9.3
scikit-image==0.19.3
tensorflow==2.10.0
pandas==1.5.3
flask==2.2.2
scikit-learn==1.2.2
```

---

## 📦 Dataset Preparation

1. Place signature images in:

```
data/real/   → genuine signatures  
data/forged/ → forged signatures
```

2. Use 12 users with at least 5 images each:

   * 3 for training
   * 2 for testing

```
data/
├── real/
│   ├── 001_1.png
│   ├── 001_2.png
│   └── ...
└── forged/
    ├── 001_1.png
    ├── 001_2.png
    └── ...
```

3. Update these paths in `signature_fraud_detection.py` if necessary:

```python
genuine_image_paths = r"path/to/your/real"
forged_image_paths = r"path/to/your/forged"
```

---

## 📊 Generate CSV Features

```bash
python signature_fraud_detection.py
```

This will:

* Preprocess all images
* Extract features
* Save training/testing CSVs to `Features/`

---

## 🌐 Web Application

### HTML Templates (`/templates`)

---

## ▶️ Run the Application

```bash
python signature_fraud_detection.py
```

Then open your browser at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

Upload a signature and enter the corresponding user ID for prediction.

---

## 🧪 Testing

* Upload a signature
* The system:

  * Preprocesses the image
  * Extracts features
  * Classifies using the model
* Displays:

  * User ID
  * Prediction (Genuine/Forged)
  * Probabilities

---

## 📌 Notes

* Model is trained **per user** using extracted features
* Dataset path should match in script
* `makeCSV()` handles CSV creation
* `predict()` classifies uploaded signature
* Adjust the system for more users or a different dataset if needed
* Secure the Flask app for production (e.g., disable debug mode, use WSGI)

---

## 🤝 Contributing

1. Fork this repo
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m "Add YourFeature"`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a Pull Request

---
 

## 🙏 Acknowledgments

* Research in AI-based signature verification
* Open-source datasets like [CEDAR](http://www.cedar.buffalo.edu/) and SigComp

---

## 📬 Contact

For questions or feedback, open an issue or reach out via GitHub.

tharunkumarbommineni@gmail.com


