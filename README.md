# ♻️ WasteVision — AI-Assisted Waste Segregation

WasteVision is an **AI-powered waste classification platform** that uses **Computer Vision, Deep Learning, and TensorFlow Lite** to automatically classify waste into two categories:

- 🟤 **Dry Waste**
- 🟢 **Wet Waste**

The system is designed as a lightweight AI-assisted solution for faster and smarter waste segregation.

---

## 🚀 Features

- 📷 Image-based waste classification
- 🤖 Deep Learning-based image recognition
- 🧠 TensorFlow / TensorFlow Lite inference
- ⚡ Quantized TFLite model for lightweight deployment
- 🎯 Dry vs Wet waste classification
- 📊 Model evaluation with Accuracy, Precision, Recall and F1-Score
- 📈 Confusion Matrix visualization
- 🌐 Flask-based web application
- 🖥️ Simple browser-based interface
- ☁️ Deployable on Render
- 📱 Lightweight inference suitable for practical applications

---

## 🧠 AI & Machine Learning

WasteVision uses a **Deep Learning Computer Vision model** to classify waste images.

### Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep Learning framework |
| TensorFlow Lite | Lightweight model inference |
| NumPy | Numerical processing |
| Pillow | Image processing |
| Scikit-learn | Model evaluation |
| Flask | Web backend |
| Flask-CORS | Cross-Origin Resource Sharing |
| Gunicorn | Production WSGI server |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |

---

## 🔄 End-to-End Workflow

```text
User uploads waste image
          ↓
Image preprocessing
          ↓
Resize to 224 × 224
          ↓
RGB conversion
          ↓
Pixel normalization [-1, 1]
          ↓
TensorFlow Lite model
          ↓
Class prediction
          ↓
Dry Waste / Wet Waste
          ↓
Prediction displayed to user
```

---

## 🏷️ Waste Categories

The dataset used for testing originally contains categories such as:

- `recyclable`
- `organic`

For the WasteVision application, these categories are mapped to:

```text
recyclable → Dry Waste
organic    → Wet Waste
```

This mapping allows the application to present predictions in terms of the intended waste-segregation categories.

---

## 📦 Dataset

The evaluation dataset contains more than **23,000 images** distributed across the two classes.

```text
Dry Waste  → 9,157 images
Wet Waste  → 13,880 images
--------------------------------
Total      → 23,037 images
```

The complete dataset was kept locally for testing and was **not uploaded to the deployment repository**, keeping the deployed application lightweight.

---

## 🧪 Model Evaluation

The model was evaluated using a balanced test subset of:

```text
500 Dry Waste images
500 Wet Waste images
--------------------
1,000 total images
```

The images were selected directly from the existing test dataset. No images were moved or copied into a separate evaluation dataset.

### Overall Performance

| Metric | Score |
|---|---:|
| Test Images | **1,000** |
| Accuracy | **88.20%** |
| Macro F1-Score | **88.19%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Dry Waste | **86.59%** | **90.40%** | **88.45%** | 500 |
| Wet Waste | **89.96%** | **86.00%** | **87.93%** | 500 |

---

## 📊 Confusion Matrix

```text
                 Predicted
              Dry      Wet
Actual Dry     452      48
Actual Wet      70     430
```

The model correctly classified:

- **452 / 500 Dry Waste images**
- **430 / 500 Wet Waste images**

Overall:

```text
Correct predictions = 882
Total images        = 1000
Accuracy            = 88.20%
```

---

## 📈 Evaluation Results

The evaluation script automatically generates:

### Confusion Matrix

```text
confusion_matrix.png
```

### Metrics Summary

```text
metrics_summary.png
```

These visualizations provide a quick overview of the model's classification performance.

---

## ⚙️ Image Preprocessing

Each input image is processed before being passed to the model.

```text
Input Image
    ↓
RGB Conversion
    ↓
Resize → 224 × 224
    ↓
Convert to Float32
    ↓
Normalize Pixel Values
    ↓
[-1, 1]
    ↓
Model Input
```

The model expects the following input shape:

```text
[1, 224, 224, 3]
```

Input datatype:

```text
float32
```

The model produces two output classes:

```text
0 → Dry Waste
1 → Wet Waste
```

---

## ⚡ TensorFlow Lite Model

The deployed model is:

```text
waste_model_quant.tflite
```

TensorFlow Lite is used to provide lightweight inference compared with a full training/deep-learning environment.

The quantized model is approximately:

```text
2.5 MB
```

This makes the model suitable for lightweight deployment and practical waste-classification applications.

---

## 📁 Project Structure

```text
waste-ai-demo/
│
├── app.py
│
├── index.html
│
├── waste_model_quant.tflite
│
├── evaluate_model.py
│
├── MODEL_EVALUATION.md
│
├── confusion_matrix.png
│
├── metrics_summary.png
│
├── background.png
│
├── image.png
│
├── requirements.txt
│
├── start_server.bat
│
├── .python-version
│
└── README.md
```

The large local test dataset is intentionally excluded from the repository.

---

## 🛠️ Installation

```bash
git clone https://github.com/GursimarpreetSingh-coder/waste-ai-demo.git
cd waste-ai-demo

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Flask application:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000
```

or:

```text
http://localhost:5000
```

Open the address in your browser and upload a waste image.

---

## 🧪 Run Model Evaluation

The evaluation script can calculate:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1-Score
- Confusion Matrix

Run:

```bash
python evaluate_model.py
```

The current evaluation configuration tests:

```text
500 Dry Waste images
500 Wet Waste images
```

for a total of:

```text
1,000 images
```

---

## 📊 Evaluation Metrics Explained

### Accuracy

Measures the percentage of all predictions that were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many images predicted as a particular class actually belong to that class.

### Recall

Measures how many actual images of a class were correctly detected.

### F1-Score

The harmonic mean of Precision and Recall.

```text
F1 = 2 × (Precision × Recall)
     ---------------------------
       Precision + Recall
```

### Macro F1

Calculates the F1-score independently for each class and then averages them.

---

## 🌐 Deployment

WasteVision is designed to run as a Flask web application and can be deployed using **Render**.

### Render Configuration

```text
Runtime:
Python 3

Python Version:
3.13

Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app:app

Root Directory:
(blank)
```

The `.python-version` file specifies the Python 3.13 runtime used during deployment.

---

## 📋 Requirements

The main dependencies include:

```text
Flask
Flask-CORS
TensorFlow
NumPy
Pillow
Gunicorn
```

Additional libraries are included in `requirements.txt` for model evaluation and visualization.

---

## 🔬 Model Evaluation Summary

The final evaluation produced the following results:

```text
Test Images       : 1000
Dry Waste         : 500
Wet Waste         : 500

Accuracy          : 88.20%
Macro F1          : 88.19%

Dry Waste
Precision         : 86.59%
Recall            : 90.40%
F1-Score          : 88.45%

Wet Waste
Precision         : 89.96%
Recall            : 86.00%
F1-Score          : 87.93%
```

These results were obtained using a **balanced 1,000-image evaluation subset** and should not be interpreted as performance across all 23,037 available images.

---

## 💡 Applications

WasteVision can be adapted for:

- 🏠 Smart household waste segregation
- 🏫 Educational institutions
- 🏢 Offices and commercial buildings
- 🏭 Industrial waste-management systems
- ♻️ Smart waste bins
- 📱 Mobile waste-classification applications
- 🤖 AI-assisted recycling systems
- 🌱 Smart-city waste-management solutions

---

## 🔮 Future Scope

Potential future improvements include:

- 📚 Training with a larger and more diverse dataset
- 🗑️ Multi-class waste classification
- 🧴 Plastic detection
- 📄 Paper/cardboard detection
- 🥫 Metal detection
- 🍾 Glass detection
- 🧪 Hazardous waste detection
- 📱 Mobile deployment using TensorFlow Lite
- 📷 Real-time camera-based classification
- 🤖 Integration with automated smart bins
- 🔌 IoT-based waste monitoring
- 📊 Real-time waste analytics dashboard
- 🌍 Deployment across large-scale waste-management systems

---

## ⚠️ Limitations

The current model performs binary classification between Dry Waste and Wet Waste.

Performance can vary depending on:

- Image quality
- Lighting conditions
- Background complexity
- Object visibility
- Waste type
- Dataset distribution
- Unseen real-world objects

The reported **88.20% accuracy** was obtained from a balanced 1,000-image evaluation subset.

---

## 📌 Project Highlights

```text
✔ AI-powered waste classification
✔ Computer Vision
✔ Deep Learning
✔ TensorFlow
✔ TensorFlow Lite
✔ Quantized ML model
✔ Flask web application
✔ 23,037-image dataset
✔ 1,000-image balanced evaluation
✔ 88.20% accuracy
✔ 88.19% Macro F1
✔ Precision / Recall / F1 analysis
✔ Confusion Matrix
✔ Render deployment ready
```

---

## 👨‍💻 Author

**Gursimarpreet Singh**

B.Tech — Electrical Engineering  
Dr. B.R. Ambedkar National Institute of Technology, Jalandhar

Interested in:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Web Development
- Quantum Computing

---

## 🤝 Support

If you find this project useful or have suggestions for improvement, feel free to open an issue or contribute to the project.

---

## 📄 License

This project is provided for educational and research purposes.

See the repository license for the applicable terms.

---

## ⭐ Acknowledgement

WasteVision combines modern **Artificial Intelligence, Computer Vision, Deep Learning, and lightweight TensorFlow Lite inference** to demonstrate how machine learning can be applied to real-world environmental challenges such as automated waste segregation.

---

## ♻️ WasteVision

**Smarter Classification. Cleaner Segregation. Better Future.**
