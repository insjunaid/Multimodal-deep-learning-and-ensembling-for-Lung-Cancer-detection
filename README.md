# Multimodal Deep Learning and Ensembling for Early Lung Cancer Detection

## 🌟 Project Overview

This repository hosts a cutting-edge multimodal deep learning system designed for early lung cancer detection. By integrating three modalities—**Chest X-rays**, **CT Scans**, and **Metadata (symptoms)**—we employ an ensemble of pre-trained models to deliver high-accuracy predictions of "Cancer" or "Not Cancer." 

The system enhances interpretability using **Grad-CAM** and **LIME** to address the black-box nature of machine learning models. It features a sleek Flask-based web interface, a Gemini API-powered chatbot, and a secure SQL database for user authentication.

Our solution combines advanced computer vision and neural network techniques, making it a robust tool for healthcare applications.

## 🔑 Key Features

### Multimodal Analysis
- Processes Chest X-rays and CT Scans using convolutional neural networks (CNNs)
- Analyzes symptom metadata with a Multilayer Perceptron (MLP)

### Model Ensemble
- **Four models for Chest X-rays**: EfficientNetB3, InceptionV3, ResNet50, DenseNet121
- **Four models for CT Scans**: Same architectures as above
- **Metadata Model**: Custom MLP for symptom-based predictions
- Combines predictions for robust final output

### Evaluation Metrics
- High accuracy (>90% on validation sets)
- Metrics: Kappa coefficient, ROC curves, confusion matrices, classification reports

### Explainable AI
- **Grad-CAM**: Visualizes key image regions influencing predictions
- **LIME**: Provides feature importance for individual predictions

### Web Application
- Built with Flask for seamless model integration
- Modern, responsive UI for data upload and result visualization

### Chatbot
- Integrated via Gemini API for user assistance and query resolution

### Database
- SQL database for secure storage of user login details

## 🛠️ Technologies Used

- **Deep Learning**:Pytorch for CNNs and MLP
- **Pre-trained Models**: EfficientNetB3, InceptionV3, ResNet50, DenseNet121
- **Explainability**: Grad-CAM, LIME
- **Backend**: Flask for API and model deployment
- **Frontend**: HTML, CSS, JavaScript (Bootstrap for styling)
- **Chatbot**: Gemini API
- **Database**: SQL (SQLite/MySQL)
- **Others**: Scikit-learn, Matplotlib, Seaborn, Pandas, NumPy

## 📥 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-lung-cancer-detection.git
cd multimodal-lung-cancer-detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
- Initialize the database:
  ```bash
  python init_db.py
  ```
- Update database credentials in `config.py`



## 🖥️ Usage

### 1. Run the Application
```bash
python app.py
```

### 2. Access the UI
Navigate to `http://localhost:5000`

### 3. Upload Data
- Upload Chest X-ray, CT Scan images, and symptom metadata via the web interface
- View predictions ("Cancer" or "Not Cancer") with explainability visualizations

### 4. Explore Explanations
- **Grad-CAM**: Heatmaps highlighting critical image regions
- **LIME**: Feature importance for predictions

### 5. Chatbot Interaction
Ask questions like "What does this prediction mean?" via the Gemini-powered chatbot

### 6. User Authentication
Register/login to access personalized prediction history stored in the SQL database

## 📈 Model Training & Evaluation

### Training
- **Image augmentation**: rotation, flip, zoom
- Fine-tuned pre-trained models on lung cancer datasets (e.g., LIDC-IDRI, ChestX-ray14)
- MLP trained on symptom metadata

### Evaluation
- **Accuracy**: >90% on validation sets
- **Metrics**: Kappa coefficient, ROC curves (high AUC), confusion matrices, classification reports


## 🔍 Explainability

- **Grad-CAM**: Generates heatmaps to highlight tumor-like regions in images
- **LIME**: Explains predictions by identifying influential features (e.g., symptoms or image regions)

These tools ensure transparency, making the model trustworthy for clinical use.

## 🤖 Chatbot & UI

- **UI**: Clean, responsive design with intuitive forms and result dashboards
- **Chatbot**: Powered by Gemini API, answers user queries about predictions and model functionality

## 🗄️ Database

- Securely stores user login details (username, hashed password)
- Supports personalized prediction history

## 👥 Contributors

- **Junaid Shariff - Team Lead
- **Md Adan**
- **Punith J**
- **Md Yousuf**
Contributions are welcome! Open issues or submit pull requests.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

We extend our heartfelt gratitude to **Dr. Victor AI** for his invaluable guidance and mentorship throughout this project. His insights and encouragement were instrumental in shaping our approach to explainable AI and healthcare innovation.

## 🌟 Support

If you find this project useful, give it a ⭐ on GitHub! For questions or feedback, create an issue.

---

