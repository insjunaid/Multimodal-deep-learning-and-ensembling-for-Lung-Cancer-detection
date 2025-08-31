from flask import Flask, render_template, request, jsonify, send_file, abort, redirect, url_for, session
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import joblib
import logging
from uuid import uuid4
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
from functools import lru_cache
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from datetime import datetime

# Logging setup must come before LIME import to define logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# LIME imports (robust)
LIME_AVAILABLE = False
try:
    import lime
    from lime import lime_image
    from lime.lime_tabular import LimeTabularExplainer
    from skimage.segmentation import mark_boundaries, quickshift, slic
    LIME_AVAILABLE = True
    logger.debug("LIME imported successfully")
except Exception as e:
    logging.warning(f"LIME import failed: {e}. Please install with `pip install lime` if needed.")

# Gemini imports
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    logger.debug("Gemini imported successfully")
except Exception as e:
    logging.warning(f"Gemini import failed: {e}. Please install with `pip install google-generativeai` if needed.")

# App config
app = Flask(__name__)
app.secret_key = 'LungVision2025!AI@Rock'  # Your memorable key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# SQLite Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lungvision.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(60), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # Patient, Clinician, Researcher
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create DB tables if they don't exist
with app.app_context():
    db.create_all()

# Performance-related defaults
DEFAULT_LIME_NUM_SAMPLES = 100  # Increased for better results
LIME_NUM_FEATURES = 8  # Increased for better feature coverage
MAX_PROCESS_TIME = 30

# Load scaler
SCALER_PATH = "scaler.pkl"
if not os.path.exists(SCALER_PATH):
    logger.error(f"Scaler file not found: {SCALER_PATH}. Place your scaler.pkl in the working folder.")
else:
    scaler = joblib.load(SCALER_PATH)
    logger.debug("Scaler loaded successfully. Note: Version mismatch with scikit-learn 1.7.1 detected. Regenerate scaler.pkl if issues arise.")

logger.debug(f"Current working directory: {os.getcwd()}")

# Model transforms
model_transforms = {
    'EfficientNetB3': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'DenseNet121': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'InceptionV3': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
    'ResNet50': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Model filenames
model_paths = {
    'ct': {
        'EfficientNetB3': ('models/ct_models/eff_ctscan_model.pth', models.efficientnet_b3, 'features'),
        'DenseNet121': ('models/ct_models/best_ctscan_densenet_model.pth', models.densenet121, 'features.denseblock4'),
        'InceptionV3': ('models/ct_models/best_ctscan_inceptionv3_model.pth', models.inception_v3, 'Mixed_7c'),
        'ResNet50': ('models/ct_models/best_ctscan_resnet50_model.pth', models.resnet50, 'layer4'),
    },
    'xray': {
        'EfficientNetB3': ('models/xray_models/87xray_effb3.pth', models.efficientnet_b3, 'features'),
        'DenseNet121': ('models/xray_models/xray_densenet121.pth', models.densenet121, 'features.denseblock4'),
        'InceptionV3': ('models/xray_models/xray_inceptionv3.pth', models.inception_v3, 'Mixed_7c'),
        'ResNet50': ('models/xray_models/xray_resnet50.pth', models.resnet50, 'layer4'),
    }
}

metadata_model_path = 'models/csv_best_model.pth'

class_names = {
    'ct': ["Lung Cancer", "Normal", "UNKNOWN"],
    'xray': ["Lung Cancer X-ray", "Normal", "UNKNOWN"],
    'metadata': ["YES", "NO"]
}

metadata_fields = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Preload models
preloaded_models = {}
def preload_models():
    for scan_type in model_paths:
        preloaded_models[scan_type] = {}
        for name, (path, arch_fn, _) in model_paths[scan_type].items():
            try:
                model = load_image_model(path, arch_fn, name)
                model = model.to(device)
                preloaded_models[scan_type][name] = model
                logger.debug(f"Preloaded model {name} for {scan_type}")
            except Exception as e:
                logger.error(f"Failed to preload model {name} for {scan_type}: {e}")

def load_image_model(model_path, model_arch, model_name, num_classes=3):
    logger.debug(f"Loading image model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = model_arch(weights=None)
    if isinstance(model, models.DenseNet):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif isinstance(model, models.EfficientNet):
        try:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        except Exception:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif isinstance(model, models.Inception3):
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_metadata_model(model_path, input_size=15):
    logger.debug(f"Loading metadata model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Metadata model not found: {model_path}")
    class MetadataModel(nn.Module):
        def __init__(self, input_size):
            super(MetadataModel, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)
            )
        def forward(self, x):
            return self.net(x)
    model = MetadataModel(input_size=input_size)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    logger.debug(f"Preprocessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)  # Early resizing
    tensor = transform(image).unsqueeze(0)
    return tensor, image

def preprocess_metadata(form_data):
    logger.debug(f"Preprocessing metadata: {form_data}")
    features = []
    for field in metadata_fields:
        value = form_data.get(field, 'No')
        if field == "GENDER":
            features.append(0 if value in ["M", "m", "Male", "male"] else 1)
        elif field == "AGE":
            try:
                features.append(float(value))
            except Exception:
                features.append(0.0)
        else:
            features.append(2 if str(value).strip().lower() in ['yes', '1', 'true', 'y'] else 1)
    features = np.array(features).reshape(1, -1)
    if 'scaler' in globals():
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    return torch.tensor(features_scaled.flatten(), dtype=torch.float32).unsqueeze(0)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        try:
            self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
            self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        except Exception as e:
            logger.warning(f"GradCAM hook registration warning: {e}")

    def _save_activation(self, module, input, output):
        try:
            self.activations = output.detach().clone()
        except Exception:
            self.activations = None

    def _save_gradient(self, module, grad_input, grad_output):
        try:
            self.gradients = grad_output[0].detach().clone()
        except Exception:
            self.gradients = None

    def generate(self, input_tensor, class_idx):
        try:
            self.model.zero_grad()
            x = input_tensor.clone().detach().requires_grad_(True).to(device)
            output = self.model(x)
            if isinstance(output, tuple):
                output = output[0]
            loss = output[0, class_idx]
            loss.backward()
            if self.gradients is None or self.activations is None:
                raise RuntimeError("GradCAM: missing gradients or activations")
            pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
            activation = self.activations.squeeze(0).clone()
            for i in range(pooled_gradients.shape[0]):
                activation[i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activation, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (np.max(heatmap) + 1e-8)
            return heatmap
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {e}")
            return None

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def __del__(self):
        self.remove_hooks()

def make_predict_fn_image_optimized(model, transform, device_str):
    """Optimized predict function for LIME with faster processing"""
    model.eval()
    
    def predict_fn(images):
        try:
            if len(images) == 0:
                return np.zeros((0, 3))
            
            # Process images in batches for efficiency
            batch_size = min(16, len(images))
            all_probs = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                tensors = []
                
                for img in batch_images:
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = img
                    
                    tensor = transform(pil_img)
                    tensors.append(tensor)
                
                if tensors:
                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        outputs = model(batch_tensor)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        all_probs.append(batch_probs)
            
            if all_probs:
                return np.vstack(all_probs)
            else:
                return np.ones((len(images), 3)) / 3
                
        except Exception as e:
            logger.error(f"LIME predict_fn error: {e}")
            return np.ones((len(images), 3)) / 3
    
    return predict_fn

def make_predict_fn_tabular(model, device=torch.device('cpu')):
    model = model.to(device)
    model.eval()
    def predict_fn(X):
        try:
            Xt = torch.tensor(np.array(X), dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(Xt)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs
        except Exception as e:
            logger.error(f"LIME tabular predict_fn error: {e}")
            n = len(X)
            C = 2
            fallback = np.ones((n, C)) / C
            return fallback
    return predict_fn

def overlay_heatmap(img_pil, heatmap):
    try:
        img = np.array(img_pil)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        return overlay
    except Exception as e:
        logger.error(f"Failed to overlay heatmap: {e}")
        return None

def create_enhanced_lime_overlay(image_pil, mask, save_path):
    """Enhanced LIME overlay with better visualization"""
    try:
        image_array = np.array(image_pil)
        
        # Create a more sophisticated overlay
        overlay = image_array.copy().astype(np.float32)
        
        # Normalize mask to better range
        mask_normalized = mask.copy()
        if np.max(np.abs(mask_normalized)) > 0:
            mask_normalized = mask_normalized / np.max(np.abs(mask_normalized))
        
        # Create positive and negative regions
        positive_mask = mask_normalized > 0
        negative_mask = mask_normalized < 0
        
        # Apply strong highlighting for important regions
        if np.any(positive_mask):
            # Green for positive regions (supporting prediction)
            overlay[positive_mask] = overlay[positive_mask] * 0.3 + np.array([0, 255, 0]) * 0.7
        
        if np.any(negative_mask):
            # Red for negative regions (opposing prediction)  
            overlay[negative_mask] = overlay[negative_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
        
        # Convert back to uint8
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Save the image
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return save_path
        
    except Exception as e:
        logger.error(f"Failed to create enhanced LIME overlay: {e}")
        return None

def get_optimal_segmentation(image_array, scan_type):
    """Get optimal segmentation parameters based on scan type"""
    try:
        if scan_type == 'xray':
            # For X-rays, use SLIC with parameters optimized for chest X-rays
            segments = slic(image_array, n_segments=150, compactness=10, sigma=1, start_label=1)
        else:  # CT scans
            # For CT scans, use quickshift which works better with medical CT images
            segments = quickshift(image_array, kernel_size=3, max_dist=6, ratio=0.5)
        
        return segments
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        # Fallback to simple SLIC
        try:
            return slic(image_array, n_segments=100, compactness=10, sigma=1, start_label=1)
        except:
            # Ultimate fallback - create simple grid segments
            h, w = image_array.shape[:2]
            segments = np.zeros((h, w), dtype=np.int32)
            seg_h, seg_w = h // 10, w // 10
            for i in range(10):
                for j in range(10):
                    segments[i*seg_h:(i+1)*seg_h, j*seg_w:(j+1)*seg_w] = i*10 + j
            return segments

def enhanced_prediction_with_explanations(image_path, model, model_name, transform, target_layer, class_names_list,
                                         scan_type, lime_num_samples=DEFAULT_LIME_NUM_SAMPLES):
    results = {}
    start_time = time.time()
    logger.info(f"Starting prediction for {model_name} on {image_path}")
    try:
        # Preprocess
        preprocess_start = time.time()
        input_tensor, original_img = preprocess_image(image_path, transform)
        logger.info(f"Preprocessing took {time.time() - preprocess_start:.2f}s")

        # Model inference
        inference_start = time.time()
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.to(device))
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1).squeeze(0)
        conf, pred = torch.max(probs, 0)
        class_probs = {class_names_list[i]: probs[i].item() for i in range(len(class_names_list))}
        logger.info(f"Inference took {time.time() - inference_start:.2f}s")

        # GradCAM
        gradcam_start = time.time()
        gradcam = GradCAM(model, target_layer)
        gradcam_filename = None
        try:
            heatmap = gradcam.generate(input_tensor.to(device), pred.item())
            if heatmap is not None:
                overlay = overlay_heatmap(original_img, heatmap)
                if overlay is not None:
                    gradcam_filename = f"{uuid4()}_gradcam.jpg"
                    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                    if overlay.dtype != np.uint8:
                        overlay = ((overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8) * 255).astype(np.uint8)
                    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.error(f"GradCAM failure for {model_name}: {e}")
        finally:
            gradcam.remove_hooks()
        logger.info(f"GradCAM took {time.time() - gradcam_start:.2f}s")

        results.update({
            'name': model_name,
            'label': class_names_list[pred.item()],
            'confidence': float(conf.item()),
            'class_probs': class_probs,
            'gradcam_filename': gradcam_filename,
            'overlay_filename': gradcam_filename
        })

        # Enhanced LIME
        lime_start = time.time()
        lime_filename = None
        if LIME_AVAILABLE:
            try:
                # Optimize samples based on device and time constraints
                effective_samples = lime_num_samples if device.type == 'cuda' else max(50, lime_num_samples // 2)
                
                # Create optimized predict function
                predict_fn = make_predict_fn_image_optimized(model, transform, str(device))
                
                # Initialize LIME explainer
                explainer = lime_image.LimeImageExplainer()
                
                # Prepare image
                img_np = np.array(original_img)
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
                
                # Get optimal segmentation
                def custom_segmentation(image):
                    return get_optimal_segmentation(image, scan_type)
                
                # Generate explanation
                explanation = explainer.explain_instance(
                    img_np,
                    predict_fn,
                    top_labels=1,
                    hide_color=0,
                    num_samples=effective_samples,
                    num_features=LIME_NUM_FEATURES,
                    segmentation_fn=custom_segmentation
                )
                
                if explanation is not None:
                    top_label = explanation.top_labels[0]
                    temp, mask = explanation.get_image_and_mask(
                        top_label,
                        positive_only=False,
                        num_features=LIME_NUM_FEATURES,
                        hide_rest=False,
                        min_weight=0.01  # Lower threshold to capture more features
                    )
                    
                    lime_filename = f"{uuid4()}_lime.jpg"
                    lime_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_filename)
                    saved = create_enhanced_lime_overlay(original_img, mask, lime_path)
                    if not saved:
                        lime_filename = None
                else:
                    logger.error(f"LIME explanation is None for {model_name}")
                    
            except Exception as e:
                logger.error(f"LIME image explanation failed for {model_name}: {e}")
                lime_filename = None
                
        results['lime_filename'] = lime_filename
        logger.info(f"LIME took {time.time() - lime_start:.2f}s")

        processing_time = time.time() - start_time
        logger.info(f"Total processing time for {model_name}: {processing_time:.2f}s")
        if processing_time > MAX_PROCESS_TIME:
            logger.warning(f"Processing time ({processing_time:.2f}s) exceeded target ({MAX_PROCESS_TIME}s) for {model_name}")
        return results
        
    except Exception as e:
        logger.error(f"Enhanced prediction failed for {model_name}: {e}")
        return {
            'name': model_name,
            'error': str(e),
            'gradcam_filename': None,
            'lime_filename': None,
            'overlay_filename': None
        }

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('diagnosis.html', error="File too large. Maximum size is 16MB.", metadata_fields=metadata_fields), 413

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    return "Image not found", 404

@app.route('/predict_metadata_form', methods=['POST'])
def predict_metadata_form():
    logger.debug("Entered /predict_metadata_form route")
    try:
        form_data = {field: request.form.get(field) for field in metadata_fields}
        for field in metadata_fields:
            if form_data.get(field) is None:
                return render_template('diagnosis.html', error=f"Missing required field: {field}", metadata_fields=metadata_fields)

        input_tensor = preprocess_metadata(form_data)
        metadata_model = load_metadata_model(metadata_model_path)
        metadata_model = metadata_model.to(device)
        with torch.no_grad():
            output = metadata_model(input_tensor.to(device))
            probs = torch.softmax(output, dim=1)
            raw_output = probs[0][1].item()
            predicted_label = "YES" if raw_output >= 0.5 else "NO"
            confidence = raw_output if raw_output >= 0.5 else 1.0 - raw_output

        metadata_result = {
            'label': predicted_label,
            'confidence': confidence,
            'input': form_data,
            'raw_output': raw_output
        }

        if LIME_AVAILABLE:
            try:
                lime_explainer = LimeTabularExplainer(
                    training_data=np.random.randn(100, len(metadata_fields)),
                    feature_names=metadata_fields,
                    class_names=['No Cancer', 'Cancer'],
                    mode='classification'
                )
                features_array = input_tensor.cpu().numpy()
                explanation = lime_explainer.explain_instance(features_array[0], make_predict_fn_tabular(metadata_model, device=device), num_features=len(metadata_fields))
                fig = explanation.as_pyplot_figure()
                lime_meta_filename = f"{uuid4()}_metadata_lime.jpg"
                lime_meta_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_meta_filename)
                fig.savefig(lime_meta_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                metadata_result['lime_filename'] = lime_meta_filename
            except Exception as e:
                logger.error(f"Metadata LIME failed: {e}")
                metadata_result['lime_filename'] = None

        return render_template('diagnosis.html', metadata_result=metadata_result, metadata_fields=metadata_fields)
    except Exception as e:
        logger.error(f"Error in /predict_metadata_form: {e}")
        return render_template('diagnosis.html', error=f"Error processing metadata: {e}", metadata_fields=metadata_fields)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    logger.debug("Entered /diagnosis route")
    if request.method == 'POST':
        scan_type = request.form.get('scan_type')

        if scan_type == 'metadata':
            if 'csv_file' not in request.files:
                return render_template('diagnosis.html', error="No CSV file uploaded", metadata_fields=metadata_fields)
            csv_file = request.files['csv_file']
            if csv_file.filename == '':
                return render_template('diagnosis.html', error="No CSV file selected", metadata_fields=metadata_fields)
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if not all(col in df.columns for col in metadata_fields):
                    return render_template('diagnosis.html', error="CSV must contain all required columns", metadata_fields=metadata_fields)
                form_data = {}
                for field in metadata_fields:
                    val = df[field].iloc[0]
                    if field == 'GENDER':
                        form_data[field] = 'M' if str(val).strip().upper() == 'M' else 'F'
                    elif field == 'AGE':
                        form_data[field] = val
                    else:
                        if str(val).strip() in ['2', 'Yes', 'yes', 'TRUE', 'True', 'true', '1']:
                            form_data[field] = 'Yes'
                        else:
                            form_data[field] = 'No'
                input_tensor = preprocess_metadata(form_data)
                metadata_model = load_metadata_model(metadata_model_path)
                metadata_model = metadata_model.to(device)
                with torch.no_grad():
                    output = metadata_model(input_tensor.to(device))
                    probs = torch.softmax(output, dim=1)
                    raw_output = probs[0][1].item()
                    predicted_label = "YES" if raw_output >= 0.5 else "NO"
                    confidence = raw_output if raw_output >= 0.5 else 1.0 - raw_output
                metadata_result = {'label': predicted_label, 'confidence': confidence, 'input': form_data, 'raw_output': raw_output}
                return render_template('diagnosis.html', metadata_result=metadata_result, metadata_fields=metadata_fields)
            except Exception as e:
                logger.error(f"Error processing CSV: {e}")
                return render_template('diagnosis.html', error=f"Error processing CSV: {e}", metadata_fields=metadata_fields)

        elif scan_type == 'metadata_form':
            return render_template('diagnosis.html', metadata_fields=metadata_fields)

        if 'image' not in request.files:
            return render_template('diagnosis.html', error="No image uploaded", metadata_fields=metadata_fields)
        file = request.files['image']
        if file.filename == '':
            return render_template('diagnosis.html', error="No image selected", metadata_fields=metadata_fields)
        try:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid4()}.jpg")
            file.save(image_path)

            results = []
            probabilities = []
            model_dict = model_paths.get(scan_type)
            if model_dict is None:
                return render_template('diagnosis.html', error="Invalid scan type", metadata_fields=metadata_fields)
            class_names_list = class_names[scan_type]

            for name, (_, _, target_layer_name) in model_dict.items():
                transform = model_transforms.get(name, transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))
                model = preloaded_models.get(scan_type, {}).get(name)
                if model is None:
                    logger.error(f"Preloaded model {name} not found for {scan_type}")
                    continue

                try:
                    if target_layer_name == 'features':
                        target_layer = model.features
                    elif target_layer_name == 'features.denseblock4':
                        target_layer = model.features.denseblock4
                    elif target_layer_name == 'layer4':
                        target_layer = model.layer4
                    elif target_layer_name == 'Mixed_7c':
                        target_layer = model.Mixed_7c
                    else:
                        target_layer = list(model.children())[-1]
                except Exception:
                    target_layer = list(model.children())[-1]

                model_result = enhanced_prediction_with_explanations(
                    image_path, model, name, transform, target_layer, class_names_list,
                    scan_type, lime_num_samples=DEFAULT_LIME_NUM_SAMPLES
                )

                if 'error' not in model_result:
                    results.append({
                        'name': model_result.get('name'),
                        'label': model_result.get('label'),
                        'confidence': model_result.get('confidence'),
                        'class_probs': model_result.get('class_probs'),
                        'gradcam_filename': model_result.get('gradcam_filename') or '',
                        'overlay_filename': model_result.get('overlay_filename') or '',
                        'lime_filename': model_result.get('lime_filename') or ''
                    })
                    input_tensor, _ = preprocess_image(image_path, transform)
                    with torch.no_grad():
                        out = model(input_tensor.to(device))
                        if isinstance(out, tuple):
                            out = out[0]
                        probs = torch.softmax(out, dim=1).squeeze(0)
                        probabilities.append(probs)
                else:
                    logger.error(f"Model {name} failed: {model_result.get('error')}")

            if probabilities:
                avg_probs = torch.stack(probabilities).mean(dim=0)
                consensus_conf, consensus_pred = torch.max(avg_probs, 0)
                consensus_label = class_names_list[consensus_pred.item()]
                consensus_class_probs = {class_names_list[i]: avg_probs[i].item() for i in range(len(class_names_list))}
            else:
                consensus_label = "Error"
                consensus_conf = 0.0
                consensus_class_probs = {}

            return render_template('diagnosis.html',
                                   results=results,
                                   consensus_label=consensus_label,
                                   consensus_conf=consensus_conf.item() if hasattr(consensus_conf, 'item') else float(consensus_conf),
                                   consensus_class_probs=consensus_class_probs,
                                   original_image_filename=os.path.basename(image_path),
                                   metadata_fields=metadata_fields,
                                   lime_available=LIME_AVAILABLE)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return render_template('diagnosis.html', error=f"Error processing image: {e}", metadata_fields=metadata_fields)

    return render_template('diagnosis.html', metadata_fields=metadata_fields)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/moreinfo')
def moreinfo():
    return render_template('moreinfo.html')

@app.route('/documentation')
def serve_documentation():
    try:
        return send_file(os.path.join('static', 'LungVision_AI_documentation.pdf'))
    except Exception:
        abort(404)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.json
        full_name = data.get('fullName')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')
        
        if not all([full_name, email, password, role]):
            return jsonify({'success': False, 'error': 'Please fill all fields.'}), 400
        
        # Validate password
        if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'\d', password) or not re.search(r'[!@#$%^&*]', password):
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters with 1 uppercase, 1 number, and 1 special character.'}), 400
        
        # Check if email exists
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered.'}), 400
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=hashed_password,
            role=role
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True})
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('diagnosis'))
    
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'success': False, 'error': 'account_not_found'}), 400
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
            return jsonify({'success': False, 'error': 'incorrect_password'}), 400
        
        session['user_id'] = user.id
        return jsonify({'success': True})
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/forgot_password')
def forgot_password():
    # Placeholder for password reset (implement as needed, e.g., email reset link)
    return "Password reset functionality coming soon."

@app.route('/get_explanation/<explanation_type>/<filename>')
def get_explanation_details(explanation_type, filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Explanation file not found'}), 404

GEMINI_API_KEY = 'AIzaSyDjQhVToC-7h6OYtJ-XraLwT_MgWkb4DgU'  

@app.route('/chat', methods=['POST'])
def chat():
    if not GEMINI_AVAILABLE:
        return jsonify({'response': 'Sorry, the chatbot is not available right now. Please try later.'}), 500

    data = request.json
    user_message = data.get('message', '').lower()

    # Expanded project-specific keywords to catch more variations
    project_keywords = [
        'app works', 'project works', 'how to use', 'lungvision', 'detection', 
        'web page', 'how this works', 'tell me how this works', 'how does this work', 
        'explain this', 'about this app', 'how it functions'
    ]
    is_project_query = any(keyword in user_message for keyword in project_keywords)

    if is_project_query:
        response = (
    "<div style='font-family: Arial, sans-serif; line-height: 1.6;'>"
    "<h3>LungVision AI Workflow</h3>"
    "<ol>"
    "<li><strong>Upload:</strong> Securely upload your chest X-ray or CT scan.</li>"
    "<li><strong>AI Analysis:</strong> Our models (EfficientNetB3, DenseNet121, etc.) analyze the image for early signs of lung cancer.</li>"
    "<li><strong>Results:</strong> Receive a report with predictions, confidence scores, and visual explanations using Grad-CAM and LIME.</li>"
    "</ol>"
    "<p>This tool is built on a Flask backend with PyTorch for model inference. It also supports metadata input via forms or CSV files.</p>"
    "<p><em>Note:</em> This is a decision-support tool, not a diagnostic system. Always consult a medical professional for interpretation.</p>"
    "</div>"
)

    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Provide some context to Gemini to avoid generic responses
            context = "You are a helpful assistant for the LungVision AI web app, a tool for early lung cancer detection. If the user asks about 'how this works' without context, assume they mean the web app and provide a brief overview."
            prompt = f"{context}\nUser: {user_message}"
            gemini_response = model.generate_content(prompt)
            response = gemini_response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            response = "Sorry, I couldn't process that right now. Please try again."

    return jsonify({'response': response})

if __name__ == '__main__':
    torch.set_num_threads(4)  # Optimize CPU usage
    if not LIME_AVAILABLE:
        logger.warning("LIME not available. Install with: pip install lime")
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available. Install with: pip install google-generativeai")
    preload_models()
    app.run(host='0.0.0.0', port=5000, debug=True)