import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, send_from_directory, abort
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from uuid import uuid4
import logging
import time
import joblib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the scaler
scaler = joblib.load("scaler.pkl")  # Load the scaler here
logger.debug("Scaler loaded successfully")

# Log current working directory
logger.debug(f"Current working directory: {os.getcwd()}")

# Verify directory permissions
try:
    test_file = os.path.join(app.config['UPLOAD_FOLDER'], 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    logger.debug(f"Successfully verified write permissions for {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Failed to verify write permissions for {app.config['UPLOAD_FOLDER']}: {str(e)}")

# Define model-specific transforms for image models
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

# Model paths for CT and X-ray
model_paths = {
    'ct': {
        'EfficientNetB3': ('models/ct_models/eff_ctscan_model.pth', models.efficientnet_b3, 'features'),
        'DenseNet121': ('models/ct_models/best_ctscan_densenet_model.pth', models.densenet121, 'features'),
        'InceptionV3': ('models/ct_models/best_ctscan_inceptionv3_model.pth', models.inception_v3, 'Mixed_7c'),
        'ResNet50': ('models/ct_models/best_ctscan_resnet50_model.pth', models.resnet50, 'layer4'),
    },
    'xray': {
        'EfficientNetB3': ('models/xray_models/87xray_effb3.pth', models.efficientnet_b3, 'features'),
        'DenseNet121': ('models/xray_models/xray_densenet121.pth', models.densenet121, 'features'),
        'InceptionV3': ('models/xray_models/xray_inceptionv3.pth', models.inception_v3, 'Mixed_7c'),
        'ResNet50': ('models/xray_models/xray_resnet50.pth', models.resnet50, 'layer4'),
    }
}

# Metadata model path
metadata_model_path = 'models/csv_best_model.pth'

# Class names for predictions
class_names = {
    'ct': ["Lung Cancer", "Normal", "UNKNOWN"],
    'xray': ["Lung Cancer X-ray", "Normal", "UNKNOWN"],
    'metadata': ["YES", "NO"]
}

# Metadata fields based on CSV
metadata_fields = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

def load_image_model(model_path, model_arch, model_name, num_classes=3):
    logger.debug(f"Loading image model: {model_path}")
    try:
        model = model_arch(pretrained=False)
        if isinstance(model, models.DenseNet):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model, models.EfficientNet):
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif isinstance(model, models.Inception3):
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif isinstance(model, models.ResNet):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load image model {model_path}: {str(e)}")
        raise

def load_metadata_model(model_path, input_size=15):
    logger.debug(f"Loading metadata model: {model_path}")
    try:
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
                    nn.Linear(32, 2)  # Assuming 2-class output
                )

            def forward(self, x):
                return self.net(x)

        model = MetadataModel(input_size=input_size)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load metadata model {model_path}: {str(e)}")
        raise

def preprocess_image(image_path, transform):
    logger.debug(f"Preprocessing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        return tensor, image
    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {str(e)}")
        raise

def preprocess_metadata(form_data):
    logger.debug(f"Preprocessing metadata: {form_data}")
    try:
        features = []
        for field in metadata_fields:
            value = form_data.get(field, 'No')
            logger.debug(f"Processing field {field}: {value}")
            if field == "GENDER":
                features.append(0 if value == "M" else 1)
            elif field == "AGE":
                try:
                    age = float(value)
                    features.append(age)
                except ValueError as e:
                    logger.error(f"Invalid age value: {value}, defaulting to 0")
                    features.append(0)
            else:
                features.append(2 if value == "Yes" else 1)
        logger.debug(f"Preprocessed features (before scaling): {features}")
        
        # Apply scaling
        features = np.array(features).reshape(1, -1)  # Reshape for scaler
        features_scaled = scaler.transform(features)  # Scale the features
        features_scaled = features_scaled.flatten().tolist()  # Convert back to list
        logger.debug(f"Scaled features: {features_scaled}")
        
        return torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        logger.error(f"Failed to preprocess metadata: {str(e)}")
        raise

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(self.target_layer.register_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            loss = output[0, class_idx]
            loss.backward()
            pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
            activation = self.activations.squeeze(0)
            for i in range(pooled_gradients.shape[0]):
                activation[i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activation, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (np.max(heatmap) + 1e-8)
            return heatmap
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {str(e)}")
            raise

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

def overlay_heatmap(img_pil, heatmap):
    try:
        img = np.array(img_pil)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        return overlay
    except Exception as e:
        logger.error(f"Failed to overlay heatmap: {str(e)}")
        raise

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        logger.debug(f"Serving uploaded image: {file_path}")
        return send_file(file_path, mimetype='image/jpeg')
    else:
        logger.error(f"File not found for serving: {file_path}")
        return "Image not found", 404

@app.route('/predict_metadata_form', methods=['POST'])
def predict_metadata_form():
    logger.debug("Entered /predict_metadata_form route")
    try:
        form_data = {}
        for field in metadata_fields:
            form_data[field] = request.form.get(field)
        logger.debug(f"Form data collected: {form_data}")

        for field in metadata_fields:
            if not form_data[field]:
                logger.error(f"Missing required field: {field}")
                return render_template('diagnosis.html', error=f"Missing required field: {field}", metadata_fields=metadata_fields)

        input_data = {}
        features = []
        for field in metadata_fields:
            if field == "GENDER":
                input_data[field] = form_data[field]
                features.append(0 if form_data[field] == "M" else 1)
            elif field == "AGE":
                try:
                    input_data[field] = int(form_data[field])
                    features.append(float(form_data[field]))
                except ValueError as e:
                    logger.error(f"Invalid age value: {form_data[field]}, defaulting to 0")
                    input_data[field] = 0
                    features.append(0.0)
            else:
                input_data[field] = form_data[field]
                features.append(2 if form_data[field] == "Yes" else 1)

        # Apply scaling
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.flatten().tolist()
        logger.debug(f"Scaled features: {features_scaled}")

        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)
        logger.debug(f"Input tensor: {input_tensor}")

        if not os.path.exists(metadata_model_path):
            logger.error(f"Metadata model file not found at: {metadata_model_path}")
            return render_template('diagnosis.html', error=f"Metadata model file not found at: {metadata_model_path}", metadata_fields=metadata_fields)

        metadata_model = load_metadata_model(metadata_model_path)
        logger.debug("Metadata model loaded successfully")

        with torch.no_grad():
            output = metadata_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            raw_output = probs[0][1].item()
            predicted_label = "YES" if raw_output >= 0.5 else "NO"
            confidence = raw_output if raw_output >= 0.5 else 1.0 - raw_output
        logger.debug(f"Raw output (YES prob): {raw_output}, Prediction: {predicted_label}, Confidence: {confidence}")

        metadata_result = {
            'label': predicted_label,
            'confidence': confidence,
            'input': input_data,
            'raw_output': raw_output
        }
        logger.debug(f"Metadata result: {metadata_result}")

        return render_template('diagnosis.html', metadata_result=metadata_result, metadata_fields=metadata_fields)

    except Exception as e:
        logger.error(f"Error in /predict_metadata_form: {str(e)}")
        return render_template('diagnosis.html', error=f"Error processing metadata: {str(e)}", metadata_fields=metadata_fields)

@app.route('/', methods=['GET'])
def index():
    logger.debug("Rendering index.html for GET request")
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    logger.debug("Entered /diagnosis route")
    if request.method == 'POST':
        scan_type = request.form.get('scan_type')
        logger.debug(f"Scan type: {scan_type}")

        if scan_type == 'metadata':
            logger.debug("Processing metadata CSV upload")
            if 'csv_file' not in request.files:
                logger.error("No CSV file uploaded")
                return render_template('diagnosis.html', error="No CSV file uploaded", metadata_fields=metadata_fields)

            csv_file = request.files['csv_file']
            if csv_file.filename == '':
                logger.error("No CSV file selected")
                return render_template('diagnosis.html', error="No CSV file selected", metadata_fields=metadata_fields)

            try:
                # Read CSV file
                df = pd.read_csv(csv_file)
                logger.debug(f"CSV data: {df.to_dict()}")

                # Validate columns
                required_columns = metadata_fields
                if not all(col in df.columns for col in required_columns):
                    logger.error("CSV file missing required columns")
                    return render_template('diagnosis.html', error="CSV file must contain all required columns", metadata_fields=metadata_fields)

                # Preprocess the CSV data for the model
                features = []
                for field in metadata_fields:
                    if field == "GENDER":
                        features.append(0 if df[field].iloc[0] == "M" else 1)
                    elif field == "AGE":
                        features.append(float(df[field].iloc[0]))
                    else:
                        features.append(df[field].iloc[0])

                # Create a tensor for the model
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                logger.debug(f"Input tensor: {input_tensor}")

                # Load the metadata model
                metadata_model = load_metadata_model(metadata_model_path)
                logger.debug("Metadata model loaded")

                # Make prediction
                with torch.no_grad():
                    output = metadata_model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    raw_output = float(probs[0][1])
                    predicted_label = "YES" if raw_output >= 0.5 else "NO"
                    confidence = raw_output if raw_output >= 0.5 else 1.0 - raw_output

                logger.debug(f"Raw output: {raw_output}, Prediction: {predicted_label}, Confidence: {confidence}")

                # Prepare the input data for display
                input_data = df.iloc[0].to_dict()
                for field in metadata_fields:
                    if field not in ["GENDER", "AGE"]:
                        input_data[field] = "Yes" if input_data[field] == 2 else "No"

                metadata_result = {
                    'label': predicted_label,
                    'confidence': confidence,
                    'input': input_data,
                    'raw_output': raw_output
                }
                logger.debug(f"Metadata result: {metadata_result}")

                return render_template('diagnosis.html', metadata_result=metadata_result, metadata_fields=metadata_fields)

            except Exception as e:
                logger.error(f"Error processing CSV: {str(e)}")
                return render_template('diagnosis.html', error=f"Error processing CSV: {str(e)}", metadata_fields=metadata_fields)

        elif scan_type == 'metadata_form':
            logger.debug("Metadata form selected, waiting for Predict button submission")
            return render_template('diagnosis.html', metadata_fields=metadata_fields)

        # Image processing (xray, ct)
        logger.debug("Processing image (xray/ct)")
        if 'image' not in request.files:
            logger.error("No image uploaded")
            return render_template('diagnosis.html', error="No image uploaded", metadata_fields=metadata_fields)

        file = request.files['image']
        if file.filename == '':
            logger.error("No image selected")
            return render_template('diagnosis.html', error="No image selected", metadata_fields=metadata_fields)

        try:
            # Save uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid4()}.jpg")
            file.save(image_path)
            logger.debug(f"Uploaded image saved at: {image_path}")

            # Process image
            results = []
            probabilities = []
            model_dict = model_paths[scan_type]
            class_names_list = class_names[scan_type]

            for name, (path, arch_fn, target_layer_name) in model_dict.items():
                logger.debug(f"Processing model: {name}")
                transform = model_transforms[name]
                input_tensor, original_img = preprocess_image(image_path, transform)
                model = load_image_model(path, arch_fn, name)

                if target_layer_name == 'features':
                    target_layer = model.features
                elif target_layer_name == 'layer4':
                    target_layer = model.layer4
                elif target_layer_name == 'Mixed_7c':
                    target_layer = model.Mixed_7c
                else:
                    target_layer = list(model.children())[-1]

                grad_cam = GradCAM(model, target_layer)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).squeeze(0)
                    probabilities.append(probs)

                conf, pred = torch.max(probs, 0)
                # Store probabilities for all classes
                class_probs = {class_names_list[i]: probs[i].item() for i in range(len(class_names_list))}
                heatmap = grad_cam.generate(input_tensor, pred.item())
                logger.debug(f"Heatmap shape: {heatmap.shape}, min: {heatmap.min()}, max: {heatmap.max()}")

                overlay = overlay_heatmap(original_img, heatmap)
                logger.debug(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}, min: {overlay.min()}, max: {overlay.max()}")

                # Save overlay image
                overlay_filename = f"{uuid4()}_overlay.jpg"
                overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
                try:
                    if overlay.dtype != np.uint8:
                        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8) * 255
                        overlay = overlay.astype(np.uint8)
                        logger.debug(f"Converted overlay to uint8: shape: {overlay.shape}, dtype: {overlay.dtype}")

                    success = cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    if success:
                        logger.debug(f"Grad-CAM overlay saved at: {overlay_path}")
                        with open(overlay_path, 'ab') as f:
                            f.flush()
                            os.fsync(f.fileno())
                        time.sleep(0.1)
                        if os.path.exists(overlay_path):
                            file_size = os.path.getsize(overlay_path)
                            logger.debug(f"File exists: {overlay_path}, size: {file_size} bytes")
                        else:
                            logger.error(f"File does not exist after saving: {overlay_path}")
                            continue
                    else:
                        logger.error(f"Failed to save Grad-CAM overlay at: {overlay_path}")
                        continue

                    results.append({
                        'name': name,
                        'label': class_names_list[pred.item()],
                        'confidence': conf.item(),
                        'class_probs': class_probs,  # Add probabilities for all classes
                        'overlay_filename': overlay_filename,
                        'absolute_path': overlay_path
                    })
                    logger.debug(f"Result for {name}: {results[-1]}")
                except Exception as e:
                    logger.error(f"Error saving Grad-CAM image for {name}: {str(e)}")
                    continue

            # Consensus Analysis
            if not probabilities:
                logger.error("No models processed successfully")
                return render_template('diagnosis.html', error="No models processed successfully", metadata_fields=metadata_fields)

            avg_probs = torch.stack(probabilities).mean(dim=0)
            consensus_conf, consensus_pred = torch.max(avg_probs, 0)
            consensus_label = class_names_list[consensus_pred.item()]
            # Store consensus probabilities for all classes
            consensus_class_probs = {class_names_list[i]: avg_probs[i].item() for i in range(len(class_names_list))}
            logger.debug(f"Consensus prediction: {consensus_label}, Confidence: {consensus_conf.item()}")

            # Log the final results being passed to the template
            logger.debug(f"Rendering template with results: {results}")
            logger.debug(f"Consensus label: {consensus_label}, Consensus confidence: {consensus_conf.item()}")

            return render_template('diagnosis.html', 
                                 results=results,
                                 consensus_label=consensus_label,
                                 consensus_conf=consensus_conf.item(),
                                 consensus_class_probs=consensus_class_probs,  # Add consensus probabilities
                                 original_image_filename=os.path.basename(image_path),  # Pass original image filename
                                 metadata_fields=metadata_fields)

        except Exception as e:
            logger.error(f"Error processing image (xray/ct): {str(e)}")
            return render_template('diagnosis.html', error=f"Error processing image: {str(e)}", metadata_fields=metadata_fields)

    logger.debug("Rendering diagnosis.html for GET request")
    return render_template('diagnosis.html', metadata_fields=metadata_fields)

# Existing routes for other pages
@app.route('/about')
def about():
    logger.debug("Entered /about route")
    return render_template('about.html')

@app.route('/home')
def home():
    logger.debug("Entered /home route")
    return render_template('index.html')

@app.route('/moreinfo')
def moreinfo():
    logger.debug("Entered /moreinfo route")
    return render_template('moreinfo.html')

# New route to serve the documentation PDF
@app.route('/documentation')
def serve_documentation():
    try:
        # Serve the PDF from the static directory
        return send_from_directory('static', 'LungVision_AI_documentation.pdf', as_attachment=False)
    except FileNotFoundError:
        logger.error("Documentation PDF not found in static directory")
        abort(404)
    except Exception as e:
        logger.error(f"Error serving documentation PDF: {str(e)}")
        abort(500)

if __name__ == '__main__':
    app.run(debug=True)