import os
import zipfile
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ========== CONFIG ==========
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

# ========== HELPER: JSON SERIALIZATION ==========
def make_json_serializable(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj

# ========== LOAD MODELS ==========
print("Loading YOLO models...")
yolo_detect = YOLO('weights/ringspot_best.pt')       # detection (class-based)
yolo_seg = YOLO('weights/yellow_best_seg.pt')        # segmentation for yellow leaf
print("YOLO models loaded.")

print("Extracting and loading TabNet models...")
if os.path.exists("weights/tabnet_model.zip"):
    with zipfile.ZipFile("weights/tabnet_model.zip", "r") as zip_ref:
        zip_ref.extractall("weights")
# load tabnet models (you said you already have these directories/files)
tabnet_models = {}
for disease in ['yellow', 'ring']:
    model_path = f"weights/tabnet_{disease}/tabnet_model.pkl"
    features_path = f"weights/tabnet_{disease}/tabnet_features.pkl"
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"TabNet files for {disease} not found at {model_path} or {features_path}")
    tabnet_models[disease] = {
        'model': joblib.load(model_path),
        'features': joblib.load(features_path)   # features is expected to be list of feature names in order
    }
print("TabNet models loaded.")

# Load questionnaire (actual questions used as features)
yellow_questions_df = pd.read_csv('data/tabnet_yellow_features.csv')
ring_questions_df = pd.read_csv('data/tabnet_ring_features.csv')
yellow_questions = yellow_questions_df['Question'].tolist()
ring_questions = ring_questions_df['Question'].tolist()

# ========== HELPERS ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_detection(image_path, results, out_path):
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for r in results:
        for box in getattr(r, 'boxes', []):
            try:
                xy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
            except Exception:
                continue
            draw.rectangle(xy, outline="red", width=3)
            draw.text((xy[0], max(0, xy[1] - 12)), f"Obj ({conf:.2f})", fill="red")
    im.save(out_path)

def draw_segmentation(image_path, results, out_path):
    im = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size)
    for r in results:
        if getattr(r, 'masks', None):
            masks = r.masks.data.cpu().numpy()
            for mask in masks:
                mask_img = Image.fromarray((mask * 255).astype('uint8')).resize(im.size)
                yellow_overlay = Image.new("RGBA", im.size, (255, 255, 0, 120))
                overlay.paste(yellow_overlay, (0, 0), mask_img)
    Image.alpha_composite(im, overlay).convert("RGB").save(out_path)

def yellow_leaf_detected(seg_results, min_masks=1):
    return len(seg_results) > 0 and getattr(seg_results[0], 'masks', None) and seg_results[0].masks.data.shape[0] >= min_masks

def tabnet_predict_from_answers(answers, model_key):
    model = tabnet_models[model_key]['model']
    fv = np.array(answers).reshape(1, -1)
    prob = model.predict_proba(fv)[0][1]
    return bool(prob > 0.5), float(prob)

# ========== ROUTES ==========
@app.route('/')
def index():
    # index provides JS that handles photo -> detection -> questions -> predict
    return render_template('index.html',
                           yellow_questions=yellow_questions,
                           ring_questions=ring_questions)

@app.route('/upload', methods=['POST'])
def upload():
    """
    Accepts form-data with key 'images' (one or multiple).
    Runs YOLO detection + segmentation and returns:
    - detected_classes: ['yellow','ring'] (any subset)
    - urls for det_image and seg_image per uploaded file
    - the question sets for the detected diseases (so frontend can show only relevant)
    """
    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files uploaded'}), 400

    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(path)
            saved_files.append(path)

    # We'll process only the first image for detection/question selection (you can expand to all)
    primary = saved_files[0]
    base_name = Path(primary).stem
    det_out = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_det.jpg")
    seg_out = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_seg.jpg")

    # run yolov8 detection (class detections)
    det_res = yolo_detect(primary)     # returns Results list
    draw_detection(primary, det_res, det_out)

    # run segmentation for yellow leaf
    seg_res = yolo_seg(primary)
    draw_segmentation(primary, seg_res, seg_out)

    # decide detected classes
    detected = []
    # If segmentation has masks -> yellow found
    if yellow_leaf_detected(seg_res):
        detected.append('yellow')
    # For ring detection: if YOLO detection returned boxes, consider it ring (you might filter by class index)
    ring_detected = False
    if len(det_res) and getattr(det_res[0], 'boxes', None):
        # if your detection model has classes, you could check box.cls values.
        # For robustness, treat any box as ring-spot detection (or adapt as per your model)
        if det_res[0].boxes.shape[0] > 0:
            ring_detected = True
    if ring_detected:
        detected.append('ring')

    # If none detected, don't aggressively assume — but we'll still give user option
    if len(detected) == 0:
        detected = []  # empty list, frontend will show 'none' branch

    # Build response
    resp = {
        'detected_classes': detected,
        'images': {
            'det_image': f"/outputs/{os.path.basename(det_out)}",
            'seg_image': f"/outputs/{os.path.basename(seg_out)}"
        },
        # attach the tabnet question sets (frontend will pick the right ones)
        'questions': {
            'yellow': yellow_questions,
            'ring': ring_questions
        }
    }
    return jsonify(make_json_serializable(resp))

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    """
    Expects JSON:
    {
      "answers": {
         "yellow": [0/1, 0/1, ...],   # optional
         "ring": [0/1, 0/1, ...]      # optional
      }
    }
    Runs TabNet for each present disease and returns probabilities + detected boolean.
    """
    data = request.get_json()
    if not data or 'answers' not in data:
        return jsonify({'error': 'No answers provided'}), 400

    answers = data['answers']
    out = {}
    for disease in ['yellow', 'ring']:
        if disease in answers:
            # ensure length matches expected
            expected_len = len(tabnet_models[disease]['features'])
            arr = answers[disease]
            if len(arr) != expected_len:
                return jsonify({'error': f'Incorrect number of answers for {disease} (expected {expected_len})'}), 400
            detected, prob = tabnet_predict_from_answers(arr, disease)
            out[disease] = {'detected': bool(detected), 'prob': float(prob)}
    return jsonify(make_json_serializable(out))

@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# Optional: serve the uploads for debugging/viewing
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    # small convenience: open browser (local dev)
    import webbrowser
    from threading import Timer
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
