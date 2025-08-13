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
    if isinstance(obj, (np.generic,)):  # np.float32, np.int64, etc.
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
yolo_detect = YOLO('weights/ringspot_best.pt')
yolo_seg = YOLO('weights/yellow_best_seg.pt')
print("YOLO models loaded.")

print("Extracting and loading TabNet models...")
with zipfile.ZipFile("weights/tabnet_model.zip", "r") as zip_ref:
    zip_ref.extractall("weights")

tabnet_models = {}
for disease in ['yellow', 'ring']:
    model_path = f"weights/tabnet_{disease}/tabnet_model.pkl"
    features_path = f"weights/tabnet_{disease}/tabnet_features.pkl"
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"TabNet files for {disease} not found.")
    tabnet_models[disease] = {
        'model': joblib.load(model_path),
        'features': joblib.load(features_path)
    }
print("TabNet models loaded.")

# Load questionnaire questions
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
            xy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            draw.rectangle(xy, outline="red", width=3)
            draw.text((xy[0], xy[1] - 12), f"Ringspot ({conf:.2f})", fill="red")
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
    prob = model.predict_proba(np.array(answers).reshape(1, -1))[0][1]
    return bool(prob > 0.5), float(prob)

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html', yellow_questions=yellow_questions, ring_questions=ring_questions)


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    if not files:
        return render_template('output.html', results=[])

    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(path)
            saved_files.append(path)

    results_info = []
    for path in saved_files:
        base_name = Path(path).stem
        det_out = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_det.jpg")
        seg_out = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_seg.jpg")

        det_res = yolo_detect(path)
        draw_detection(path, det_res, det_out)
        seg_res = yolo_seg(path)
        draw_segmentation(path, seg_res, seg_out)

        ring_detected = any(int(c) == 0 for c in det_res[0].boxes.cls.cpu().numpy()) if len(det_res) else False
        yellow_detected = yellow_leaf_detected(seg_res)

        yellow_prob = float(tabnet_models['yellow']['model'].predict_proba(
            np.zeros((1, len(tabnet_models['yellow']['features'])))
        )[0][1])
        ring_prob = float(tabnet_models['ring']['model'].predict_proba(
            np.zeros((1, len(tabnet_models['ring']['features'])))
        )[0][1])

        results_info.append({
            'filename': os.path.basename(path),
            'det_image': f"/outputs/{os.path.basename(det_out)}",
            'seg_image': f"/outputs/{os.path.basename(seg_out)}",
            'final_decision': {
                'ring_spot': ring_detected,
                'yellow_leaf': yellow_detected,
                'ring_prob': ring_prob,
                'yellow_prob': yellow_prob,
                'overall': ring_detected or yellow_detected
            }
        })

    return render_template('output.html', results=results_info)

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.get_json()
    if not data or 'answers' not in data:
        return jsonify({'error': 'No answers provided'}), 400

    answers = data['answers']
    if len(answers) != len(tabnet_models['yellow']['features']) + len(tabnet_models['ring']['features']):
        return jsonify({'error': 'Incorrect number of answers'}), 400

    yellow_ans = answers[:len(tabnet_models['yellow']['features'])]
    ring_ans = answers[len(tabnet_models['yellow']['features']):]

    yellow_detected, yellow_prob = tabnet_predict_from_answers(yellow_ans, 'yellow')
    ring_detected, ring_prob = tabnet_predict_from_answers(ring_ans, 'ring')

    return jsonify(make_json_serializable({
        'yellow_detected': yellow_detected,
        'yellow_prob': yellow_prob,
        'ring_detected': ring_detected,
        'ring_prob': ring_prob
    }))

@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
