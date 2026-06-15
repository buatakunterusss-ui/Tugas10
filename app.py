"""
app.py - RupiahVision + DollarVision
Klasifikasi Uang Kertas Rupiah & Dollar USD
"""

import os, json
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = (224, 224)

# ─── NOMINAL INFO ─────────────────────────────────────────────────────────────
RUPIAH_INFO = {
    "1000"  : {"label": "Rp 1.000",   "warna": "Abu-abu", "tokoh": "Tjut Meutia"},
    "2000"  : {"label": "Rp 2.000",   "warna": "Abu-abu", "tokoh": "M.H. Thamrin"},
    "5000"  : {"label": "Rp 5.000",   "warna": "Coklat",  "tokoh": "Idham Chalid"},
    "10000" : {"label": "Rp 10.000",  "warna": "Ungu",    "tokoh": "Frans Kaisiepo"},
    "20000" : {"label": "Rp 20.000",  "warna": "Hijau",   "tokoh": "Sam Ratulangi"},
    "50000" : {"label": "Rp 50.000",  "warna": "Biru",    "tokoh": "Djuanda Kartawidjaja"},
    "100000": {"label": "Rp 100.000", "warna": "Merah",   "tokoh": "Soekarno & Hatta"},
}

DOLLAR_INFO = {
    "1"  : {"label": "$1",   "warna": "Hijau", "tokoh": "George Washington"},
    "2"  : {"label": "$2",   "warna": "Hijau", "tokoh": "Thomas Jefferson"},
    "5"  : {"label": "$5",   "warna": "Hijau", "tokoh": "Abraham Lincoln"},
    "10" : {"label": "$10",  "warna": "Hijau", "tokoh": "Alexander Hamilton"},
    "20" : {"label": "$20",  "warna": "Hijau", "tokoh": "Andrew Jackson"},
    "50" : {"label": "$50",  "warna": "Hijau", "tokoh": "Ulysses S. Grant"},
    "100": {"label": "$100", "warna": "Hijau", "tokoh": "Benjamin Franklin"},
}

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
models = {}
class_indices = {}

print("[INFO] Loading Rupiah model...")
if os.path.exists("model/rupiah_cnn.h5"):
    models["rupiah"] = load_model("model/rupiah_cnn.h5")
    with open("model/class_indices.json") as f:
        class_indices["rupiah"] = {v: k for k, v in json.load(f).items()}
    print("[INFO] Rupiah model loaded!")
else:
    print("[WARN] Rupiah model tidak ditemukan!")

print("[INFO] Loading Dollar model...")
if os.path.exists("model/dollar_cnn.h5"):
    models["dollar"] = load_model("model/dollar_cnn.h5")
    with open("model/dollar_class_indices.json") as f:
        class_indices["dollar"] = {v: k for k, v in json.load(f).items()}
    print("[INFO] Dollar model loaded!")
else:
    print("[WARN] Dollar model tidak ditemukan! Jalankan train_dollar.py dulu.")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img_path, currency):
    if currency not in models:
        return {"error": f"Model {currency} belum tersedia."}
    
    arr  = preprocess(img_path)
    pred = models[currency].predict(arr, verbose=0)[0]
    idx  = np.argmax(pred)
    conf = float(pred[idx]) * 100
    cls  = class_indices[currency][idx]
    info = (RUPIAH_INFO if currency == "rupiah" else DOLLAR_INFO).get(cls, {})

    top3_idx = np.argsort(pred)[::-1][:3]
    top3 = [
        {
            "nominal": (RUPIAH_INFO if currency=="rupiah" else DOLLAR_INFO)
                       .get(class_indices[currency][i], {})
                       .get("label", class_indices[currency][i]),
            "confidence": round(float(pred[i])*100, 2)
        }
        for i in top3_idx
    ]
    return {
        "nominal"   : info.get("label", cls),
        "confidence": round(conf, 2),
        "warna"     : info.get("warna", "-"),
        "tokoh"     : info.get("tokoh", "-"),
        "top3"      : top3,
        "currency"  : currency
    }

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    available = list(models.keys())
    return render_template("index.html", available=available)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file."}), 400
    file     = request.files["file"]
    currency = request.form.get("currency", "rupiah")
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung."}), 400
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    try:
        result = predict(filepath, currency)
        result["image_url"] = url_for("static", filename=f"uploads/{filename}")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
