# OrangDesaJugaNgertiDolar 💵
**Klasifikasi Nominal Uang Kertas Rupiah menggunakan CNN MobileNetV2**

Aplikasi web berbasis Flask yang menggunakan Convolutional Neural Network (MobileNetV2 Transfer Learning) untuk mengklasifikasikan nominal uang kertas Rupiah Indonesia.

## Kelas yang Didukung
Rp1.000 | Rp2.000 | Rp5.000 | Rp10.000 | Rp20.000 | Rp50.000 | Rp100.000

## Teknologi
- Python 3.10+
- TensorFlow / Keras (MobileNetV2 Transfer Learning)
- Flask (Backend)
- Bootstrap 5 (Frontend)
- scikit-learn (Evaluasi)

## Cara Menjalankan

### 1. Clone & Install
```bash
git clone https://github.com/username/rupiah-classifier.git
cd rupiah-classifier
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Dataset
Download dari Kaggle: https://www.kaggle.com/datasets/anidwiastuti/rupiah-banknotes-dataset
Ekstrak ke folder `dataset/` dengan struktur:
```
dataset/
  train/
    1000/ 2000/ 5000/ 10000/ 20000/ 50000/ 100000/
  test/
    1000/ 2000/ 5000/ 10000/ 20000/ 50000/ 100000/
```

### 3. Training Model
```bash
python train_model.py
```
Model akan disimpan di `model/rupiah_cnn.h5`

### 4. Jalankan Aplikasi Web
```bash
python app.py
```
Buka browser: http://localhost:5000

## Deployment ke Railway
1. Push ke GitHub
2. Connect repo di railway.app
3. Set Start Command: `gunicorn app:app`
4. Deploy!

## Struktur Project
```
rupiah-classifier/
├── dataset/              # Dataset Kaggle
├── model/
│   ├── rupiah_cnn.h5     # Model terlatih
│   └── class_indices.json
├── static/uploads/       # Gambar yang diunggah user
├── templates/
│   ├── index.html        # Halaman utama
│   └── about.html        # Halaman tentang
├── train_model.py        # Script training CNN
├── app.py                # Flask backend
├── requirements.txt
└── Procfile              # Untuk Railway/Heroku
```

## Akademik
- **Mata Kuliah**: Praktikum Kecerdasan Buatan
- **Kampus**: Universitas Bale Bandung (UNIBBA)
- **Tugas**: Tugas 10 – CNN Web Application
