# Meme vs Non-Meme Detection 🖼️🧮

This repository focuses on **detecting whether a given image is a meme or non-meme** using a combination of handcrafted visual features (SIFT, LBP, ColorHist, Wavelet) and deep-learning-based models such as Siamese networks and Canonical Correlation Analysis (CCA).

---

## 📁 Project Structure

```
meme-filter/
├── data/                    # Dataset folders (not tracked in git)
│   ├── processed/
│   └── raw/
├── docs/                    # Documentation, results, plots
├── models/                  # Trained models (.pkl, .pt)
│   └── cca_1.pkl
├── notebooks/               # Experiment notebooks
├── scripts/                 # (To be added: training/eval scripts)
├── src/
│   └── meme_filter/
│       ├── __init__.py
│       ├── features/        # Handcrafted feature extractors
│       │   ├── colorhist.py
│       │   ├── lbp.py
│       │   ├── sift.py
│       │   └── wavelet.py
│       └── models/          # ML/DL models
│           ├── cca.py
│           └── siamese.py
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies
```bash
pip install -e .
```
or manually:
```bash
pip install numpy scikit-learn opencv-python pillow torch
```

---

## 🚀 Usage

### Feature Extraction
Run individual feature extractors:
```bash
python -m meme_filter.features.lbp --input <path_to_image>
python -m meme_filter.features.sift --input <path_to_image>
```

### Model Evaluation
Evaluate meme/non-meme classifier:
```bash
python -m meme_filter.models.cca --test_data <path_to_data>
```

Siamese network example:
```bash
python -m meme_filter.models.siamese --mode eval --weights models/siamese_weights.pt
```

---

## 🧩 Features
- Classical handcrafted feature-based meme detection
- CCA and Siamese deep learning models
- Modular feature pipelines
- Ready for integration into multimodal systems

---

## 🔬 Future Work
- Merge with emotion classifier for end-to-end meme understanding  
- Add dataset management utilities  
- Extend Siamese training script and logging

---

## 🪪 License
MIT License © 2025 Chhavi Sharma
