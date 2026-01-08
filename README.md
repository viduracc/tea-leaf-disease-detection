<p align="center">
  <h1 align="center">Tea Leaf Disease Detection</h1>
  <p align="center">
    AI-powered tea leaf disease classification with explainable diagnostics
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License">
</p>

---

An end-to-end deep learning system that classifies tea leaf diseases from images using an **EfficientNet-B3** backbone, with **Grad-CAM** heatmaps for model interpretability. The application features a polished Streamlit UI for interactive use and a FastAPI REST API for programmatic access.

## Features

- **Disease Classification** — Identifies 4 conditions (Algal Leaf, Healthy, Red Leaf Spot, White Spot) with confidence tiers (high / medium / low)
- **Out-of-Distribution Rejection** — Automatically rejects non-leaf images using entropy-based filtering and leaf segmentation
- **Grad-CAM Explainability** — Generates heatmaps masked to the leaf region, showing exactly which areas the model focused on
- **User Authentication** — Email/password registration and login with bcrypt hashing
- **Prediction History** — Stores past analyses with original images and heatmaps per user
- **REST API** — FastAPI endpoints for health checks, predictions, and Grad-CAM overlays

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client Layer                       │
│          Streamlit UI  ·  FastAPI REST API               │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                    ML Pipeline                          │
│  Image → Resize → HSV Leaf Segmentation → BG Neutral   │
│  → EfficientNet-B3 → Confidence Tiers → Grad-CAM       │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                   Data Layer                            │
│          SQLite (Users + Predictions)                    │
│          Local file storage (uploads + heatmaps)         │
└─────────────────────────────────────────────────────────┘
```

## Model Performance

**Model v5.1** — EfficientNet-B3, trained with Focal Loss, MixUp augmentation, and background-swap regularisation.

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Algal Leaf | 0.783 | 0.783 | 0.783 | 23 |
| Healthy | 0.923 | 0.857 | 0.889 | 14 |
| Red Leaf Spot | 0.810 | 1.000 | 0.895 | 34 |
| White Spot | 0.909 | 0.690 | 0.784 | 29 |
| **Overall Accuracy** | | | **0.840** | **100** |

<details>
<summary><b>Robustness Evaluation</b></summary>

| Scenario | Accuracy | Macro F1 |
|----------|----------|----------|
| Original (baseline) | 88.0% | 0.881 |
| Background swapped (shortcut test) | 80.0% | 0.778 |
| Brightness −30% | 70.0% | 0.714 |
| 90° rotation | 84.0% | 0.848 |

</details>

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | TensorFlow / Keras, EfficientNet-B3 |
| **Computer Vision** | OpenCV, Pillow |
| **Web UI** | Streamlit |
| **REST API** | FastAPI, Uvicorn |
| **Database** | SQLAlchemy + SQLite |
| **Authentication** | Passlib (bcrypt) |
| **Scientific** | NumPy, Pandas, SciPy |

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/viduracc/tea-leaf-disease-detection.git
cd tea-leaf-disease-detection

# Install dependencies
pip install -r requirements.txt
```

### Model Setup

Download the trained model files and place them in the `models/` directory:

- `v5.1.keras` — EfficientNet-B3 classifier
- `backbone_extractor.keras` — Feature extractor for Grad-CAM

### Run

```bash
# Streamlit app (interactive UI)
streamlit run streamlit_app.py

# FastAPI server (REST API)
uvicorn main:app --reload --port 8000
```

## Project Structure

```
├── streamlit_app.py              # Streamlit UI with auth, prediction, and history
├── main.py                       # FastAPI REST API
├── requirements.txt              # Python dependencies
├── services/
│   ├── ml_service.py             # ML pipeline: preprocessing, inference, Grad-CAM
│   ├── database.py               # SQLAlchemy models and database setup
│   └── auth.py                   # User authentication (bcrypt)
├── models/
│   ├── v5.1.keras                # Trained EfficientNet-B3 classifier
│   ├── backbone_extractor.keras  # Grad-CAM feature extractor
│   └── class_indices.json        # Class label → index mapping
├── notebook/
│   ├── train.py                  # Full training pipeline (Colab)
│   └── results/                  # Training logs, confusion matrix, metrics
├── static/
│   └── BGImage_web.jpg           # Hero banner background
└── app_data/                     # Created at runtime
    ├── tea_leaf.db               # SQLite database
    └── uploads/                  # Uploaded images and heatmaps
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model status and Grad-CAM availability |
| `POST` | `/predict` | Upload image → JSON prediction with confidence tier |
| `POST` | `/gradcam` | Upload image → Grad-CAM heatmap overlay (PNG) |

<details>
<summary><b>Example Response — /predict</b></summary>

```json
{
  "status": "high",
  "prediction": "Red Leaf Spot",
  "confidence": 0.9234,
  "message": "Detected: Red Leaf Spot",
  "disclaimer": "This system is a research tool...",
  "all_probs": null
}
```

</details>

## Training

The model was trained on Google Colab using a two-phase approach:

1. **Phase 1** — Head training at 224×224 (frozen backbone)
2. **Phase 2** — Fine-tuning at 300×300 (top 60 layers unfrozen)

Key training techniques:
- Focal Loss for class imbalance
- MixUp augmentation (α = 0.2)
- Background-swap regularisation to prevent shortcut learning
- Cosine learning rate decay
- Class-weighted training

The complete training script is available in [`notebook/train.py`](notebook/train.py).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Disclaimer: This system is a research tool. Always consult a qualified expert for treatment decisions.</i>
</p>
