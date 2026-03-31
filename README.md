Emotion Detection from Text

A Python NLP project that detects emotions — **joy, sadness, anger, fear, surprise, disgust, and neutral** — from raw text. It offers three interchangeable backends (rule-based lexicon, ML classifier, and transformer) behind a clean API and a browser-based UI.

---

##  Problem Statement

Understanding the emotional tone of text is valuable across many domains — mental health support, customer feedback analysis, social media monitoring, and more. This project builds an end-to-end pipeline that takes any raw text as input and outputs the dominant emotion with confidence scores.

---

##  Features

| Feature | Details |
|---|---|
| 3 detection backends | Lexicon (fast), TF-IDF + Logistic Regression (ML), HuggingFace Transformer (accurate) |
| REST API | Flask endpoints: `/predict`, `/predict/batch`, `/health` |
| Browser UI | No framework needed — open and type |
| CLI | Analyse text or `.txt` files directly from terminal |
| Trainable | Bring your own CSV; synthetic data generator included |
| Tested | Pytest suite with 20+ unit tests |

---

##  Project Structure

```
emotion-detection/
│
├── src/
│   └── emotion_detector.py     # Core library (LexiconDetector, MLDetector, TransformerDetector)
│
├── tests/
│   └── test_emotion_detector.py
│
├── models/                     # Saved .joblib model files (after training)
├── data/                       # Place your CSV datasets here
│
├── app.py                      # Flask web app + REST API
├── train_model.py              # Training script
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/emotion-detection.git
cd emotion-detection
pip install -r requirements.txt
```

### 2. Run the web app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser, type any sentence, and hit **Detect Emotion**.

### 3. Use the CLI

```bash
# Single text
python src/emotion_detector.py "I can't believe how amazing this is!"

# Analyse a file
python src/emotion_detector.py --file my_text.txt

# Interactive mode
python src/emotion_detector.py
```

### 4. Use as a Python library

```python
from src.emotion_detector import EmotionDetector

det = EmotionDetector(method="lexicon")   # or "ml" / "transformer" / "auto"

result = det.predict("I am so happy today!")
print(result)
# {
#   "text": "I am so happy today!",
#   "predicted_emotion": "joy",
#   "confidence": 0.8571,
#   "scores": {"joy": 0.8571, "sadness": 0.0, ...},
#   "method": "lexicon"
# }

# Batch prediction
results = det.predict_batch(["Great day!", "I feel terrible.", "The train is late."])

# Document-level analysis
doc_result = det.analyse_document("I was happy. Then things went wrong. Now I'm angry.")
print(doc_result["overall_emotion"])   # → anger
print(doc_result["emotion_distribution"])
```

---

##  Training the ML Model

```bash
# With your own data (CSV with 'text' and 'label' columns)
python train_model.py --data data/emotions.csv --output models/emotion_model.joblib

# With built-in synthetic data (no CSV needed)
python train_model.py --samples 500
```

Then use the trained model:

```python
det = EmotionDetector(method="ml", model_path="models/emotion_model.joblib")
print(det.predict("This ruined my whole day."))
```

---

##  REST API Reference

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel fantastic!"}'
```

**Response:**
```json
{
  "text": "I feel fantastic!",
  "predicted_emotion": "joy",
  "confidence": 0.75,
  "scores": { "joy": 0.75, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0, "disgust": 0.25 },
  "method": "lexicon"
}
```

### `POST /predict/batch`

```bash
curl -X POST http://127.0.0.1:5000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["I am happy!", "This is terrible.", "The sky is blue."]}'
```

### `GET /health`

Returns `{"status": "ok", "method": "lexicon"}`.

---

##  Running Tests

```bash
pytest tests/ -v
```

Expected output: **20+ tests passing**.

---

##  Switching Backends

| Backend | Speed | Accuracy | Requirements |
|---|---|---|---|
| `lexicon` |  Instant | Good for clear sentiment | None |
| `ml` |  Fast | Better | scikit-learn + trained model |
| `transformer` |  Slower | Best | transformers + torch |
| `auto` | — | Best available | Tries transformer → ml → lexicon |

To enable transformer support:
```bash
pip install transformers torch
```

Then change `method="transformer"` in `app.py` or your script.

---

##  Emotion Categories

| Emotion | Example |
|---|---|
|  joy | "I am so happy and excited!" |
|  sadness | "I feel so lonely and hopeless." |
|  anger | "I am furious about this injustice!" |
|  fear | "I'm terrified of what might happen." |
|  surprise | "I can't believe this happened!" |
|  disgust | "That was absolutely revolting." |
|  neutral | "The meeting is at 3 PM." |

---

##  Tech Stack

- **Python 3.10+**
- **scikit-learn** — TF-IDF vectorisation & Logistic Regression
- **Flask** — REST API & web UI
- **HuggingFace Transformers** *(optional)* — `distilroberta-base` fine-tuned on emotions
- **pytest** — testing

---

##  License

MIT License — free to use, modify, and distribute.

---

##  Author

Built as a Bring Your Own Project (BYOP) capstone submission.
