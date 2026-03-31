"""
Emotion Detection from Text
============================
Detects emotions (joy, sadness, anger, fear, surprise, disgust, neutral)
from raw text using a hybrid approach:
  - Rule-based lexicon (fast, interpretable baseline)
  - ML classifier using TF-IDF + Logistic Regression (trained on NRC / custom data)
  - Optional: HuggingFace transformer model (high accuracy)

Author: BYOP Project
"""

import re
import string
import json
from collections import Counter
from pathlib import Path

import numpy as np

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ── Emotion Lexicon (curated seed words) ─────────────────────────────────────
EMOTION_LEXICON: dict[str, list[str]] = {
    "joy": [
        "happy", "happiness", "joyful", "excited", "wonderful", "fantastic",
        "great", "love", "amazing", "cheerful", "delight", "elated", "glad",
        "pleased", "thrilled", "ecstatic", "blissful", "jubilant", "content",
        "smile", "laugh", "celebrate", "awesome", "excellent", "brilliant",
    ],
    "sadness": [
        "sad", "unhappy", "depressed", "miserable", "grief", "sorrow",
        "heartbroken", "disappointed", "lonely", "cry", "tears", "mourn",
        "gloomy", "hopeless", "despair", "melancholy", "sorrowful", "hurt",
        "regret", "loss", "miss", "pain", "suffer", "unfortunate",
    ],
    "anger": [
        "angry", "furious", "rage", "hate", "annoyed", "frustrated",
        "irritated", "hostile", "mad", "outraged", "enraged", "bitter",
        "resentful", "aggressive", "violent", "livid", "irate", "fuming",
        "infuriated", "disgusted", "offensive", "terrible", "awful",
    ],
    "fear": [
        "afraid", "scared", "fearful", "terrified", "anxious", "worried",
        "nervous", "panic", "dread", "horror", "frighten", "tremble",
        "phobia", "uneasy", "apprehensive", "tense", "alarmed", "startled",
        "threat", "danger", "unsafe", "vulnerable", "shaky",
    ],
    "surprise": [
        "surprised", "astonished", "amazed", "shocked", "unexpected",
        "unbelievable", "incredible", "wow", "sudden", "startled",
        "astounding", "breathtaking", "stunned", "speechless", "jaw-dropping",
        "unanticipated", "extraordinary", "wonder",
    ],
    "disgust": [
        "disgusting", "revolting", "gross", "nasty", "repulsive", "vile",
        "yuck", "horrible", "repelled", "nauseous", "sick", "filthy",
        "foul", "putrid", "loathe", "detest", "abhor", "awful", "dreadful",
    ],
}

NEGATIONS = {"not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
             "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't",
             "wasn't", "aren't", "weren't", "shouldn't", "couldn't", "wouldn't"}

INTENSIFIERS = {"very", "extremely", "really", "incredibly", "absolutely",
                "deeply", "utterly", "terribly", "awfully", "super", "quite"}


# ─────────────────────────────────────────────────────────────────────────────
class LexiconDetector:
    """Rule-based emotion detector using the seed lexicon."""

    def __init__(self, lexicon: dict[str, list[str]] | None = None):
        self.lexicon = lexicon or EMOTION_LEXICON
        # Build reverse map: word → emotion
        self._word_map: dict[str, str] = {}
        for emotion, words in self.lexicon.items():
            for w in words:
                self._word_map[w] = emotion

    # ------------------------------------------------------------------
    def _preprocess(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s'-]", " ", text)  # keep apostrophes & hyphens
        return text.split()

    def predict(self, text: str) -> dict:
        """Return emotion scores and predicted label for a text snippet."""
        tokens = self._preprocess(text)
        scores: Counter = Counter()
        negation_active = False

        for i, token in enumerate(tokens):
            if token in NEGATIONS:
                negation_active = True
                continue

            # Reset negation after 3 tokens
            if negation_active and i > 0:
                distance = i - tokens.index(token)  # approximate
                if distance > 3:
                    negation_active = False

            if token in self._word_map:
                emotion = self._word_map[token]
                weight = 1.5 if (i > 0 and tokens[i - 1] in INTENSIFIERS) else 1.0
                if negation_active:
                    # Flip joy ↔ sadness; anger ↔ fear; else reduce score
                    emotion = {"joy": "sadness", "sadness": "joy",
                               "anger": "fear", "fear": "anger"}.get(emotion, emotion)
                    weight *= 0.5
                    negation_active = False
                scores[emotion] += weight

        # Normalise to probabilities
        total = sum(scores.values()) or 1
        probs = {e: round(scores.get(e, 0) / total, 4) for e in self.lexicon}

        predicted = max(probs, key=probs.get) if any(scores.values()) else "neutral"
        if not any(scores.values()):
            probs["neutral"] = 1.0

        return {
            "text": text,
            "predicted_emotion": predicted,
            "confidence": round(probs.get(predicted, 0), 4),
            "scores": probs,
            "method": "lexicon",
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
class MLDetector:
    """
    TF-IDF + Logistic Regression classifier.
    Train with `fit()` or load a pre-saved model with `load()`.
    """

    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=30_000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ])
        self.classes_: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "MLDetector":
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.classes_)
        self._fitted = True
        print(f"[MLDetector] Trained on {len(texts)} samples | classes: {self.classes_}")
        return self

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        from sklearn.metrics import classification_report, accuracy_score
        preds = self.pipeline.predict(texts)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, output_dict=True)
        print(f"[MLDetector] Accuracy: {acc:.4f}")
        return {"accuracy": acc, "report": report}

    def predict(self, text: str) -> dict:
        if not self._fitted:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        proba = self.pipeline.predict_proba([text])[0]
        scores = {c: round(float(p), 4) for c, p in zip(self.classes_, proba)}
        predicted = max(scores, key=scores.get)
        return {
            "text": text,
            "predicted_emotion": predicted,
            "confidence": scores[predicted],
            "scores": scores,
            "method": "ml",
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def save(self, path: str | Path) -> None:
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required. pip install joblib")
        joblib.dump(self.pipeline, path)
        print(f"[MLDetector] Model saved → {path}")

    def load(self, path: str | Path) -> "MLDetector":
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required. pip install joblib")
        self.pipeline = joblib.load(path)
        self.classes_ = list(self.pipeline.classes_)
        self._fitted = True
        print(f"[MLDetector] Model loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
class TransformerDetector:
    """
    Wraps a HuggingFace model for emotion detection.
    Default: j-hartmann/emotion-english-distilroberta-base
    """

    DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, model_name: str | None = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. pip install transformers torch")
        self.model_name = model_name or self.DEFAULT_MODEL
        print(f"[TransformerDetector] Loading model: {self.model_name} …")
        self._pipe = hf_pipeline("text-classification", model=self.model_name,
                                  return_all_scores=True)
        print("[TransformerDetector] Ready.")

    def predict(self, text: str) -> dict:
        results = self._pipe(text[:512])[0]  # truncate to model max length
        scores = {r["label"].lower(): round(r["score"], 4) for r in results}
        predicted = max(scores, key=scores.get)
        return {
            "text": text,
            "predicted_emotion": predicted,
            "confidence": scores[predicted],
            "scores": scores,
            "method": "transformer",
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
class EmotionDetector:
    """
    High-level facade that selects the backend automatically:
      - 'lexicon'     → fast, no dependencies
      - 'ml'          → requires scikit-learn + a fitted model
      - 'transformer' → requires transformers + torch (most accurate)
      - 'auto'        → tries transformer → ml → lexicon (in that order)
    """

    def __init__(self, method: str = "lexicon", model_path: str | None = None,
                 transformer_model: str | None = None):
        self.method = method.lower()
        self._detector = None

        if self.method == "lexicon":
            self._detector = LexiconDetector()

        elif self.method == "ml":
            self._detector = MLDetector()
            if model_path:
                self._detector.load(model_path)

        elif self.method == "transformer":
            self._detector = TransformerDetector(transformer_model)

        elif self.method == "auto":
            for attempt in ("transformer", "ml", "lexicon"):
                try:
                    self.__init__(attempt, model_path, transformer_model)
                    self.method = attempt
                    break
                except Exception:
                    continue

    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """Analyse a single text and return emotion prediction."""
        if not text or not text.strip():
            return {"text": text, "predicted_emotion": "neutral",
                    "confidence": 1.0, "scores": {}, "method": self.method}
        return self._detector.predict(text)

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Analyse a list of texts."""
        return self._detector.predict_batch(texts)

    def analyse_document(self, text: str, sentence_split: bool = True) -> dict:
        """
        Analyse a multi-sentence document.
        Returns per-sentence results and an overall emotion summary.
        """
        if sentence_split:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        else:
            sentences = [text]

        results = self.predict_batch(sentences)
        emotion_counts: Counter = Counter(r["predicted_emotion"] for r in results)
        dominant = emotion_counts.most_common(1)[0][0]

        return {
            "overall_emotion": dominant,
            "sentence_count": len(sentences),
            "emotion_distribution": dict(emotion_counts),
            "sentence_results": results,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Emotion Detection from Text")
    parser.add_argument("text", nargs="?", help="Text to analyse")
    parser.add_argument("--method", default="lexicon",
                        choices=["lexicon", "ml", "transformer", "auto"])
    parser.add_argument("--model", default=None, help="Path to saved ML model (.joblib)")
    parser.add_argument("--file", default=None, help="Path to a .txt file to analyse")
    args = parser.parse_args()

    detector = EmotionDetector(method=args.method, model_path=args.model)

    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        result = detector.analyse_document(content)
        print(json.dumps(result, indent=2))
    elif args.text:
        result = detector.predict(args.text)
        print(json.dumps(result, indent=2))
    else:
        # Interactive demo
        print("=== Emotion Detector (type 'quit' to exit) ===")
        while True:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            result = detector.predict(user_input)
            print(f"  Emotion  : {result['predicted_emotion'].upper()}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Scores   : {result['scores']}")
