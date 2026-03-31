"""
train_model.py
==============
Train and evaluate the TF-IDF + Logistic Regression emotion classifier.

Usage:
    python train_model.py --data data/emotions.csv --output models/emotion_model.joblib

CSV format expected:
    text,label
    "I am so happy today!",joy
    "This is terrible news.",sadness
    ...
"""

import argparse
import csv
import random
import sys
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from emotion_detector import MLDetector, LexiconDetector, EMOTION_LEXICON


# ── Synthetic data generator (fallback when no CSV is provided) ───────────────
TEMPLATES: dict[str, list[str]] = {
    "joy": [
        "I am so {w} today!",
        "This is absolutely {w}!",
        "Feeling {w} and grateful.",
        "What a {w} day this has been.",
        "I can't stop smiling, everything is {w}!",
        "Life is {w} right now.",
    ],
    "sadness": [
        "I feel so {w} and alone.",
        "This is such a {w} situation.",
        "I can't help feeling {w} about everything.",
        "My heart is {w} right now.",
        "Why does life have to be so {w}?",
        "I'm overwhelmed with {w}.",
    ],
    "anger": [
        "I am absolutely {w} about this!",
        "This makes me so {w}.",
        "How dare they do this? I'm {w}!",
        "I'm so {w} I can barely speak.",
        "This situation is {w} and unfair.",
        "I can't believe how {w} I feel right now.",
    ],
    "fear": [
        "I'm so {w} about what might happen.",
        "The thought of it makes me {w}.",
        "I feel deeply {w} and uncertain.",
        "Something {w} is going to happen, I just know it.",
        "I'm {w} to even think about it.",
        "The situation is making me {w}.",
    ],
    "surprise": [
        "I'm completely {w} by this news!",
        "Wow, that was totally {w}!",
        "Nobody expected this — it's truly {w}.",
        "I'm {w} at how things turned out.",
        "That was an {w} turn of events!",
        "I never saw that coming — truly {w}!",
    ],
    "disgust": [
        "That is absolutely {w}.",
        "I find this whole situation {w}.",
        "The smell / sight of it was {w}.",
        "I'm completely {w} by their behaviour.",
        "How {w} and shameful.",
        "This is {w} beyond words.",
    ],
}

NEUTRAL_SENTENCES = [
    "The weather today is partly cloudy.",
    "I went to the store to buy some milk.",
    "The meeting is scheduled for 3 PM.",
    "She read the book on the table.",
    "The train arrives at platform 4.",
    "He turned off the lights before leaving.",
    "The report needs to be submitted by Friday.",
    "There are five apples in the basket.",
    "The car is parked outside.",
    "We discussed the agenda for next week.",
]


def generate_synthetic_data(samples_per_class: int = 300) -> tuple[list[str], list[str]]:
    """Generate synthetic labelled training data from the lexicon + templates."""
    texts, labels = [], []

    for emotion, words in EMOTION_LEXICON.items():
        templates = TEMPLATES.get(emotion, ["{w}"])
        for _ in range(samples_per_class):
            word = random.choice(words)
            template = random.choice(templates)
            texts.append(template.format(w=word))
            labels.append(emotion)

    # Neutral class
    for _ in range(samples_per_class):
        texts.append(random.choice(NEUTRAL_SENTENCES))
        labels.append("neutral")

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def load_csv_data(path: str) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("text", "").strip()
            l = row.get("label", "").strip().lower()
            if t and l:
                texts.append(t)
                labels.append(l)
    return texts, labels


def train_test_split(texts, labels, test_size=0.2, seed=42):
    random.seed(seed)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = int(len(combined) * (1 - test_size))
    train, test = combined[:split], combined[split:]
    tr_t, tr_l = zip(*train)
    te_t, te_l = zip(*test)
    return list(tr_t), list(tr_l), list(te_t), list(te_l)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument("--data", default=None,
                        help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--output", default="models/emotion_model.joblib",
                        help="Where to save the trained model")
    parser.add_argument("--samples", type=int, default=300,
                        help="Samples per class when using synthetic data")
    args = parser.parse_args()

    # ── Load / generate data
    if args.data:
        print(f"Loading data from: {args.data}")
        texts, labels = load_csv_data(args.data)
    else:
        print(f"No CSV provided. Generating synthetic data ({args.samples} samples/class)…")
        texts, labels = generate_synthetic_data(args.samples)

    print(f"Dataset size : {len(texts)}")
    print(f"Class distribution: {dict(Counter(labels))}")

    # ── Split
    tr_texts, tr_labels, te_texts, te_labels = train_test_split(texts, labels)
    print(f"Train: {len(tr_texts)} | Test: {len(te_texts)}")

    # ── Train
    model = MLDetector()
    model.fit(tr_texts, tr_labels)

    # ── Evaluate
    metrics = model.evaluate(te_texts, te_labels)
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")

    # ── Compare with lexicon baseline
    print("\n--- Lexicon baseline ---")
    lexicon = LexiconDetector()
    correct = sum(
        lexicon.predict(t)["predicted_emotion"] == l
        for t, l in zip(te_texts, te_labels)
    )
    lex_acc = correct / len(te_texts)
    print(f"Lexicon accuracy: {lex_acc:.4f}")
    print(f"ML improvement  : {(metrics['accuracy'] - lex_acc) * 100:+.1f}%")

    # ── Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"\nModel saved → {args.output}")


if __name__ == "__main__":
    main()
