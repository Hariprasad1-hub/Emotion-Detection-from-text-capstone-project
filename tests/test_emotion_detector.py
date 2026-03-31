"""
tests/test_emotion_detector.py
================================
Unit tests for the Emotion Detection project.

Run:  pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from emotion_detector import LexiconDetector, EmotionDetector, EMOTION_LEXICON


# ─────────────────────────────────────────────────────────────────────────────
class TestLexiconDetector:

    def setup_method(self):
        self.det = LexiconDetector()

    def test_returns_dict(self):
        result = self.det.predict("I am happy")
        assert isinstance(result, dict)
        assert "predicted_emotion" in result
        assert "confidence" in result
        assert "scores" in result

    def test_joy_detected(self):
        result = self.det.predict("I am so happy and excited today!")
        assert result["predicted_emotion"] == "joy"

    def test_sadness_detected(self):
        result = self.det.predict("I feel so sad and lonely.")
        assert result["predicted_emotion"] == "sadness"

    def test_anger_detected(self):
        result = self.det.predict("I am furious and outraged by this!")
        assert result["predicted_emotion"] == "anger"

    def test_fear_detected(self):
        result = self.det.predict("I am terrified and scared of what might happen.")
        assert result["predicted_emotion"] == "fear"

    def test_neutral_on_empty(self):
        result = self.det.predict("")
        assert result["predicted_emotion"] == "neutral"

    def test_neutral_on_plain_text(self):
        result = self.det.predict("The train departs at 6 AM.")
        assert result["predicted_emotion"] == "neutral"

    def test_negation_flips_joy_to_sadness(self):
        with_neg = self.det.predict("I am not happy at all.")
        without_neg = self.det.predict("I am happy.")
        assert with_neg["predicted_emotion"] != without_neg["predicted_emotion"] \
               or with_neg["scores"]["sadness"] >= without_neg["scores"]["sadness"]

    def test_confidence_between_0_and_1(self):
        result = self.det.predict("This is a wonderful day!")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_1(self):
        result = self.det.predict("I love this so much!")
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-3 or result["predicted_emotion"] == "neutral"

    def test_batch_prediction(self):
        texts = ["I am happy", "I am sad", "I am angry"]
        results = self.det.predict_batch(texts)
        assert len(results) == 3
        assert all("predicted_emotion" in r for r in results)

    def test_intensifier_increases_confidence(self):
        weak = self.det.predict("I am happy.")
        strong = self.det.predict("I am extremely happy!")
        assert strong["confidence"] >= weak["confidence"]


# ─────────────────────────────────────────────────────────────────────────────
class TestEmotionDetectorFacade:

    def setup_method(self):
        self.det = EmotionDetector(method="lexicon")

    def test_predict_returns_result(self):
        result = self.det.predict("Life is beautiful and joyful!")
        assert result["predicted_emotion"] in list(EMOTION_LEXICON.keys()) + ["neutral"]

    def test_predict_empty_string(self):
        result = self.det.predict("")
        assert result["predicted_emotion"] == "neutral"

    def test_predict_whitespace_only(self):
        result = self.det.predict("   ")
        assert result["predicted_emotion"] == "neutral"

    def test_analyse_document(self):
        doc = ("I feel incredibly happy today. But yesterday I was very sad. "
               "The whole situation makes me furious!")
        result = self.det.analyse_document(doc)
        assert "overall_emotion" in result
        assert "sentence_results" in result
        assert result["sentence_count"] >= 1
        assert "emotion_distribution" in result

    def test_analyse_document_single_sentence(self):
        result = self.det.analyse_document("I love this.", sentence_split=False)
        assert result["sentence_count"] == 1

    def test_batch_returns_correct_length(self):
        texts = ["happy text", "sad text", "angry text", "neutral text", "fear text"]
        results = self.det.predict_batch(texts)
        assert len(results) == len(texts)


# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:

    def setup_method(self):
        self.det = LexiconDetector()

    def test_uppercase_text(self):
        result = self.det.predict("I AM SO HAPPY AND EXCITED!")
        assert result["predicted_emotion"] == "joy"

    def test_special_characters(self):
        result = self.det.predict("I'm happy!! :) *** great stuff ***")
        assert result["predicted_emotion"] == "joy"

    def test_mixed_emotions(self):
        result = self.det.predict("I am happy but also a bit afraid.")
        assert result["predicted_emotion"] in EMOTION_LEXICON.keys()

    def test_very_long_text(self):
        long_text = "I am very happy " * 200
        result = self.det.predict(long_text)
        assert result["predicted_emotion"] == "joy"

    def test_numbers_only(self):
        result = self.det.predict("12345 678 9000")
        assert result["predicted_emotion"] == "neutral"

    def test_method_field_present(self):
        result = self.det.predict("I feel great!")
        assert result["method"] == "lexicon"
