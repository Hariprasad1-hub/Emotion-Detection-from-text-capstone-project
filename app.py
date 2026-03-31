"""
Emotion Detection Web App
==========================
A lightweight Flask REST API + simple HTML front-end.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, request, jsonify, render_template_string
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from emotion_detector import EmotionDetector

app = Flask(__name__)
detector = EmotionDetector(method="lexicon")  # switch to "ml" or "transformer" as needed

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Emotion Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; min-height: 100vh;
           display: flex; align-items: center; justify-content: center; }
    .card { background: white; border-radius: 16px; padding: 2.5rem; max-width: 680px;
            width: 95%; box-shadow: 0 4px 24px rgba(0,0,0,.08); }
    h1 { font-size: 1.8rem; color: #1a202c; margin-bottom: .4rem; }
    p.sub { color: #718096; margin-bottom: 1.6rem; }
    textarea { width: 100%; height: 130px; border: 2px solid #e2e8f0; border-radius: 10px;
               padding: .9rem; font-size: 1rem; resize: vertical; transition: border .2s; }
    textarea:focus { outline: none; border-color: #667eea; }
    button { margin-top: 1rem; width: 100%; padding: .85rem; background: #667eea;
             color: white; border: none; border-radius: 10px; font-size: 1rem;
             cursor: pointer; transition: background .2s; }
    button:hover { background: #5a67d8; }
    #result { margin-top: 1.5rem; display: none; }
    .emotion-label { font-size: 2rem; font-weight: 700; text-transform: capitalize; }
    .confidence { color: #718096; margin: .3rem 0 1rem; }
    .bar-container { margin: .35rem 0; }
    .bar-label { display: flex; justify-content: space-between; font-size: .85rem; color: #4a5568; }
    .bar-bg { background: #edf2f7; border-radius: 99px; height: 10px; margin-top: 3px; }
    .bar-fill { height: 10px; border-radius: 99px; transition: width .5s; }
    .emoji { font-size: 3rem; margin-bottom: .5rem; }
    .error { color: #e53e3e; margin-top: 1rem; }
    .method-badge { display: inline-block; background: #ebf8ff; color: #2b6cb0;
                    border-radius: 99px; padding: .2rem .7rem; font-size: .8rem; margin-top: .5rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1> Emotion Detector</h1>
    <p class="sub">Analyse the emotion behind any text using NLP.</p>
    <textarea id="inputText" placeholder="Type or paste your text here…"></textarea>
    <button onclick="analyse()">Detect Emotion</button>

    <div id="result"></div>
    <p id="error" class="error"></p>
  </div>

  <script>
    const COLORS = {
      joy: '#f6ad55', sadness: '#63b3ed', anger: '#fc8181',
      fear: '#9f7aea', surprise: '#68d391', disgust: '#a0aec0', neutral: '#cbd5e0'
    };
    const EMOJIS = {
      joy, sadness, anger, fear,
      surprise, disgust, neutral
    };

    async function analyse() {
      const text = document.getElementById('inputText').value.trim();
      document.getElementById('error').textContent = '';
      document.getElementById('result').style.display = 'none';

      if (!text) {
        document.getElementById('error').textContent = 'Please enter some text first.';
        return;
      }
      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        renderResult(data);
      } catch (e) {
        document.getElementById('error').textContent = 'Error: ' + e.message;
      }
    }

    function renderResult(data) {
      const emotion = data.predicted_emotion;
      const scores  = data.scores;
      const sorted  = Object.entries(scores).sort((a, b) => b[1] - a[1]);

      let bars = sorted.map(([e, v]) => `
        <div class="bar-container">
          <div class="bar-label"><span>${e}</span><span>${(v*100).toFixed(1)}%</span></div>
          <div class="bar-bg"><div class="bar-fill"
               style="width:${v*100}%;background:${COLORS[e]||'#a0aec0'}"></div></div>
        </div>`).join('');

      document.getElementById('result').innerHTML = `
        <div class="emoji">${EMOJIS[emotion] || }</div>
        <div class="emotion-label" style="color:${COLORS[emotion]||'#1a202c'}">${emotion}</div>
        <p class="confidence">Confidence: ${(data.confidence*100).toFixed(1)}%</p>
        <span class="method-badge">Method: ${data.method}</span>
        <div style="margin-top:1.2rem">${bars}</div>`;
      document.getElementById('result').style.display = 'block';
    }

    document.getElementById('inputText').addEventListener('keydown', e => {
      if (e.ctrlKey && e.key === 'Enter') analyse();
    });
  </script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = detector.predict(text)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Provide a 'texts' list"}), 400
    results = detector.predict_batch(texts)
    return jsonify(results)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "method": detector.method})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
