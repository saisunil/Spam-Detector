"""
Spam Email Detection — Flask Web Application
=============================================
Serves the web UI and exposes prediction + stats API endpoints.
"""

import json
import os

from flask import Flask, render_template, request, jsonify
from spam_detector import predict_message, load_model, load_data, train_model, save_model

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# Global cache
_model = None
_vectorizer = None
_metrics = None


def _ensure_model():
    """Load or train model on first request."""
    global _model, _vectorizer, _metrics

    if _model is not None:
        return

    # Try loading saved model
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        _model, _vectorizer = load_model()
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                _metrics = json.load(f)
        else:
            # Re-compute metrics
            df = load_data()
            _, _, _metrics = train_model(df)
            with open(METRICS_PATH, "w") as f:
                json.dump(_metrics, f)
    else:
        # Train from scratch
        print("No saved model found - training now...")
        df = load_data()
        _model, _vectorizer, _metrics = train_model(df)
        save_model(_model, _vectorizer)
        with open(METRICS_PATH, "w") as f:
            json.dump(_metrics, f)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a message and return the prediction."""
    _ensure_model()
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    result = predict_message(message, _model, _vectorizer)
    return jsonify(result)


@app.route("/stats")
def stats():
    """Return model performance metrics."""
    _ensure_model()
    return jsonify(_metrics)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _ensure_model()
    print("Server running at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
