import joblib
import numpy as np
import os


# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


def load_artifacts():
    """
    Load trained model and vectorizer using joblib
    """

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    return model, vectorizer


def get_probabilities(model, features):
    """
    Return (fake_prob, real_prob) as floats between 0 and 1

    Supports:
    - models with predict_proba()
    - models with decision_function()
    - fallback prediction-only models
    """

    # Case 1: model supports probability output
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(features)[0]

        classes = list(model.classes_)

        fake_idx = classes.index(0) if 0 in classes else 0
        real_idx = classes.index(1) if 1 in classes else 1

        return float(probs[fake_idx]), float(probs[real_idx])

    # Case 2: model supports decision_function (LinearSVC etc.)
    if hasattr(model, "decision_function"):

        score = model.decision_function(features)[0]

        real_prob = float(1 / (1 + np.exp(-score)))
        fake_prob = 1 - real_prob

        return fake_prob, real_prob

    # Case 3: fallback hard prediction
    pred = int(model.predict(features)[0])

    if pred == 1:
        return 0.0, 1.0
    else:
        return 1.0, 0.0


def get_top_words(vectorizer, features, n=15):
    """
    Extract top TF-IDF weighted words
    """

    feature_names = np.array(vectorizer.get_feature_names_out())

    tfidf_scores = features.toarray()[0]

    nonzero_idx = np.where(tfidf_scores > 0)[0]

    if len(nonzero_idx) == 0:
        return []

    top_idx = nonzero_idx[
        np.argsort(tfidf_scores[nonzero_idx])[::-1][:n]
    ]

    return [
        (feature_names[i], float(tfidf_scores[i]))
        for i in top_idx
    ]


def predict(text: str) -> dict:
    """
    Run prediction pipeline

    Returns dictionary:

    label
    confidence
    fake_prob
    real_prob
    top_words
    word_count
    """

    if not text.strip():
        raise ValueError("Input text is empty.")

    model, vectorizer = load_artifacts()

    word_count = len(text.split())

    features = vectorizer.transform([text])

    prediction = int(model.predict(features)[0])

    label = "FAKE" if prediction == 0 else "REAL"

    fake_prob, real_prob = get_probabilities(model, features)

    confidence = fake_prob if label == "FAKE" else real_prob

    top_words = get_top_words(vectorizer, features)

    return {
        "label": label,
        "confidence": confidence,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "top_words": top_words,
        "word_count": word_count,
    }