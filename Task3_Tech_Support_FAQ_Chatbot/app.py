"""
FAQ Chatbot - CodeAlpha AI Internship (Task 3)
------------------------------------------------
A simple Flask web app that answers tech-support FAQs using
TF-IDF vectorization + cosine similarity (no paid API required).

Run with: python app.py
"""

import json
import os
import re
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

# Similarity threshold below which we fall back to a default reply.
# TF-IDF cosine similarity scores range from 0 (no match) to 1 (identical).
SIMILARITY_THRESHOLD = 0.30

FALLBACK_RESPONSE = (
    "I'm not sure about that — please contact our support team for further help."
)

app = Flask(__name__)


# ----------------------------------------------------------------------
# Load & prepare FAQ data
# ----------------------------------------------------------------------
def load_faqs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    """Lowercase, normalize common variants, and strip punctuation for matching."""
    text = text.lower()
    # Normalize common tech spelling variants so "wifi" and "wi-fi" line up.
    text = re.sub(r"\bwi[\s-]?fi\b", "wifi", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


faqs = load_faqs(FAQ_PATH)
faq_questions_raw = [item["question"] for item in faqs]
faq_questions_clean = [clean_text(q) for q in faq_questions_raw]

# Fit a single TF-IDF vectorizer over all known FAQ questions.
# English stop words are removed since they add noise, not meaning.
vectorizer = TfidfVectorizer(stop_words="english")
faq_vectors = vectorizer.fit_transform(faq_questions_clean)


def find_best_match(user_message):
    """
    Vectorize the user's message with the same fitted TF-IDF vectorizer,
    compute cosine similarity against every FAQ question, and return the
    best match (answer, score, matched_question) if it clears the
    similarity threshold. Otherwise return the fallback response.
    """
    cleaned = clean_text(user_message)

    if not cleaned:
        return FALLBACK_RESPONSE, 0.0, None

    user_vector = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score >= SIMILARITY_THRESHOLD:
        return faqs[best_index]["answer"], float(best_score), faq_questions_raw[best_index]

    return FALLBACK_RESPONSE, float(best_score), None


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please type a question so I can help you."})

    answer, score, matched_question = find_best_match(user_message)

    return jsonify({
        "reply": answer,
        "matched_question": matched_question,
        "confidence": round(score, 3),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
