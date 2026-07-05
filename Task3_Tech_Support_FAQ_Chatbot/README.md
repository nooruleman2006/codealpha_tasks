# TechHelp FAQ Chatbot — CodeAlpha AI Internship (Task 3)

A lightweight Flask web app that answers common tech-support questions
(password resets, Wi-Fi issues, slow computers, printers, viruses, backups,
browser problems, and basic hardware troubleshooting) using **NLP-based
question matching** — no paid API required.

---

## How it works

1. **FAQ knowledge base** — All questions and answers live in
   `data/faqs.json` as simple `{"question": ..., "answer": ...}` pairs.

2. **Text matching (TF-IDF + Cosine Similarity)**
   - On startup, `app.py` loads every FAQ question and cleans it
     (lowercasing, stripping punctuation).
   - A `TfidfVectorizer` (from scikit-learn) is fit on all FAQ questions,
     turning each one into a vector that reflects which words are
     distinctive/important across the dataset.
   - When a user sends a message, it's cleaned the same way and
     transformed into a vector using the **same fitted vectorizer**.
   - **Cosine similarity** is computed between the user's message vector
     and every FAQ question vector — this gives a score from 0 (completely
     different) to 1 (identical wording/meaning of words used).
   - The FAQ with the **highest similarity score** is selected as the
     candidate answer.

3. **Similarity threshold & fallback**
   - If the best match's score is **below 0.30** (configurable in
     `app.py` via `SIMILARITY_THRESHOLD`), the bot doesn't trust the
     match and instead replies:
     > "I'm not sure about that — please contact our support team for
     > further help."
   - This prevents the bot from confidently giving a wrong answer to an
     unrelated question.

4. **Flask backend**
   - `GET /` — serves the chat UI (`templates/index.html`).
   - `POST /api/chat` — accepts `{ "message": "<user text>" }` as JSON,
     runs the matching logic, and returns
     `{ "reply": "...", "matched_question": "...", "confidence": 0.42 }`.

5. **Frontend**
   - A chat-bubble interface built with plain HTML/CSS/JS (no frameworks).
   - Includes a dark/light mode toggle, quick-topic shortcut buttons, and
     a typing indicator while waiting for a response.
   - JavaScript (`static/script.js`) sends the user's message to
     `/api/chat` via `fetch()` and renders the bot's reply as a new chat
     bubble.

---

## Project structure

```
faq_chatbot/
├── app.py                 # Flask app + TF-IDF matching logic
├── requirements.txt
├── README.md
├── data/
│   └── faqs.json           # 20 tech-support Q&A pairs
├── templates/
│   └── index.html          # Chat UI markup
└── static/
    ├── style.css            # Chat bubble styling, dark/light theme
    └── script.js            # Chat interactions, theme toggle, fetch calls
```

---

## Setup & run locally

**1. Clone / copy the project folder, then create a virtual environment
(recommended):**

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run the app:**

```bash
python app.py
```

**4. Open your browser at:**

```
http://127.0.0.1:5000
```

You should see the chat interface. Try asking things like:

- "How do I reset my password?"
- "My Wi-Fi keeps disconnecting"
- "Why is my laptop so slow?"
- "My printer says offline"
- "I think I have a virus"
- "How do I back up my files?"

Ask something unrelated (e.g. "what's the weather today?") to see the
fallback response in action.

---

## Customizing the FAQ dataset

Simply edit `data/faqs.json` and add more `{"question": "...", "answer": "..."}`
objects. No retraining step is needed — the TF-IDF vectorizer is refit
automatically each time the Flask app starts.

## Adjusting match sensitivity

If the bot feels too strict (giving the fallback too often) or too loose
(matching unrelated questions), adjust `SIMILARITY_THRESHOLD` in `app.py`:

- Lower it (e.g. `0.2`) → more lenient matching.
- Raise it (e.g. `0.4`) → stricter matching, more fallbacks.

---

## Tech stack

- **Backend:** Python, Flask
- **NLP:** scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Data:** JSON file (no database required)

Built for the **CodeAlpha AI Internship — Task 3 (FAQ Chatbot)**.
