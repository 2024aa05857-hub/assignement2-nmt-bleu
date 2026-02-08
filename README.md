# Assignment 2 - Neural Machine Translation with BLEU Evaluation

## Overview
This repository contains a Streamlit application that:
1) Translates user-provided source text using a Transformer-based NMT model (MarianMT / Helsinki-NLP).
2) Evaluates translation quality using BLEU, including:
   - brevity penalty (BP)
   - modified n-gram precisions (1-gram to 4-gram)
   - candidate-wise evaluation for multiple generated translations

Domain focus: News / Journalism style sentences.

---

## Local Setup

### 1) Create environment (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the app
```bash
streamlit run app.py
```

The first run will download the selected MarianMT model weights.

---

## How to use
1) Select the language pair model from the sidebar (e.g., `Helsinki-NLP/opus-mt-en-hi`).
2) Paste a news-style source sentence/paragraph.
3) Provide reference translation(s):
   - Upload a UTF-8 `.txt` file with one reference per line, OR
   - Paste one or more reference translations (one per line).
4) Click **Translate + Evaluate**.
5) The app will show:
   - the best candidate translation
   - BLEU + BP + n-gram precision table
   - summary table for all candidate translations

---

## Notes
- Tokenization is performed using SacreBLEU `13a` tokenizer (WMT-style).
- BLEU is computed with a clear, didactic implementation (BP + clipped precision + geometric mean).
- SacreBLEU is also shown as a cross-check for the best candidate.

