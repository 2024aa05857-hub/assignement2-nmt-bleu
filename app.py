"""
BITS - Assignment 2: Neural Machine Translation with BLEU Evaluation
Part 1 - Task A: NMT + BLEU evaluation app (News/Journalism domain)

UI: Streamlit
- Input source text
- Upload / paste reference translation(s)
- Choose model (MarianMT / Helsinki-NLP)
- Generate 1..K candidate translations (beam search)
- Compute BLEU, brevity penalty, modified n-gram precisions
- Display n-gram precision table (1-4 grams) + overall BLEU

Run:
  streamlit run app.py
"""

import io
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import streamlit as st

# Transformers for MarianMT
import torch
from transformers import MarianMTModel, MarianTokenizer

# BLEU helper tokenizer (WMT style) + reference BLEU
import sacrebleu


# -----------------------------
# BLEU implementation (didactic)
# -----------------------------
@dataclass
class BleuBreakdown:
    bleu: float
    bp: float
    precisions: List[float]           # p1..p4 in %
    clipped_matches: List[int]        # for n=1..4
    total_ngrams: List[int]           # for n=1..4
    ref_len: int
    cand_len: int


def _sacre_tokenize(text: str) -> List[str]:
    """
    Use SacreBLEU's '13a' tokenization (common WMT baseline).
    Returns a list of tokens.
    """
    # sacrebleu.tokenizers.tokenizer_13a.Tokenizer13a
    tok = sacrebleu.tokenizers.TOKENIZERS["13a"]()
    return tok(text).split()


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu_breakdown(candidate: str, references: List[str], max_n: int = 4) -> BleuBreakdown:
    """
    Computes:
      - modified n-gram precision (clipped)
      - brevity penalty
      - BLEU with uniform weights (1/max_n)
    This is a compact, readable implementation for assignment reporting.
    """
    cand_toks = _sacre_tokenize(candidate.strip())
    ref_tok_lists = [_sacre_tokenize(r.strip()) for r in references if r.strip()]
    if not ref_tok_lists:
        raise ValueError("At least one non-empty reference is required.")

    cand_len = len(cand_toks)

    # Choose reference length: closest length to candidate (ties -> shorter), like BLEU.
    ref_lens = [len(rt) for rt in ref_tok_lists]
    ref_len = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))

    # Brevity penalty
    if cand_len == 0:
        bp = 0.0
    elif cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(ref_len) / float(cand_len))

    clipped_matches = []
    total_ngrams = []
    precisions = []

    for n in range(1, max_n + 1):
        cand_ngrams = _ngram_counts(cand_toks, n)
        total = sum(cand_ngrams.values())
        total_ngrams.append(total)

        # max reference counts for each n-gram
        max_ref_counts: Counter = Counter()
        for rt in ref_tok_lists:
            ref_ngrams = _ngram_counts(rt, n)
            for g, c in ref_ngrams.items():
                if c > max_ref_counts.get(g, 0):
                    max_ref_counts[g] = c

        clipped = 0
        for g, c in cand_ngrams.items():
            clipped += min(c, max_ref_counts.get(g, 0))

        clipped_matches.append(clipped)

        # Avoid division by zero (if candidate shorter than n)
        if total == 0:
            p = 0.0
        else:
            p = clipped / total
        precisions.append(p)

    # Geometric mean with smoothing (method-1 like: add epsilon if any p=0)
    eps = 1e-16
    log_p_sum = 0.0
    for p in precisions:
        log_p_sum += math.log(p if p > 0 else eps)
    geo_mean = math.exp(log_p_sum / max_n)

    bleu = 100.0 * bp * geo_mean
    precisions_pct = [100.0 * p for p in precisions]

    return BleuBreakdown(
        bleu=bleu,
        bp=bp,
        precisions=precisions_pct,
        clipped_matches=clipped_matches,
        total_ngrams=total_ngrams,
        ref_len=ref_len,
        cand_len=cand_len,
    )


# -----------------------------
# NMT (MarianMT) helper
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device


def translate_k(
    text: str,
    model_name: str,
    num_return_sequences: int = 1,
    num_beams: int = 4,
    max_new_tokens: int = 128,
    length_penalty: float = 1.0,
) -> List[str]:
    tok, model, device = load_model(model_name)
    batch = tok([text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **batch,
            num_beams=max(num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            length_penalty=length_penalty,
        )
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for d in decoded:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NMT + BLEU Evaluator", layout="wide")
st.title("Neural Machine Translation (Transformer) + BLEU Evaluation")
st.caption("Domain: News/Journalism translation | Models: MarianMT / Helsinki-NLP | Metric: BLEU (with BP + modified n-gram precisions)")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Choose MarianMT model",
        [
            # Popular Helsinki-NLP models
            "Helsinki-NLP/opus-mt-en-hi",
            "Helsinki-NLP/opus-mt-hi-en",
            "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-de-en",
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-fr-en",
        ],
        index=0,
        help="Pick a language pair. You can add more MarianMT models if needed."
    )
    k = st.slider("Number of candidate translations (K)", 1, 5, 3)
    beams = st.slider("Beam size", 1, 8, 4)
    length_penalty = st.slider("Length penalty", 0.6, 1.4, 1.0, 0.1)
    max_new_tokens = st.slider("Max new tokens", 32, 256, 128, 16)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Input Source Text")
    src = st.text_area(
        "Enter source sentence/paragraph (news style)",
        height=170,
        placeholder="Example: The central bank raised interest rates to curb inflation amid global uncertainty."
    )

    st.subheader("2) Reference Translation(s)")
    st.write("Upload a reference translation file (txt) **or** paste one or more references (one per line).")
    uploaded = st.file_uploader("Upload reference .txt", type=["txt"])
    refs_text = st.text_area(
        "Paste reference translation(s), one per line",
        height=140,
        placeholder="Reference 1...\nReference 2 (optional)..."
    )

    references: List[str] = []
    if uploaded is not None:
        try:
            references = uploaded.read().decode("utf-8").splitlines()
        except Exception:
            st.error("Could not read uploaded file. Please upload a UTF-8 .txt file.")
    else:
        references = refs_text.splitlines() if refs_text.strip() else []

    st.info("Tip: Provide 2-3 references for more stable BLEU (optional).")

    run = st.button("Translate + Evaluate", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")
    if run:
        if not src.strip():
            st.error("Please enter source text.")
        elif not any(r.strip() for r in references):
            st.error("Please provide at least one reference translation.")
        else:
            with st.spinner("Generating translations... (first run may download the model)"):
                cands = translate_k(
                    src.strip(),
                    model_name=model_name,
                    num_return_sequences=k,
                    num_beams=beams,
                    max_new_tokens=max_new_tokens,
                    length_penalty=length_penalty,
                )

            st.success(f"Generated {len(cands)} candidate translation(s).")

            # Evaluate each candidate
            rows = []
            breakdowns: List[BleuBreakdown] = []
            for i, cand in enumerate(cands, 1):
                b = compute_bleu_breakdown(cand, references, max_n=4)
                breakdowns.append(b)
                rows.append(
                    {
                        "Candidate": f"C{i}",
                        "BLEU": round(b.bleu, 2),
                        "BP": round(b.bp, 4),
                        "cand_len": b.cand_len,
                        "ref_len": b.ref_len,
                        "p1(%)": round(b.precisions[0], 2),
                        "p2(%)": round(b.precisions[1], 2),
                        "p3(%)": round(b.precisions[2], 2),
                        "p4(%)": round(b.precisions[3], 2),
                    }
                )

            # Best candidate by BLEU
            best_idx = max(range(len(breakdowns)), key=lambda i: breakdowns[i].bleu)

            st.markdown("### Best Candidate (by BLEU)")
            st.code(cands[best_idx], language=None)

            st.markdown("### Candidate-wise BLEU Summary")
            st.dataframe(rows, use_container_width=True)

            st.markdown("### Detailed n-gram precision table (Best candidate)")
            b = breakdowns[best_idx]
            table = []
            for n in range(1, 5):
                table.append(
                    {
                        "n-gram": f"{n}-gram",
                        "clipped_matches": b.clipped_matches[n-1],
                        "total_ngrams": b.total_ngrams[n-1],
                        "modified_precision(%)": round(b.precisions[n-1], 2),
                    }
                )
            st.table(table)

            st.markdown("### Final BLEU computation details (Best candidate)")
            st.write(
                {
                    "BLEU": round(b.bleu, 4),
                    "Brevity Penalty (BP)": round(b.bp, 6),
                    "Candidate length": b.cand_len,
                    "Reference length (closest)": b.ref_len,
                    "Weights": "uniform (0.25 each for 1-4 grams)",
                    "Tokenization": "SacreBLEU 13a",
                }
            )

            # Cross-check with SacreBLEU (for sanity)
            sb = sacrebleu.corpus_bleu([cands[best_idx]], [references], tokenize="13a")
            st.caption(f"SacreBLEU cross-check (tokenize=13a): BLEU={sb.score:.4f} | BP={sb.bp:.4f} | precisions={sb.precisions}")

    else:
        st.write("Enter source text and reference translation, then click **Translate + Evaluate**.")
