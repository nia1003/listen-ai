"""
label_and_train.py

1. Labels all posts from the DB using an expanded lexicon (ground truth).
2. Evaluates the CURRENT (small lexicon) algorithm against that ground truth.
3. Trains a TF-IDF + Logistic Regression model on the labeled data.
4. Evaluates the new model on a held-out test set.
5. Prints comparison table and inference timing.
6. Saves the trained model to nlp/model.pkl.
"""

import os
import re
import sqlite3
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# Expanded lexicons (ground-truth labelling)
# ─────────────────────────────────────────────
POSITIVE_ZH = {
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒", "最佳", "快速", "順暢",
    "可靠", "滿意", "推薦", "幸福", "感動", "超棒", "好玩", "有趣", "精彩", "推", "愛",
    "美味", "順利", "滿足", "期待", "興奮", "快樂", "享受", "感謝", "厲害", "溫暖",
    "友善", "可愛", "漂亮", "輕鬆", "放鬆", "安心", "不錯", "好吃", "好喝", "完美",
    "成功", "積極", "正向", "愉快", "歡樂", "喜悅", "感恩", "支持", "佩服", "欣賞",
    "嗨", "驚喜", "讚嘆", "美好", "棒棒", "超讚", "很棒", "頗棒", "酷", "帥", "萌",
    "cute", "yay",
}

NEGATIVE_ZH = {
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題", "生氣", "壞掉", "負面",
    "昂貴", "失望", "難過", "煩", "累", "壓力", "焦慮", "崩潰", "爛", "痛", "辛苦",
    "困難", "害怕", "擔心", "麻煩", "後悔", "沮喪", "挫折", "憂鬱", "可惜", "難受",
    "傷心", "悲傷", "哭", "煩惱", "無聊", "乏味", "憤怒", "厭惡", "抱怨", "怨", "恨",
    "怒", "恐懼", "慌張", "緊張", "不滿", "不好", "很差", "爛透", "垃圾", "超差",
    "掰", "哀",
}

NEGATION_ZH = {"不", "沒", "無", "未", "別", "不是", "沒有", "並不", "毫不"}

# ─────────────────────────────────────────────
# Tokeniser (mirrors app.py's logic)
# ─────────────────────────────────────────────
_ALL_LEXICON_TERMS = sorted(
    POSITIVE_ZH | NEGATIVE_ZH | NEGATION_ZH, key=len, reverse=True
)


def _tokenize_cjk_segment(segment: str) -> list:
    tokens = []
    idx = 0
    while idx < len(segment):
        match = ""
        for term in _ALL_LEXICON_TERMS:
            if segment.startswith(term, idx):
                match = term
                break
        if match:
            tokens.append(match)
            idx += len(match)
        else:
            tokens.append(segment[idx])
            idx += 1
    return tokens


def tokenize(text: str) -> list:
    raw_tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for raw in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_tokenize_cjk_segment(raw))
        else:
            tokens.append(raw)
    return tokens


def label_with_expanded_lexicon(text: str) -> str:
    """Label a text using the expanded lexicon (ground truth)."""
    tokens = tokenize(text)
    score = 0
    prev = ["", ""]
    for token in tokens:
        is_negated = any(p in NEGATION_ZH for p in prev)
        if token in POSITIVE_ZH:
            score += -1 if is_negated else 1
        elif token in NEGATIVE_ZH:
            score += 1 if is_negated else -1
        prev = [prev[-1], token]
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# Current (small lexicon) app.py classifier
# ─────────────────────────────────────────────
POSITIVE_OLD = {
    "good", "great", "excellent", "love", "awesome", "happy", "amazing", "nice",
    "best", "positive", "fast", "smooth", "reliable",
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒", "最佳", "正面",
    "快速", "順暢", "可靠", "滿意", "推薦",
}
NEGATIVE_OLD = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs", "issue",
    "issues", "angry", "broken", "negative", "expensive",
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題", "生氣", "壞掉",
    "負面", "昂貴", "失望", "卡頓",
}
NEGATION_OLD = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}

_OLD_LEXICON_TERMS = sorted(
    POSITIVE_OLD | NEGATIVE_OLD | NEGATION_OLD, key=len, reverse=True
)


def _tokenize_cjk_old(segment: str) -> list:
    tokens = []
    idx = 0
    while idx < len(segment):
        match = ""
        for term in _OLD_LEXICON_TERMS:
            if segment.startswith(term, idx):
                match = term
                break
        if match:
            tokens.append(match)
            idx += len(match)
        else:
            tokens.append(segment[idx])
            idx += 1
    return tokens


def classify_old(text: str) -> str:
    raw_tokens = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for raw in raw_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            tokens.extend(_tokenize_cjk_old(raw))
        else:
            tokens.append(raw)

    score = 0
    prev = ["", ""]
    for token in tokens:
        is_negated = any(p in NEGATION_OLD for p in prev)
        if token in POSITIVE_OLD:
            score += -1 if is_negated else 1
        elif token in NEGATIVE_OLD:
            score += 1 if is_negated else -1
        prev = [prev[-1], token]
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    DB_PATH = "/Users/nia/listen-ai/data/listenai.db"
    MODEL_PATH = Path(__file__).parent / "model.pkl"

    print("=" * 60)
    print("Loading posts from database …")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT content FROM posts").fetchall()
    conn.close()

    texts = [r[0] for r in rows]
    print(f"  Loaded {len(texts)} posts")

    # ── Step 1: Label with expanded lexicon ──────────────────────
    print("\nLabelling with expanded lexicon (ground truth) …")
    labels = [label_with_expanded_lexicon(t) for t in texts]
    from collections import Counter
    dist = Counter(labels)
    print(f"  Distribution: {dict(dist)}")

    # ── Step 2: Evaluate OLD algorithm vs ground truth ────────────
    print("\n" + "=" * 60)
    print("Evaluating OLD (small lexicon) algorithm vs expanded-lexicon ground truth")
    old_preds = [classify_old(t) for t in texts]

    old_acc = accuracy_score(labels, old_preds)
    old_macro_f1 = f1_score(labels, old_preds, average="macro", zero_division=0)
    old_per_class = classification_report(labels, old_preds, zero_division=0)
    print(f"\n  Accuracy : {old_acc:.4f}")
    print(f"  Macro F1 : {old_macro_f1:.4f}")
    print("\nPer-class report:")
    print(old_per_class)

    # ── Step 3: Train TF-IDF + LogReg ────────────────────────────
    print("=" * 60)
    print("Training TF-IDF + Logistic Regression …")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=20000,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(X_train, y_train)
    new_preds = pipeline.predict(X_test)

    new_acc = accuracy_score(y_test, new_preds)
    new_macro_f1 = f1_score(y_test, new_preds, average="macro", zero_division=0)
    new_per_class = classification_report(y_test, new_preds, zero_division=0)
    print(f"\n  Accuracy : {new_acc:.4f}")
    print(f"  Macro F1 : {new_macro_f1:.4f}")
    print("\nPer-class report:")
    print(new_per_class)

    # ── Step 4: Save model ────────────────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── Step 5: Comparison table ──────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 52)
    print(f"{'Old (small lexicon)  [all]':<30} {old_acc:>10.4f} {old_macro_f1:>10.4f}")
    # Evaluate new model on full dataset for a fair comparison
    new_preds_all = pipeline.predict(texts)
    new_acc_all = accuracy_score(labels, new_preds_all)
    new_macro_f1_all = f1_score(labels, new_preds_all, average="macro", zero_division=0)
    print(f"{'New (TF-IDF + LogReg)[all]':<30} {new_acc_all:>10.4f} {new_macro_f1_all:>10.4f}")
    print(f"{'New (TF-IDF + LogReg)[test]':<30} {new_acc:>10.4f} {new_macro_f1:>10.4f}")

    # ── Step 6: Inference timing ──────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENCE TIMING (1000 texts)")
    sample = (texts * 1000)[:1000]

    t0 = time.perf_counter()
    _ = [classify_old(t) for t in sample]
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pipeline.predict(sample)
    t_new = time.perf_counter() - t0

    print(f"  Old lexicon : {t_old*1000:.1f} ms  ({t_old/1000*1e6:.2f} µs/text)")
    print(f"  New model   : {t_new*1000:.1f} ms  ({t_new/1000*1e6:.2f} µs/text)")
    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(f"  Speedup     : {speedup:.2f}x {'(new faster)' if speedup > 1 else '(old faster)'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
