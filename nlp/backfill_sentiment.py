"""
backfill_sentiment.py

Reads all posts without a sentiment value from the production DB,
classifies them using the trained ML model (or lexicon fallback),
and bulk-updates the sentiment column.
"""

import sqlite3
import sys
import time
from pathlib import Path

# Try to load the trained model
_model = None
try:
    import joblib
    model_path = Path(__file__).parent / "model.pkl"
    _model = joblib.load(model_path)
    print(f"[backfill] Using ML model: {model_path}")
except Exception as exc:
    print(f"[backfill] ML model unavailable ({exc}); falling back to lexicon")

# Lexicon fallback (same as app.py)
import re

POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "awesome", "happy", "amazing", "nice",
    "best", "positive", "fast", "smooth", "reliable",
    "好", "很好", "優秀", "喜歡", "讚", "開心", "高興", "棒", "最佳", "正面",
    "快速", "順暢", "可靠", "滿意", "推薦",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "slow", "bug", "bugs", "issue",
    "issues", "angry", "broken", "negative", "expensive",
    "差", "糟糕", "很糟", "討厭", "最差", "慢", "錯誤", "問題", "生氣", "壞掉",
    "負面", "昂貴", "失望", "卡頓",
}
NEGATION_WORDS = {"not", "never", "no", "hardly", "不", "沒", "無", "未", "別", "不是"}
ALL_TERMS = sorted(POSITIVE_WORDS | NEGATIVE_WORDS | NEGATION_WORDS, key=len, reverse=True)


def _tokenize_cjk(segment: str) -> list:
    tokens, idx = [], 0
    while idx < len(segment):
        matched = ""
        for term in ALL_TERMS:
            if segment.startswith(term, idx):
                matched = term
                break
        if matched:
            tokens.append(matched)
            idx += len(matched)
        else:
            tokens.append(segment[idx])
            idx += 1
    return tokens


def classify_lexicon(text: str) -> str:
    raw = re.findall(r"[a-zA-Z']+|[\u4e00-\u9fff]+", text.lower())
    tokens = []
    for r in raw:
        if re.fullmatch(r"[\u4e00-\u9fff]+", r):
            tokens.extend(_tokenize_cjk(r))
        else:
            tokens.append(r)
    score, prev = 0, ["", ""]
    for tok in tokens:
        neg = any(p in NEGATION_WORDS for p in prev)
        if tok in POSITIVE_WORDS:
            score += -1 if neg else 1
        elif tok in NEGATIVE_WORDS:
            score += 1 if neg else -1
        prev = [prev[-1], tok]
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def classify(texts: list) -> list:
    if _model is not None:
        return list(_model.predict(texts))
    return [classify_lexicon(t) for t in texts]


def backfill(db_path: str, batch_size: int = 500) -> None:
    conn = sqlite3.connect(db_path)

    # Count posts that need backfill
    total = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE sentiment IS NULL"
    ).fetchone()[0]
    print(f"[backfill] Posts without sentiment: {total}")

    if total == 0:
        print("[backfill] Nothing to do.")
        conn.close()
        return

    updated = 0
    t0 = time.perf_counter()

    while True:
        rows = conn.execute(
            "SELECT id, content FROM posts WHERE sentiment IS NULL LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        labels = classify(texts)

        conn.executemany(
            "UPDATE posts SET sentiment = ? WHERE id = ?",
            [(label, pid) for label, pid in zip(labels, ids)],
        )
        conn.commit()
        updated += len(rows)
        elapsed = time.perf_counter() - t0
        rate = updated / elapsed if elapsed > 0 else 0
        print(
            f"  {updated:>6}/{total}  ({rate:.0f} posts/s)",
            end="\r",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    print(f"\n[backfill] Done. Updated {updated} posts in {elapsed:.2f}s "
          f"({updated/elapsed:.0f} posts/s)")
    conn.close()


if __name__ == "__main__":
    DB_PATH = "/Users/nia/listen-ai/data/listenai.db"
    print(f"[backfill] Target DB: {DB_PATH}")
    backfill(DB_PATH)

    # Also backfill the worktree DB
    WORKTREE_DB = "/Users/nia/listen-ai/.claude/worktrees/funny-lumiere/data/listenai.db"
    print(f"\n[backfill] Target DB: {WORKTREE_DB}")
    backfill(WORKTREE_DB)
