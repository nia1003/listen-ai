"""
benchmark_cache.py

Benchmarks the speedup from caching sentiment in the database.

For three dataset sizes (5 000, 100 000, 1 000 000 posts):
  - OLD approach: fetch posts + compute sentiment via lexicon (simulating NLP)
  - NEW approach: fetch posts only (sentiment already stored in DB column)

A separate temporary SQLite file is used; it is removed on exit.
"""

import os
import random
import re
import sqlite3
import time
from pathlib import Path

# ── Lexicon (same as app.py fallback) ──────────────────────────────────────
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


# ── DB helpers ──────────────────────────────────────────────────────────────
PROD_DB = "/Users/nia/listen-ai/data/listenai.db"
BENCH_DB = "/tmp/listenai_benchmark.db"
SIZES = [5_000, 100_000, 1_000_000]
LABELS = ["positive", "neutral", "negative"]


def load_real_posts() -> list:
    conn = sqlite3.connect(PROD_DB)
    rows = conn.execute("SELECT content FROM posts LIMIT 5000").fetchall()
    conn.close()
    return [r[0] for r in rows]


def build_benchmark_db(real_posts: list) -> None:
    if os.path.exists(BENCH_DB):
        os.remove(BENCH_DB)

    conn = sqlite3.connect(BENCH_DB)
    conn.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT,
            author TEXT,
            content TEXT,
            created_at TEXT,
            sentiment TEXT DEFAULT NULL
        )
    """)

    # We'll insert up to max(SIZES) rows in one go, but store sentinel marks
    # so we can SELECT LIMIT N for each benchmark.
    max_size = max(SIZES)
    print(f"  Inserting {max_size:,} rows into benchmark DB …")

    batch = []
    for i in range(max_size):
        content = real_posts[i % len(real_posts)]
        sentiment = random.choice(LABELS)
        batch.append(("bench", "user", content, "2024-01-01T00:00:00Z", sentiment))
        if len(batch) == 10_000:
            conn.executemany(
                "INSERT INTO posts (platform, author, content, created_at, sentiment) VALUES (?,?,?,?,?)",
                batch,
            )
            conn.commit()
            batch = []
    if batch:
        conn.executemany(
            "INSERT INTO posts (platform, author, content, created_at, sentiment) VALUES (?,?,?,?,?)",
            batch,
        )
        conn.commit()
    conn.close()
    print("  Benchmark DB ready.")


def benchmark_size(n: int) -> dict:
    conn = sqlite3.connect(BENCH_DB)

    # ── OLD: fetch content only, then classify each ──────────────
    t0 = time.perf_counter()
    rows = conn.execute(
        "SELECT content FROM posts ORDER BY id LIMIT ?", (n,)
    ).fetchall()
    texts = [r[0] for r in rows]
    for t in texts:
        classify_lexicon(t)
    t_old = time.perf_counter() - t0

    # ── NEW: fetch content + precomputed sentiment ────────────────
    t0 = time.perf_counter()
    rows2 = conn.execute(
        "SELECT content, sentiment FROM posts ORDER BY id LIMIT ?", (n,)
    ).fetchall()
    # Just verify sentiment is present (no classification needed)
    _ = [(r[0], r[1]) for r in rows2]
    t_new = time.perf_counter() - t0

    conn.close()
    return {"n": n, "t_old": t_old, "t_new": t_new}


def main():
    print("=" * 70)
    print("CACHE BENCHMARK")
    print("=" * 70)

    real_posts = load_real_posts()
    print(f"Loaded {len(real_posts)} real posts for seeding.")

    build_benchmark_db(real_posts)

    results = []
    for size in SIZES:
        print(f"\nBenchmarking {size:>9,} posts …", end=" ", flush=True)
        r = benchmark_size(size)
        results.append(r)
        speedup = r["t_old"] / r["t_new"] if r["t_new"] > 0 else float("inf")
        print(f"done  (old={r['t_old']:.2f}s  new={r['t_new']:.2f}s  speedup={speedup:.1f}x)")

    print("\n" + "=" * 70)
    print(f"{'Posts':>12}  {'OLD (s)':>10}  {'NEW (s)':>10}  {'Speedup':>10}  {'OLD ms/post':>12}  {'NEW ms/post':>12}")
    print("-" * 70)
    for r in results:
        n = r["n"]
        speedup = r["t_old"] / r["t_new"] if r["t_new"] > 0 else float("inf")
        print(
            f"{n:>12,}  {r['t_old']:>10.3f}  {r['t_new']:>10.3f}  {speedup:>9.1f}x"
            f"  {r['t_old']/n*1000:>11.3f}  {r['t_new']/n*1000:>11.3f}"
        )
    print("=" * 70)

    # Clean up
    if os.path.exists(BENCH_DB):
        os.remove(BENCH_DB)
    print("\nBenchmark DB removed.")


if __name__ == "__main__":
    main()
