"""TF-IDF 수치 검증과 고정 분할의 검색/분류 실험."""

import argparse
import json
from pathlib import Path
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from bow import NumpyTfidf, SearchIndex, BM25
from preprocess import tokenize, preview

QUERIES = [
    "good acting interesting story",
    "bad boring predictable plot",
    "beautiful visual effect music",
]


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, result):
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def verify(records, output):
    """동일 어휘/토큰/IDF/정규화의 라이브러리 결과와 비교한다."""
    texts = [r["text"] for r in records]
    results = []
    for tf in ("raw", "log"):
        for smooth in (True, False):
            for norm in (None, "l2"):
                manual = NumpyTfidf(tf=tf, smooth_idf=smooth, norm=norm)
                matrix = manual.fit_transform(texts)
                reference = TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    lowercase=False,
                    vocabulary=manual.vocabulary_,
                    smooth_idf=smooth,
                    sublinear_tf=tf == "log",
                    norm=norm,
                )
                expected = reference.fit_transform(texts).toarray()
                error = float(np.max(np.abs(matrix - expected)))
                if error > 1e-6:
                    raise AssertionError(f"TF-IDF 오차 {error}")
                results.append(
                    {
                        "tf": tf,
                        "smooth_idf": smooth,
                        "norm": norm,
                        "shape": list(matrix.shape),
                        "max_absolute_error": error,
                    }
                )
    save(output / "verification.json", results)
    print(json.dumps(results), flush=True)


def classification(records, output):
    """분류용 TF/IDF는 각 Train에서만 학습한다."""
    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records])
    tr, te = train_test_split(
        np.arange(len(records)), test_size=0.2, random_state=42, stratify=labels
    )
    rows = []
    errors = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, tf in zip(axes, ("raw", "log", "double")):
        vectorizer = NumpyTfidf(tf=tf)
        train = vectorizer.fit_transform([texts[i] for i in tr])
        test = vectorizer.transform([texts[i] for i in te])
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
            train, labels[tr]
        )
        prediction = model.predict(test)
        rows.append(
            {
                "tf": tf,
                "train_rows": len(tr),
                "test_rows": len(te),
                "vocabulary": len(vectorizer.vocabulary_),
                "accuracy": float(accuracy_score(labels[te], prediction)),
                "f1": float(f1_score(labels[te], prediction)),
                "confusion_matrix": confusion_matrix(labels[te], prediction).tolist(),
            }
        )
        ConfusionMatrixDisplay.from_predictions(
            labels[te], prediction, ax=ax, colorbar=False
        )
        ax.set_title(tf)
        if tf == "log":
            for pos in np.flatnonzero(prediction != labels[te])[:10]:
                contribution = test[pos] * model.coef_[0]
                order = np.argsort(contribution)
                errors.append(
                    {
                        "id": records[te[pos]]["id"],
                        "row": int(te[pos]),
                        "true": int(labels[te[pos]]),
                        "predicted": int(prediction[pos]),
                        "snippet": preview(texts[te[pos]], 500),
                        "positive_terms": [
                            [str(vectorizer.features_[j]), float(contribution[j])]
                            for j in order[-5:][::-1]
                        ],
                        "negative_terms": [
                            [str(vectorizer.features_[j]), float(contribution[j])]
                            for j in order[:5]
                        ],
                    }
                )
    fig.tight_layout()
    fig.savefig(output / "confusion.png", dpi=150)
    plt.close(fig)
    result = {
        "primary_tf": "log chosen before comparison",
        "split": "stratified80/20 random_state42",
        "classification": rows,
        "misclassified_primary": errors,
    }
    save(output / "classification.json", result)
    save(output / "split.json", {"train": tr.tolist(), "test": te.tolist()})
    print(json.dumps(result), flush=True)


def search(records, output):
    """같은3쿼리로 TF변형과 BM25의 순위를 비교한다."""
    texts = [r["text"] for r in records]
    results = []
    reference = {}
    for tf in ("raw", "log", "double"):
        index = SearchIndex(texts, tf=tf)
        for query in QUERIES:
            hits = index.top(query)
            ids = [records[i]["id"] for i, _ in hits]
            if tf == "log":
                reference[query] = set(ids)
            results.append(
                {
                    "method": tf,
                    "query": query,
                    "hits": [
                        {
                            "score": score,
                            "id": records[i]["id"],
                            "label": records[i]["label"],
                            "snippet": preview(records[i]["text"]),
                        }
                        for i, score in hits
                    ],
                }
            )
    bm = BM25(texts)
    for query in QUERIES:
        hits = bm.top(query)
        results.append(
            {
                "method": "BM25",
                "query": query,
                "hits": [
                    {
                        "score": score,
                        "id": records[i]["id"],
                        "label": records[i]["label"],
                        "snippet": preview(records[i]["text"]),
                    }
                    for i, score in hits
                ],
            }
        )
    for row in results:
        ids = {r["id"] for r in row["hits"]}
        baseline = reference[row["query"]]
        row["top5_jaccard_with_log"] = len(ids & baseline) / len(ids | baseline)
    save(output / "search.json", results)
    print(json.dumps(results), flush=True)


def benchmark(records, output):
    """같은 쿼리의 전 스캔과 전치 목록을 반복 측정한다."""
    texts = [r["text"] for r in records]
    index = SearchIndex(texts)
    rng = np.random.default_rng(42)
    queries = [
        " ".join(rng.choice(index.vectorizer.features_, 3, replace=False))
        for _ in range(20)
    ]
    error = max(
        float(np.max(np.abs(index.scores(q) - index.scores(q, True)))) for q in queries
    )
    if error > 1e-12:
        raise AssertionError("역색인 점수 불일치")
    measured = {}
    for name, flag in (("dense_scan", False), ("inverted", True)):
        for q in queries:
            index.scores(q, flag)
        times = []
        for _ in range(5):
            begin = time.perf_counter()
            for _ in range(10):
                for q in queries:
                    index.scores(q, flag)
            times.append((time.perf_counter() - begin) / 200)
        measured[name] = {
            "seconds_per_query_batches": times,
            "median_seconds": statistics.median(times),
        }
    result = {
        "documents": len(records),
        "vocabulary": len(index.vectorizer.vocabulary_),
        "nonzero_entries": int(np.count_nonzero(index.matrix)),
        "queries": 20,
        "batches": 5,
        "calls_per_batch": 200,
        "maximum_score_error": error,
        "measurements": measured,
        "speed_ratio_dense_over_inverted": measured["dense_scan"]["median_seconds"]
        / measured["inverted"]["median_seconds"],
    }
    save(output / "benchmark.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task", choices=["verify", "classification", "search", "benchmark"]
    )
    parser.add_argument("--data", type=Path, default=Path("data/reviews.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {
        "verify": verify,
        "classification": classification,
        "search": search,
        "benchmark": benchmark,
    }[args.task](load(args.data), args.output)


if __name__ == "__main__":
    main()
