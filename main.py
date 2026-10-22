"""코사인/BM25 상위 문서의 점수/ID/스니펫을 출력한다."""

import argparse
import json
from pathlib import Path
from bow import SearchIndex, BM25
from preprocess import preview


def search(query, records, topk=5, method="log"):
    """(점수, 문서 ID, 스니펫) 목록을 반환한다."""
    texts = [r["text"] for r in records]
    index = BM25(texts) if method == "bm25" else SearchIndex(texts, tf=method)
    return [
        (score, records[i]["id"], preview(records[i]["text"]))
        for i, score in index.top(query, topk)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--method", choices=["raw", "log", "double", "bm25"], default="log"
    )
    parser.add_argument("--data", type=Path, default=Path("data/reviews.jsonl"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.data.read_text().splitlines()]
    for score, identity, snippet in search(args.query, records, args.topk, args.method):
        print(f"{score:.6f}\t{identity}\t{snippet}")


if __name__ == "__main__":
    main()
