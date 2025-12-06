from __future__ import annotations

import argparse

from src.agent.pipeline import ask_kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the full Constructor KB (pdf + excel + web) with RAG.")
    parser.add_argument("--q", "--query", dest="query", required=True, help="Your question")
    parser.add_argument("--source", default="all", choices=["all", "pdf", "excel", "web"], help="Source filter")
    parser.add_argument("--topk", type=int, default=12, help="Top-k chunks used for context (answer-mode)")
    args = parser.parse_args()

    res = ask_kb(args.query, source=args.source, top_k=args.topk)
    print("\nQUESTION:", args.query)
    print("SOURCE FILTER:", args.source)
    print("MODE:", res["mode"])
    print("\nANSWER:\n", res["answer"])


if __name__ == "__main__":
    main()
