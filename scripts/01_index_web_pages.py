from __future__ import annotations

from src.indexing.index_build import index_web


if __name__ == "__main__":
    stats = index_web()
    print(stats)
