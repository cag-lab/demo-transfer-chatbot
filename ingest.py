"""Build the search index from everything in ./docs (.md, .txt, .pdf, .docx).

Run this again whenever you add, change or remove documents:
    python ingest.py
"""
import shutil
import sys

from rag import DOCS_DIR, INDEX_DIR, build_index

if __name__ == "__main__":
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)  # always rebuild from scratch so deleted docs disappear
    try:
        n = build_index()
    except Exception as e:  # show a plain message instead of a long traceback
        print(f"Ingest failed: {e}")
        sys.exit(1)
    print(f"Indexed {n} passages from {DOCS_DIR} into {INDEX_DIR}")
