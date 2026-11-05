"""Read PDFs in ./kb_docs and create embeddings in FAISS.

Run:  python build_embeddings.py            # sync (only new/changed files are embedded)
      python build_embeddings.py --rebuild  # wipe the index and re-embed everything
      python build_embeddings.py --query "washing machine E21"   # test retrieval afterwards

Add PDF manuals to kb_docs/ or upload them in the admin Knowledge Base tab.
Needs OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import os
import shutil

import config
import knowledge_base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="delete the index first")
    ap.add_argument("--query", help="run a test similarity search after indexing")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")

    pdfs = []
    if os.path.isdir(config.KB_DIR):
        pdfs = [name for name in os.listdir(config.KB_DIR)
                if name.lower().endswith(knowledge_base.SUPPORTED)]
    if not pdfs:
        raise SystemExit(f"No PDF files found in {config.KB_DIR}. Add a PDF and try again.")

    if args.rebuild and os.path.isdir(config.FAISS_DIR):
        shutil.rmtree(config.FAISS_DIR)
        print("Deleted existing FAISS index.")

    vs = knowledge_base.get_vectorstore()

    stats = knowledge_base.sync_folder(vs)
    print("sync:", stats)
    for f, info in sorted(knowledge_base.indexed_files(vs).items()):
        print(f"  {f}: {info['chunks']} chunk(s)")
    print("total chunks:", knowledge_base._count_docs(vs))

    if args.query:
        for d in vs.similarity_search(args.query, k=3):
            print(f"\n[{d.metadata['title']}] (file={d.metadata['file']}, page={d.metadata['page']})")
            print(d.page_content[:300])


if __name__ == "__main__":
    main()
