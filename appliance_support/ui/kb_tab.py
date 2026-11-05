"""Knowledge-base management tab (PDF folder -> FAISS vector DB)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
import knowledge_base
from runtime import RUNTIME


def render() -> None:
    vs = RUNTIME.vectorstore
    st.subheader("Knowledge base (PDF manuals → FAISS embeddings)")
    st.caption(f"Folder: `{config.KB_DIR}` · {knowledge_base._count_docs(vs)} chunks embedded")

    c1, c2 = st.columns([1, 3])
    if c1.button("🔄 Re-sync folder"):
        with st.spinner("Reading PDFs and creating embeddings…"):
            stats = knowledge_base.sync_folder(vs)
        c2.success(f"Added {stats['added']}, updated {stats['updated']}, removed {stats['removed']}, "
                   f"unchanged {stats['skipped']} · {stats['chunks']} new chunks")

    files = st.file_uploader("Upload PDF manuals / FAQs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Save to folder & embed"):
        with st.spinner("Embedding…"):
            n = 0
            for f in files:
                knowledge_base.save_upload(f.name, f.getvalue())
                n += knowledge_base.ingest_file(vs, f.name, f.getvalue())
        st.success(f"Saved {len(files)} file(s) and created {n} embeddings.")

    idx = knowledge_base.indexed_files(vs)
    if idx:
        st.dataframe(pd.DataFrame([{"file": f, "chunks": i["chunks"]} for f, i in sorted(idx.items())]),
                     use_container_width=True, hide_index=True)

    q = st.text_input("Test a search")
    if q:
        for doc in vs.similarity_search(q, k=3):
            st.markdown(f"**{doc.metadata.get('title')}** · `{doc.metadata.get('file')}` p.{doc.metadata.get('page')}")
            st.write(doc.page_content)
