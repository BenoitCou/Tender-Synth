"""
Thin wrapper around an already-instantiated ConversationalRetrievalChain.

Responsibilities
----------------
* Take a raw question string.
* Call the QA chain stored in ``st.session_state.qa``.
* Re-package the answer together with the list of source documents
  (PDF page numbers included) and any images attached to those docs.
"""

from typing import Tuple, List, Dict, Any
import streamlit as st

def ask_question(question: str) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Run the RAG chain with question and extract useful metadata.

    Parameters
    ----------
    question : str
        The user query to send to the chain.

    Returns
    -------
    answer : str
        The chain’s textual answer (stripped of trailing whitespace).
    docs_used : list[str]
        Human-readable labels of the source documents, e.g.
        ``["report.pdf (page 3)", "spec.docx"]``.
    images_used : list[dict]
        One dict per image referenced, each containing:
        ``file`` (str), ``page`` (int / str), ``image_bytes`` (bytes).
    """

    # ── Call the ConversationalRetrievalChain already stored in session state ──
    result = st.session_state.qa({"question": question})
    answer = result.get("answer", "")
    docs_used = []
    images_used = []

    # ------------------------Iterate over every source document 
    # returned by the chain and build human-friendly references + image payloads.
    for doc in result.get("source_documents", []):
        file = doc.metadata.get("file")
        if not file:
            continue
        
        if file.lower().endswith(".pdf"):
            # PDFs → keep page number
            page = doc.metadata.get("page", "N/A")
            docs_used.append(f"{file} (page {page})")

        else:
            # Word / Excel / etc. → file name only
            page = doc.metadata.get("page", 1)
            docs_used.append(file)
        
        # Harvest image bytes if the retriever attached any
        if "images" in doc.metadata and doc.metadata["images"]:
            for img_bytes in doc.metadata["images"]:
                images_used.append({
                    "file": file,
                    "page": page,
                    "image_bytes": img_bytes,
                    })
                
    return answer.strip(), docs_used, images_used

