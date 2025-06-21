"""
Shared utilities for accurate token counting and for splitting long
passages into sliding-window chunks that respect a user-defined token
budget.

Both the Streamlit UI and the backend summariser rely on these
helpers so we keep them in a tiny, self-contained module.
"""

from typing import List

import tiktoken
import streamlit as st

encoding = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, encoding=encoding) -> int:
    """
    Return the exact number of BPE tokens in text.

    Parameters
    ----------
    text : str
        The input string to measure.
    encoding : tiktoken.Encoding, optional
        Pre-initialised encoder; defaults to the module-level ``encoding``.
    """
    return len(encoding.encode(text))

def split_text(text: str, max_tokens: int | None = None, overlap: int | None = None, encoding= None) -> List[str]:
    """
    Slice text into overlapping chunks that never exceed max_tokens.

    Parameters
    ----------
    text : str
        Input document.
    max_tokens : int, optional
        Chunk size; defaults to ``st.session_state.max_tokens``.
    overlap : int, optional
        Number of tokens to re-use from the previous chunk; defaults to
        ``st.session_state.overlap``.
    encoding : tiktoken.Encoding, optional
        Tokeniser; defaults to ``st.session_state.encoding`` or the module
        default.

    Returns
    -------
    list[str]
        Ordered list of decoded string chunks covering the entire input.
    """

    # Default values picked up from Streamlit session (set in app.py)
    if max_tokens is None:
        max_tokens = st.session_state.max_tokens

    if overlap is None:
        overlap = st.session_state.overlap

    if encoding is None:
        encoding = st.session_state.encoding

    tokens  = encoding.encode(text)
    chunks  = []
    start   = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start = end - overlap
        
    return chunks

