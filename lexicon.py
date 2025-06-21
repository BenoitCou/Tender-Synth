"""
Parse the project-specific controlled vocabulary (`LEXICON_PATH`)
and expose four helper utilities:

* ``normalize(text)``        – replace synonyms by the canonical form.
* ``rerank(query_norm, docs)`` – demote documents that violate disallowed pairs.
* ``parse_lexicon(path)``    – low-level DOCX parser (called once at import).
* ``ConstraintRetriever``    – wrapper that plugs the two steps above into any
                               LangChain ``BaseRetriever``.

The goal is to guarantee consistent wording in the user query and, at the same
time, avoid returning documents that use mutually-exclusive terminology unless
the query explicitly contains both terms.
"""

import re
from typing import Dict, List, Tuple

from docx import Document as DocxDocument
from pydantic import ConfigDict
from langchain.schema import BaseRetriever   

from config import LEXICON_PATH

def parse_lexicon(path: str) -> Tuple[Dict[str, str], List[Tuple[str,str]], List[str]]:
    """
    Read the docx lexicon and return:
      * canonical     : synonym -> canonical form
      * disallowed    : list of pairs (a, b) that should not co-occur
      * all_terms     : every term seen in the document
    """    
    doc            = DocxDocument(path)
    canonical      = {}
    disallowed     = []
    all_terms_set  = set()

    for p in doc.paragraphs:

        line = p.text.strip().lower()
        if not line or line.startswith("lexique"):
            # Skip empty lines or headings
            continue

        if "=" in line:     # synonym line                            
            variants = [v.strip() for v in line.split("=")]
            canon    = variants[0]
            for v in variants:
                canonical[v] = canon
                all_terms_set.add(v)
    
        elif "/" in line:   # disallowed pair            
            parts = [x.strip() for x in line.split("/")]
            if len(parts) >= 2:
                a, b = parts[0], parts[1]
                disallowed.append((a, b))
                all_terms_set.update([a, b])

    return canonical, disallowed, list(all_terms_set)

# Parse once at import time so every module gets the same data ----------------
CANONICAL, DISALLOWED_PAIRS, ALL_TERMS = parse_lexicon(LEXICON_PATH)

# Pre-compile a regex that matches any synonym (longest first to avoid overlap)
_pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(CANONICAL, key=len, reverse=True))) + r")\b",
                      flags=re.IGNORECASE)

def normalize(text: str) -> str:
    """
    Return text with every synonym replaced by its canonical form.

    The match is case-insensitive; output is lower-cased to keep things simple.
    """    
    return _pattern.sub(lambda m: CANONICAL[m.group(0).lower()], text.lower())

def rerank(query_norm: str, docs):
    """
    Re-order docs according to disallowed pairs defined in the lexicon.

    Logic
    -----
    * If the query contains both A and B, we keep the original ranking
      (the user explicitly asked for the full context).
    * Otherwise, for each document:
        - if it contains **B but not A** (when only A was asked, or vice-versa)
          we multiply its score by ``0.3`` (arbitrary penalty).
    * Documents are returned sorted by the penalised score.
    """

     # User mentions both terms → no penalty.
    for a, b in DISALLOWED_PAIRS:
        if a in query_norm and b in query_norm:
            return docs

    scored = []
    for d in docs:
        score = 1.0
        terms = set(d.metadata.get("present_terms", []))
        for a, b in DISALLOWED_PAIRS:
            # Query mentions *one* term, document mentions the *other*
            if a in query_norm and b in terms and a not in terms:
                score *= 0.3
        scored.append((d, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]


class ConstraintRetriever(BaseRetriever):
    """
    Drop-in wrapper for any LangChain retriever that enforces:

    1. Synonym normalisation on the incoming query.
    2. Custom re-ranking based on disallowed term pairs.

    Instantiate with: ``ConstraintRetriever(base=my_retriever)``
    """

    # pydantic model fields ---------------------------------------------------
    base: BaseRetriever
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # synchronous
    def get_relevant_documents(self, query: str):
        q_norm = normalize(query)
        docs   = self.base.get_relevant_documents(q_norm)
        return rerank(q_norm, docs)

    # asynchronous
    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)
