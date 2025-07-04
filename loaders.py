"""
Unified file-to-plain-text helpers used throughout the project.

Supported formats
-----------------
* PDF   – text is extracted with `PyPDFLoader`; if the page is scanned and
              contains no embedded text, we fall back to page-level OCR
              (Tesseract via Pillow).
* DOCX  – single pass through python-docx.
* XLS/XLSX – every sheet is concatenated into one long string; NaNs are
                 replaced with empty strings and excessive whitespace collapsed.

These functions are deliberately kept free of LangChain/Streamlit dependencies
(except `PyPDFLoader` for convenience) so they can be unit-tested in isolation.
"""

import re
import fitz                     
import pandas as pd
#import pytesseract
import easyocr
import numpy as np

from PIL import Image
from docx import Document as DocxDocument
from langchain.document_loaders import PyPDFLoader

reader = easyocr.Reader(['fr'], gpu=False)

def load_pdf(file_path: str) -> str:
    """
    Return the entire textual content of file_path.

    Workflow
    --------
    1. Try `PyPDFLoader` – fast, structured, preserves page order.
    2. If the resulting string is empty → treat the PDF as scanned images:
       iterate over pages, rasterise them, run Tesseract OCR.

    Notes
    -----
    * Page breaks are preserved by joining with ``\n``.
    * The function purposefully prints “OCR” when fallback is triggered; this
      helps diagnose performance hits on large scans.
    """

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs]).strip()
    
    if text:      # embedded text found → done
        return text
    
    else:
        print("OCR fallback for scanned PDF")

        doc         = fitz.open(file_path)
        ocr_texts   = []

        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            #page_text = pytesseract.image_to_string(img)
            image_np = np.array(img)
            results = reader.readtext(image_np)
            page_text = " ".join([res[1] for res in results]).strip()
            ocr_texts.append(page_text)

        text = "\n".join(ocr_texts)
        print(text)
        return text

def load_docx(file_path: str) -> str:
    """
    Extract plain text from a Word ``.docx`` file.

    Each paragraph is joined with a newline.  No styling is preserved.
    """
    doc = DocxDocument(file_path)
    text = "\n".join([p.text for p in doc.paragraphs]).strip()
    return text

def load_excel(file_path: str) -> str:
    """
    Flatten an Excel workbook into a plain-text string.

    * .xlsx → parsed with openpyxl engine.  
    * .xls  → parsed with xlrd engine (legacy).  
    * Each sheet is read into a DataFrame, NaNs are replaced by ``""``,
      then ``df.to_string()`` is run and all excessive whitespace collapsed.

    Returns an empty string on error and prints the exception for debugging.
    """
    try:
        texts = []

        if file_path.lower().endswith('.xlsx'):
            # New-style workbooks (zip archive)

            with pd.ExcelFile(file_path, engine="openpyxl") as xls:
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
                    df.fillna("", inplace=True)
                    texts.append(re.sub(r'\s+', ' ', df.to_string()))

        elif file_path.lower().endswith('.xls'):
            # Legacy BIFF format

            with pd.ExcelFile(file_path, engine="xlrd") as xls:
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet, engine="xlrd")
                    df.fillna("", inplace=True)
                    texts.append(re.sub(r'\s+', ' ', df.to_string()))
        else:
            return ""

        # Concatenate all sheets and collapse whitespace one last time
        text = "\n".join(texts)
        text = re.sub(r'\s+', ' ', text)

    except Exception as e:
        print(f"Erreur lors du chargement du fichier Excel {file_path} : {e}")
    return ""

