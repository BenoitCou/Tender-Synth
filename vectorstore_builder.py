"""

Build a hybrid FAISS index (text + embedded images) from a folder of
arbitrary documents and provide a helper that runs GPT-4o-mini to create
natural-language captions for the retrieved pictures.
"""

import os
import io
import zipfile
import hashlib
import base64

import fitz
#import pytesseract
import easyocr
from PIL import Image
import numpy as np

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LC_Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

from lexicon import normalize, ALL_TERMS, rerank
from loaders import load_docx, load_excel
from config  import OPENAI_API_KEY
from token_utils import split_text, count_tokens, encoding


client_openai = OpenAI(api_key=OPENAI_API_KEY)
reader = easyocr.Reader(['fr'], gpu=False)

def build_vectorstore(folder_path: str, embeddings, min_width: int = 400, min_height: int = 400, chunk_size: int = 2048, chunk_overlap: int = 200, drawings_threshold: int = 500,):
    """
    Index every file under folder_path into a FAISS vector store.

    Parameters
    ----------
    folder_path : str
        Directory containing PDFs, DOCX, XLS, XLSX files to ingest.
    embeddings : langchain.embeddings.base.Embeddings
        Embedding object with ``embed_documents`` / ``embed_query`` methods.
    min_width, min_height : int, default 400
        Ignore images smaller than this (thumbnails, logos, …).
    chunk_size, chunk_overlap : int
        Parameters passed to RecursiveCharacterTextSplitter.
    drawings_threshold : int
        Heuristic: if a PDF page contains more than this number of vector
        drawings we rasterise the whole page instead of trying to extract
        individual images – cheaper and avoids missing schema diagrams.

    Returns
    -------
    FAISS
        The populated vector store.
    """

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_documents = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        ext = file.lower()

        # ---------- PDF ----------------------------------------------------
        if ext.endswith(".pdf"):
            doc = fitz.open(file_path)

            for page_number, page in enumerate(doc, start=1):
                # 1️. text extraction (OCR fallback)
                text = page.get_text().strip()
                if not text:
                    pix = page.get_pixmap()
                    #text = pytesseract.image_to_string(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    image_np = np.array(image)
                    results = reader.readtext(image_np)
                    text = " ".join([res[1] for res in results]).strip()

                # 2️. image harvesting with dedup + size filter
                images, seen_hashes = [], set()
                if len(page.get_drawings()) > drawings_threshold:
                    images.append(page.get_pixmap(dpi=300).tobytes("png"))

                else:
                    for img in page.get_images(full=True):
                        xref = img[0]
                        data = doc.extract_image(xref)["image"]
                        h = hashlib.sha256(data).hexdigest()
                        if h in seen_hashes:
                            continue
                        with Image.open(io.BytesIO(data)) as im:
                            if im.width < min_width or im.height < min_height:
                                continue
                        images.append(data)
                        seen_hashes.add(h)

                # 3️. split text + create LC_Document objects
                for idx, chunk in enumerate(splitter.split_text(text), start=1):
                    chunk_norm = normalize(chunk)
                    present = [t for t in ALL_TERMS if t in chunk_norm]
                    meta = {"file": file, "page": page_number, "chunk": idx, "present_terms": present}
                    if images and idx == 1:
                        meta["images"] = images
                    all_documents.append(LC_Document(page_content=chunk_norm, metadata=meta))
            doc.close()

        # ---------- DOCX ---------------------------------------------------
        elif ext.endswith(".docx"):
            doc_text = load_docx(file_path)
            images, seen_hashes = [], set()

            with zipfile.ZipFile(file_path) as z:
                for name in (n for n in z.namelist() if n.startswith("word/media/")):
                    data = z.read(name)
                    h = hashlib.sha256(data).hexdigest()

                    if h in seen_hashes:
                        continue

                    with Image.open(io.BytesIO(data)) as im:
                        if im.width < min_width or im.height < min_height:
                            continue

                    images.append(data)
                    seen_hashes.add(h)

            for idx, chunk in enumerate(splitter.split_text(doc_text), start=1):
                chunk_norm = normalize(chunk)
                present = [t for t in ALL_TERMS if t in chunk_norm]
                meta = {"file": file, "page": page_number, "chunk": idx, "present_terms": present}
                
                if images and idx == 1:
                    meta["images"] = images

                all_documents.append(LC_Document(page_content=chunk_norm, metadata=meta))

        # ---------- XLS / XLSX --------------------------------------------
        elif ext.endswith((".xlsx", ".xls")):
            doc_text = load_excel(file_path)
            images = []
            if ext.endswith(".xlsx"):
                seen_hashes = set()
                with zipfile.ZipFile(file_path) as z:
                    for name in (n for n in z.namelist() if n.startswith("xl/media/")):
                        data = z.read(name)
                        h = hashlib.sha256(data).hexdigest()
                        if h in seen_hashes:
                            continue
                        with Image.open(io.BytesIO(data)) as im:
                            if im.width < min_width or im.height < min_height:
                                continue
                        images.append(data)
                        seen_hashes.add(h)

            for idx, chunk in enumerate(splitter.split_text(doc_text), start=1):
                chunk_norm = normalize(chunk)
                present = [t for t in ALL_TERMS if t in chunk_norm]
                meta = {"file": file, "page": page_number, "chunk": idx, "present_terms": present}
                if images and idx == 1:
                    meta["images"] = images
                all_documents.append(LC_Document(page_content=chunk_norm, metadata=meta))

    # ----------------------- Bulk-insert into FAISS ------------------------
    BATCH_DOCS = 100             
    vector_store = None
    batch = []

    for doc in all_documents:
        batch.append(doc)
        if len(batch) >= BATCH_DOCS:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            batch = []

    if batch:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    return vector_store

def generate_image_descriptions(query: str, vectorstore):
    """
    Retrieve pages relevant to query and let the LLM generate a short
    French title for each unique image.

    The LLM is instructed to return “NONE” for non-relevant pictures; these
    are filtered out.

    Returns
    -------
    list[dict]
        ``[{"image_name": "...", "description": "...", "file": "...", "page": 3,
            "image_bytes": b"...", "text": "page content"}]``
    """

    if vectorstore is None:
        raise ValueError("Le vectorstore fourni est None. Veuillez l'initialiser correctement.")

    # 1️. similarity search + custom lexicon re-rank
    k = len(vectorstore.docstore._dict)
    query_norm   = normalize(query)
    docs_initial = vectorstore.similarity_search(query_norm, k=k)
    relevant_docs = rerank(query_norm, docs_initial)
    
    final_results = []
    seen_hashes = set()
    
    for doc in relevant_docs:
        metadata = doc.metadata

        if "images" in metadata and metadata["images"]:
            for idx, image_bytes in enumerate(metadata["images"]):
                # Skip duplicates across pages
                img_hash = hashlib.sha256(image_bytes).hexdigest()

                if img_hash in seen_hashes:
                    continue

                seen_hashes.add(img_hash)
                
                # Build multi-modal OpenAI payload
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                text = doc.page_content
                prompt_text = (
                    f"Tu réponds toujours en français."
                    "Reponds uniquement si l'image est pertinente par rapport au texte SYNTHESE suivant :\n{query}\n\n"
                    "Si l'image reçue ne semble pas pertinente, reponds uniquement et sans mise en forme par 'NONE'.\n"
                    "Si l'image est pertinente, crée un parallèle entre le texte DESCRIPTION (en majorité), le texte "
                    "SYNTHSESE et ce que tu vois pour donner un titre de quelques mots (pas plus de 25 mots) à l'image." 
                    "Ne me donne rien d'autre que le titre de l'image en une unique phrase.\n"
                    "Voici le texte DESCRIPTION :\n" + text + "\n\n"
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
                
                try:
                    response = client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    description = response.choices[0].message.content.strip()

                except Exception as e:
                    print(f"Erreur lors de la génération de la description pour l'image {metadata.get('file', 'inconnue')} page {metadata.get('page', '1')}: {str(e)}")
                    description = "Erreur"
                
                if description.upper() != "NONE":
                    image_name = f"{metadata.get('file', 'unknown')}_page{metadata.get('page', '1')}_img{idx+1}"
                    final_results.append({
                        "image_name": image_name,
                        "description": description,
                        "file": metadata.get("file", "unknown"),
                        "page": metadata.get("page", 1),
                        "image_bytes": image_bytes,
                        "text": text
                    })
                    
    return final_results

