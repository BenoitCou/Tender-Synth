# Tender-Synth 📄🔍

**Tender-Synth** is a Streamlit
application that helps French restoration company 
work faster on public tenders.

- **Upload** any mix of PDF, Word (.docx) or Excel (.xls/.xlsx) documents  
  (CCTP, lot list, drawings, price schedules, …)
- **Detect** which lots really match the company’s skills  
- **Summarise** hundreds of pages into a clean Word report, including images with relevant captions
- **Chat** with the documents through a Retrieval-Augmented Generation
  chain that includes sources qnd relevant images
- Everything happens in **French** because that is the client’s language 
---

## 1. Quick start
```bash
# Clone and enter the repo
git clone https://github.com/BenoitCou/tender-synth.git
cd tender-synth

# Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# ➊ Add your secrets -------------------------------------
New-Item -Path .env -ItemType File -Force          # then edit with your keys

# ➋ Launch the UI ----------------------------------------
streamlit run app.py
---
```

## 2. Folder structure
```
├── app.py # Streamlit front-end
├── chat_backend.py # Thin wrapper around the RAG chain
├── summariser.py # High-level orchestration (lots, summaries, docx)
├── vectorstore_builder.py # FAISS + image captioning helpers
├── loaders.py # Format-agnostic text extraction
├── lexicon.py # Controlled vocabulary & custom re-ranker
├── image_utils.py
├── token_utils.py
├── config.py # Centralised settings + .env loader
└── requirements.txt
---
```

## 3. Configuration – `.env` file 🔑

Create a `.env` file at the repository root (already in `.gitignore`) and add:

| Variable name        | Example value                       | Mandatory | Description |
|----------------------|-------------------------------------|-----------|-------------|
| `OPENAI_API_KEY`     | `sk-xxxxxxxxxxxxxxxx`          | ✅ | GPT-4o-mini + embeddings + image captions |
| `MISTRAL_API_KEY`    | `abcdefghijklxxxxxx`          | ✅ | Mistral-small chat + embeddings |
| `CLIENT`             | `Company`                  | ✅ | Company name (shown in prompts) |
| `CLIENT_DESCRIPTION` | *Long French paragraph…*            | ✅ | Expertise of the company, fed to the LLM |
| `LEXICON_PATH`       | `./lexique.docx`                    | ✅ | DOCX file with synonyms & exclusions, specific to the context of the client |

