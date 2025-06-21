# Tender-Synth 📄🔍

**Tender-Synth** is a Streamlit
application that helps French restoration company 
work faster on public tenders.

- **Upload** any mix of PDF, Word (.docx) or Excel (.xls/.xlsx) documents  
  (CCTP, lot list, drawings, price schedules, …)
- **Detect** which lots really match the company’s skills  
- **Summarise** hundreds of pages into a clean Word report
- **Chat** with the documents through a Retrieval-Augmented Generation
  chain
- **Caption** relevant images
- Everything happens in **French** because that is the client’s language 🥖
---

## 1. Quick start

```bash
# Clone and enter the repo
git clone https://github.com/your-org/tender-synth.git
cd tender-synth

# Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# ➊ Add your secrets -------------------------------------
cp .env.example .env          # then edit with your keys

# ➋ Launch the UI ----------------------------------------
streamlit run app.py
