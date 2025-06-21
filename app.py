"""
Streamlit UI entry-point.
Launch with:
    streamlit run app.py
"""

import os
import io
import tempfile
import summariser
import streamlit as st

# LangChain imports ---------------------------------------------------------
from langchain.memory    import ConversationBufferMemory
from langchain.prompts   import PromptTemplate
from langchain.chains    import ConversationalRetrievalChain
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai    import ChatOpenAI,   OpenAIEmbeddings

# Internal modules ----------------------------------------------------------
from config            import OPENAI_API_KEY, MISTRAL_API_KEY, CLIENT
from summariser        import extract_interesting_lots, summarize_documents, final_summary_and_export
from vectorstore_builder import build_vectorstore
from lexicon           import ConstraintRetriever
from chat_backend      import ask_question
from token_utils       import encoding

# Helper functions ----------------------------------------------------------
def submit_question() -> None:
    """
    Callback fired when the user presses <Enter> in the text-input.

    • Runs the RAG chain (st.session_state.qa) if it exists.  
    • Stores answer, document list, image list into session state.  
    • Clears the input box afterwards.
    """ 

    question = st.session_state.question_input

    if question:
        if st.session_state.qa is not None:
            
            with st.spinner("Le LLM réfléchit..."):
                answer, docs_used, images_used = ask_question(question)

            # Persist chat turn + references
            st.session_state.chat_history.append({"user": question, "response": answer})
            st.session_state.used_docs = docs_used  
            st.session_state.used_images = images_used

        else:
            st.error("Veuillez d'abord déposer et traiter les documents dans la zone de droite.")

        st.session_state.question_input = ""

def initialize_embeddings():
    """
    Return the embedding client selected in the modal.

    Uses the model name stored in *st.session_state.llm_choice*.
    """

    if st.session_state.llm_choice == "🇫🇷 ChatMistralAI (mistral-small-latest)":
        return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    
    elif st.session_state.llm_choice == "🇺🇸 ChatOpenAI (gpt-4o-mini)":
        return OpenAIEmbeddings(chunk_size=1000)
    
    else:
        st.error("Aucun modèle d'embedding n'a été sélectionné.")
        st.stop()

def initialize_llm():
    """
    Instantiate the chat model chosen by the user and
    store token-related limits in session state.
    """
    if st.session_state.llm_choice == "🇫🇷 ChatMistralAI (mistral-small-latest)":

        st.session_state.encoding = encoding
        st.session_state.max_tokens = 29000
        st.session_state.overlap = 1000

        return ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.2,
            mistral_api_key=MISTRAL_API_KEY, 
            timeout=180, 
            max_retries=10
        )
    
    elif st.session_state.llm_choice == "🇺🇸 ChatOpenAI (gpt-4o-mini)":

        st.session_state.encoding = encoding
        st.session_state.max_tokens = 100000
        st.session_state.overlap = 5000

        return ChatOpenAI(
            model_name="gpt-4o-mini", 
            temperature=0.2
            )
    
    else:
        st.error("Aucun modèle LLM n'a été sélectionné.")
        st.stop()

def show_llm_choice_modal() -> None:
    """
    First-run modal asking the user to pick an LLM family (OpenAI vs Mistral).

    On confirmation the choice is stored in session state and the
    page reloads so that `initialize_llm()` and friends can run.
    """
    
    st.markdown("<h2 style='text-align: center;'>Bienvenue 👋</h2>", unsafe_allow_html=True)
    st.write("Veuillez choisir un modèle LLM pour continuer.")

    choice = st.radio(
        "Choisissez le modèle LLM :",
        ["🇫🇷 ChatMistralAI (mistral-small-latest)", "🇺🇸 ChatOpenAI (gpt-4o-mini)"],
        key="llm_choice_modal"
    )

    if st.button("Confirmer"):
        st.session_state.llm_choice             = choice
        st.session_state.llm_initialized        = False  
        st.session_state.embeddings_initialized = False  
        st.rerun()

# Page-wide Streamlit configuration ------------------------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    h1 { font-size: 2rem; }
    h2 { font-size: 1.5rem; }
    h3 { font-size: 1.2rem; }
    a[href^="https://share.streamlit.io"] { display: none; }
    </style>
    """, unsafe_allow_html=True
    )

# -------------------- Session-state initialisation -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "used_docs" not in st.session_state:
    st.session_state.used_docs = []  
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa" not in st.session_state:
    st.session_state.qa = None
if "used_images" not in st.session_state:
    st.session_state.used_images = []
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = None
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if "embeddings_initialized" not in st.session_state:
    st.session_state.embeddings_initialized = False

# ---------------------- LLM & embedding selection --------------------------
if st.session_state.llm_choice is None:
    show_llm_choice_modal()
    st.stop()

if not st.session_state.llm_initialized:
    st.session_state.llm = initialize_llm()
    summariser.set_llm(st.session_state.llm)
    st.session_state.llm_initialized = True

if not st.session_state.embeddings_initialized:
    st.session_state.embeddings = initialize_embeddings()
    st.session_state.embeddings_initialized = True

llm         = st.session_state.llm
embeddings  = st.session_state.embeddings

st.write(f"✅ Modèle sélectionné : {st.session_state.llm_choice}")

#  Page layout (3 columns) ----------------------------------------------------
left_col, center_col, right_col = st.columns([1, 2, 1])

# Right column – uploads ----------------------------------------------------
with right_col:

    st.header("Zone de dépôt des documents")
    st.subheader("Tous les documents")

    all_docs_files = st.file_uploader("Glissez/déposez vos fichiers PDF, Word ou Excel", type=["pdf", "docx", "xlsx", "xls"], accept_multiple_files=True)
    
    st.subheader("Document contenant les lots")

    lots_file = st.file_uploader("Déposez le document des lots (PDF, Word ou Excel)", type=["pdf", "docx", "xlsx", "xls"])
    
    # 1️. Full workflow: build synthesis + RAG chain ------------------ #
    
    if st.button("Créer la synthèse et dialoguer avec les documents"):

        if all_docs_files and lots_file:

            with tempfile.TemporaryDirectory() as tmpdirname:            
                # Persist uploaded files to temporary folder ----------------

                for uploaded_file in all_docs_files:
                    file_path = os.path.join(tmpdirname, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                lots_file_path = os.path.join(tmpdirname, lots_file.name)

                with open(lots_file_path, "wb") as f:
                    f.write(lots_file.getbuffer())

                # NLP pipeline ---------------------------------------------
                interesting_lots = str(extract_interesting_lots(lots_file_path)).replace("{", "[").replace("}", "]")
                summaries = summarize_documents(tmpdirname, interesting_lots)
                vstore = build_vectorstore(tmpdirname, embeddings=st.session_state.embeddings)
                
                st.session_state.vector_store = vstore

                # Build constrained retriever ------------------------------
                retriever = ConstraintRetriever(base=st.session_state.vector_store.as_retriever(search_kwargs={"k": 15}))
                
                # Build Word synthesis document ----------------------------
                synthesis_doc = final_summary_and_export(summaries, st.session_state.vector_store)
                
                # Build ConversationalRetrievalChain -----------------------
                memory = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True, 
                    input_key='question',
                    output_key='answer'
                    )
                template = (
                            f"Tu es l'assistant personnalisé de {CLIENT}\n"
                        +
                            """\
                        Utilise le CONTEXTE pour répondre à la QUESTION. N'utilise pas de savoir externe.
                        Si tu ne sais pas, dis que tu ne sais pas et demande des précisions à l'utilisateur.

                        CONTEXTE : {context}

                        QUESTION : {question}
                        """
                        )
                
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                st.session_state.qa = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
                    output_key='answer',
                    return_source_documents=True
                )

                # Offer synthesis download ---------------------------------
                buffer = io.BytesIO()
                synthesis_doc.save(buffer)
                buffer.seek(0)

                st.success("Synthèse créée avec succès !")
                st.download_button(
                    label="Télécharger la synthèse",
                    data=buffer,
                    file_name="synthese.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        else:
            st.error("Veuillez déposer à la fois les fichiers 'Tous les documents' et le 'Document contenant les lots'.")

    # 2️. Build RAG chain only (no synthesis) --------------------------

    if st.button("Dialoguer avec les documents sans créer la synthèse"):
        
        if all_docs_files and lots_file:
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Persist uploaded files to temporary folder ----------------
                for uploaded_file in all_docs_files:
                    
                    file_path = os.path.join(tmpdirname, uploaded_file.name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                lots_file_path = os.path.join(tmpdirname, lots_file.name)
                
                with open(lots_file_path, "wb") as f:
                    f.write(lots_file.getbuffer())

                # NLP pipeline ---------------------------------------------
                vstore = build_vectorstore(tmpdirname, embeddings=st.session_state.embeddings)
                st.session_state.vector_store = vstore

                # Build constrained retriever ------------------------------
                retriever = ConstraintRetriever(base=st.session_state.vector_store.as_retriever(search_kwargs={"k": 15}))
                
                # Build ConversationalRetrievalChain -----------------------
                memory = ConversationBufferMemory(
                    memory_key="chat_history", 
                    return_messages=True, 
                    input_key='question', 
                    output_key='answer'
                    )
                
                template = (
                            f"Tu es l'assistant personnalisé de {CLIENT}\n"
                        +
                            """\
                        Utilise le CONTEXTE pour répondre à la QUESTION. N'utilise pas de savoir externe.
                        Si tu ne sais pas, dis que tu ne sais pas et demande des précisions à l'utilisateur.

                        CONTEXTE : {context}

                        QUESTION : {question}
                        """
                        )
                
                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                st.session_state.qa = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
                    output_key='answer',
                    return_source_documents=True
                )
        else:
            st.error("Veuillez déposer à la fois les fichiers 'Tous les documents' et le 'Document contenant les lots'.")


# Left column – sources used in answers ------------------------------------

with left_col:

    st.header("Documents utilisés pour répondre")
    if st.session_state.used_docs:
        for doc_info in st.session_state.used_docs:
            st.write(f"• {doc_info}")
    else:
        st.write("Aucun document utilisé pour le moment.")

    st.header("Images présentes dans les documents utilisés pour répondre")
    if st.session_state.used_images:
        for image in st.session_state.used_images:
            caption = f"{image['file']} (page {image['page']})"
            st.image(image["image_bytes"], caption=caption)
    else:
        st.write("Aucune image utilisée pour le moment.")

# Centre column – chat interface -------------------------------------------

with center_col:

    st.header("Dialogue avec les documents")
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            st.chat_message("user").write(entry["user"])
            st.chat_message("assistant").write(entry["response"])
    
    # Input field: <Enter> triggers submit_question()
    st.text_input(
        "Posez votre question :", 
        key="question_input",
        on_change=submit_question
    )

