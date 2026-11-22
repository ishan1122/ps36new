import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
st.set_page_config(page_title="Research Paper Q&A | PS36", page_icon="📚", layout="wide")

SYSTEM = """You are an academic research paper assistant. Answer ONLY using the retrieved excerpts.
Do not use outside knowledge or invent facts, methods, datasets, metrics, or citations.
Explain clearly while preserving academic accuracy. Cite the PDF page numbers supplied.
If the excerpts do not contain enough information, say:
'This information was not found in the uploaded research paper.'"""

PRESETS = {
    "Method": "Explain the research methodology, approach and experimental setup used in this paper.",
    "Dataset": "What dataset or data sources were used? Include size, features and split only if stated.",
    "Results": "Summarize the main experimental results and reported performance metrics.",
    "Limitations": "What limitations, assumptions or future work are explicitly mentioned by the authors?"
}

@st.cache_resource(show_spinner=False)
def make_index(pdf_bytes: bytes, api_key: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        pages = PyPDFLoader(path).load()
        if not pages or not any(p.page_content.strip() for p in pages):
            raise ValueError("No selectable text found. This may be a scanned PDF; OCR is not included.")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, add_start_index=True
        ).split_documents(pages)
        if not chunks:
            raise ValueError("No text chunks could be created.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=api_key
        )
        return FAISS.from_documents(chunks, embeddings), len(pages), len(chunks)
    finally:
        try: os.remove(path)
        except OSError: pass

def ask_paper(question, store, api_key, model):
    docs = store.similarity_search(question, k=3)
    excerpts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else "unknown"
        excerpts.append(f"[Source {i} | PDF page {page_num}]\\n{doc.page_content}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Retrieved paper excerpts:\\n\\n{context}\\n\\nQuestion: {question}\\nAnswer with page citations.")
    ])
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.1)
    result = llm.invoke(prompt.format_messages(context="\\n\\n".join(excerpts), question=question))
    answer = result.content
    if isinstance(answer, list):
        answer = "\\n".join(x.get("text", str(x)) if isinstance(x, dict) else str(x) for x in answer)
    sources, seen = [], set()
    for doc in docs:
        page = doc.metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else None
        key = (page_num, doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            sources.append({"page": page_num, "excerpt": doc.page_content.strip()})
    return str(answer), sources

st.title("📚 Research Paper Q&A")
st.caption("PS36 · AI-powered academic paper analysis using Retrieval-Augmented Generation (RAG)")
st.write("Upload a research paper and ask about its methodology, dataset, results and limitations.")

with st.sidebar:
    st.header("Project information")
    st.markdown("**Project:** Research Paper Q&A")
    st.markdown("**Pipeline:** PDF → chunks → Gemini embeddings → FAISS → Gemini answer")
    st.divider()
    key = st.text_input("Gemini API key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    model = st.text_input("Gemini model", value="gemini-2.5-flash")
    st.caption("Alternatively, set GOOGLE_API_KEY in your local .env file.")

upload = st.file_uploader("Upload research paper (PDF)", type=["pdf"])
if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.signature = None
    st.session_state.messages = []
    st.session_state.paper_info = None

if upload:
    data = upload.getvalue()
    signature = (upload.name, len(data))
    if signature != st.session_state.signature:
        st.session_state.store = None
        st.session_state.messages = []
        st.session_state.signature = signature
        st.session_state.paper_info = None
    if st.session_state.store is None:
        if st.button("Process research paper", type="primary", disabled=not bool(key)):
            try:
                with st.spinner("Extracting text, creating embeddings and indexing the paper..."):
                    store, pages, chunks = make_index(data, key)
                st.session_state.store = store
                st.session_state.paper_info = (upload.name, pages, chunks)
                st.success("Paper processed successfully.")
            except Exception as e:
                st.error(f"Could not process PDF: {e}")
        if not key:
            st.info("Enter your Gemini API key in the sidebar to process the PDF.")
    else:
        name, pages, chunks = st.session_state.paper_info
        st.success(f"Ready: {name} · {pages} pages · {chunks} text chunks")

if st.session_state.store is not None:
    st.subheader("Explore the paper")
    cols = st.columns(4)
    for col, (label, q) in zip(cols, PRESETS.items()):
        if col.button(label, use_container_width=True):
            st.session_state.pending_question = q

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Retrieved source excerpts"):
                    for src in msg["sources"]:
                        st.markdown(f"**PDF page {src['page'] if src['page'] else 'unavailable'}**")
                        st.caption(src["excerpt"])

    typed = st.chat_input("Ask a question about this research paper...")
    question = st.session_state.pop("pending_question", None) or typed
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the paper and generating an answer..."):
                    answer, sources = ask_paper(question, st.session_state.store, key, model)
                st.markdown(answer)
                with st.expander("Retrieved source excerpts"):
                    for src in sources:
                        st.markdown(f"**PDF page {src['page'] if src['page'] else 'unavailable'}**")
                        st.caption(src["excerpt"])
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                error = f"An error occurred while answering: {e}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("Upload a PDF and select Process research paper to begin.")
    st.markdown("**Preset questions:** Method · Dataset · Results · Limitations")

st.divider()
st.caption("Verify important claims against the original paper. Answers may omit information not present in retrieved passages.")
