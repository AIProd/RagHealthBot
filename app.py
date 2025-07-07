# ──────────────────────────────────────────────────────────────────────────────
# app.py · Healthcare Research Bot (RAG) · Streamlit
# Author: Mohit S
# ──────────────────────────────────────────────────────────────────────────────
#   pip install streamlit langchain langchain-community langchain-openai
#               faiss-cpu pypdf tiktoken cryptography>=42
#
#   Put your Azure OpenAI creds in .streamlit/secrets.toml (or env vars):
#     AZURE_ENDPOINT = "https://<name>.openai.azure.com/"
#     AZURE_API_KEY  = "sk-..."
#     API_VERSION    = "2024-12-01-preview"
# ──────────────────────────────────────────────────────────────────────────────
import os, base64, tempfile, textwrap
from pathlib import Path
from typing import List, Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Secrets / environment
# ──────────────────────────────────────────────────────────────────────────────
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", os.getenv("AZURE_ENDPOINT"))
AZURE_API_KEY  = st.secrets.get("AZURE_API_KEY",  os.getenv("AZURE_API_KEY"))
API_VERSION    = st.secrets.get("API_VERSION",    os.getenv("API_VERSION", "2024-12-01-preview"))

CHAT_DEPLOYMENT = "gpt-4o"                 # chat completion deployment name
EMBED_MODEL     = "text-embedding-3-large" # embedding model / deployment

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _write_uploaded_file(uploaded_file) -> Path:
    """Persist Streamlit UploadedFile to disk & return its Path."""
    suffix = Path(uploaded_file.name).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.read())
    return Path(tmp_path)


def _build_index(pdf_paths: List[Path], progress_bar: Optional[st.progress] = None):
    """Embed PDFs, build FAISS index, return (RAG-chain, vectordb)."""
    # 2·1 Load & split
    docs = []
    for p in pdf_paths:
        docs.extend(PyPDFLoader(str(p)).load_and_split())
    splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=800)
    splits = splitter.split_documents(docs)

    # 2·2 Embed in batches with visual progress
    # embeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint=AZURE_ENDPOINT,
    #     api_key=AZURE_API_KEY,
    #     model=EMBED_MODEL,
    #     chunk_size=2048,
    # )
    embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    openai_api_version=API_VERSION,          # ← add this
    azure_deployment=EMBED_MODEL,            # ← prefer this alias
    chunk_size=2048,
    )
    vectordb = FAISS()
    batch_size = 64
    total = len(splits)
    for i in range(0, total, batch_size):
        batch = splits[i : i + batch_size]
        vectordb.add_documents(batch, embeddings)
        if progress_bar:
            progress_bar.progress(min((i + batch_size) / total, 1.0))

    vectordb.save_local("faiss_index")  # persist across sessions

    # 2·3 Build RAG chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        openai_api_version=API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=0,
        max_tokens=4096,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
    prompt_template = (
        "You are Healthcare Research Bot by Mohit S.\n\n"
        "Use the following context to answer the user's question.\n"
        "If the answer is not in the context, say you do not know—do not fabricate.\n"
        "Respond in 30–50 words.\n\n"
        "{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User question: {question}\n"
        "Answer:"
    )
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template,
    )
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        output_key="answer",
    )
    return rag_chain, vectordb, len(splits)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Streamlit layout & global CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Research Bot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* body tweaks */
        main {max-width: 1100px; margin: 0 auto;}
        footer {visibility: hidden;}
        .user-bubble, .bot-bubble {
            padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;
            line-height: 1.45;
        }
        .user-bubble {background:#e0f7fa; align-self:flex-end;}
        .bot-bubble  {background:#f1f8ff;}
        .chat-row {display:flex; gap:8px; align-items:flex-start;}
        .avatar {font-size:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="text-align:center;
               font-size:2.6rem;
               font-weight:800;
               background:linear-gradient(90deg,#00b4db,#0083b0);
               -webkit-background-clip:text;
               -webkit-text-fill-color:transparent;
               margin:0 0 4px 0;">
        🩺 Healthcare Research Bot
    </h1>
    <div style="text-align:center;font-size:0.95rem;color:#666;margin-bottom:28px;">
         — Ask and cite medical literature instantly by Mohit&nbsp;S
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Sidebar – file upload & controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📄 Document Index")
    api_key_input = st.text_input(
        "🔑 OpenAI / Azure API key",
        type="password",
        placeholder="sk-… or <azure-key>",
    )
    if api_key_input:
        AZURE_API_KEY = api_key_input.strip()

    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s) to index",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files processed only for this session.",
    )

    build_clicked = st.button("🔄 Build / Update Index", type="primary")
    clear_clicked = st.button("🗑️ Clear Conversation", type="secondary")

    # Show current indexed PDFs if any
    if "pdf_list" in st.session_state:
        st.write("### ✅ Indexed files")
        for fn in st.session_state["pdf_list"]:
            st.write(f"• {fn}")

# ──────────────────────────────────────────────────────────────────────────────
# 5.  Build / load index
# ──────────────────────────────────────────────────────────────────────────────
if build_clicked:
    if not uploaded_pdfs:
        st.error("Please upload at least one PDF first.")
    else:
        with st.spinner("Building embeddings…"):
            tmp_paths = [_write_uploaded_file(f) for f in uploaded_pdfs]
            prog = st.progress(0.0)
            chain, db, pages = _build_index(tmp_paths, progress_bar=prog)
            st.session_state["rag_chain"] = chain
            st.session_state["vectordb"]  = db
            st.session_state["pdf_list"]  = [p.name for p in tmp_paths]
        st.success(f"Index built ({pages} pages)! Ask away →")

# Auto-load existing index if present
if "rag_chain" not in st.session_state and Path("faiss_index").exists():
    with st.spinner("Loading existing index…"):
        emb = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            openai_api_version=API_VERSION,          # ← add
            azure_deployment=EMBED_MODEL,            # ← change param name
            chunk_size=2048,
        )
        db = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            openai_api_version=API_VERSION,
            azure_deployment=CHAT_DEPLOYMENT,
            temperature=0,
            max_tokens=4096,
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
        )
        prompt_template = (
            "You are Healthcare Research Bot by Mohit S.\n\n"
            "Use the following context to answer the user's question.\n"
            "If the answer is not in the context, say you do not know—do not fabricate.\n"
            "Respond in 30–50 words.\n\n"
            "{context}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "User question: {question}\n"
            "Answer:"
        )
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template,
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="answer",
        )
        st.session_state["rag_chain"] = chain
        st.session_state["vectordb"]  = db

# Clear conversation
if clear_clicked:
    for k in ("messages", "rag_chain"):
        st.session_state.pop(k, None)
    st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  Chat UI helpers
# ──────────────────────────────────────────────────────────────────────────────
def _render_msg(role: str, text: str):
    avatar = "👤" if role == "user" else "🩺"
    bubble_cls = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(
        f"""
        <div class="chat-row">
            <div class="avatar">{avatar}</div>
            <div class="{bubble_cls}">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for m in st.session_state["messages"]:
    _render_msg(m["role"], m["content"])

# Onboarding hint
if "rag_chain" not in st.session_state:
    st.info("🧠 **First time here?**\n\n1. Upload a PDF in the sidebar\n2. Click **Build / Update Index**\n3. Ask a question below!")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  Chat loop
# ──────────────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask something about the indexed documents…")
if prompt:
    if "rag_chain" not in st.session_state:
        st.warning("Please build or load an index first (see sidebar).")
    else:
        # show user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        _render_msg("user", prompt)

        with st.spinner("Searching & reasoning…"):
            result = st.session_state["rag_chain"].invoke({"question": prompt})
            answer = textwrap.fill(result["answer"], width=90)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        _render_msg("assistant", answer)

        # Optional sources dropdown
        if result.get("source_documents"):
            with st.expander("📑 Sources", expanded=False):
                for doc in result["source_documents"]:
                    src  = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    snip = doc.page_content[:200].replace("\n", " ")
                    st.markdown(f"**{src}** (p.{page}) — {snip}…")
