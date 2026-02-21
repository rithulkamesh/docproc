"""NotebookLM-style frontend for DocProc v2."""

import os
import time
import httpx
import streamlit as st

st.set_page_config(page_title="DocProc v2", layout="wide", initial_sidebar_state="expanded")

# NotebookLM-inspired styling
st.markdown("""
<style>
    .stApp { max-width: 100%; }
    .main .block-container { padding-top: 2rem; padding-bottom: 4rem; }
    div[data-testid="stSidebar"] { background: #f8f9fa; }
    .stChatMessage { padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)

api_url = os.getenv("DOCPROC_API_URL", "http://localhost:8000")
base = api_url.rstrip("/")
SUPPORTED = ["pdf", "docx", "pptx", "xlsx"]

# Session state
if "docs" not in st.session_state:
    st.session_state.docs = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "refresh_docs" not in st.session_state:
    st.session_state.refresh_docs = False
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None


def fetch_docs():
    try:
        r = httpx.get(f"{base}/documents/", timeout=10.0)
        r.raise_for_status()
        data = r.json()
        return data.get("documents", [])
    except Exception:
        return []


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## Sources")
    st.caption("Documents in your library")
    if st.button("Refresh", use_container_width=True):
        st.session_state.refresh_docs = True
    if not st.session_state.docs or st.session_state.refresh_docs:
        st.session_state.docs = fetch_docs()
        st.session_state.refresh_docs = False

    if not st.session_state.docs:
        st.caption("No documents yet. Upload above.")
    for doc in st.session_state.docs:
        if doc.get("status") != "completed":
            continue
        label = doc.get("filename", doc.get("id", ""))[:50]
        if st.button(label, key=doc["id"], use_container_width=True):
            # Fetch full doc for full_text
            try:
                r = httpx.get(f"{base}/documents/{doc['id']}", timeout=10.0)
                if r.status_code == 200:
                    st.session_state.selected_doc = r.json()
                else:
                    st.session_state.selected_doc = doc
            except Exception:
                st.session_state.selected_doc = doc
            st.rerun()

    st.divider()
    st.markdown("**Upload**")
    file = st.file_uploader("Add document", type=SUPPORTED, label_visibility="collapsed", key="doc_upload")
    if file:
        file_key = (file.name, file.size)
        if file_key != st.session_state.last_uploaded:
            try:
                r = httpx.post(
                    f"{base}/documents/upload",
                    files={"file": (file.name, file.getvalue())},
                    timeout=60.0,
                )
                r.raise_for_status()
                data = r.json()
                doc_id = data.get("id")
                if doc_id:
                    progress_placeholder = st.empty()
                    while True:
                        r2 = httpx.get(f"{base}/documents/{doc_id}", timeout=10.0)
                        d = r2.json()
                        status = d.get("status", "")
                        prog = d.get("progress", {})
                        page = prog.get("page", 0)
                        total = max(1, prog.get("total", 1))
                        msg = prog.get("message", "")
                        with progress_placeholder.container():
                            st.progress(min(1.0, page / total), text=msg or status)
                        if status in ("completed", "failed"):
                            break
                        time.sleep(0.5)
                    if status == "completed":
                        st.success(f"Added: {file.name}")
                    else:
                        st.warning(f"Status: {status}" + (f" — {d.get('error', '')}" if status == "failed" else ""))
                    st.session_state.refresh_docs = True
                st.session_state.last_uploaded = file_key
            except Exception as e:
                st.error(str(e))

    st.divider()
    with st.expander("Settings"):
        st.text_input("API URL", value=base, key="api_url_override", disabled=True)


# ---- Main ----
view = st.radio("View", ["Chat", "Library"], horizontal=True, label_visibility="collapsed")

if view == "Library":
    if st.session_state.selected_doc:
        doc = st.session_state.selected_doc
        st.markdown(f"### {doc.get('filename', 'Document')}")
        st.caption(f"{doc.get('pages', 0)} pages · ID: {doc.get('id', '')}")
        full = doc.get("full_text", "")
        if full:
            st.markdown("---")
            st.markdown(full)
        else:
            st.info("No full text available. Re-upload to index.")
    else:
        st.markdown("### Library")
        st.caption("Select a document from the sidebar to view its full content.")

else:
    # Chat
    st.markdown("### Chat with your documents")
    st.caption("Ask questions about your uploaded documents. Answers are grounded in your sources.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        fn = s.get("filename", "—")
                        st.markdown(f"**{fn}**")
                        st.text(s.get("content", "")[:300] + ("…" if len(s.get("content", "")) > 300 else ""))

    prompt = st.chat_input("Ask a question…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    r = httpx.post(
                        f"{base}/query",
                        json={"prompt": prompt, "top_k": 5},
                        timeout=60.0,
                    )
                    r.raise_for_status()
                    data = r.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                fn = s.get("filename", "—")
                                st.markdown(f"**{fn}**")
                                st.text(s.get("content", "")[:300] + ("…" if len(s.get("content", "")) > 300 else ""))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(str(e))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": str(e),
                        "sources": [],
                    })
        st.rerun()
