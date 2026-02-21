"""Minimal Streamlit frontend for DocProc v2."""

import streamlit as st

st.set_page_config(page_title="DocProc v2", layout="wide")
st.title("DocProc v2 - Document Intelligence")

api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

tab1, tab2, tab3 = st.tabs(["Upload", "Query", "Models"])

with tab1:
    st.header("Upload Document")
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        if st.button("Process"):
            import httpx
            try:
                r = httpx.post(
                    f"{api_url.rstrip('/')}/documents/upload",
                    files={"file": (file.name, file.getvalue(), "application/pdf")},
                    timeout=60.0,
                )
                r.raise_for_status()
                data = r.json()
                st.success(f"Document ID: {data['id']} - Status: {data['status']}")
                if data.get("status") == "completed":
                    doc = httpx.get(f"{api_url.rstrip('/')}/documents/{data['id']}").json()
                    st.json(doc.get("regions", [])[:5])
            except Exception as e:
                st.error(str(e))

with tab2:
    st.header("RAG Query")
    prompt = st.text_area("Question")
    if st.button("Query"):
        if prompt:
            import httpx
            try:
                r = httpx.post(
                    f"{api_url.rstrip('/')}/query",
                    json={"prompt": prompt, "top_k": 5},
                    timeout=60.0,
                )
                r.raise_for_status()
                data = r.json()
                st.write("**Answer:**")
                st.write(data.get("answer", ""))
                with st.expander("Retrieved context"):
                    st.write(data.get("retrieved", []))
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Enter a question")

with tab3:
    st.header("Available Models")
    import httpx
    try:
        r = httpx.get(f"{api_url.rstrip('/')}/models")
        r.raise_for_status()
        data = r.json()
        st.json(data)
    except Exception as e:
        st.error(str(e))
