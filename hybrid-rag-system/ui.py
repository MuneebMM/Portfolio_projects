"""Streamlit UI for the Hybrid RAG System."""

import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def api(method: str, path: str, timeout: int = 60, **kwargs):
    base = st.session_state.get("api_url", "http://localhost:8000")
    try:
        resp = requests.request(method, f"{base}{path}", timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the server. Is it running?"
    except requests.exceptions.HTTPError as e:
        return None, f"Server error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, str(e)


def check_health():
    data, err = api("GET", "/health")
    return data is not None and data.get("status") == "healthy", err


# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Hybrid RAG")
    st.divider()

    # Server config
    st.subheader("Server")
    st.session_state.api_url = st.text_input(
        "API URL", value=st.session_state.api_url, label_visibility="collapsed"
    )

    healthy, err = check_health()
    if healthy:
        st.success("Server online", icon="✅")
    else:
        st.error(err or "Server offline", icon="🔴")

    st.divider()

    # Ingest
    st.subheader("Documents")
    if st.button("Ingest Documents", use_container_width=True, type="primary"):
        with st.spinner("Ingesting... (this may take a few minutes for large documents)"):
            data, err = api("POST", "/ingest", timeout=600)
        if err:
            st.error(err)
        else:
            st.success(
                f"✅ {data['documents_ingested']} doc(s) · {data['chunks_created']} chunks"
            )

    st.divider()

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Hybrid RAG · Dense + Sparse · Cohere Rerank")

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Ask a Question")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks · {msg.get('retrieval_count', '?')} retrieved)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**{i}. {src['source']}** — score `{src['score']}`")
                    st.caption(src["content"])
                    if i < len(msg["sources"]):
                        st.divider()

# Chat input
if prompt := st.chat_input("Ask anything about your documents..."):
    if not healthy:
        st.error("Server is offline. Start the FastAPI server first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                data, err = api("POST", "/query", json={"query": prompt, "top_k": 5})

            if err:
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
            else:
                answer = data["answer"]
                sources = data.get("sources", [])
                retrieval_count = data.get("retrieval_count", 0)

                st.markdown(answer)
                if sources:
                    with st.expander(f"Sources ({len(sources)} chunks · {retrieval_count} retrieved)"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**{i}. {src['source']}** — score `{src['score']}`")
                            st.caption(src["content"])
                            if i < len(sources):
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieval_count": retrieval_count,
                })
