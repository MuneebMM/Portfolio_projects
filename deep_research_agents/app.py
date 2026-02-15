import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎")

title_html = """
<div style="display: flex; justify-content: center; align-items: center; width: 100%; padding: 32px 0 24px 0;">
    <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: bold;">
        <span style="font-size:2.5rem;">🔎</span> Deep Research Agent
    </h1>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_URL)
    st.divider()

    st.header("About")
    st.markdown(
        """
    This application is powered by a **LangGraph** research pipeline:
    - **Searcher**: Finds and extracts information via Firecrawl, stores in Qdrant
    - **Analyst**: Synthesizes and interprets findings with Qdrant retrieval
    - **Writer**: Produces a final, polished report

    **Tech Stack**: LangChain, LangGraph, GPT-5-mini, Scrapy + Playwright, Qdrant, PostgreSQL, Redis, LangSmith
    """
    )
    st.divider()

    st.header("Past Reports")
    try:
        reports_response = requests.get(f"{api_url}/reports", timeout=5)
        if reports_response.status_code == 200:
            reports = reports_response.json()
            for report in reports[:10]:
                with st.expander(f"📄 {report['topic'][:50]}..."):
                    st.caption(f"Created: {report['created_at']}")
                    if st.button("View", key=f"view_{report['id']}"):
                        detail = requests.get(
                            f"{api_url}/reports/{report['id']}"
                        ).json()
                        st.markdown(detail.get("report_content", "No content"))
    except requests.exceptions.ConnectionError:
        st.info("API not connected. Start the FastAPI backend first.")

    st.markdown("---")
    st.markdown("Built with LangGraph + FastAPI + Streamlit")

user_input = st.chat_input("What topic would you like to research?")

if user_input:
    try:
        with st.status("Executing research pipeline...", expanded=True) as status:
            status.write("🧠 **Searching** — Gathering information from the web...")

            response = requests.post(
                f"{api_url}/research",
                json={"topic": user_input},
                stream=True,
                timeout=300,
            )

            full_report = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    node = data.get("node", "")

                    if node == "searcher":
                        status.write(
                            "🔬 **Analyzing** — Synthesizing research findings..."
                        )
                    elif node == "analyst":
                        status.write(
                            "✍️ **Writing** — Producing final report..."
                        )
                    elif node == "writer":
                        full_report = data.get("data", {}).get("report", "")
                    elif node == "saved":
                        status.write(
                            f"💾 Report saved (ID: {data.get('report_id')})"
                        )

            status.update(label="Research complete!", state="complete")

        if full_report:
            st.markdown(full_report)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. Make sure the FastAPI backend is running: "
            "`python api.py` or `docker-compose up`"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
