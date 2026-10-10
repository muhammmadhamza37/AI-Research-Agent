import os
import streamlit as st
from agent import run_research

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")
st.title("🔎 AI Research Agent")
st.caption("Powered by CrewAI + Groq (gpt-oss-120b) + DuckDuckGo")


def get_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()
if not api_key:
    api_key = st.sidebar.text_input("Groq API Key", type="password",
                                    help="Get a free key at console.groq.com")

topic = st.text_input("Research topic", placeholder="e.g. Latest advances in solid-state batteries")

if st.button("Generate Report", type="primary"):
    if not api_key:
        st.error("Please add your Groq API key first.")
    elif not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Researching... this can take 30-90 seconds."):
            try:
                st.session_state["report"] = run_research(topic.strip(), api_key)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])
    st.download_button("⬇️ Download report (.md)", data=st.session_state["report"],
                       file_name="research_report.md", mime="text/markdown")
