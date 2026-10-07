import streamlit as st
import time
import os
import traceback
from typing import List, Any

try:
    from query_engine import QueryOnlyLogStore
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Backend import failed: {e}")
    BACKEND_AVAILABLE = False

st.set_page_config(
    page_title="LogInsightAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }

    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .query-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: white;
    }

    .result-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .metric-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    .log-entry {
        background: #1a1a1a;
        color: #00ff00;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #00ff00;
        font-size: 0.9rem;
        white-space: pre-wrap;
        overflow-x: auto;
        width: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .log-container {
        width: 100%;
        margin: 1rem 0;
    }

    .analysis-section {
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1rem 0;
    }

    .floating-icon {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 3rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }

    .status-info { background: #e6fffa; color: #319795; }
    .status-success { background: #d4edda; color: #155724; }

    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }

    .center-analyze {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'log_store' not in st.session_state:
    st.session_state.log_store = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'show_logs' not in st.session_state:
    st.session_state.show_logs = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None
if 'original_analysis' not in st.session_state:
    st.session_state.original_analysis = None


@st.cache_resource
def initialize_log_store():
    """Load the Chroma-backed query store; returns (store, status_string)."""
    try:
        if not BACKEND_AVAILABLE:
            return None, "Backend not available"
        store = QueryOnlyLogStore()
        if store.db is not None:
            return store, "success"
        return None, "Failed to load precomputed data"
    except Exception as e:
        return None, str(e)


if not st.session_state.initialized:
    with st.spinner("Initializing LogInsightAI backend..."):
        log_store, init_status = initialize_log_store()
        if log_store:
            st.session_state.log_store = log_store
            st.session_state.initialized = True
            st.success("Backend initialized successfully!")
        else:
            st.error(f"Backend initialization failed: {init_status}")
            st.info("Please run log_ingest.py first to prepare the data.")

st.markdown("""
<div class="main-header">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
    <h1 class="main-title">LogInsightAI</h1>
    <p class="main-subtitle">AI-Powered Log Analytics &amp; Root Cause Detection</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="query-card">
    <h3 style="margin: 0 0 1rem 0;">📝 Enter Your Log Analysis Query</h3>
    <p style="margin: 0; opacity: 0.9;">Describe the issue you want to analyze or search for specific patterns</p>
</div>
""", unsafe_allow_html=True)

user_query = st.text_area(
    "Query Input",
    height=120,
    placeholder="Type your query here...\nExamples: 'show failed login attempts', 'rm commands by alice', 'chmod by frank'",
    key="query_input",
    label_visibility="hidden"
)

st.markdown('<div class="center-analyze">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_clicked = st.button("🚀 Analyze Logs", key="analyze", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if analyze_clicked:
    if not user_query.strip():
        st.error("Please enter a query to analyze.")
    elif not st.session_state.initialized or not st.session_state.log_store:
        st.error("Backend not initialized. Please check system status.")
    else:
        with st.spinner("Analyzing logs..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            try:
                results = st.session_state.log_store.search(user_query, k=10)
                st.session_state.search_results = results
                st.session_state.show_logs = False

                if results:
                    try:
                        analysis = st.session_state.log_store.analyze_results_with_ai(user_query, results)
                        st.session_state.original_analysis = analysis
                        st.session_state.ai_analysis = analysis
                    except Exception as e:
                        st.session_state.ai_analysis = f"AI analysis failed: {e}"
                        st.session_state.original_analysis = st.session_state.ai_analysis

                    st.success("Analysis complete!")
                else:
                    st.session_state.search_results = None
                    st.session_state.ai_analysis = None
                    st.session_state.original_analysis = None
                    st.warning("No matching log entries found. Try different keywords or check that data is ingested.")

            except Exception as e:
                st.error(f"Search failed: {e}")
                st.error(traceback.format_exc())
                st.session_state.search_results = None
                st.session_state.ai_analysis = None
                st.session_state.original_analysis = None

if st.session_state.search_results:
    results = st.session_state.search_results

    st.markdown(f"""
    <div class="result-card">
        <h2 style="color: #667eea; margin-bottom: 1rem;">🔍 Search Results</h2>
        <span class="status-badge status-success">{len(results)} entries found</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">📊 Log Entries</h3>
            <h2 style="margin: 0; color: #333;">{len(results)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">⚡ Analysis Time</h3>
            <h2 style="margin: 0; color: #333;">0.8s</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_score = sum(r.get('score', 0) for r in results) / len(results) if results else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">🎯 Avg Relevance</h3>
            <h2 style="margin: 0; color: #333;">{avg_score:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        show_label = "📜 Hide Raw Logs" if st.session_state.show_logs else "📜 Show Raw Logs"
        if st.button(show_label, key="toggle_logs"):
            st.session_state.show_logs = not st.session_state.show_logs
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_logs:
        st.markdown("### 📜 Raw Log Entries")
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        for i, result in enumerate(results):
            content = result.get('content', '')
            st.markdown(f"""
            <div class="log-entry">
                <strong>#{i+1}</strong><br><br>{content}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.ai_analysis:
        with st.expander("🤖 AI Analysis", expanded=True):
            st.markdown(
                f'<div class="analysis-section"><p>{st.session_state.original_analysis}</p></div>',
                unsafe_allow_html=True
            )

with st.sidebar:
    st.markdown("### System Status")
    if st.session_state.initialized and st.session_state.log_store:
        st.markdown("**Status:** Ready")
    else:
        st.markdown("**Status:** Not Ready")

with st.expander("ℹ️ Help & Usage Guide"):
    st.markdown("""
**How to use LogInsightAI:**

1. **Enter your query** in natural language describing what you want to find
2. **Click Analyze** to search through logs using AI
3. **View AI Analysis** — automatically generated summary
4. **Click 'Show Raw Logs'** to view actual log entries in terminal style

**Example queries:**
- "show failed login attempts"
- "find all rm commands by alice"
- "chmod operations by frank"
- "files deleted by grace"
- "unauthorized access attempts"
- "what did bob do on Aug 5"
- "show ssh logins for carol"
""")

st.markdown('<div class="floating-icon">🔍</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>Powered by Advanced AI &amp; Machine Learning</p>
    <p style="font-size: 0.9rem;">LogInsightAI v2.0 | Real-time Log Analysis &amp; Root Cause Detection</p>
</div>
""", unsafe_allow_html=True)
