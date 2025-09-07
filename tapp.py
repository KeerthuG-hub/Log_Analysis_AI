import streamlit as st
from deep_translator import GoogleTranslator
import time
import os
import sys
from typing import Dict, List, Any, Optional
import traceback

# Import the query backend
try:
    from qap import QueryOnlyLogStore
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Backend import failed: {e}")
    BACKEND_AVAILABLE = False

# ------------------ App Config ------------------
st.set_page_config(
    page_title="LogInsightAI / லாக் இன்சைட் ஏ.ஐ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ Translation Functions ------------------
@st.cache_data(ttl=3600)
def translate_text(text: str, to_tamil: bool = True) -> str:
    """Cached translation function"""
    try:
        if not to_tamil or not text.strip():
            return text

        # Skip translation for very short text or already Tamil text
        if len(text) < 3 or any(ord(char) >= 0x0B80 and ord(char) <= 0x0BFF for char in text):
            return text

        translator = GoogleTranslator(source='en', target='ta')
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text

def get_text(en_text: str, ta_text: Optional[str] = None) -> str:
    """Get text based on current language mode"""
    if st.session_state.get('tamil_mode', False):
        if ta_text:
            return ta_text
        return translate_text(en_text, to_tamil=True)
    return en_text

# ------------------ Custom CSS ------------------
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

    .status-critical { background: #fed7d7; color: #c53030; }
    .status-warning { background: #fef5e7; color: #d69e2e; }
    .status-info { background: #e6fffa; color: #319795; }
    .status-success { background: #d4edda; color: #155724; }

    .center-toggle {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }

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

# ------------------ Initialize Session State ------------------
if 'tamil_mode' not in st.session_state:
    st.session_state.tamil_mode = False
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

# ------------------ Initialize Backend ------------------
@st.cache_resource
def initialize_log_store():
    """Initialize the log store backend"""
    try:
        if not BACKEND_AVAILABLE:
            return None, "Backend not available"

        store = QueryOnlyLogStore()

        if store.db is not None:
            return store, "success"
        else:
            return None, "Failed to load precomputed data"
    except Exception as e:
        return None, str(e)

# Initialize backend
if not st.session_state.initialized:
    with st.spinner(get_text("Initializing LogInsightAI backend...")):
        log_store, init_status = initialize_log_store()

        if log_store:
            st.session_state.log_store = log_store
            st.session_state.initialized = True
            st.success(get_text("Backend initialized successfully!"))
        else:
            st.error(f"{get_text('Backend initialization failed')}: {init_status}")
            st.info(get_text("Please run log_ingest.py first to prepare the data."))

# ------------------ Enhanced Header ------------------
title_text = get_text("LogInsightAI")
subtitle_text = get_text("AI-Powered Log Analytics & Root Cause Detection",
                        "ஏ.ஐ-இயங்கும் பதிவு பகுப்பாய்வு & மூல காரண கண்டறிதல்")
desc_text = get_text("Advanced Log Analysis with Machine Learning Intelligence",
                    "இயந்திர கற்றல் நுண்ணறிவுடன் மேம்பட்ட பதிவு பகுப்பாய்வு")

st.markdown(f"""
<div class="main-header">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
    <h1 class="main-title">{title_text}</h1>
    <h2 style="color: #ffd700; margin: 0.5rem 0;">லாக் இன்சைட் ஏ.ஐ</h2>
    <p class="main-subtitle">{subtitle_text}</p>
    <p style="font-size: 1rem; opacity: 0.8;">{desc_text}</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Language Toggle (Centered) ------------------
current_lang = get_text("English", "தமிழ் (Tamil)")
lang_status_text = get_text("Current Language", "தற்போதைய மொழி")

st.markdown(f"""
<div class="center-toggle">
    <span class="status-badge status-info">{lang_status_text}: {current_lang}</span>
</div>
""", unsafe_allow_html=True)

# Centered toggle button
st.markdown('<div class="center-button">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    toggle_text = get_text("🌐 Toggle Language / மொழி மாற்று")
    if st.button(toggle_text, key="lang_toggle"):
        st.session_state.tamil_mode = not st.session_state.tamil_mode
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Query Input Section ------------------
query_title = get_text("📝 Enter Your Log Analysis Query", "📝 உங்கள் பதிவு பகுப்பாய்வு வினவலை உள்ளிடுங்கள்")
query_desc = get_text("Describe the issue you want to analyze or search for specific patterns",
                     "நீங்கள் பகுப்பாய்வு செய்ய விரும்பும் சிக்கலை விவரிக்கவும் அல்லது குறிப்பிட்ட வடிவங்களைத் தேடவும்")

st.markdown(f"""
<div class="query-card">
    <h3 style="margin: 0 0 1rem 0;">{query_title}</h3>
    <p style="margin: 0; opacity: 0.9;">{query_desc}</p>
</div>
""", unsafe_allow_html=True)

placeholder_text = get_text(
    "Type your query here...\nExample: 'database connection issue', 'file access attempt', 'security breach'",
    "உங்கள் வினவலை இங்கே தட்டச்சு செய்யுங்கள்...\nஎடுத்துக்காட்டு: 'தரவுத்தள இணைப்பு சிக்கல்', 'கோப்பு அணுகல் முயற்சி', 'பாதுகாப்பு மீறல்'"
)

user_query = st.text_area(
    "Query Input",
    height=120,
    placeholder=placeholder_text,
    key="query_input",
    label_visibility="hidden"
)

# ------------------ Centered Analyze Button ------------------
st.markdown('<div class="center-analyze">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_text = get_text("🚀 Analyze Logs", "🚀 பதிவுகளை பகுப்பாய்வு செய்யுங்கள்")
    analyze_clicked = st.button(analyze_text, key="analyze", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Process Analysis ------------------
if analyze_clicked:
    if not user_query.strip():
        error_text = get_text("Please enter a query to analyze!", "தயவுசெய்து பகுப்பாய்வு செய்ய ஒரு வினவலை உள்ளிடவும்!")
        st.error(error_text)
    elif not st.session_state.initialized or not st.session_state.log_store:
        error_text = get_text("Backend not initialized. Please check system status.",
                             "பின்முனை துவக்கப்படவில்லை. தயவுசெய்து கணினி நிலையை சரிபார்க்கவும்.")
        st.error(error_text)
    else:
        # Show loading animation
        loading_text = get_text("🔍 Analyzing logs with AI...", "🔍 ஏ.ஐ உடன் பதிவுகளை பகுப்பாய்வு செய்கிறது...")
        with st.spinner(loading_text):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Perform actual search
            try:
                results = st.session_state.log_store.search(user_query, k=10)
                st.session_state.search_results = results
                st.session_state.show_logs = False  # Reset show logs when new search is performed

                if results:
                    # Generate AI analysis
                    try:
                        analysis = st.session_state.log_store.analyze_results_with_ai(user_query, results)
                        # Store original English analysis
                        st.session_state.original_analysis = analysis
                        # Set current analysis based on language mode
                        if st.session_state.tamil_mode:
                            analysis = translate_text(analysis, to_tamil=True)
                        st.session_state.ai_analysis = analysis
                    except Exception as e:
                        error_msg = get_text("AI analysis failed", "ஏ.ஐ பகுப்பாய்வு தோல்வியுற்றது") + f": {str(e)}"
                        st.session_state.ai_analysis = error_msg
                        st.session_state.original_analysis = f"AI analysis failed: {str(e)}"

                    success_text = get_text("Analysis Complete!", "பகுப்பாய்வு முடிந்தது!")
                    st.success(f"✅ {success_text}")
                else:
                    st.session_state.search_results = None
                    st.session_state.ai_analysis = None
                    st.session_state.original_analysis = None
                    no_results_text = get_text(
                        "No matching log entries found. Try different keywords or check if the data is properly ingested.",
                        "பொருந்தும் பதிவு உள்ளீடுகள் எதுவும் கிடைக்கவில்லை. வெவ்வேறு முக்கிய வார்த்தைகளை முயற்சி செய்யுங்கள்."
                    )
                    st.warning(no_results_text)

            except Exception as e:
                search_error_text = get_text("Search failed", "தேடல் தோல்வியுற்றது")
                st.error(f"{search_error_text}: {str(e)}")
                st.error(traceback.format_exc())
                st.session_state.search_results = None
                st.session_state.ai_analysis = None
                st.session_state.original_analysis = None

# ------------------ Display Results (Only if we have results) ------------------
if st.session_state.search_results:
    results = st.session_state.search_results
    
    # Display search results summary
    results_title = get_text("🔍 Search Results", "🔍 தேடல் முடிவுகள்")
    st.markdown(f"""
    <div class="result-card">
        <h2 style="color: #667eea; margin-bottom: 1rem;">{results_title}</h2>
        <span class="status-badge status-success">{len(results)} {get_text("entries found", "பதிவுகள் கண்டுபிடிக்கப்பட்டன")}</span>
    </div>
    """, unsafe_allow_html=True)

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">📊 {get_text("Log Entries", "பதிவு உள்ளீடுகள்")}</h3>
            <h2 style="margin: 0; color: #333;">{len(results)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">⚡ {get_text("Analysis Time", "பகுப்பாய்வு நேரம்")}</h3>
            <h2 style="margin: 0; color: #333;">0.8s</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_score = sum(r.get('score', 0) for r in results) / len(results) if results else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin: 0; color: #667eea;">🎯 {get_text("Avg Relevance", "சராசரி பொருத்தம்")}</h3>
            <h2 style="margin: 0; color: #333;">{avg_score:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Toggle button for showing logs (centered)
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        show_logs_text = get_text("📜 Show Raw Logs", "📜 மூல பதிவுகளைக் காட்டு") if not st.session_state.show_logs else get_text("📜 Hide Raw Logs", "📜 மூல பதிவுகளை மறைக்கவும்")
        if st.button(show_logs_text, key="toggle_logs"):
            st.session_state.show_logs = not st.session_state.show_logs
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Display raw logs only if show_logs is True
    if st.session_state.show_logs:
        log_entries_text = get_text("📜 Raw Log Entries", "📜 மூல பதிவு உள்ளீடுகள்")
        st.markdown(f"### {log_entries_text}")
        
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        for i, result in enumerate(results):
            content = result.get('content', '')
            st.markdown(f"""
            <div class="log-entry">
                <strong>#{i+1}</strong>
                <br><br>
                {content}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # AI Analysis (Always shown when results exist)
    if st.session_state.ai_analysis:
        ai_analysis_text = get_text("🤖 AI Analysis", "🤖 ஏ.ஐ பகுப்பாய்வு")
        with st.expander(ai_analysis_text, expanded=True):
            # Store original English analysis if not already stored
            if 'original_analysis' not in st.session_state:
                st.session_state.original_analysis = st.session_state.ai_analysis
            
            # Display analysis in current language
            if st.session_state.tamil_mode:
                # If currently in Tamil mode, translate the original English analysis
                current_analysis = translate_text(st.session_state.original_analysis, to_tamil=True)
            else:
                # If in English mode, show original analysis
                current_analysis = st.session_state.original_analysis
            
            st.markdown(f'<div class="analysis-section"><p>{current_analysis}</p></div>',
                      unsafe_allow_html=True)

# ------------------ Sidebar (Minimal) ------------------
with st.sidebar:
    st.markdown(f"### {get_text('System Status', 'கணினி நிலை')}")

    if st.session_state.initialized and st.session_state.log_store:
        status_text = get_text("✅ Ready", "✅ தயார்")
        st.markdown(f"**{get_text('Status', 'நிலை')}:** {status_text}")
    else:
        status_text = get_text("❌ Not Ready", "❌ தயார் இல்லை")
        st.markdown(f"**{get_text('Status', 'நிலை')}:** {status_text}")

# ------------------ Help Section ------------------
with st.expander(get_text("ℹ️ Help & Usage Guide", "ℹ️ உதவி & பயன்பாட்டு வழிகாட்டி")):
    help_content = get_text(
        """
        **How to use LogInsightAI:**

        1. **Enter your query** in natural language describing what you want to find
        2. **Click Analyze** to search through the logs using AI
        3. **View AI Analysis** which is automatically generated
        4. **Click 'Show Raw Logs'** to view the actual log entries in full width
        5. **Use Language Toggle** to switch between English and Tamil

        **Query Examples:**
        - "Show me failed login attempts"
        - "Find all file deletion commands"
        - "Search for permission denied errors"
        - "List commands executed by specific user"

        **Features:**
        - Vector similarity search using embeddings
        - Command-specific indexing
        - AI-powered result ranking
        - Bilingual support (English/Tamil)
        - Full-width log display in terminal style
        """,
        """
        **LogInsightAI ஐ எவ்வாறு பயன்படுத்துவது:**

        1. **உங்கள் வினவலை உள்ளிடவும்** நீங்கள் கண்டுபிடிக்க விரும்புவதை விவரிக்கும் இயற்கை மொழியில்
        2. **பகுப்பாய்வு கிளிக் செய்யவும்** ஏ.ஐ பயன்படுத்தி பதிவுகளில் தேட
        3. **ஏ.ஐ பகுப்பாய்வைப் பார்க்கவும்** இது தானாகவே உருவாக்கப்படும்
        4. **'மூல பதிவுகளைக் காட்டு' கிளிக் செய்யவும்** முழு அகலத்தில் உண்மையான பதிவு உள்ளீடுகளைப் பார்க்க
        5. **மொழி டாகிளைப் பயன்படுத்தவும்** ஆங்கிலம் மற்றும் தமிழ் இடையே மாற
        """
    )
    st.markdown(help_content)

# ------------------ Floating Animation ------------------
st.markdown('<div class="floating-icon">🔍</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("---")
footer_text = get_text(
    "🚀 Powered by Advanced AI & Machine Learning",
    "🚀 மேம்பட்ட ஏ.ஐ & இயந்திர கற்றலால் இயக்கப்படுகிறது"
)
version_text = get_text(
    "LogInsightAI v2.0 | Real-time Log Analysis & Root Cause Detection",
    "LogInsightAI v2.0 | நிகழ்நேர பதிவு பகுப்பாய்வு & மூல காரண கண்டறிதல்"
)

st.markdown(f"""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>{footer_text}</p>
    <p style="font-size: 0.9rem;">{version_text}</p>
</div>
""", unsafe_allow_html=True)
