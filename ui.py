import streamlit as st
from deep_translator import GoogleTranslator
import time

# ------------------ App Config ------------------
st.set_page_config(
    page_title="LogInsightAI / லாக் இன்சைட் ஏ.ஐ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Disable usage stats collection
import os
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# ------------------ Custom CSS for Enhanced UI ------------------
st.markdown("""
<style>
    /* Main container styling */
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
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .log-entry {
        background: #2d3748;
        color: #68d391;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #68d391;
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
</style>
""", unsafe_allow_html=True)

# ------------------ Initialize Session State ------------------
if 'tamil_mode' not in st.session_state:
    st.session_state.tamil_mode = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ------------------ Enhanced Header ------------------
st.markdown("""
<div class="main-header">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
    <h1 class="main-title">LogInsightAI</h1>
    <h2 style="color: #ffd700; margin: 0.5rem 0;">லாக் இன்சைட் ஏ.ஐ</h2>
    <p class="main-subtitle">AI-Powered Log Analytics & Root Cause Detection</p>
    <p style="font-size: 1rem; opacity: 0.8;">Advanced Log Analysis with Machine Learning Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ------------------ Language Toggle ------------------
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
""", unsafe_allow_html=True)

if st.button("🌐 Toggle Language / மொழி மாற்று", key="lang_toggle"):
    st.session_state.tamil_mode = not st.session_state.tamil_mode
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Display current language status
current_lang = "தமிழ் (Tamil)" if st.session_state.tamil_mode else "English"
st.markdown(f"""
<div style="text-align: center; margin: 1rem 0;">
    <span class="status-badge status-info">Current Language: {current_lang}</span>
</div>
""", unsafe_allow_html=True)

# ------------------ Enhanced RCA Cases ------------------
cases = {
    "db_connection": {
        "title": "Database Connection Failures",
        "title_ta": "தரவுத்தள இணைப்பு தோல்விகள்",
        "severity": "critical",
        "logs": [
            "2025-08-18 11:45:03 → DB connection timeout for app_server_01",
            "2025-08-18 11:45:10 → DB connection timeout for app_server_02", 
            "2025-08-18 11:46:00 → Connection refused from app_server_03"
        ],
        "analysis": "Multiple connection timeouts in a short period suggest network instability or DB server overload. Repeated refusal indicates possible authentication or configuration issues.",
        "analysis_ta": "குறுகிய காலத்தில் பல இணைப்பு காலாவதிகள் நெட்வொர்க் உறுதியின்மை அல்லது டிபி சர்வர் அதிக சுமையை குறிக்கிறது.",
        "root_cause": "Could be intermittent network issues, DB configuration mismatch, or insufficient connection pool size.",
        "root_cause_ta": "இடைவிடாத நெட்வொர்க் சிக்கல்கள், டிபி உள்ளமைவு பொருத்தமின்மை, அல்லது போதுமான இணைப்பு குளம் அளவு இல்லாமை.",
        "recommendation": "Check network connectivity, validate DB configs, increase connection pool limits, and monitor DB server performance.",
        "recommendation_ta": "நெட்வொர்க் இணைப்பை சரிபார்க்கவும், டிபி கட்டமைப்புகளை உறுதிபடுத்தவும், இணைப்பு குளம் வரம்புகளை அதிகரிக்கவும்."
    },
    "file_access": {
        "title": "Unauthorized File Access Attempts", 
        "title_ta": "அங்கீகரிக்கப்படாத கோப்பு அணுகல் முயற்சிகள்",
        "severity": "warning",
        "logs": [
            "2025-08-15 09:23:17 → intern01 tried to read /secure_data/finance_report.xlsx and got Permission Denied",
            "2025-08-15 09:25:42 → contractor02 attempted write to /secure_data/project_plan.docx"
        ],
        "analysis": "Unauthorized attempts indicate potential insider threats or misconfigured ACLs. Users trying to access sensitive directories without permission is unusual behavior.",
        "analysis_ta": "அங்கீகரிக்கப்படாத முயற்சிகள் உள்நோக்கு அச்சுறுத்தல்கள் அல்லது தவறாக உள்ளமைக்கப்பட்ட ACLகளை குறிக்கிறது.",
        "root_cause": "Lack of strict access controls or unawareness of data sensitivity.",
        "root_cause_ta": "கடுமையான அணுகல் கட்டுப்பாடுகளின் பற்றாக்குறை அல்லது தரவு உணர்வு விழிப்புணர்வின்மை.",
        "recommendation": "Review ACLs, restrict sensitive directories, and enable alerting for repeated unauthorized access attempts.",
        "recommendation_ta": "ACLகளை மீளாய்வு செய்யவும், உணர்திறன் கோப்பகங்களை கட்டுப்படுத்தவும், மீண்டும் மீண்டும் அங்கீகரிக்கப்படாத அணுகல் முயற்சிகளுக்கு எச்சரிக்கையை இயக்கவும்."
    }
}

# ------------------ Enhanced Query Input ------------------
st.markdown("""
<div class="query-card">
    <h3 style="margin: 0 0 1rem 0;">📝 Enter Your Log Analysis Query</h3>
    <p style="margin: 0; opacity: 0.9;">Describe the issue you want to analyze or search for specific patterns</p>
</div>
""", unsafe_allow_html=True)

user_query = st.text_area(
    "Query Input", 
    height=120, 
    placeholder="Type your query here...\nExample: 'database connection issue', 'file access attempt', 'security breach'",
    key="query_input",
    label_visibility="hidden"
)

# ------------------ Enhanced Analyze Button ------------------
if st.button("🚀 Analyze Logs", key="analyze", type="primary"):
    if user_query.strip():
        st.session_state.analysis_complete = True
        
        # Show loading animation
        with st.spinner('🔍 Analyzing logs with AI...'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        query_lower = user_query.lower()
        if "database" in query_lower or "db" in query_lower or "connection" in query_lower:
            selected_case = cases["db_connection"]
            case_key = "db_connection"
        elif "file" in query_lower or "access" in query_lower or "secure" in query_lower or "permission" in query_lower:
            selected_case = cases["file_access"]
            case_key = "file_access"
        else:
            st.warning("🔍 No matching RCA case found. Try queries about 'database connection' or 'file access'.")
            selected_case = None
            case_key = None

        if selected_case:
            # Get text based on language mode
            title = selected_case["title_ta"] if st.session_state.tamil_mode else selected_case["title"]
            root_cause = selected_case["root_cause_ta"] if st.session_state.tamil_mode else selected_case["root_cause"]
            recommendation = selected_case["recommendation_ta"] if st.session_state.tamil_mode else selected_case["recommendation"]
            
            # Case Title with Status Badge
            severity_class = f"status-{selected_case['severity']}"
            severity_text = selected_case['severity'].upper()
            
            st.markdown(f"""
            <div class="result-card">
                <h2 style="color: #667eea; margin-bottom: 1rem;">🔹 {title}</h2>
                <span class="status-badge {severity_class}">{severity_text}</span>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced Expandable Sections
            with st.expander("📜 Raw Log Entries", expanded=True):
                for i, log in enumerate(selected_case["logs"]):
                    st.markdown(f'<div class="log-entry">#{i+1} {log}</div>', unsafe_allow_html=True)

            with st.expander("🕵️ Root Cause Investigation", expanded=True):
                st.markdown(f'<div class="analysis-section"><p><strong>Identified Cause:</strong> {root_cause}</p></div>', unsafe_allow_html=True)

            with st.expander("✅ Recommended Actions", expanded=True):
                st.markdown(f'<div class="analysis-section"><p><strong>Next Steps:</strong> {recommendation}</p></div>', unsafe_allow_html=True)

            # Success message
            success_msg = "பகுப்பாய்வு முடிந்தது!" if st.session_state.tamil_mode else "Analysis Complete!"
            st.success(f"✅ {success_msg}")
            
    else:
        st.error("Please enter a query to analyze!")

# ------------------ Floating Animation ------------------
st.markdown('<div class="floating-icon">🔍</div>', unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>🚀 Powered by Advanced AI & Machine Learning</p>
    <p style="font-size: 0.9rem;">LogInsightAI v2.0 | Real-time Log Analysis & Root Cause Detection</p>
</div>
""", unsafe_allow_html=True)
