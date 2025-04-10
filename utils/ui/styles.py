"""
Shared CSS styles for the Beyond Notes application.
Centralizes styling to maintain consistent appearance across pages.
"""

def get_base_styles():
    """Return core CSS styles used on all pages."""
    return """
<style>
    /* --- Main Layout --- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
        padding: 0.5rem 0;
    }
    
    .subheader {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* --- Section Headers with Numbers --- */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 2.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-number {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 38px;
        height: 38px;
        background-color: #2196F3;
        color: white;
        border-radius: 50%;
        text-align: center;
        margin-right: 12px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* --- Document Preview --- */
    .document-meta {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .document-meta-label {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    
    .document-meta-value {
        font-weight: 500;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    /* --- Progress Status --- */
    .progress-container {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .progress-stage {
        margin-top: 0.8rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .progress-message {
        opacity: 0.8;
        margin-top: 0.3rem;
        font-size: 0.95rem;
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div > div {
        background-color: #2196F3;
        background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
        background-size: 40px 40px;
        animation: progress-bar-stripes 2s linear infinite;
    }
    
    @keyframes progress-bar-stripes {
        0% {background-position: 40px 0;}
        100% {background-position: 0 0;}
    }

    /* Enhanced typography */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }

    /* Improved button styling */
    .stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
    }

    .stButton > button[data-baseweb="button"]:focus {
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.5) !important;
    }

    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: #2196F3 !important;
        border-color: #2196F3 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1976D2 !important;
        border-color: #1976D2 !important;
    }

    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Radio buttons, checkboxes */
    .stRadio > div[role="radiogroup"] > label > div[data-baseweb="radio"] > div {
        background-color: rgba(33, 150, 243, 0.8) !important;
    }

    .stCheckbox > div[role="checkbox"] > div[data-baseweb="checkbox"] > div {
        background-color: rgba(33, 150, 243, 0.8) !important;
    }

    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 6px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #2196F3 !important;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.5) !important;
    }

    /* Select boxes */
    .stSelectbox > div > div > div {
        border-radius: 6px !important;
    }

    .stSelectbox > div > div > div:focus {
        border-color: #2196F3 !important;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.5) !important;
    }

    /* Metrics styling */
    .stMetric {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 10px;
        transition: transform 0.2s ease;
    }

    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Code blocks */
    .stCode > div {
        border-radius: 8px !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: #2196F3 !important;
        border-radius: 6px !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(33, 150, 243, 0.1) !important;
    }

    .streamlit-expanderContent {
        border-radius: 0 0 6px 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #2196F3 !important;
    }

    .stSlider > div > div > div > div > div {
        background-color: #2196F3 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #2196F3 !important;
        background-image: linear-gradient(
            45deg,
            rgba(255, 255, 255, 0.15) 25%,
            transparent 25%,
            transparent 50%,
            rgba(255, 255, 255, 0.15) 50%,
            rgba(255, 255, 255, 0.15) 75%,
            transparent 75%,
            transparent
        ) !important;
        background-size: 40px 40px !important;
        animation: progress-bar-stripes 2s linear infinite !important;
    }

    @keyframes progress-bar-stripes {
        0% {background-position: 40px 0;}
        100% {background-position: 0 0;}
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(33, 150, 243, 0.1) !important;
        color: #2196F3 !important;
    }

    /* Sidebar improvements */
    .css-1d391kg, .css-hxt7ib {
        background-color: rgba(0, 0, 0, 0.1) !important;
    }

    /* File uploader */
    .stFileUploader > div {
        border-radius: 8px !important;
    }

    .stFileUploader > div:hover {
        border-color: #2196F3 !important;
    }

    /* Error and warning messages */
    .stAlert {
        border-radius: 8px !important;
    }

    [data-testid="stNotificationContent"] {
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
</style>
"""

def get_issues_styles():
    """Return CSS styles specific to issue assessment display."""
    return """
<style>
    /* --- Issue Card Styling --- */
    .issue-card {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        border-left: 4px solid #666;
        transition: transform 0.2s ease;
    }
    
    .issue-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Severity-based colors */
    .issue-critical {
        border-left-color: #F44336;
    }
    
    .issue-high {
        border-left-color: #FF9800;
    }
    
    .issue-medium {
        border-left-color: #2196F3;
    }
    
    .issue-low {
        border-left-color: #4CAF50;
    }
    
    /* Issue header */
    .issue-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    
    .issue-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-right: 1rem;
    }
    
    .issue-meta {
        display: flex;
        gap: 0.6rem;
    }
    
    .issue-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    /* Category badges */
    .badge-technical {
        background-color: rgba(156, 39, 176, 0.2);
        border: 1px solid rgba(156, 39, 176, 0.5);
    }
    
    .badge-process {
        background-color: rgba(33, 150, 243, 0.2);
        border: 1px solid rgba(33, 150, 243, 0.5);
    }
    
    .badge-resource {
        background-color: rgba(255, 152, 0, 0.2);
        border: 1px solid rgba(255, 152, 0, 0.5);
    }
    
    .badge-quality {
        background-color: rgba(76, 175, 80, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.5);
    }
    
    .badge-risk {
        background-color: rgba(244, 67, 54, 0.2);
        border: 1px solid rgba(244, 67, 54, 0.5);
    }
    
    .badge-compliance {
        background-color: rgba(63, 81, 181, 0.2);
        border: 1px solid rgba(63, 81, 181, 0.5);
    }
    
    /* Issue content */
    .issue-description {
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .issue-impact {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .issue-impact-label {
        font-weight: 500;
        margin-bottom: 0.4rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Terminal-like output styling for processing stages */
    .stage-progress {
        margin-bottom: 8px;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .stage-message {
        opacity: 0.8;
        font-size: 0.9rem;
        margin-left: 24px;
        margin-top: 2px;
    }
    
    .completed-stage {
        border-left: 3px solid #4CAF50;
    }
    
    .running-stage {
        border-left: 3px solid #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
    }
    
    .failed-stage {
        border-left: 3px solid #F44336;
        background-color: rgba(244, 67, 54, 0.1);
    }
    
    /* Copy button styling */
    .copy-button {
        background-color: rgba(33, 150, 243, 0.2);
        border: 1px solid rgba(33, 150, 243, 0.5);
        border-radius: 4px;
        padding: 0.5rem 1rem;
        color: white;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .copy-button:hover {
        background-color: rgba(33, 150, 243, 0.4);
        transform: translateY(-1px);
    }
    
    .copy-button svg {
        margin-right: 0.5rem;
    }
</style>
"""

def get_actions_styles():
    """Return CSS styles specific to action item extraction display."""
    return """
<style>
    /* Action items styling will go here */
    /* To be implemented when needed */
</style>
"""

def get_analysis_styles():
    """Return CSS styles specific to framework analysis display."""
    return """
<style>
    /* Framework analysis styling will go here */
    /* To be implemented when needed */
</style>
"""
