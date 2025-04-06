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
    
    /* Recommendations section */
    .recommendations-header {
        font-size: 1rem;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .recommendations-list {
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .recommendations-list li {
        margin-bottom: 0.5rem;
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