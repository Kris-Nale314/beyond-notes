"""
Enhanced UI components for the Beyond Notes application.
These components provide a more polished and professional look and feel.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

# Configure logger
logger = logging.getLogger("beyond-notes.enhanced-ui")

# =====================================================================
# Page Layout Components
# =====================================================================

def enhanced_page_header(title: str, icon: str, subtitle: str, color_gradient: Tuple[str, str] = ("#2196F3", "#673AB7")):
    """
    Render a visually appealing page header with gradient background for dark theme.
    
    Args:
        title: The page title
        icon: Icon emoji to display next to title
        subtitle: Page subtitle or description
        color_gradient: Tuple of (start_color, end_color) for gradient
    """
    # Darken the gradient colors slightly to work better with dark mode
    # Convert from hex to rgba with some transparency
    start_color = color_gradient[0]
    end_color = color_gradient[1]
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {start_color}40, {end_color}40);
                padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center;
                border: 1px solid {start_color}30;">
        <h1 style="font-size: 2.2rem; margin-bottom: 0.5rem; color: white;">
            {icon} {title}
        </h1>
        <p style="font-size: 1.05rem; opacity: 0.9; color: rgba(255, 255, 255, 0.8); max-width: 800px; margin: 0 auto;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def enhanced_section_header(title: str, number: Optional[int] = None, icon: Optional[str] = None):
    """
    Render an enhanced section header with optional number and icon.
    
    Args:
        title: Section title
        number: Optional section number
        icon: Optional icon emoji
    """
    # Combine icon and number if both present
    prefix = ""
    if number is not None:
        prefix += f'<span class="section-number">{number}</span>'
    if icon is not None:
        if number is not None:
            # If we have both, use the icon inside the number circle
            prefix = f'<span class="section-number">{icon}</span>'
        else:
            # Otherwise just add the icon
            prefix += f'<span class="section-icon">{icon}</span>'
    
    st.markdown(f'<div class="section-header">{prefix}{title}</div>', unsafe_allow_html=True)

# =====================================================================
# Document Preview Component
# =====================================================================

def enhanced_document_preview(document):
    """
    Display an enhanced document preview with dark theme styling.
    
    Args:
        document: Document object with filename, text, and metadata
    """
    st.markdown("""
    <div style="background: linear-gradient(145deg, rgba(30, 30, 30, 0.7), rgba(50, 50, 50, 0.7)); 
                border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid rgba(70, 70, 70, 0.7);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);">
        <h3 style="margin-top: 0; color: #2196F3;">Document Details</h3>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h4 style='color: rgba(255, 255, 255, 0.85);'>Document Preview</h4>", unsafe_allow_html=True)
        
        preview_length = min(1000, len(document.text))
        preview_text = document.text[:preview_length]
        if len(document.text) > preview_length:
            preview_text += "..."
        
        # Escape HTML characters properly
        preview_text_escaped = preview_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
        
        st.markdown(f"""
        <div style="background-color: rgba(30, 30, 30, 0.9); border-radius: 8px; height: 200px; overflow-y: auto; 
                  padding: 15px; font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9rem; white-space: pre-wrap;
                  border: 1px solid rgba(70, 70, 70, 0.7); color: rgba(255, 255, 255, 0.85);">
            {preview_text_escaped}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # File info card
        st.markdown("<h4 style='color: rgba(255, 255, 255, 0.85);'>File Information</h4>", unsafe_allow_html=True)
        
        # Create metrics for file details
        import os
        file_size = os.path.getsize(str(document.file_path)) if hasattr(document, 'file_path') and document.file_path else 0
        
        metrics = [
            {"name": "File Name", "value": document.filename, "icon": "📄"},
            {"name": "Word Count", "value": f"{document.word_count:,}", "icon": "📝"},
            {"name": "File Size", "value": f"{file_size / 1024:.1f} KB", "icon": "💾"}
        ]
        
        for metric in metrics:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 15px; 
                      background-color: rgba(40, 40, 40, 0.9); padding: 10px; border-radius: 8px;
                      border: 1px solid rgba(70, 70, 70, 0.7);">
                <div style="font-size: 1.5rem; margin-right: 15px;">{metric['icon']}</div>
                <div>
                    <div style="font-size: 0.8rem; opacity: 0.7; color: rgba(255, 255, 255, 0.7);">{metric['name']}</div>
                    <div style="font-weight: 500; color: rgba(255, 255, 255, 0.9);">{metric['value']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# Progress Indicators
# =====================================================================

def animated_progress_indicator(session_state):
    """
    Display an animated progress bar with enhanced visual styling.
    
    Args:
        session_state: Streamlit session state containing progress info
    """
    progress_value = float(session_state.get("current_progress", 0.0))
    progress_message = session_state.get("progress_message", "Processing...")
    current_stage = session_state.get("current_stage", "")
    
    # Add styling for progress elements
    st.markdown("""
    <style>
        /* Progress container */
        .enhanced-progress-container {
            background-color: rgba(33, 150, 243, 0.05);
            border-radius: 8px;
            padding: 1.2rem;
            margin: 1.2rem 0;
            border: 1px solid rgba(33, 150, 243, 0.2);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        /* Stage name styling */
        .enhanced-current-stage {
            font-weight: 600;
            margin: 0.75rem 0 0.25rem 0;
            color: #2196F3;
        }
        
        /* Message styling */
        .enhanced-progress-message {
            margin-bottom: 0.75rem;
            opacity: 0.9;
        }
        
        /* Processing animation */
        @keyframes pulse {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }
        
        .enhanced-processing-indicator {
            animation: pulse 1.5s infinite;
            color: #2196F3;
            display: flex;
            align-items: center;
            padding: 0.5rem;
            border-radius: 4px;
            margin-top: 0.7rem;
            background-color: rgba(33, 150, 243, 0.1);
            font-weight: 500;
            width: fit-content;
        }
        
        .enhanced-processing-icon {
            margin-right: 0.5rem;
        }
        
        /* Animated progress bar */
        .enhanced-progress-bar-container {
            height: 10px;
            background-color: rgba(0, 0, 0, 0.1);
            border-radius: 5px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .enhanced-progress-bar {
            height: 100%;
            border-radius: 5px;
            background-color: #2196F3;
            background-image: linear-gradient(
                -45deg,
                rgba(255, 255, 255, 0.2) 25%,
                transparent 25%,
                transparent 50%,
                rgba(255, 255, 255, 0.2) 50%,
                rgba(255, 255, 255, 0.2) 75%,
                transparent 75%,
                transparent
            );
            background-size: 50px 50px;
            animation: progress-animation 2s linear infinite;
            transition: width 0.3s ease;
        }
        
        @keyframes progress-animation {
            0% { background-position: 0 0; }
            100% { background-position: 50px 0; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create progress container
    st.markdown("<div class='enhanced-progress-container'>", unsafe_allow_html=True)
    
    # Create animated progress bar
    progress_percentage = int(progress_value * 100)
    st.markdown(f"""
    <div class="enhanced-progress-bar-container">
        <div class="enhanced-progress-bar" style="width: {progress_percentage}%;"></div>
    </div>
    <div style="text-align: right; font-size: 0.9rem; opacity: 0.7; margin-top: -0.5rem;">
        {progress_percentage}%
    </div>
    """, unsafe_allow_html=True)
    
    # Show current stage with proper formatting
    if current_stage:
        stage_display = current_stage.replace('_', ' ').title()
        st.markdown(f"<div class='enhanced-current-stage'>{stage_display}</div>", unsafe_allow_html=True)
    
    # Show progress message
    st.markdown(f"<div class='enhanced-progress-message'>{progress_message}</div>", unsafe_allow_html=True)
    
    # Add animated processing indicator
    st.markdown(
        """<div class="enhanced-processing-indicator">
            <span class="enhanced-processing-icon">⚙️</span> Processing document...
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Display current active agent if available
    if "current_agent" in session_state:
        current_agent = session_state.current_agent
        current_task = session_state.get("current_agent_task", "Working...")
        display_agent_card(current_agent, current_task)
    
    # Close container
    st.markdown("</div>", unsafe_allow_html=True)

def display_agent_card(agent_name, task, show_animation=True):
    """
    Display a card showing the currently active agent with nice styling.
    
    Args:
        agent_name: Name of the active agent
        task: Current task being performed
        show_animation: Whether to show animation
    """
    # Map agent names to friendly display names and icons
    agent_display_names = {
        "PlannerAgent": ("Planner", "🧭"),
        "ExtractorAgent": ("Extractor", "🔍"),
        "SummarizerAgent": ("Summarizer", "📝"),
        "AggregatorAgent": ("Aggregator", "🧩"),
        "EvaluatorAgent": ("Evaluator", "⚖️"),
        "FormatterAgent": ("Formatter", "📊"),
        "ReviewerAgent": ("Reviewer", "🔍")
    }
    
    # Get display name and icon
    display_name, icon = agent_display_names.get(agent_name, ("Agent", "🤖"))
    
    # Create animation class if enabled
    animation_class = "agent-pulse" if show_animation else ""
    
    # Render the agent card
    st.markdown(f"""
    <style>
        .agent-card {{
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            padding: 12px;
            margin-top: 16px;
            border-left: 4px solid #2196F3;
            display: flex;
            align-items: center;
        }}
        
        .agent-card.agent-pulse {{
            animation: card-pulse 2s infinite ease-in-out;
        }}
        
        @keyframes card-pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.4); }}
            70% {{ box-shadow: 0 0 0 10px rgba(33, 150, 243, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }}
        }}
        
        .agent-icon {{
            width: 40px;
            height: 40px;
            background-color: #2196F3;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 12px;
            font-size: 20px;
        }}
        
        .agent-info {{
            flex: 1;
        }}
        
        .agent-name {{
            font-weight: 600;
            margin-bottom: 4px;
        }}
        
        .agent-task {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .agent-status {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #4CAF50;
            margin-left: 12px;
            animation: status-pulse 1.5s infinite;
        }}
        
        @keyframes status-pulse {{
            0% {{ opacity: 0.6; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.6; }}
        }}
    </style>
    
    <div class="agent-card {animation_class}">
        <div class="agent-icon">{icon}</div>
        <div class="agent-info">
            <div class="agent-name">{display_name} Agent</div>
            <div class="agent-task">{task}</div>
        </div>
        <div class="agent-status"></div>
    </div>
    """, unsafe_allow_html=True)

def advanced_progress_detail(session_state):
    """
    Display detailed progress information with better visual organization.
    
    Args:
        session_state: Streamlit session state containing progress info
    """
    stages_info = session_state.get("stages_info", {})
    if not stages_info:
        return
    
    # Add custom styling
    st.markdown("""
    <style>
        .stage-container {
            margin-bottom: 12px;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .stage-header {
            padding: 10px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px 8px 0 0;
        }
        
        .stage-name {
            display: flex;
            align-items: center;
            font-weight: 500;
        }
        
        .stage-icon {
            margin-right: 8px;
            font-size: 1.1rem;
        }
        
        .stage-status {
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .status-completed {
            background-color: rgba(76, 175, 80, 0.2);
            color: #2E7D32;
        }
        
        .status-running {
            background-color: rgba(33, 150, 243, 0.2);
            color: #1565C0;
            animation: pulse 1.5s infinite;
        }
        
        .status-failed {
            background-color: rgba(244, 67, 54, 0.2);
            color: #C62828;
        }
        
        .status-pending {
            background-color: rgba(158, 158, 158, 0.2);
            color: #424242;
        }
        
        .stage-body {
            padding: 8px 12px 12px 32px;
            font-size: 0.9rem;
            opacity: 0.9;
            border-radius: 0 0 8px 8px;
            background-color: rgba(0, 0, 0, 0.02);
            border-left: 1px solid rgba(0, 0, 0, 0.05);
            border-right: 1px solid rgba(0, 0, 0, 0.05);
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .stage-progress-bar {
            height: 6px;
            background-color: rgba(0, 0, 0, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin: 8px 0;
        }
        
        .stage-progress-fill {
            height: 100%;
            border-radius: 3px;
        }
        
        @keyframes pulse {
            0% { opacity: 0.7; }
            50% { opacity: 1; }
            100% { opacity: 0.7; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Expected stage order
    stage_order = [
        "document_analysis", "chunking", "planning", 
        "extraction", "aggregation", "evaluation", 
        "formatting", "review"
    ]
    
    # Display stages in order
    for stage_name in stage_order:
        if stage_name in stages_info:
            stage_info = stages_info[stage_name]
            status = stage_info.get("status", "not_started")
            progress = stage_info.get("progress", 0)
            message = stage_info.get("message", "")
            
            # Format display name
            display_name = stage_name.replace("_", " ").title()
            
            # Determine icon and status class
            if status == "completed":
                icon = "✅"
                status_class = "status-completed"
            elif status == "running":
                icon = "⏳"
                status_class = "status-running"
            elif status == "failed":
                icon = "❌"
                status_class = "status-failed"
            else:
                icon = "⏱️"
                status_class = "status-pending"
                
            # Determine progress color
            if status == "completed":
                progress_color = "#4CAF50"  # Green
            elif status == "running":
                progress_color = "#2196F3"  # Blue
            elif status == "failed":
                progress_color = "#F44336"  # Red
            else:
                progress_color = "#9E9E9E"  # Grey
                
            # Format progress percentage
            progress_pct = progress * 100
            
            # Render stage container
            st.markdown(f"""
            <div class="stage-container">
                <div class="stage-header">
                    <div class="stage-name">
                        <span class="stage-icon">{icon}</span>
                        {display_name}
                    </div>
                    <div class="stage-status {status_class}">{status.upper()}</div>
                </div>
                
                <div class="stage-progress-bar">
                    <div class="stage-progress-fill" style="width: {progress_pct}%; background-color: {progress_color};"></div>
                </div>
                
                <div class="stage-body">
                    {message}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# Option Selection Components
# =====================================================================

def enhanced_option_selector(options, title, key_prefix, selected_key):
    """
    Enhanced selection component with visual cards for options.
    
    Args:
        options: Dictionary of options {key: {title: str, description: str}}
        title: Title for the selector
        key_prefix: Prefix for Streamlit radio keys
        selected_key: Currently selected option key
    """
    st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
    
    # Add custom styling for the selector
    st.markdown("""
    <style>
        .option-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        
        .option-card {
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            padding: 16px;
            border: 2px solid transparent;
            transition: all 0.2s ease;
            height: 100%;
        }
        
        .option-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .option-card.selected {
            border-color: #2196F3;
            background-color: rgba(33, 150, 243, 0.1);
        }
        
        .option-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 8px;
            color: #2196F3;
        }
        
        .option-description {
            font-size: 0.9rem;
            opacity: 0.8;
            line-height: 1.4;
        }
        
        /* Hide the actual radio buttons */
        .option-radio-container {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Start option container
    st.markdown('<div class="option-container">', unsafe_allow_html=True)
    
    # Create invisible container for the actual radio buttons
    st.markdown('<div class="option-radio-container">', unsafe_allow_html=True)
    
    # Create the actual radio buttons (invisible)
    option_choice = st.radio(
        "Select Option",
        options=list(options.keys()),
        index=list(options.keys()).index(selected_key) if selected_key in options else 0,
        key=f"{key_prefix}_radio",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create option cards
    for option_key, option_info in options.items():
        # Get option values
        option_title = option_info.get("title", option_key)
        option_description = option_info.get("description", "")
        
        # Check if this option is selected
        is_selected = option_key == option_choice
        selected_class = "selected" if is_selected else ""
        
        # Create the option card with onclick handler
        st.markdown(f"""
        <div class="option-card {selected_class}" 
             onclick="document.querySelector('input[name=\"{key_prefix}_radio\"][value=\"{option_key}\"]').click()">
            <div class="option-title">{option_title}</div>
            <div class="option-description">{option_description}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # End option container
    st.markdown('</div>', unsafe_allow_html=True)
    
    return option_choice

def format_selector(selected_format="executive"):
    """
    Display a selector for summary formats with enhanced styling.
    
    Args:
        selected_format: Currently selected format
        
    Returns:
        Selected format key
    """
    # Format options with clear descriptions
    format_options = {
        "executive": {
            "title": "Executive Summary",
            "description": "A concise overview highlighting only the most important information (5-10% of original length)."
        },
        "comprehensive": {
            "title": "Comprehensive Summary",
            "description": "A detailed summary covering all significant aspects of the document (15-25% of original length)."
        },
        "bullet_points": {
            "title": "Key Points",
            "description": "Important information organized into easy-to-scan bullet points grouped by topic."
        },
        "narrative": {
            "title": "Narrative Summary",
            "description": "A flowing narrative that captures the document's content while maintaining its tone."
        }
    }
    
    return enhanced_option_selector(format_options, "Summary Format", "format", selected_format)

def detail_level_selector(selected_level="standard"):
    """
    Display a selector for detail levels with enhanced styling.
    
    Args:
        selected_level: Currently selected level
        
    Returns:
        Selected detail level key
    """
    # Detail level options with descriptions
    detail_options = {
        "essential": {
            "title": "Essential Issues Only",
            "description": "Focus only on the most significant issues and problems."
        },
        "standard": {
            "title": "Standard Assessment",
            "description": "Balanced analysis of issues with moderate detail."
        },
        "comprehensive": {
            "title": "Comprehensive Analysis",
            "description": "In-depth assessment of all potential issues and risks."
        }
    }
    
    return enhanced_option_selector(detail_options, "Detail Level", "detail", selected_level)

# =====================================================================
# Result Display Components
# =====================================================================

# Dark theme summary results
def display_summary_result(data):
    """
    Display summary results with dark theme styling and proper markdown rendering.
    
    Args:
        data: Summary data dictionary from DataAccessor
    """
    # Extract key data
    format_type = data.get("format_type", "executive")
    summary_content = data.get("summary_content", "")
    executive_summary = data.get("executive_summary", "")
    key_points = data.get("key_points", [])
    topics = data.get("topics", [])
    
    # Format names for display
    format_display_names = {
        "executive": "Executive Summary",
        "comprehensive": "Comprehensive Summary",
        "bullet_points": "Key Points Summary",
        "narrative": "Narrative Summary"
    }
    
    # Function to convert basic markdown to HTML
    def process_markdown(text):
        if not text:
            return ""
        
        # Process bold formatting
        text = text.replace("**", "<strong>", 1)
        while "**" in text:
            text = text.replace("**", "</strong>", 1)
            if "**" in text:
                text = text.replace("**", "<strong>", 1)
        
        # Process italic formatting
        text = text.replace("*", "<em>", 1)
        while "*" in text:
            text = text.replace("*", "</em>", 1)
            if "*" in text:
                text = text.replace("*", "<em>", 1)
        
        # Process underline (not standard markdown but sometimes used)
        text = text.replace("__", "<u>", 1)
        while "__" in text:
            text = text.replace("__", "</u>", 1)
            if "__" in text:
                text = text.replace("__", "<u>", 1)
                
        # Process links
        import re
        # Find markdown links [text](url)
        link_pattern = r'\[(.*?)\]\((.*?)\)'
        text = re.sub(link_pattern, r'<a href="\2" target="_blank" style="color: #2196F3; text-decoration: underline;">\1</a>', text)
        
        # Process paragraphs (blank lines)
        text = text.replace("\n\n", "</p><p>")
        
        # Process line breaks
        text = text.replace("\n", "<br>")
        
        return f"<p>{text}</p>"
    
    # Process markdown in content
    summary_content_html = process_markdown(summary_content)
    executive_summary_html = process_markdown(executive_summary)
    
    # Create summary container
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(20, 55, 97, 0.5), rgba(42, 36, 83, 0.5)); 
                border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid rgba(33, 150, 243, 0.3);">
        <h2 style="color: #2196F3; margin-bottom: 15px;">{format_display_names.get(format_type, "Document Summary")}</h2>
    """, unsafe_allow_html=True)
    
    # Display executive summary if available
    if executive_summary:
        st.markdown(f"""
        <div style="background-color: rgba(33, 150, 243, 0.15); padding: 15px; border-radius: 8px; 
                  margin-bottom: 20px; border-left: 3px solid #2196F3; color: rgba(255, 255, 255, 0.9);">
            {executive_summary_html}
        </div>
        """, unsafe_allow_html=True)
    
    # Display content based on format type
    if format_type == "bullet_points":
        # Display topics if available
        if topics and isinstance(topics, list) and len(topics) > 0:
            for i, topic in enumerate(topics):
                if isinstance(topic, dict):
                    topic_name = topic.get("topic", "")
                    if topic_name:
                        st.markdown(f"<h3 style='color: #2196F3;'>{topic_name}</h3>", unsafe_allow_html=True)
                        
                        # Display topic key points
                        topic_points = topic.get("key_points", [])
                        if topic_points and isinstance(topic_points, list):
                            for j, point in enumerate(topic_points):
                                if isinstance(point, str):
                                    text = process_markdown(point)
                                elif isinstance(point, dict) and "text" in point:
                                    text = process_markdown(point["text"])
                                else:
                                    continue
                                    
                                # Remove outer <p> tags if present for better rendering
                                text = text.replace("<p>", "", 1) if text.startswith("<p>") else text
                                text = text.replace("</p>", "", 1) if text.endswith("</p>") else text
                                
                                st.markdown(f"""
                                <div style="padding: 10px 15px; background-color: rgba(40, 40, 40, 0.7); border-radius: 8px; margin-bottom: 10px; 
                                          border-left: 3px solid #2196F3; display: flex; align-items: center; color: rgba(255, 255, 255, 0.9);">
                                    <div style="background-color: #2196F3; color: white; border-radius: 50%; width: 24px; height: 24px; 
                                              display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">
                                        {j+1}
                                    </div>
                                    <div style="flex: 1;">{text}</div>
                                </div>
                                """, unsafe_allow_html=True)
        # If no topics, display flat key points
        elif key_points and isinstance(key_points, list):
            for i, point in enumerate(key_points):
                if isinstance(point, dict):
                    text = point.get("text", "")
                    importance = point.get("importance", "medium")
                elif isinstance(point, str):
                    text = point
                    importance = "medium"
                else:
                    continue
                
                # Process markdown in point text
                text_html = process_markdown(text)
                
                # Remove outer <p> tags if present for better rendering
                text_html = text_html.replace("<p>", "", 1) if text_html.startswith("<p>") else text_html
                text_html = text_html.replace("</p>", "", 1) if text_html.endswith("</p>") else text_html
                    
                importance_colors = {
                    "high": "#F44336",
                    "medium": "#2196F3", 
                    "low": "#4CAF50"
                }
                color = importance_colors.get(importance, "#2196F3")
                
                st.markdown(f"""
                <div style="padding: 12px 15px; background-color: rgba(40, 40, 40, 0.7); border-radius: 8px; margin-bottom: 10px; 
                          border-left: 3px solid {color}; display: flex; align-items: center; color: rgba(255, 255, 255, 0.9);">
                    <div style="background-color: {color}; color: white; border-radius: 50%; width: 24px; height: 24px; 
                              display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">
                        {i+1}
                    </div>
                    <div style="flex: 1;">{text_html}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # For executive, comprehensive and narrative formats
        if summary_content:
            st.markdown(f"""
            <div style="background-color: rgba(40, 40, 40, 0.7); padding: 20px; border-radius: 8px; line-height: 1.6; 
                      margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); color: rgba(255, 255, 255, 0.9);">
                {summary_content_html}
            </div>
            """, unsafe_allow_html=True)
    
    # End summary container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display statistics
    # End summary container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display statistics with better data extraction
    statistics = data.get("statistics", {})
    metadata = data.get("metadata", {})
    
    # Get original word count - look in multiple places for it
    original_words = statistics.get("original_word_count", 0)
    if original_words == 0:
        # Try getting it from metadata or document_info
        document_info = metadata.get("document_info", {})
        original_words = document_info.get("word_count", 0)
    
    # Get summary word count
    summary_words = statistics.get("summary_word_count", 0)
    if summary_words == 0 and summary_content:
        # Calculate it directly from content
        summary_words = len(summary_content.split())
    
    # Calculate compression ratio if needed
    compression = statistics.get("compression_ratio", 0)
    if compression == 0 and original_words > 0 and summary_words > 0:
        compression = (summary_words / original_words) * 100
    
    # Only display if we have meaningful stats
    if original_words > 0 or summary_words > 0:
        st.markdown("<h3 style='color: rgba(255, 255, 255, 0.9);'>Summary Statistics</h3>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Original Document", f"{original_words:,} words")
            
        with cols[1]:
            st.metric("Summary Length", f"{summary_words:,} words")
            
        with cols[2]:
            if original_words and summary_words:
                st.metric("Compression Ratio", f"{compression:.1f}%")
            else:
                st.metric("Compression Ratio", "N/A")


# Dark theme issues results
def display_issues_result(data):
    """
    Display issues assessment results with dark theme styling and proper markdown rendering.
    
    Args:
        data: Issues data dictionary from DataAccessor
    """
    # Extract key data
    issues = data.get("issues", [])
    executive_summary = data.get("executive_summary", "")
    
    # Function to convert basic markdown to HTML
    def process_markdown(text):
        if not text:
            return ""
        
        # Process bold formatting
        text = text.replace("**", "<strong>", 1)
        while "**" in text:
            text = text.replace("**", "</strong>", 1)
            if "**" in text:
                text = text.replace("**", "<strong>", 1)
        
        # Process italic formatting
        text = text.replace("*", "<em>", 1)
        while "*" in text:
            text = text.replace("*", "</em>", 1)
            if "*" in text:
                text = text.replace("*", "<em>", 1)
        
        # Process underline (not standard markdown but sometimes used)
        text = text.replace("__", "<u>", 1)
        while "__" in text:
            text = text.replace("__", "</u>", 1)
            if "__" in text:
                text = text.replace("__", "<u>", 1)
                
        # Process links
        import re
        # Find markdown links [text](url)
        link_pattern = r'\[(.*?)\]\((.*?)\)'
        text = re.sub(link_pattern, r'<a href="\2" target="_blank" style="color: #2196F3; text-decoration: underline;">\1</a>', text)
        
        # Process paragraphs (blank lines)
        text = text.replace("\n\n", "</p><p>")
        
        # Process line breaks
        text = text.replace("\n", "<br>")
        
        return f"<p>{text}</p>"
    
    # Process markdown in executive summary
    executive_summary_html = process_markdown(executive_summary)
    
    # Create issues container
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(97, 20, 20, 0.5), rgba(97, 55, 20, 0.5)); 
                border-radius: 10px; padding: 20px; margin: 20px 0; border: 1px solid rgba(244, 67, 54, 0.3);">
        <h2 style="color: #F44336; margin-bottom: 15px;">Issue Assessment Results</h2>
    """, unsafe_allow_html=True)
    
    # Display executive summary if available
    if executive_summary:
        st.markdown(f"""
        <div style="background-color: rgba(244, 67, 54, 0.15); padding: 15px; border-radius: 8px; 
                  margin-bottom: 20px; border-left: 3px solid #F44336; color: rgba(255, 255, 255, 0.9);">
            {executive_summary_html}
        </div>
        """, unsafe_allow_html=True)
    
    # Group issues by severity
    issues_by_severity = {"critical": [], "high": [], "medium": [], "low": []}
    for issue in issues:
        if isinstance(issue, dict):
            # Use evaluated_severity if available, otherwise use severity
            severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
            if severity in issues_by_severity:
                issues_by_severity[severity].append(issue)
    
    # Severity colors and icons
    severity_styles = {
        "critical": {"color": "#F44336", "icon": "🔴", "border": "2px solid #F44336"},
        "high": {"color": "#FF9800", "icon": "🟠", "border": "2px solid #FF9800"},
        "medium": {"color": "#2196F3", "icon": "🔵", "border": "1px solid #2196F3"},
        "low": {"color": "#4CAF50", "icon": "🟢", "border": "1px solid #4CAF50"}
    }
    
    # Display issue count
    total_issues = len(issues)
    st.markdown(f"<h3 style='color: rgba(255, 255, 255, 0.9);'>Total Issues Found: {total_issues}</h3>", unsafe_allow_html=True)
    
    # Display each severity group
    for severity, issues_list in issues_by_severity.items():
        if issues_list:
            style = severity_styles.get(severity)
            
            st.markdown(f"""
            <h3 style="color: {style['color']}; display: flex; align-items: center; margin-top: 25px;">
                <span style="margin-right: 10px;">{style['icon']}</span>
                {severity.upper()} Issues ({len(issues_list)})
            </h3>
            <hr style="height: 2px; border: none; background-color: {style['color']}; margin-bottom: 15px;">
            """, unsafe_allow_html=True)
            
            for issue in issues_list:
                title = issue.get("title", "Untitled Issue")
                description = issue.get("description", "")
                category = issue.get("category", "uncategorized").upper()
                
                # Process markdown in description
                description_html = process_markdown(description)
                
                # Remove outer <p> tags if present for better rendering
                description_html = description_html.replace("<p>", "", 1) if description_html.startswith("<p>") else description_html
                description_html = description_html.replace("</p>", "", 1) if description_html.endswith("</p>") else description_html
                
                st.markdown(f"""
                <div style="padding: 15px; background-color: rgba(40, 40, 40, 0.7); border-radius: 8px; margin-bottom: 15px; 
                          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); border-left: 4px solid {style['color']};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <h4 style="color: {style['color']}; margin: 0;">{title}</h4>
                        <span style="background-color: rgba(60, 60, 60, 0.7); padding: 2px 8px; border-radius: 4px; 
                             font-size: 0.8rem; font-weight: 500; color: rgba(255, 255, 255, 0.9);">{category}</span>
                    </div>
                    <div style="margin-bottom: 0; color: rgba(255, 255, 255, 0.9);">{description_html}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show impact and recommendations in an expander
                impact = issue.get("potential_impact", issue.get("impact", ""))
                recommendations = issue.get("recommendations", issue.get("suggested_recommendations", []))
                
                if impact or recommendations:
                    with st.expander("View Details", expanded=False):
                        if impact:
                            st.markdown("**Potential Impact:**")
                            st.info(process_markdown(impact))
                        
                        if recommendations:
                            st.markdown("**Recommendations:**")
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
    
    # End issues container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display statistics
    statistics = data.get("statistics", {})
    if statistics:
        by_severity = statistics.get("by_severity", {})
        by_category = statistics.get("by_category", {})
        
        if by_severity:
            st.markdown("<h3 style='color: rgba(255, 255, 255, 0.9);'>Issue Distribution</h3>", unsafe_allow_html=True)
            
            # Create bar chart for severity distribution
            for severity, count in by_severity.items():
                if severity in severity_styles:
                    style = severity_styles.get(severity)
                    percentage = (count / total_issues * 100) if total_issues > 0 else 0
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <div style="display: flex; align-items: center; color: rgba(255, 255, 255, 0.9);">
                                <span style="margin-right: 8px;">{style['icon']}</span>
                                <span style="text-transform: capitalize;">{severity}</span>
                            </div>
                            <div style="color: rgba(255, 255, 255, 0.9);">{count} ({percentage:.1f}%)</div>
                        </div>
                        <div style="height: 8px; background-color: rgba(60, 60, 60, 0.7); border-radius: 4px; overflow: hidden;">
                            <div style="width: {percentage}%; height: 100%; background-color: {style['color']}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# =====================================================================
# Utility Functions
# =====================================================================

def get_enhanced_styles():
    """
    Get enhanced CSS styles for consistent styling across pages.
    
    Returns:
        CSS styles as a string
    """
    return """
    <style>
        /* Global typography improvements */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 600;
        }
        
        /* Custom section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            display: flex;
            align-items: center;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .section-number {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 36px;
            height: 36px;
            background-color: #2196F3;
            color: white;
            border-radius: 50%;
            margin-right: 12px;
            font-weight: 600;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        .section-icon {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 36px;
            height: 36px;
            background-color: rgba(33, 150, 243, 0.2);
            color: #2196F3;
            border-radius: 50%;
            margin-right: 12px;
            font-weight: 600;
        }
        
        /* Enhanced buttons */
        .stButton > button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Card styling */
        .card {
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 500 !important;
            color: #2196F3 !important;
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
    </style>
    """

def apply_enhanced_theme():
    """Apply enhanced theme to the current page."""
    st.markdown(get_enhanced_styles(), unsafe_allow_html=True)

def get_severity_color(severity):
    """Get color for severity level."""
    colors = {
        "critical": "#F44336",
        "high": "#FF9800",
        "medium": "#2196F3",
        "low": "#4CAF50"
    }
    return colors.get(severity, "#2196F3")