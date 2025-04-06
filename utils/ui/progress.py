"""
Progress tracking components for the Beyond Notes application.
Visualizes pipeline stages and processing status.
"""

import streamlit as st

def render_pipeline_status(session_state):
    """Display the current pipeline status with enhanced visuals."""
    progress_value = float(session_state.get("current_progress", 0.0))
    
    # Progress container with better styling
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    
    # Enhanced progress bar
    progress_bar = st.progress(progress_value)
    
    # Current stage and progress message
    current_stage = session_state.get("current_stage", "")
    progress_message = session_state.get("progress_message", "Waiting to start...")
    
    if current_stage:
        st.markdown(f'<div class="progress-stage">{current_stage.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="progress-message">{progress_message}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show detailed stage information if available
    stages_info = session_state.get("stages_info", {})
    if stages_info:
        with st.expander("View detailed progress", expanded=False):
            for stage_name, stage_info in stages_info.items():
                status = stage_info.get("status", "not_started")
                progress = float(stage_info.get("progress", 0.0))
                message = stage_info.get("message", "")
                
                # Determine emoji based on status
                if status == "completed": emoji = "✅"
                elif status == "running": emoji = "⏳"
                elif status == "failed": emoji = "❌"
                else: emoji = "⏱️"
                
                display_name = stage_name.replace("_", " ").title()
                progress_pct = f"{int(progress * 100)}%" if progress > 0 else ""
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{emoji} **{display_name}** {progress_pct}")
                    if message:
                        st.caption(f"> {message[:100]}")
                with col2:
                    st.markdown(f"`{status.upper()}`")

def render_detailed_progress(session_state):
    """Display detailed progress information with terminal-like output."""
    progress_value = float(session_state.get("current_progress", 0.0))
    
    # Create main progress bar
    st.progress(progress_value)
    
    # Show current stage and message
    current_stage = session_state.get("current_stage", "")
    progress_message = session_state.get("progress_message", "Waiting to start...")
    
    if current_stage:
        st.markdown(f"**Current Stage:** {current_stage.replace('_', ' ').title()}")
    st.caption(progress_message)
    
    # Show detailed stage information
    stages_info = session_state.get("stages_info", {})
    if stages_info:
        with st.expander("View Processing Details", expanded=True):
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
                    
                    # Determine status icon
                    if status == "completed":
                        icon = "✅"
                    elif status == "running":
                        icon = "⏳"
                    elif status == "failed":
                        icon = "❌"
                    else:
                        icon = "⏱️"
                    
                    # Display stage status
                    display_name = stage_name.replace("_", " ").title()
                    progress_pct = f"{int(progress * 100)}%" if progress > 0 else ""
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"{icon} **{display_name}** {progress_pct}")
                        if message:
                            st.caption(f"> {message}")
                    with col2:
                        st.markdown(f"`{status.upper()}`")