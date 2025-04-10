# pages/05_Refine.py
import streamlit as st
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-refine")

# Import components
from utils.paths import AppPaths
from utils.formatting import format_assessment_report

# Ensure directories exist
AppPaths.ensure_dirs()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Refine Results",
    page_icon="✏️",
    layout="wide",
)

# Define accent colors
PRIMARY_COLOR = "#4CAF50"  # Green
SECONDARY_COLOR = "#2196F3"  # Blue
ACCENT_COLOR = "#FF9800"  # Orange
ERROR_COLOR = "#F44336"  # Red

# CSS to customize the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .item-editor {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #2196F3;
    }
    .json-editor {
        font-family: monospace;
        height: 400px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_refine_page():
    """Initialize the session state variables for the refine page."""
    if "current_assessment" not in st.session_state:
        st.session_state.current_assessment = None
    
    if "edited_assessment" not in st.session_state:
        st.session_state.edited_assessment = None
    
    if "editing_mode" not in st.session_state:
        st.session_state.editing_mode = "visual"  # visual or json
    
    if "edited_item_index" not in st.session_state:
        st.session_state.edited_item_index = None

def load_recent_assessments():
    """Load a list of recent assessments."""
    assessment_types = ["distill", "extract", "assess", "analyze"]
    recent_assessments = []
    
    # Look for assessment results in output directories
    for assessment_type in assessment_types:
        output_dir = AppPaths.get_assessment_output_dir(assessment_type)
        
        if output_dir.exists():
            # Look for JSON result files
            for file_path in output_dir.glob(f"{assessment_type}_result_*.json"):
                try:
                    # Get file stats
                    stat = file_path.stat()
                    
                    # Try to load basic metadata
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            metadata = data.get("metadata", {})
                            
                            # Create assessment entry
                            assessment = {
                                "id": file_path.stem,
                                "path": file_path,
                                "type": assessment_type,
                                "display_name": metadata.get("assessment_display_name", assessment_type.title()),
                                "document_name": metadata.get("document_info", {}).get("filename", "Unknown Document"),
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "data": data
                            }
                            
                            recent_assessments.append(assessment)
                        except json.JSONDecodeError:
                            # Skip invalid JSON files
                            logger.warning(f"Invalid JSON in file: {file_path}")
                            continue
                except Exception as e:
                    logger.warning(f"Error loading assessment file {file_path}: {str(e)}")
                    continue
    
    # Sort by modification time (most recent first)
    recent_assessments.sort(key=lambda x: x["modified"], reverse=True)
    
    return recent_assessments

def save_assessment(assessment_data, original_path):
    """Save the edited assessment data."""
    try:
        # Create a new file with timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        assessment_type = assessment_data.get("metadata", {}).get("assessment_type", "unknown")
        
        # Create output directory if it doesn't exist
        output_dir = AppPaths.get_assessment_output_dir(assessment_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with edited suffix
        filename = f"{assessment_type}_result_{timestamp}_edited.json"
        output_path = output_dir / filename
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(assessment_data, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report as well
        try:
            report_md = format_assessment_report(assessment_data, assessment_type)
            md_filename = f"{assessment_type}_report_{timestamp}_edited.md"
            md_path = output_dir / md_filename
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(report_md)
                
            return output_path, md_path
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            return output_path, None
            
    except Exception as e:
        logger.error(f"Error saving assessment: {str(e)}")
        return None, None

def get_items_key(assessment_type):
    """Get the key for items based on assessment type."""
    return {
        "distill": "key_points",
        "extract": "action_items",
        "assess": "issues",
        "analyze": "evidence"
    }.get(assessment_type, "items")

def get_item_editor_fields(assessment_type):
    """Get the fields to display in the item editor based on assessment type."""
    if assessment_type == "distill":
        return {
            "text": {"type": "text_area", "label": "Key Point Text"},
            "importance": {"type": "select", "label": "Importance", "options": ["High", "Medium", "Low"]},
            "point_type": {"type": "select", "label": "Type", "options": ["Fact", "Decision", "Question", "Insight", "Quote"]},
            "topic": {"type": "text", "label": "Topic"}
        }
    elif assessment_type == "extract":
        return {
            "description": {"type": "text_area", "label": "Action Item Description"},
            "owner": {"type": "text", "label": "Owner"},
            "due_date": {"type": "text", "label": "Due Date"},
            "priority": {"type": "select", "label": "Priority", "options": ["high", "medium", "low"]}
        }
    elif assessment_type == "assess":
        return {
            "title": {"type": "text", "label": "Issue Title"},
            "description": {"type": "text_area", "label": "Issue Description"},
            "severity": {"type": "select", "label": "Severity", "options": ["critical", "high", "medium", "low"]},
            "category": {"type": "select", "label": "Category", "options": ["technical", "process", "resource", "quality", "risk", "compliance"]},
            "impact": {"type": "text_area", "label": "Impact"},
            "recommendations": {"type": "text_area", "label": "Recommendations"}
        }
    elif assessment_type == "analyze":
        return {
            "dimension": {"type": "text", "label": "Dimension"},
            "criteria": {"type": "text", "label": "Criteria"},
            "evidence_text": {"type": "text_area", "label": "Evidence Text"},
            "commentary": {"type": "text_area", "label": "Commentary"},
            "maturity_rating": {"type": "number", "label": "Maturity Rating", "min": 1, "max": 5}
        }
    else:
        return {
            "text": {"type": "text_area", "label": "Item Text"}
        }

def display_visual_editor(assessment_data):
    """Display a visual editor for the assessment data."""
    assessment_type = assessment_data.get("metadata", {}).get("assessment_type", "unknown")
    result = assessment_data.get("result", {})
    
    # Get items key based on assessment type
    items_key = get_items_key(assessment_type)
    items = result.get(items_key, [])
    
    st.markdown(f"### Edit {items_key.replace('_', ' ').title()}")
    
    if not items:
        st.warning(f"No {items_key} found in this assessment.")
        return
    
    # Display items in a list with edit buttons
    for i, item in enumerate(items):
        with st.expander(f"Item {i+1}: {item.get('title', item.get('description', item.get('text', f'Item {i+1}')))}", expanded=st.session_state.edited_item_index == i):
            # Display item editor if this is the selected item
            if st.session_state.edited_item_index == i:
                edit_item(item, assessment_type, i)
            else:
                # Display summary and edit button
                col1, col2 = st.columns([3, 1])
                with col1:
                    for key, value in item.items():
                        if key not in ["id", "source", "evidence", "chunk_index"]:
                            if isinstance(value, list):
                                st.markdown(f"**{key.replace('_', ' ').title()}**: {', '.join(str(v) for v in value)}")
                            else:
                                st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
                
                with col2:
                    if st.button("Edit", key=f"edit_btn_{i}", use_container_width=True):
                        st.session_state.edited_item_index = i
                        st.rerun()
    
    # Add a new item button
    if st.button("Add New Item", type="primary"):
        # Create a new blank item
        new_item = {}
        # Add required fields based on assessment type
        field_defs = get_item_editor_fields(assessment_type)
        for field, config in field_defs.items():
            if config["type"] == "text" or config["type"] == "text_area":
                new_item[field] = ""
            elif config["type"] == "select" and "options" in config:
                new_item[field] = config["options"][0]
            elif config["type"] == "number":
                new_item[field] = config.get("min", 1)
        
        # Add ID
        new_item["id"] = f"item-{len(items)}"
        
        # Add to items list
        items.append(new_item)
        
        # Select the new item for editing
        st.session_state.edited_item_index = len(items) - 1
        
        # Update the assessment data
        result[items_key] = items
        st.session_state.edited_assessment = assessment_data
        
        st.rerun()

def edit_item(item, assessment_type, index):
    """Display an editor for a single item."""
    # Get fields to edit based on assessment type
    fields = get_item_editor_fields(assessment_type)
    
    # Store original values for comparison
    original_item = item.copy()
    
    # Create form for editing
    with st.form(key=f"item_edit_form_{index}"):
        edited_item = item.copy()
        
        # Create input fields for each editable field
        for field, config in fields.items():
            if config["type"] == "text":
                edited_item[field] = st.text_input(config["label"], value=item.get(field, ""))
            elif config["type"] == "text_area":
                edited_item[field] = st.text_area(config["label"], value=item.get(field, ""), height=100)
            elif config["type"] == "select":
                edited_item[field] = st.selectbox(config["label"], options=config["options"], index=config["options"].index(item.get(field, config["options"][0])) if item.get(field) in config["options"] else 0)
            elif config["type"] == "number":
                edited_item[field] = st.number_input(config["label"], min_value=config.get("min", 0), max_value=config.get("max", 100), value=int(item.get(field, config.get("min", 1))))
        
        # Add save and cancel buttons
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.form_submit_button("Save Changes", type="primary", use_container_width=True)
        with col2:
            cancel_button = st.form_submit_button("Cancel", type="secondary", use_container_width=True)
    
    # Handle form submission
    if save_button:
        # Update item with edited values
        for field in fields.keys():
            item[field] = edited_item[field]
        
        # Update assessment data
        assessment_data = st.session_state.edited_assessment
        assessment_type = assessment_data.get("metadata", {}).get("assessment_type", "unknown")
        items_key = get_items_key(assessment_type)
        assessment_data["result"][items_key][index] = item
        
        # Reset edit mode
        st.session_state.edited_item_index = None
        
        # Show success message
        st.success("Item updated successfully!")
        st.rerun()
    
    elif cancel_button:
        # Reset edit mode without saving
        st.session_state.edited_item_index = None
        st.rerun()

def display_json_editor(assessment_data):
    """Display a JSON editor for the assessment data."""
    # Convert assessment data to formatted JSON string
    assessment_json = json.dumps(assessment_data, indent=2)
    
    # Create a text area for editing
    edited_json = st.text_area("Edit Assessment JSON", assessment_json, height=600, key="json_editor")
    
    # Add save button
    if st.button("Save JSON Changes", type="primary"):
        try:
            # Parse the edited JSON
            updated_assessment = json.loads(edited_json)
            
            # Basic validation - ensure it has result and metadata
            if "result" not in updated_assessment or "metadata" not in updated_assessment:
                st.error("Invalid assessment structure. Must contain 'result' and 'metadata' keys.")
                return
            
            # Update session state
            st.session_state.edited_assessment = updated_assessment
            
            # Show success message
            st.success("JSON updated successfully!")
            
            # Switch back to visual mode
            st.session_state.editing_mode = "visual"
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {str(e)}")
        except Exception as e:
            st.error(f"Error updating assessment: {str(e)}")

def display_summary_editor(assessment_data):
    """Display an editor for summary fields like executive_summary."""
    assessment_type = assessment_data.get("metadata", {}).get("assessment_type", "unknown")
    result = assessment_data.get("result", {})
    
    # Fields to edit vary by assessment type
    if assessment_type == "distill":
        summary_fields = {
            "summary": {"type": "text_area", "label": "Document Summary"},
            "key_themes": {"type": "text_area", "label": "Key Themes"}
        }
    elif assessment_type in ["extract", "assess"]:
        summary_fields = {
            "executive_summary": {"type": "text_area", "label": "Executive Summary"}
        }
    elif assessment_type == "analyze":
        summary_fields = {
            "executive_summary": {"type": "text_area", "label": "Executive Summary"},
            "assessment_methodology": {"type": "text_area", "label": "Assessment Methodology"}
        }
    else:
        summary_fields = {
            "summary": {"type": "text_area", "label": "Summary"}
        }
    
    st.markdown("### Edit Summary Information")
    
    # Create form for editing summary fields
    with st.form(key="summary_edit_form"):
        updated = False
        
        for field, config in summary_fields.items():
            if field in result:
                if config["type"] == "text_area":
                    new_value = st.text_area(config["label"], value=result.get(field, ""), height=200)
                    if new_value != result.get(field):
                        result[field] = new_value
                        updated = True
        
        # Add save button
        save_button = st.form_submit_button("Save Summary Changes", type="primary")
    
    # Handle form submission
    if save_button:
        if updated:
            # Update assessment data
            assessment_data["result"] = result
            st.session_state.edited_assessment = assessment_data
            
            # Show success message
            st.success("Summary updated successfully!")
            st.rerun()
        else:
            st.info("No changes were made.")

def main():
    """Main function for the refine page."""
    # Initialize page
    initialize_refine_page()
    
    # Header
    st.markdown('<div class="main-header">Refine Assessment Results</div>', unsafe_allow_html=True)
    st.markdown("Edit and refine your assessment results before sharing them.")
    
    # Sidebar for assessment selection
    with st.sidebar:
        st.markdown("### Select Assessment")
        
        # Load recent assessments
        recent_assessments = load_recent_assessments()
        
        if not recent_assessments:
            st.warning("No assessments found. Process a document in the Assess tab first.")
        else:
            # Group by assessment type
            assessment_types = {}
            for assessment in recent_assessments:
                a_type = assessment.get("type")
                if a_type not in assessment_types:
                    assessment_types[a_type] = []
                assessment_types[a_type].append(assessment)
            
            # Let user select assessment type first
            selected_type = st.selectbox(
                "Assessment Type",
                options=list(assessment_types.keys()),
                format_func=lambda x: {
                    "distill": "📝 Summarization",
                    "extract": "📋 Action Items",
                    "assess": "⚠️ Issues & Risks",
                    "analyze": "📊 Framework Analysis"
                }.get(x, x.title())
            )
            
            if selected_type:
                # Then show assessments of that type
                type_assessments = assessment_types.get(selected_type, [])
                
                assessment_options = {
                    f"{a['document_name']} ({a['modified'].strftime('%Y-%m-%d %H:%M')})": i
                    for i, a in enumerate(type_assessments)
                }
                
                selected_assessment_key = st.selectbox(
                    "Document",
                    options=list(assessment_options.keys()),
                    format_func=lambda x: x
                )
                
                if selected_assessment_key:
                    assessment_idx = assessment_options.get(selected_assessment_key)
                    selected_assessment = type_assessments[assessment_idx]
                    
                    # Load assessment data if not already loaded
                    if (not st.session_state.current_assessment or 
                        st.session_state.current_assessment.get("id") != selected_assessment.get("id")):
                        
                        # Reset any editing state
                        st.session_state.edited_item_index = None
                        st.session_state.editing_mode = "visual"
                        
                        # Store assessment info in session state
                        st.session_state.current_assessment = selected_assessment
                        st.session_state.edited_assessment = selected_assessment.get("data").copy()
                        
                        # Show success message
                        st.success(f"Loaded assessment: {selected_assessment.get('display_name')}")
        
        # Editing mode toggle
        if st.session_state.current_assessment:
            st.markdown("### Editing Mode")
            edit_mode = st.radio(
                "Select Editing Mode",
                options=["Visual Editor", "JSON Editor"],
                index=0 if st.session_state.editing_mode == "visual" else 1,
                horizontal=True
            )
            
            # Update editing mode
            st.session_state.editing_mode = "visual" if edit_mode == "Visual Editor" else "json"
    
    # Main content - Editor
    if not st.session_state.current_assessment or not st.session_state.edited_assessment:
        st.info("Select an assessment from the sidebar to start editing.")
    else:
        # Display current assessment info
        assessment_data = st.session_state.edited_assessment
        assessment_type = assessment_data.get("metadata", {}).get("assessment_type", "unknown")
        document_name = assessment_data.get("metadata", {}).get("document_info", {}).get("filename", "Unknown Document")
        
        st.markdown(f"### Editing: {document_name}")
        st.caption(f"Assessment Type: {assessment_type.title()}")
        
        # Save and export section
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Save Changes", type="primary", use_container_width=True):
                # Save the edited assessment
                json_path, md_path = save_assessment(
                    assessment_data, 
                    st.session_state.current_assessment.get("path")
                )
                
                if json_path:
                    st.success(f"Assessment saved successfully to {json_path.name}")
                    
                    # Provide download links
                    st.markdown("### Download Files")
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(json_path, "rb") as f:
                            st.download_button(
                                label="Download JSON",
                                data=f,
                                file_name=json_path.name,
                                mime="application/json"
                            )
                    
                    if md_path and md_path.exists():
                        with col2:
                            with open(md_path, "rb") as f:
                                st.download_button(
                                    label="Download Markdown",
                                    data=f,
                                    file_name=md_path.name,
                                    mime="text/markdown"
                                )
                else:
                    st.error("Failed to save assessment. Check logs for details.")
        
        with col2:
            if st.button("Discard Changes", type="secondary", use_container_width=True):
                # Reload the original assessment
                st.session_state.edited_assessment = st.session_state.current_assessment.get("data").copy()
                st.session_state.edited_item_index = None
                st.success("Changes discarded. Reverted to original assessment.")
                st.rerun()
        
        # Editor sections
        if st.session_state.editing_mode == "visual":
            # Visual editor with tabs for different sections
            tabs = st.tabs(["Summary", f"{get_items_key(assessment_type).replace('_', ' ').title()}"])
            
            with tabs[0]:
                display_summary_editor(assessment_data)
            
            with tabs[1]:
                display_visual_editor(assessment_data)
        else:
            # JSON editor
            display_json_editor(assessment_data)

if __name__ == "__main__":
    main()