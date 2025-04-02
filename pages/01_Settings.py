# pages/01_Settings.py
import streamlit as st
import json
import logging
import copy
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-settings")

# Import components
from assessments.loader import AssessmentLoader
from utils.paths import AppPaths

# Ensure directories exist
AppPaths.ensure_dirs()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Settings",
    page_icon="⚙️",
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
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .template-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
    }
    .base-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        background-color: rgba(255, 255, 255, 0.05);
        margin-bottom: 1rem;
    }
    .json-editor {
        font-family: monospace;
        height: 400px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_settings_page():
    """Initialize the session state variables for the settings page."""
    if "selected_config_id" not in st.session_state:
        st.session_state.selected_config_id = None
    
    if "selected_config" not in st.session_state:
        st.session_state.selected_config = None
    
    if "editing_mode" not in st.session_state:
        st.session_state.editing_mode = False
    
    if "creating_template" not in st.session_state:
        st.session_state.creating_template = False
    
    if "edited_config" not in st.session_state:
        st.session_state.edited_config = None

def load_assessment_loader():
    """Load or get the assessment loader."""
    if "assessment_loader" not in st.session_state:
        try:
            st.session_state.assessment_loader = AssessmentLoader()
            logger.info("Assessment loader initialized")
        except Exception as e:
            st.error(f"Error initializing assessment loader: {str(e)}")
            logger.error(f"Error initializing assessment loader: {e}", exc_info=True)
            return None
    
    return st.session_state.assessment_loader

def load_config_by_id(assessment_id):
    """Load a specific assessment configuration."""
    try:
        loader = load_assessment_loader()
        if not loader:
            return None
        
        config = loader.load_config(assessment_id)
        if not config:
            st.error(f"Configuration '{assessment_id}' not found.")
            return None
        
        return config
    except Exception as e:
        st.error(f"Error loading configuration '{assessment_id}': {str(e)}")
        logger.error(f"Error loading configuration '{assessment_id}': {e}", exc_info=True)
        return None

def create_template_from_base(base_id, template_id, display_name, description):
    """Create a new template from a base assessment."""
    try:
        loader = load_assessment_loader()
        if not loader:
            return False
        
        new_template_id = loader.create_template_from_base(
            base_assessment_id=base_id,
            new_template_id=template_id,
            display_name=display_name,
            description=description
        )
        
        if new_template_id:
            st.success(f"New template '{template_id}' created successfully!")
            return True
        else:
            st.error("Failed to create template. Check logs for details.")
            return False
    except Exception as e:
        st.error(f"Error creating template: {str(e)}")
        logger.error(f"Error creating template: {e}", exc_info=True)
        return False

def update_template(template_id, updated_config):
    """Update an existing template configuration."""
    try:
        loader = load_assessment_loader()
        if not loader:
            return False
        
        # Ensure we're not updating critical fields
        updates = copy.deepcopy(updated_config)
        if "assessment_id" in updates:
            del updates["assessment_id"]
        
        if "metadata" in updates and "is_template" in updates["metadata"]:
            del updates["metadata"]["is_template"]
        
        success = loader.update_template(
            template_id=template_id,
            config_updates=updates
        )
        
        if success:
            st.success(f"Template '{template_id}' updated successfully!")
            return True
        else:
            st.error("Failed to update template. Check logs for details.")
            return False
    except Exception as e:
        st.error(f"Error updating template: {str(e)}")
        logger.error(f"Error updating template: {e}", exc_info=True)
        return False

def delete_template(template_id):
    """Delete a template configuration."""
    try:
        loader = load_assessment_loader()
        if not loader:
            return False
        
        success = loader.delete_template(template_id)
        
        if success:
            st.success(f"Template '{template_id}' deleted successfully!")
            return True
        else:
            st.error("Failed to delete template. Check logs for details.")
            return False
    except Exception as e:
        st.error(f"Error deleting template: {str(e)}")
        logger.error(f"Error deleting template: {e}", exc_info=True)
        return False

def display_config_editor(config):
    """Display a JSON editor for the configuration."""
    # Convert config to formatted JSON string
    config_json = json.dumps(config, indent=2)
    
    # Create a multiline text editor
    edited_json = st.text_area("Edit Configuration", config_json, height=400, key="json_editor")
    
    # Add save button
    if st.button("Save Changes", type="primary"):
        try:
            # Parse the edited JSON
            updated_config = json.loads(edited_json)
            
            # Validate basic structure
            if "assessment_id" not in updated_config:
                st.error("Missing required 'assessment_id' field.")
                return None
            
            # Check if this is a template
            is_template = updated_config.get("metadata", {}).get("is_template", False)
            template_id = updated_config.get("assessment_id")
            
            if is_template:
                # Update the template
                if update_template(template_id, updated_config):
                    # Reload the config
                    st.session_state.edited_config = updated_config
                    st.session_state.selected_config = updated_config
                    
                    # Reset editing mode
                    st.session_state.editing_mode = False
                    
                    # Rerun to update UI
                    st.rerun()
            else:
                st.error("Only template configurations can be edited.")
                return None
                
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")
            logger.error(f"Error saving configuration: {e}", exc_info=True)
            return None
    
    # Add cancel button
    if st.button("Cancel", type="secondary"):
        st.session_state.editing_mode = False
        st.rerun()

def display_config_details(config):
    """Display details of a configuration."""
    if not config:
        st.warning("No configuration selected.")
        return
    
    # Get basic info
    assessment_id = config.get("assessment_id", "Unknown")
    assessment_type = config.get("assessment_type", "Unknown")
    display_name = config.get("display_name", assessment_id)
    description = config.get("description", "No description available.")
    version = config.get("version", "Unknown")
    is_template = config.get("metadata", {}).get("is_template", False)
    
    # Button options - different for base types vs templates
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if is_template:
            # Templates can be edited
            if st.button("Edit Template", type="primary", use_container_width=True):
                st.session_state.editing_mode = True
                st.session_state.edited_config = copy.deepcopy(config)
                st.rerun()
        else:
            # Base types can be used to create templates
            if st.button("Create Template", type="primary", use_container_width=True):
                st.session_state.creating_template = True
                st.session_state.base_config_id = assessment_id
                st.rerun()
    
    with col2:
        if is_template:
            # Templates can be deleted
            if st.button("Delete Template", type="secondary", use_container_width=True):
                if delete_template(assessment_id):
                    st.session_state.selected_config_id = None
                    st.session_state.selected_config = None
                    # Reload assessment loader to refresh configs
                    st.session_state.assessment_loader.reload()
                    st.rerun()
        else:
            # Base types can be viewed in JSON
            if st.button("View JSON", type="secondary", use_container_width=True):
                st.session_state.viewing_json = True
                st.rerun()
    
    with col3:
        # Copy ID button
        if st.button("Copy ID", use_container_width=True):
            # Use streamlit's clipboard functionality
            st.write(f"ID: `{assessment_id}`")
            st.success(f"Assessment ID: {assessment_id}")
    
    # Display configuration details
    st.markdown("### Configuration Details")
    
    st.markdown(f"**ID:** {assessment_id}")
    st.markdown(f"**Type:** {assessment_type}")
    st.markdown(f"**Display Name:** {display_name}")
    st.markdown(f"**Version:** {version}")
    st.markdown(f"**Template:** {'Yes' if is_template else 'No'}")
    
    st.markdown("**Description:**")
    st.markdown(f"> {description}")
    
    # Metadata
    metadata = config.get("metadata", {})
    if metadata:
        created_date = metadata.get("created_date", "Unknown")
        modified_date = metadata.get("last_modified_date", "Unknown")
        
        with st.expander("Metadata", expanded=False):
            if is_template and "base_assessment_id" in metadata:
                st.markdown(f"**Base Configuration:** {metadata.get('base_assessment_id')}")
            
            st.markdown(f"**Created:** {created_date}")
            st.markdown(f"**Last Modified:** {modified_date}")
    
    # Configuration sections
    if "workflow" in config:
        with st.expander("Workflow Configuration", expanded=False):
            # Display enabled stages
            enabled_stages = config.get("workflow", {}).get("enabled_stages", [])
            st.markdown("**Enabled Stages:**")
            for stage in enabled_stages:
                st.markdown(f"- {stage.replace('_', ' ').title()}")
            
            # Display agent instructions if available
            agent_instructions = config.get("workflow", {}).get("agent_instructions", {})
            if agent_instructions:
                st.markdown("**Agent Instructions:**")
                for agent, instruction in agent_instructions.items():
                    with st.expander(f"{agent.title()} Agent", expanded=False):
                        st.markdown(instruction)
    
    # Main definition section based on assessment type
    definition_key = {
        "distill": "output_definition",
        "extract": "entity_definition",
        "assess": "entity_definition",
        "analyze": "framework_definition"
    }.get(assessment_type)
    
    if definition_key and definition_key in config:
        with st.expander(f"{definition_key.replace('_', ' ').title()}", expanded=True):
            definition = config.get(definition_key, {})
            st.json(definition)
    
    # Output schema
    if "output_schema" in config:
        with st.expander("Output Schema", expanded=False):
            st.json(config.get("output_schema", {}))
    
    # Extraction criteria
    if "extraction_criteria" in config:
        with st.expander("Extraction Criteria", expanded=False):
            st.json(config.get("extraction_criteria", {}))
    
    # User options
    if "user_options" in config:
        with st.expander("User Options", expanded=False):
            st.json(config.get("user_options", {}))
    
    # View full JSON
    with st.expander("Full Configuration JSON", expanded=False):
        st.json(config)

def display_create_template_form(base_id):
    """Display form for creating a new template from a base assessment."""
    st.markdown("### Create Template from Base Assessment")
    
    # Load base config for reference
    base_config = load_config_by_id(base_id)
    if not base_config:
        st.error(f"Failed to load base configuration: {base_id}")
        if st.button("Cancel"):
            st.session_state.creating_template = False
            st.rerun()
        return
    
    # Display base config info
    st.markdown(f"**Base Assessment:** {base_config.get('display_name', base_id)}")
    st.markdown(f"**Type:** {base_config.get('assessment_type', 'Unknown')}")
    st.markdown(f"**Description:** {base_config.get('description', 'No description available')}")
    
    # Form inputs
    st.markdown("#### New Template Details")
    
    # Template ID
    template_id = st.text_input(
        "Template ID",
        value=f"custom_{base_config.get('assessment_type', 'template')}_{datetime.now().strftime('%Y%m%d')}",
        help="Unique identifier for the template. Use lowercase letters, numbers, and underscores only."
    )
    
    # Display name
    display_name = st.text_input(
        "Display Name",
        value=f"Custom {base_config.get('display_name', 'Template')}",
        help="Human-readable name for the template."
    )
    
    # Description
    description = st.text_area(
        "Description",
        value=f"Custom template based on {base_config.get('display_name', base_id)}.",
        help="Detailed description of the template's purpose."
    )
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Template", type="primary", use_container_width=True):
            # Validate inputs
            if not template_id or " " in template_id or "/" in template_id or "\\" in template_id or "." in template_id:
                st.error("Invalid Template ID. Use lowercase letters, numbers, and underscores only.")
                return
            
            # Create template
            if create_template_from_base(base_id, template_id, display_name, description):
                # Reset state and reload loader
                st.session_state.creating_template = False
                st.session_state.base_config_id = None
                st.session_state.selected_config_id = template_id
                st.session_state.selected_config = load_config_by_id(template_id)
                # Reload assessment loader to refresh configs
                st.session_state.assessment_loader.reload()
                st.rerun()
    
    with col2:
        if st.button("Cancel", type="secondary", use_container_width=True):
            st.session_state.creating_template = False
            st.session_state.base_config_id = None
            st.rerun()

def main():
    """Main function for the settings page."""
    # Initialize page
    initialize_settings_page()
    
    # Load assessment loader
    loader = load_assessment_loader()
    if not loader:
        st.error("Failed to initialize assessment loader. Cannot load configurations.")
        return
    
    # Header
    st.markdown('<div class="main-header">Assessment Settings</div>', unsafe_allow_html=True)
    st.markdown("Manage assessment configurations and create custom templates.")
    
    # Create layout with sidebar for config selection
    with st.sidebar:
        st.markdown("### Assessment Configurations")
        
        # Refresh button
        if st.button("Refresh Configurations"):
            loader.reload()
            st.success("Configurations refreshed!")
            # Clear selection
            st.session_state.selected_config_id = None
            st.session_state.selected_config = None
            st.rerun()
        
        # Load all configurations
        all_configs = loader.get_assessment_configs_list()
        
        # Group by type and template status
        base_types = {}
        templates = []
        
        for config in all_configs:
            if config.get("is_template", False):
                templates.append(config)
            else:
                a_type = config.get("assessment_type", "unknown")
                if a_type not in base_types:
                    base_types[a_type] = []
                base_types[a_type].append(config)
        
        # Display base types
        st.markdown("#### Base Types")
        for a_type, configs in base_types.items():
            st.markdown(f"**{a_type.title()}**")
            for config in configs:
                if st.sidebar.button(
                    config.get("display_name", config.get("id", "Unknown")),
                    key=f"base_{config.get('id')}",
                    help=config.get("description", ""),
                    use_container_width=True
                ):
                    st.session_state.selected_config_id = config.get("id")
                    st.session_state.selected_config = load_config_by_id(config.get("id"))
                    st.session_state.editing_mode = False
                    st.session_state.creating_template = False
                    st.rerun()
        
        # Display templates
        if templates:
            st.markdown("#### Custom Templates")
            for config in templates:
                if st.sidebar.button(
                    config.get("display_name", config.get("id", "Unknown")),
                    key=f"template_{config.get('id')}",
                    help=config.get("description", ""),
                    use_container_width=True
                ):
                    st.session_state.selected_config_id = config.get("id")
                    st.session_state.selected_config = load_config_by_id(config.get("id"))
                    st.session_state.editing_mode = False
                    st.session_state.creating_template = False
                    st.rerun()
    
    # Main content area
    if st.session_state.creating_template and "base_config_id" in st.session_state:
        # Show template creation form
        display_create_template_form(st.session_state.base_config_id)
    elif st.session_state.editing_mode and st.session_state.selected_config:
        # Show configuration editor
        st.markdown(f"### Editing Template: {st.session_state.selected_config.get('display_name', st.session_state.selected_config.get('assessment_id', 'Unknown'))}")
        display_config_editor(st.session_state.selected_config)
    elif st.session_state.selected_config:
        # Show configuration details
        st.markdown(f"### Selected Configuration: {st.session_state.selected_config.get('display_name', st.session_state.selected_config.get('assessment_id', 'Unknown'))}")
        display_config_details(st.session_state.selected_config)
    else:
        # No configuration selected
        st.info("Select an assessment configuration from the sidebar to view or edit its details.")
        
        st.markdown("### Assessment Configurations Overview")
        st.markdown("""
        This page allows you to manage assessment configurations used by Beyond Notes:
        
        - **View details** of base assessment types
        - **Create custom templates** based on base types
        - **Edit templates** to customize their behavior
        - **Delete templates** that are no longer needed
        
        Select a configuration from the sidebar to get started.
        """)
        
        # Quick stats
        st.markdown("### Configuration Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Types", len(all_configs) - len(templates))
        with col2:
            st.metric("Custom Templates", len(templates))

if __name__ == "__main__":
    main()