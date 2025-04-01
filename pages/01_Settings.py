import os
import sys
import json
import streamlit as st
import logging
from pathlib import Path
import copy
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project components
from assessments.loader import AssessmentLoader
from utils.paths import AppPaths


# Page config
st.set_page_config(
    page_title="Settings - Beyond Notes",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def display_base_type(base_type, loader, config):
    """Display a base assessment type configuration."""
    st.header(f"{config.get('display_name', base_type.title())}")
    st.caption(config.get("description", ""))
    
    # Assessment info in JSON format
    if st.button("View Assessment Information", key=f"view_json_{base_type}"):
        st.json(config)
    
    # Template creation
    st.subheader("Create Template")
    st.write("Create a new template based on this assessment type.")
    
    # Template form
    template_name = st.text_input("Template Name", placeholder="E.g., Client ABC Issue Analysis", key=f"template_name_{base_type}")
    template_description = st.text_area(
        "Template Description", 
        placeholder="Describe the purpose of this template",
        key=f"template_desc_{base_type}"
    )
    
    if st.button("Create Template", key=f"create_template_{base_type}"):
        if template_name:
            # Create template
            template_id = loader.create_template_from_base(base_type, template_name, template_description)
            
            if template_id:
                st.success(f"Created template: {template_name}")
                # Reload templates
                loader.reload()
                # Set session state to view the new template
                st.session_state.selected_tab = "My Templates"
                st.session_state.selected_template = template_id
                st.experimental_rerun()
            else:
                st.error(f"Failed to create template {template_name}")

def display_template(template_id, loader, config):
    """Display a template configuration with editing capabilities."""
    display_name = config.get("display_name", template_id.replace("template:", ""))
    
    st.header(f"{display_name}")
    st.caption(config.get("description", ""))
    
    # Base type
    base_type = config.get("metadata", {}).get("base_type", "unknown")
    st.write(f"Based on: **{base_type.title()}**")
    
    # Template sections
    tabs = st.tabs(["Basic Settings", "Entity Definition", "Workflow", "Output Format"])
    
    with tabs[0]:
        # Basic information tab
        st.subheader("Basic Information")
        
        with st.form(key="basic_settings"):
            new_display_name = st.text_input("Display Name", value=display_name)
            new_description = st.text_area("Description", value=config.get("description", ""))
            
            # Save changes
            if st.form_submit_button("Save Basic Settings"):
                try:
                    # Update config
                    updates = {
                        "display_name": new_display_name,
                        "description": new_description
                    }
                    
                    # Get template name from ID
                    template_name = template_id.replace("template:", "")
                    
                    # Update template
                    if loader.update_template(template_name, updates):
                        st.success("Basic settings updated successfully")
                        # Reload the template
                        loader.reload()
                        st.experimental_rerun()
                    else:
                        st.error("Failed to update template")
                except Exception as e:
                    st.error(f"Error updating template: {str(e)}")
    
    with tabs[1]:
        # Entity Definition tab
        st.subheader("Entity Definition")
        
        # Get the entity definition based on assessment type
        entity_key = "entity_definition"
        if base_type == "analyze":
            entity_key = "framework_definition"
        elif base_type == "distill":
            entity_key = "output_definition"
        
        entity_def = config.get(entity_key, {})
        
        # Display as JSON with editing
        st.write("Edit the entity definition:")
        entity_json = st.text_area(
            "Entity Definition (JSON)",
            value=json.dumps(entity_def, indent=2),
            height=400
        )
        
        # Save changes
        if st.button("Save Entity Definition"):
            try:
                # Parse JSON
                updated_entity = json.loads(entity_json)
                
                # Update config
                updates = {
                    entity_key: updated_entity
                }
                
                # Get template name from ID
                template_name = template_id.replace("template:", "")
                
                # Update template
                if loader.update_template(template_name, updates):
                    st.success("Entity definition updated successfully")
                    # Reload the template
                    loader.reload()
                    st.experimental_rerun()
                else:
                    st.error("Failed to update template")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
            except Exception as e:
                st.error(f"Error updating template: {str(e)}")
    
    with tabs[2]:
        # Workflow tab
        st.subheader("Agent Instructions")
        
        workflow = config.get("workflow", {})
        agent_instructions = workflow.get("agent_instructions", {})
        
        # Display each agent's instructions
        for agent, instructions in agent_instructions.items():
            st.write(f"**{agent.title()} Agent**")
            new_instructions = st.text_area(
                f"Instructions for {agent.title()}",
                value=instructions,
                key=f"instr_{agent}",
                height=150
            )
            
            # Save this agent's instructions
            if st.button(f"Save {agent.title()} Instructions"):
                try:
                    # Update the instructions
                    updated_workflow = copy.deepcopy(workflow)
                    updated_workflow["agent_instructions"][agent] = new_instructions
                    
                    # Update config
                    updates = {
                        "workflow": updated_workflow
                    }
                    
                    # Get template name from ID
                    template_name = template_id.replace("template:", "")
                    
                    # Update template
                    if loader.update_template(template_name, updates):
                        st.success(f"{agent.title()} instructions updated successfully")
                        # Reload the template
                        loader.reload()
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to update {agent} instructions")
                except Exception as e:
                    st.error(f"Error updating instructions: {str(e)}")
                    
        # Stage weights (optional)
        st.subheader("Stage Weights")
        st.write("Configure the weight of each stage in the progress bar:")
        
        stage_weights = workflow.get("stage_weights", {})
        if st.checkbox("Edit Stage Weights", value=False):
            with st.form(key="stage_weights"):
                updated_weights = {}
                
                for stage, weight in stage_weights.items():
                    updated_weights[stage] = st.slider(
                        f"{stage.replace('_', ' ').title()}", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=float(weight),
                        step=0.05,
                        key=f"weight_{stage}"
                    )
                
                # Save changes
                if st.form_submit_button("Save Stage Weights"):
                    try:
                        # Update the workflow
                        updated_workflow = copy.deepcopy(workflow)
                        updated_workflow["stage_weights"] = updated_weights
                        
                        # Update config
                        updates = {
                            "workflow": updated_workflow
                        }
                        
                        # Get template name from ID
                        template_name = template_id.replace("template:", "")
                        
                        # Update template
                        if loader.update_template(template_name, updates):
                            st.success("Stage weights updated successfully")
                            # Reload the template
                            loader.reload()
                            st.experimental_rerun()
                        else:
                            st.error("Failed to update stage weights")
                    except Exception as e:
                        st.error(f"Error updating stage weights: {str(e)}")
    
    with tabs[3]:
        # Output Format tab
        st.subheader("Output Format")
        
        output_format = config.get("output_format", {})
        
        # Display as JSON with editing
        st.write("Edit the output format configuration:")
        output_json = st.text_area(
            "Output Format (JSON)",
            value=json.dumps(output_format, indent=2),
            height=300
        )
        
        # Save changes
        if st.button("Save Output Format"):
            try:
                # Parse JSON
                updated_output = json.loads(output_json)
                
                # Update config
                updates = {
                    "output_format": updated_output
                }
                
                # Get template name from ID
                template_name = template_id.replace("template:", "")
                
                # Update template
                if loader.update_template(template_name, updates):
                    st.success("Output format updated successfully")
                    # Reload the template
                    loader.reload()
                    st.experimental_rerun()
                else:
                    st.error("Failed to update output format")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
            except Exception as e:
                st.error(f"Error updating template: {str(e)}")
    
    # Template management buttons
    st.subheader("Template Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Duplicate template
        if st.button("Duplicate Template"):
            try:
                # Get template name
                template_name = template_id.replace("template:", "")
                new_name = f"{template_name} Copy"
                
                # Get full config
                template_config = loader.load_config(template_id)
                
                # Create a copy
                if template_config:
                    template_id = loader.create_template_from_base(
                        base_type, 
                        new_name, 
                        template_config.get("description", "")
                    )
                    
                    if template_id:
                        # Update the new template with all settings from original
                        template_config["display_name"] = new_name
                        if "metadata" in template_config:
                            template_config["metadata"]["created_date"] = datetime.now().isoformat()
                        
                        # Get template name from ID
                        new_template_name = template_id.replace("template:", "")
                        
                        # Update template
                        if loader.update_template(new_template_name, template_config):
                            st.success(f"Duplicated template as: {new_name}")
                            # Reload the template
                            loader.reload()
                            # Set session state to view the new template
                            st.session_state.selected_template = template_id
                            st.experimental_rerun()
                        else:
                            st.error("Failed to update duplicated template")
                    else:
                        st.error(f"Failed to create duplicate template")
                else:
                    st.error("Failed to load template for duplication")
            except Exception as e:
                st.error(f"Error duplicating template: {str(e)}")
    
    with col2:
        # Delete template with confirmation
        if st.button("Delete Template"):
            st.session_state.confirm_delete = template_id
    
    # Confirmation dialog
    if st.session_state.get("confirm_delete") == template_id:
        st.warning("Are you sure you want to delete this template? This action cannot be undone.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Yes, Delete"):
                try:
                    # Get template name from ID
                    template_name = template_id.replace("template:", "")
                    
                    # Delete template
                    if loader.delete_template(template_name):
                        st.success(f"Deleted template: {template_name}")
                        # Clear confirmation
                        st.session_state.pop("confirm_delete", None)
                        # Go back to base types
                        st.session_state.selected_tab = "Base Types"
                        st.session_state.pop("selected_template", None)
                        # Reload templates
                        loader.reload()
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete template {template_name}")
                except Exception as e:
                    st.error(f"Error deleting template: {str(e)}")
        
        with col2:
            if st.button("Cancel"):
                # Clear confirmation
                st.session_state.pop("confirm_delete", None)
                st.experimental_rerun()

def main():
    st.title("Assessment Library Settings")
    
    # Ensure necessary directories exist
    AppPaths.ensure_dirs()
    
    # Get assessment loader from session state or initialize
    if "assessment_loader" in st.session_state:
        loader = st.session_state.assessment_loader
    else:
        loader = AssessmentLoader()
        st.session_state.assessment_loader = loader
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Get assessment types
    base_types = loader.get_base_types()
    templates = loader.get_templates()
    
    # Tabs for base types and templates
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = "Base Types"
    
    selected_tab = st.radio(
        "View",
        ["Base Types", "My Templates"],
        index=0 if st.session_state.selected_tab == "Base Types" else 1
    )
    st.session_state.selected_tab = selected_tab
    
    if selected_tab == "Base Types":
        # Display base assessment types
        for base_type, info in base_types.items():
            with st.expander(f"{info['display_name']}", expanded=False):
                # Load full config
                config = loader.load_config(base_type)
                if config:
                    display_base_type(base_type, loader, config)
                else:
                    st.error(f"Could not load configuration for {base_type}")
    
    else:  # My Templates
        if not templates:
            st.info("No templates created yet. Create one from a base assessment type.")
        else:
            # Template selection
            template_options = {t_id: info["display_name"] for t_id, info in templates.items()}
            
            if "selected_template" in st.session_state and st.session_state.selected_template in template_options:
                default_template = st.session_state.selected_template
            else:
                default_template = next(iter(template_options.keys()), None)
            
            selected_template = st.selectbox(
                "Select Template",
                list(template_options.keys()),
                format_func=lambda x: template_options.get(x, x),
                index=list(template_options.keys()).index(default_template) if default_template else 0
            )
            
            st.session_state.selected_template = selected_template
            
            # Display selected template
            if selected_template:
                config = loader.load_config(selected_template)
                if config:
                    display_template(selected_template, loader, config)
                else:
                    st.error(f"Could not load configuration for {selected_template}")

if __name__ == "__main__":
    # Initialize session state for template deletion confirmation
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = None
    
    main()