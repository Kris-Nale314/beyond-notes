# assessments/loader.py
import os
import json
import logging
import shutil
import copy
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssessmentLoader:
    """
    Loads and manages assessment configurations (base types and user templates)
    based on a unified JSON structure where 'assessment_id' is the key identifier.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the assessment loader.

        Args:
            base_dir: Base directory containing 'types' and 'templates' folders.
                      Defaults to the 'assessments' directory relative to this file's location.
        """
        self.script_dir = Path(__file__).parent
        self.base_dir = Path(base_dir) if base_dir else self.script_dir
        self.types_path = self.base_dir / "types"
        self.templates_path = self.base_dir / "templates"
        logger.info(f"AssessmentLoader using base directory: {self.base_dir}")
        logger.info(f"Looking for types in: {self.types_path}")
        logger.info(f"Looking for templates in: {self.templates_path}")

        self._ensure_directories()

        # Cache for loaded configurations: keys are like "type:{assessment_id}" or "template:{assessment_id}"
        self.configs: Dict[str, Dict[str, Any]] = {}

        self.standard_types = {
            "distill": "Document Summarization",
            "extract": "Action Item Extraction",
            "assess": "Issue Assessment",
            "analyze": "Framework Analysis"
        }

        self.reload()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        try:
            self.types_path.mkdir(parents=True, exist_ok=True)
            self.templates_path.mkdir(parents=True, exist_ok=True)
            gitkeep_path = self.templates_path / ".gitkeep"
            if not any(f for f in self.templates_path.iterdir() if f.name != '.gitkeep') and not gitkeep_path.exists():
                 gitkeep_path.touch()
            logger.debug(f"Directories verified: {self.types_path}, {self.templates_path}")
        except OSError as e:
            logger.error(f"OS Error ensuring directories: {e}", exc_info=True)
        except Exception as e:
             logger.error(f"An unexpected error occurred during directory setup: {e}", exc_info=True)

    def _load_configs_from_dir(self, config_dir: Path, is_template: bool) -> None:
        """Helper to load configs from a specific directory (types or templates)."""
        if not config_dir.exists():
            logger.warning(f"{'Templates' if is_template else 'Types'} directory not found: {config_dir}")
            return

        glob_pattern = "*/config.json" if is_template else "*.json"
        prefix = "template" if is_template else "type"

        logger.info(f"Scanning {config_dir} for pattern '{glob_pattern}'")
        found_files = list(config_dir.glob(glob_pattern))
        if not found_files:
             logger.info(f"No configuration files found in {config_dir} matching '{glob_pattern}'")
             return
        logger.info(f"Found {len(found_files)} potential configuration files in {config_dir}.")

        for file_path in found_files:
            try:
                logger.debug(f"Attempting to load config from: {file_path}")
                config = self._load_json_file(file_path)
                if not config:
                    logger.warning(f"Could not load or empty config file: {file_path}")
                    continue

                assessment_id = config.get("assessment_id")
                if not assessment_id:
                    logger.error(f"Missing 'assessment_id' key in config file: {file_path}. Skipping.")
                    continue

                if self._validate_config(config, assessment_id, is_template, file_path):
                    config.setdefault("metadata", {})
                    config["metadata"]["is_template"] = is_template
                    config["metadata"]["loaded_from"] = str(file_path.relative_to(self.base_dir))

                    if is_template and "base_assessment_id" not in config["metadata"]:
                        base_type = config.get("assessment_type")
                        # Simple inference, assuming base IDs follow a pattern. Explicit is better.
                        inferred_base_id = f"base_{base_type}_v1"
                        config["metadata"]["base_assessment_id"] = inferred_base_id
                        logger.warning(f"Template '{assessment_id}' missing 'base_assessment_id' in metadata. Inferred as '{inferred_base_id}'. Please set explicitly.")

                    cache_key = f"{prefix}:{assessment_id}"
                    self.configs[cache_key] = config
                    logger.info(f"Successfully loaded {'template' if is_template else 'base type'}: '{assessment_id}'")
                else:
                    logger.warning(f"Invalid configuration for '{assessment_id}' in file: {file_path}. Skipping.")

            except Exception as e:
                logger.error(f"Error processing config file {file_path}: {str(e)}", exc_info=True)

    def _load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file: {file_path}. Error: {e}")
            return None
        except FileNotFoundError:
             logger.error(f"File not found: {file_path}")
             return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}", exc_info=True)
            return None

    def _validate_config(self, config: Dict[str, Any], assessment_id: str, is_template: bool, file_path: Path) -> bool:
        """Validate assessment configuration against the unified structure."""
        is_valid = True
        required_keys = ["assessment_id", "assessment_type", "version", "display_name", "description", "workflow", "output_schema", "metadata"]
        for key in required_keys:
            if key not in config:
                logger.error(f"Config '{assessment_id}' ({file_path.name}): Missing required key: '{key}'")
                is_valid = False

        expected_id_from_path = file_path.stem if not is_template else file_path.parent.name
        if config.get("assessment_id") != expected_id_from_path:
             logger.warning(f"Config '{assessment_id}' ({file_path.name}): Internal 'assessment_id' ('{config.get('assessment_id')}') does not match filename/directory name ('{expected_id_from_path}'). Convention mismatch.")

        assessment_type = config.get("assessment_type")
        if not assessment_type or assessment_type not in self.standard_types:
             logger.error(f"Config '{assessment_id}' ({file_path.name}): Invalid or missing 'assessment_type': '{assessment_type}'")
             is_valid = False

        definition_keys = ["output_definition", "entity_definition", "framework_definition"]
        present_definitions = [key for key in definition_keys if isinstance(config.get(key), dict) and config.get(key)]

        expected_definition_key = {
            "distill": "output_definition",
            "extract": "entity_definition",
            "assess": "entity_definition",
            "analyze": "framework_definition"
        }.get(assessment_type)

        if assessment_type in self.standard_types:
            if not expected_definition_key:
                logger.error(f"Config '{assessment_id}' ({file_path.name}): Internal error - No definition key mapping for valid assessment_type: '{assessment_type}'")
                is_valid = False
            elif expected_definition_key not in present_definitions:
                logger.error(f"Config '{assessment_id}' ({file_path.name}): Missing or empty expected definition block '{expected_definition_key}' for type '{assessment_type}'. Found: {present_definitions}")
                is_valid = False
            elif len(present_definitions) > 1:
                logger.warning(f"Config '{assessment_id}' ({file_path.name}): Multiple definition blocks found ({present_definitions}). Only '{expected_definition_key}' should be populated for type '{assessment_type}'.")

        if is_template:
            metadata = config.get("metadata")
            if not isinstance(metadata, dict):
                 logger.error(f"Config '{assessment_id}' ({file_path.name}): Template is missing 'metadata' dictionary.")
                 is_valid = False
            elif not metadata.get("is_template"):
                 logger.error(f"Config '{assessment_id}' ({file_path.name}): Template loaded from template directory but metadata.is_template is not true.")
                 is_valid = False

        return is_valid

    def _log_loader_status(self) -> None:
        """Log the status of loaded configurations."""
        if not self.configs:
            logger.info("AssessmentLoader: No configurations loaded.")
            return

        base_type_ids = sorted([cfg["assessment_id"] for cfg in self.configs.values() if cfg.get("assessment_id") and not cfg.get("metadata", {}).get("is_template")])
        template_ids = sorted([cfg["assessment_id"] for cfg in self.configs.values() if cfg.get("assessment_id") and cfg.get("metadata", {}).get("is_template")])

        logger.info(f"AssessmentLoader reload complete. {len(self.configs)} configurations loaded.")
        logger.info(f" Base types ({len(base_type_ids)}): {', '.join(base_type_ids) or 'None'}")
        logger.info(f" Templates ({len(template_ids)}): {', '.join(template_ids) or 'None'}")

    def get_base_types(self) -> Dict[str, Dict[str, str]]:
        """Get available base assessment types (those where metadata.is_template is false)."""
        base_types_info = {}
        for config in self.configs.values():
            if not config.get("metadata", {}).get("is_template", False):
                assessment_id = config.get("assessment_id")
                if assessment_id:
                    base_types_info[assessment_id] = {
                        "display_name": config.get("display_name", assessment_id),
                        "description": config.get("description", "")
                    }
        if not base_types_info:
            logger.warning("No base types found in cache. UI might show fallback or be empty.")
            # Keep fallback logic minimal or remove if create_default_types is reliable
        return base_types_info

    def get_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available templates (those where metadata.is_template is true)."""
        templates_info = {}
        for config in self.configs.values():
             metadata = config.get("metadata", {})
             if metadata.get("is_template", False):
                  assessment_id = config.get("assessment_id")
                  if assessment_id:
                       templates_info[assessment_id] = {
                            "display_name": config.get("display_name", assessment_id),
                            "description": config.get("description", ""),
                            "base_assessment_id": metadata.get("base_assessment_id", "Unknown"),
                            "created_date": metadata.get("created_date", "Unknown"),
                            "last_modified_date": metadata.get("last_modified_date", "Unknown")
                       }
        return templates_info

    def get_assessment_ids(self, include_types: bool = True, include_templates: bool = True) -> List[str]:
        """Get a sorted list of all available assessment IDs based on filters."""
        ids = []
        for config in self.configs.values():
             metadata = config.get("metadata", {})
             is_template = metadata.get("is_template", False)
             assessment_id = config.get("assessment_id")
             if assessment_id:
                  if include_types and not is_template:
                       ids.append(assessment_id)
                  elif include_templates and is_template:
                       ids.append(assessment_id)
        return sorted(ids)

    def get_assessment_configs_list(self) -> List[Dict[str, Any]]:
        """Get all assessment configurations (base and templates) as a list of dicts, suitable for UI."""
        configs_list = []
        for config in self.configs.values():
             assessment_id = config.get("assessment_id")
             if assessment_id:
                 metadata = config.get("metadata", {})
                 is_template = metadata.get("is_template", False)
                 configs_list.append({
                     "id": assessment_id,
                     "display_name": config.get("display_name", assessment_id),
                     "description": config.get("description", ""),
                     "is_template": is_template,
                     "assessment_type": config.get("assessment_type", "unknown"),
                     "base_assessment_id": metadata.get("base_assessment_id") if is_template else None,
                     "version": config.get("version", "unknown")
                 })
        configs_list.sort(key=lambda x: (x["is_template"], x["display_name"]))
        return configs_list

    def load_config(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific assessment configuration by its assessment_id."""
        type_key = f"type:{assessment_id}"
        template_key = f"template:{assessment_id}"

        if type_key in self.configs:
            logger.debug(f"Cache hit for '{assessment_id}' (as type).")
            return copy.deepcopy(self.configs[type_key])
        elif template_key in self.configs:
            logger.debug(f"Cache hit for '{assessment_id}' (as template).")
            return copy.deepcopy(self.configs[template_key])

        logger.warning(f"Config '{assessment_id}' not found in cache. Attempting direct file load.")
        loaded_config = None
        try:
            type_file_path = self.types_path / f"{assessment_id}.json"
            if type_file_path.exists():
                 logger.debug(f"Attempting direct load from type file: {type_file_path}")
                 config = self._load_json_file(type_file_path)
                 if config and config.get("assessment_id") == assessment_id and self._validate_config(config, assessment_id, is_template=False, file_path=type_file_path):
                      cache_key = f"type:{assessment_id}"
                      self.configs[cache_key] = config
                      logger.info(f"Loaded base type '{assessment_id}' directly from file and cached.")
                      loaded_config = copy.deepcopy(config)

            if not loaded_config:
                template_file_path = self.templates_path / assessment_id / "config.json"
                if template_file_path.exists():
                     logger.debug(f"Attempting direct load from template file: {template_file_path}")
                     config = self._load_json_file(template_file_path)
                     if config and config.get("assessment_id") == assessment_id and self._validate_config(config, assessment_id, is_template=True, file_path=template_file_path):
                          cache_key = f"template:{assessment_id}"
                          self.configs[cache_key] = config
                          logger.info(f"Loaded template '{assessment_id}' directly from file and cached.")
                          loaded_config = copy.deepcopy(config)

        except Exception as e:
            logger.error(f"Error during direct load attempt for '{assessment_id}': {str(e)}", exc_info=True)

        if not loaded_config:
             logger.error(f"Configuration could not be found or loaded for assessment_id: '{assessment_id}'")
             return None

        return loaded_config

    def create_template_from_base(self, base_assessment_id: str, new_template_id: str,
                                  display_name: Optional[str] = None,
                                  description: Optional[str] = None) -> Optional[str]:
        """Create a new template configuration file based on a base assessment ID."""
        base_config = self.load_config(base_assessment_id)
        if not base_config:
            logger.error(f"Could not load base configuration to create template from: '{base_assessment_id}'")
            return None

        if base_config.get("metadata", {}).get("is_template", False):
             logger.error(f"Cannot create template from another template: '{base_assessment_id}' is already a template.")
             return None

        if not new_template_id or "/" in new_template_id or "\\" in new_template_id or "." in new_template_id:
             logger.error(f"Invalid new_template_id: '{new_template_id}'. Cannot contain slashes or dots.")
             return None

        template_dir = self.templates_path / new_template_id
        if template_dir.exists():
            logger.error(f"Template directory already exists for ID: '{new_template_id}' at {template_dir}")
            return None

        try:
            template_dir.mkdir(parents=True)
            logger.info(f"Created template directory: {template_dir}")
            template_config = copy.deepcopy(base_config)

            template_config["assessment_id"] = new_template_id
            template_config["display_name"] = display_name or new_template_id
            if description is not None:
                template_config["description"] = description
            template_config["version"] = "1.0"

            template_config.setdefault("metadata", {})
            template_config["metadata"]["is_template"] = True
            template_config["metadata"]["base_assessment_id"] = base_assessment_id
            now_iso = datetime.now(datetime.timezone.utc).isoformat() # Use timezone-aware UTC
            template_config["metadata"]["created_date"] = now_iso
            template_config["metadata"]["last_modified_date"] = now_iso
            template_config["metadata"]["created_by"] = "User"

            config_path = template_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(template_config, f, indent=2)
            logger.info(f"Saved new template config to: {config_path}")

            cache_key = f"template:{new_template_id}"
            self.configs[cache_key] = template_config
            logger.info(f"Added new template '{new_template_id}' to cache.")

            return new_template_id

        except OSError as e:
             logger.error(f"OS Error creating template '{new_template_id}' directory or file: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error creating template '{new_template_id}': {str(e)}", exc_info=True)
            if template_dir.exists():
                try:
                    shutil.rmtree(template_dir)
                    logger.info(f"Cleaned up failed template directory: {template_dir}")
                except Exception as cleanup_e:
                     logger.error(f"Error during cleanup of failed template directory {template_dir}: {cleanup_e}")
            return None

    def update_template(self, template_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update an existing template configuration file using its assessment_id."""
        current_config = self.load_config(template_id)
        if not current_config:
            logger.error(f"Template not found for update: '{template_id}'")
            return False

        if not current_config.get("metadata", {}).get("is_template", False):
            logger.error(f"Attempted to update a non-template config as a template: '{template_id}'")
            return False

        try:
            config_updates.pop("assessment_id", None)
            config_updates.pop("assessment_type", None)
            if "metadata" in config_updates and isinstance(config_updates["metadata"], dict):
                config_updates["metadata"].pop("is_template", None)
                config_updates["metadata"].pop("base_assessment_id", None)
                config_updates["metadata"].pop("created_date", None)
                config_updates["metadata"].pop("created_by", None)
                if not config_updates["metadata"]:
                     config_updates.pop("metadata")

            updated_config = copy.deepcopy(current_config)
            self._deep_update(updated_config, config_updates)

            updated_config.setdefault("metadata", {})
            updated_config["metadata"]["last_modified_date"] = datetime.now(datetime.timezone.utc).isoformat() # Use timezone-aware UTC

            config_path = self.templates_path / template_id / "config.json"
            if not config_path.parent.exists():
                 logger.error(f"Template directory '{config_path.parent}' not found for saving update. Cannot update.")
                 return False

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(updated_config, f, indent=2)
            logger.info(f"Successfully updated template config file: {config_path}")

            cache_key = f"template:{template_id}"
            self.configs[cache_key] = updated_config
            logger.info(f"Updated template '{template_id}' in cache.")
            return True

        except Exception as e:
            logger.error(f"Error updating template '{template_id}': {str(e)}", exc_info=True)
            return False

    def delete_template(self, template_id: str) -> bool:
        """Delete a template configuration directory and remove from cache using its assessment_id."""
        template_dir = self.templates_path / template_id
        cache_key = f"template:{template_id}"
        deleted = False

        if cache_key in self.configs:
            del self.configs[cache_key]
            logger.info(f"Removed template '{template_id}' from cache.")
            deleted = True

        if template_dir.exists():
            try:
                shutil.rmtree(template_dir)
                logger.info(f"Deleted template directory: {template_dir}")
                deleted = True
            except OSError as e:
                logger.error(f"OS Error deleting template directory {template_dir}: {e}", exc_info=True)
                return False
            except Exception as e:
                logger.error(f"Unexpected error deleting template directory {template_dir}: {str(e)}", exc_info=True)
                return False
        else:
            if deleted:
                 logger.warning(f"Template directory not found for deletion: {template_dir}, but cache entry was removed.")
            else:
                 logger.error(f"Template directory not found and no cache entry existed for deletion: {template_id}")
                 return False

        return deleted

    def get_user_options(self, assessment_id: str) -> Dict[str, Any]:
        """Get user-configurable options for a specific assessment."""
        config = self.load_config(assessment_id)
        return config.get("user_options", {}) if config else {}

    def get_workflow_config(self, assessment_id: str) -> Dict[str, Any]:
        """Get workflow configuration for a specific assessment."""
        config = self.load_config(assessment_id)
        return config.get("workflow", {}) if config else {}

    def get_output_schema(self, assessment_id: str) -> Dict[str, Any]:
        """Get output schema for a specific assessment."""
        config = self.load_config(assessment_id)
        return config.get("output_schema", {}) if config else {}

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries."""
        for key, value in source.items():
            if isinstance(target.get(key), dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            elif isinstance(target.get(key), list) and isinstance(value, list):
                 target[key] = value # Replace lists for simplicity
            else:
                target[key] = value

    def create_default_types(self) -> None:
        """Create default base type JSON files if they don't exist, using the unified structure."""
        try:
            default_ids = [
                "base_distill_summary_v1",
                "base_extract_action_items_v1",
                "base_assess_issue_v1",
                "base_analyze_readiness_v1"
            ]
            created_count = 0
            self._ensure_directories() # Ensure types dir exists

            for assessment_id in default_ids:
                 config_path = self.types_path / f"{assessment_id}.json"
                 if not config_path.exists():
                    # Attempt to load content from memory/string (requires base JSONs defined elsewhere)
                    # This example assumes you have the content available as dicts
                    # Replace with actual loading of your base file content
                    base_content_dict = self._get_default_config_content(assessment_id) # Placeholder function
                    if base_content_dict:
                         try:
                              with open(config_path, 'w', encoding='utf-8') as f:
                                   json.dump(base_content_dict, f, indent=2)
                              logger.info(f"Created default base type config file: {config_path}")
                              created_count += 1
                         except Exception as e:
                              logger.error(f"Error writing default config file {config_path}: {str(e)}", exc_info=True)
                    else:
                         logger.error(f"Could not find or generate default content for '{assessment_id}'. Cannot create file.")
                 else:
                     logger.debug(f"Default config file already exists: {config_path}")

            if created_count > 0:
                 logger.info(f"Created {created_count} missing default base type config files.")
                 self.reload() # Reload configs if defaults were created

        except Exception as e:
             logger.error(f"Error during creation of default types: {e}", exc_info=True)

    def _get_default_config_content(self, assessment_id: str) -> Optional[Dict[str, Any]]:
         """Placeholder: Load the default config content, e.g., from strings or embedded resources."""
         # In a real scenario, load the JSON text you provided earlier based on assessment_id
         logger.warning(f"Placeholder function _get_default_config_content called for {assessment_id}. Implement loading of actual base JSON content.")
         # Example minimal structure:
         if assessment_id == "base_distill_summary_v1":
            # Load the full base_distill_summary_v1.json content here
            return {"assessment_id": assessment_id, "assessment_type": "distill", "version": "1.0", "display_name":"Default Distill", "description": "...", "output_definition": {}, "workflow": {}, "output_schema": {}, "metadata": {}} # Replace with actual full content
         # Add similar cases for other default IDs
         return None

    def reload(self) -> None:
        """Clear cache and reload all configurations from disk."""
        logger.info("Reloading all assessment configurations...")
        self.configs = {}
        self._load_configs_from_dir(self.types_path, is_template=False)
        self._load_configs_from_dir(self.templates_path, is_template=True)
        # Optionally run create_default_types here if you want it checked on every reload
        # self.create_default_types()
        self._log_loader_status()

    def get_standard_assessment_type_names(self) -> List[str]:
        """
        Returns a sorted list of the standard, built-in assessment type names
        (e.g., ['analyze', 'assess', 'distill', 'extract']).
        These correspond to the keys used for output directories.
        """
        return sorted(list(self.standard_types.keys()))

# Example usage
if __name__ == "__main__":
    print("--- Running AssessmentLoader Standalone Test ---")
    # Assumes loader.py is in 'assessments' folder, and 'types', 'templates' are siblings
    test_loader = AssessmentLoader() # Loads configs on init

    print("\n--- Available Configs (Summary) ---")
    configs_list = test_loader.get_assessment_configs_list()
    if configs_list:
        for cfg in configs_list:
            print(f"- ID: {cfg['id']}, Name: {cfg['display_name']}, Type: {cfg['assessment_type']}, Template: {cfg['is_template']}")
    else:
         print("No configurations loaded.")

    # ... (rest of the __main__ test block from previous version remains the same) ...
    print("\n--- Loading a Base Type ---")
    base_id = "base_extract_action_items_v1" # Use one of the base IDs
    loaded_base = test_loader.load_config(base_id)
    if loaded_base:
        print(f"Successfully loaded '{base_id}'. Display Name: {loaded_base.get('display_name')}")
    else:
        print(f"Failed to load '{base_id}'. Ensure '{base_id}.json' exists in '{test_loader.types_path}' and is valid.")

    print("\n--- Creating a Template ---")
    new_template_name = "my_test_action_template_01" # Unique name
    created_template_id = test_loader.create_template_from_base(base_id, new_template_name, "My Custom Action Template", "For testing purposes")
    if created_template_id:
         print(f"Successfully created template with ID: '{created_template_id}'")

         print("\n--- Updating the Template ---")
         updates = {
             "description": "Updated description for testing.",
             "user_options": {"detail_level": {"default": "essential"}, "new_option": {"type": "boolean", "default": True, "display_name": "New Flag"}},
             "metadata": { "custom_field": "some_value" }
         }
         if test_loader.update_template(created_template_id, updates):
              print(f"Successfully updated template '{created_template_id}'.")
              updated_config = test_loader.load_config(created_template_id)
              if updated_config:
                   print(f" Updated Description: {updated_config.get('description')}")
                   print(f" Custom Metadata: {updated_config.get('metadata', {}).get('custom_field')}")
         else:
              print(f"Failed to update template '{created_template_id}'.")

         print("\n--- Deleting the Template ---")
         if test_loader.delete_template(created_template_id):
             print(f"Successfully deleted template '{created_template_id}'.")
             if not test_loader.load_config(created_template_id):
                  print("Verified: Template cannot be loaded after deletion.")
             else:
                  print("Error: Template still loadable after deletion attempt.")
         else:
             print(f"Failed to delete template '{created_template_id}'.")
    else:
         print(f"Failed to create template '{new_template_name}'. Skipping update/delete tests.")

    print("\n--- Test Complete ---")