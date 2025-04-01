#!/usr/bin/env python
"""
Setup script to create the directory structure for Beyond Notes.
This ensures all necessary directories exist with proper organization.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project components
from utils.paths import AppPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("setup_dirs")

def create_directory_structure():
    """Create the full directory structure for the application."""
    logger.info("Creating Beyond Notes directory structure...")
    
    # Create all directories defined in AppPaths
    AppPaths.ensure_dirs()
    
    # Create additional directories if needed
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    # Create the assessments directory structure
    assessment_types_dir = AppPaths.get_types_dir()
    assessment_templates_dir = AppPaths.get_templates_dir()
    
    # Ensure .gitkeep files exist in empty directories to preserve them in git
    for directory in [
        AppPaths.UPLOADS,
        AppPaths.CACHE,
        AppPaths.TEMP,
        assessment_templates_dir
    ]:
        gitkeep_file = directory / ".gitkeep"
        if not any(f for f in directory.iterdir() if f.name != ".gitkeep") and not gitkeep_file.exists():
            gitkeep_file.touch()
            logger.info(f"Created .gitkeep in {directory}")
    
    # Log completion
    logger.info("Directory structure setup complete!")
    
    # Return the paths for reference
    return {
        "project_root": project_root,
        "data_dir": AppPaths.DATA,
        "assessments_dir": AppPaths.ASSESSMENTS,
        "types_dir": assessment_types_dir,
        "templates_dir": assessment_templates_dir,
        "output_dir": AppPaths.OUTPUT
    }

def print_structure(paths):
    """Print the directory structure in a readable format."""
    print("\nBeyond Notes Directory Structure:")
    print("================================")
    
    for name, path in paths.items():
        rel_path = path.relative_to(project_root)
        print(f"- {name}: {rel_path}")
    
    print("\nStructure is ready for use!")

if __name__ == "__main__":
    paths = create_directory_structure()
    print_structure(paths)