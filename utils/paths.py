import os
from pathlib import Path
from datetime import datetime
from typing import Tuple
import logging
from logging import getLogger
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Tuple


class AppPaths:
    """Central management of application paths."""
    
    # Check if running in Docker
    IN_DOCKER = os.environ.get('RUNNING_IN_DOCKER', 'false').lower() == 'true'
    
    # Base paths - adaptable for Docker
    if IN_DOCKER:
        # In Docker, use standard container paths
        ROOT = Path('/app')
        DATA = Path('/data')
        OUTPUT = Path('/output')
        TEMP = Path('/tmp/beyond-notes')
    else:
        # Local development paths
        ROOT = Path(__file__).parent.parent
        DATA = ROOT / "data"
        OUTPUT = ROOT / "output"
        TEMP = ROOT / "temp"
    
    # Data subpaths
    UPLOADS = DATA / "uploads"
    CACHE = DATA / "cache"
    SAMPLES = DATA / "samples"
    
    # Assessment paths
    ASSESSMENTS = ROOT / "assessments"
    ASSESSMENT_TYPES = ASSESSMENTS / "types"
    ASSESSMENT_TEMPLATES = ASSESSMENTS / "templates"
    
    # Output subpaths - all assessment types
    DISTILL_OUTPUT = OUTPUT / "distill"
    EXTRACT_OUTPUT = OUTPUT / "extract"
    ASSESS_OUTPUT = OUTPUT / "assess"
    ANALYZE_OUTPUT = OUTPUT / "analyze"
    
    # Legacy output paths (for backward compatibility)
    ISSUES_OUTPUT = OUTPUT / "issues"
    ACTION_ITEMS_OUTPUT = OUTPUT / "action_items"
    INSIGHTS_OUTPUT = OUTPUT / "insights"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all application directories exist."""
        for attr_name in dir(cls):
            if attr_name.isupper() and not attr_name.startswith('_'):
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def get_assessment_output_dir(cls, assessment_type):
        """Get output directory for a specific assessment type."""
        output_dir = cls.OUTPUT / assessment_type
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir
    
    @classmethod
    def get_types_dir(cls):
        """Get directory for assessment types."""
        cls.ASSESSMENT_TYPES.mkdir(exist_ok=True, parents=True)
        return cls.ASSESSMENT_TYPES
    
    @classmethod
    def get_templates_dir(cls):
        """Get directory for assessment templates."""
        cls.ASSESSMENT_TEMPLATES.mkdir(exist_ok=True, parents=True)
        return cls.ASSESSMENT_TEMPLATES
    
    @classmethod
    def get_upload_path(cls, filename):
        """Get path for an uploaded file."""
        cls.UPLOADS.mkdir(exist_ok=True, parents=True)
        return cls.UPLOADS / filename
    
    @classmethod
    def get_temp_path(cls, subdir=None):
        """Get a temporary directory path."""
        if subdir:
            temp_dir = cls.TEMP / subdir
            temp_dir.mkdir(exist_ok=True, parents=True)
            return temp_dir
        return cls.TEMP
    
def get_assessment_result_path(assessment_type: str, document_name: str, run_id: str = None) -> Tuple[Path, Path, Path]:
    """
    Generate standardized paths for assessment results, context, and report.
    
    Args:
        assessment_type: Type of assessment (distill, extract, assess, analyze)
        document_name: Name of the original document
        run_id: Optional unique run identifier 
        
    Returns:
        Tuple of (json_path, context_path, markdown_path)
    """
    # Ensure the output directory exists
    output_dir = get_assessment_output_dir(assessment_type)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a base filename from document name
    base_name = Path(document_name).stem
    # Clean up filename to be safe for all filesystems
    base_name = "".join(c for c in base_name if c.isalnum() or c in "._- ").strip()
    if not base_name:
        base_name = "document"
        
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{run_id}" if run_id else ""
    
    # Generate the three paths
    result_filename = f"{assessment_type}_result_{base_name}_{timestamp}{run_suffix}.json"
    context_filename = f"{assessment_type}_context_{base_name}_{timestamp}{run_suffix}.pkl"
    report_filename = f"{assessment_type}_report_{base_name}_{timestamp}{run_suffix}.md"
    
    return (
        output_dir / result_filename,
        output_dir / context_filename,
        output_dir / report_filename
    )