# utils/paths.py
import os
from pathlib import Path

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
    
    # Output subpaths
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