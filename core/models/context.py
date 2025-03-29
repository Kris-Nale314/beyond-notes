# core/models/context.py
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

class ProcessingContext:
    """
    Context object for document processing pipelines.
    Manages state, metadata, and results throughout processing.
    """
    
    def __init__(self, document_text: str, options: Optional[Dict[str, Any]] = None):
        """Initialize processing context with document and options."""
        # Core content
        self.document_text = document_text
        self.options = options or {}
        
        # Document information
        self.document_info = {
            "created_at": datetime.now().isoformat(),
            "word_count": len(document_text.split()),
            "character_count": len(document_text)
        }
        
        # Chunking information
        self.chunks = []
        self.chunk_metadata = []
        
        # Results storage by stage
        self.results = {}
        
        # Processing metadata
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"
        self.metadata = {
            "start_time": time.time(),
            "run_id": self.run_id,
            "current_stage": None,
            "stages": {},
            "errors": [],
            "warnings": [],
            "progress": 0.0,
            "progress_message": "Initializing"
        }
        
        # Progress callback
        self.progress_callback = None
    
    def set_stage(self, stage_name: str) -> None:
        """Begin a processing stage."""
        self.metadata["current_stage"] = stage_name
        self.metadata["stages"][stage_name] = {
            "status": "running",
            "start_time": time.time(),
            "progress": 0.0
        }
        
        # Update progress message
        self.update_progress(self.metadata["progress"], f"Starting {stage_name}")
    
    def complete_stage(self, stage_name: str, result: Any = None) -> None:
        """Complete a processing stage and store result."""
        if stage_name not in self.metadata["stages"]:
            self.set_stage(stage_name)
            
        # Update stage metadata
        stage = self.metadata["stages"][stage_name]
        stage["status"] = "completed"
        stage["end_time"] = time.time()
        stage["duration"] = stage["end_time"] - stage["start_time"]
        
        # Store result
        if result is not None:
            self.results[stage_name] = result
        
        # Update progress
        self.update_progress(self.metadata["progress"], f"Completed {stage_name}")
    
    def fail_stage(self, stage_name: str, error: str) -> None:
        """Mark a stage as failed."""
        # Update stage info
        if stage_name in self.metadata["stages"]:
            stage = self.metadata["stages"][stage_name]
            stage["status"] = "failed"
            stage["end_time"] = time.time()
            stage["duration"] = stage["end_time"] - stage["start_time"]
            stage["error"] = error
        else:
            # Handle failure for stage that wasn't started
            self.metadata["stages"][stage_name] = {
                "status": "failed",
                "start_time": time.time(),
                "end_time": time.time(),
                "duration": 0,
                "error": error
            }
            
        # Add to errors list
        self.metadata["errors"].append({
            "stage": stage_name,
            "message": error,
            "time": time.time()
        })
        
        # Update progress with error
        self.update_progress(self.metadata["progress"], f"Error in {stage_name}: {error}")
    
    def update_stage_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update progress for the current stage."""
        current_stage = self.metadata.get("current_stage")
        if not current_stage:
            return
            
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Update stage progress
        stage = self.metadata["stages"][current_stage]
        stage["progress"] = progress
        
        if message:
            stage["message"] = message
        
        # Update overall progress based on stage weights
        self._update_overall_progress()
    
    def update_progress(self, progress: float, message: str) -> None:
        """Update overall progress and call the progress callback if provided."""
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Update metadata
        self.metadata["progress"] = progress
        self.metadata["progress_message"] = message
        
        # Call progress callback if provided
        if self.progress_callback:
            try:
                self.progress_callback(progress, message)
            except Exception as e:
                self.add_warning(f"Error in progress callback: {str(e)}")
    
    def _update_overall_progress(self) -> None:
        """Update overall progress based on stage progress and weights."""
        # Define stage weights
        stage_weights = {
            "document_analysis": 0.05,
            "chunking": 0.10,
            "extraction": 0.40,
            "aggregation": 0.20,
            "evaluation": 0.15,
            "formatting": 0.10
        }
        
        total_weight = 0
        weighted_progress = 0
        
        for stage_name, stage_data in self.metadata["stages"].items():
            weight = stage_weights.get(stage_name, 0.1)
            total_weight += weight
            
            if stage_data["status"] == "completed":
                weighted_progress += weight
            elif stage_data["status"] == "running":
                weighted_progress += weight * stage_data.get("progress", 0)
        
        # Calculate overall progress
        if total_weight > 0:
            self.metadata["progress"] = weighted_progress / total_weight
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the context."""
        self.metadata["warnings"].append({
            "message": message,
            "time": time.time(),
            "stage": self.metadata.get("current_stage")
        })
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Set document chunks."""
        self.chunks = chunks
        self.chunk_metadata = [{
            "chunk_index": chunk.get("chunk_index", i),
            "word_count": chunk.get("word_count", len(chunk.get("text", "").split()))
        } for i, chunk in enumerate(chunks)]
    
    def get_processing_time(self) -> float:
        """Get total processing time so far."""
        return time.time() - self.metadata["start_time"]
    
    def get_final_result(self) -> Dict[str, Any]:
        """Create the final result dictionary with metadata."""
        # Get the formatted result if available
        formatted_result = self.results.get("formatting", {})
        
        # Create result dictionary
        result = {
            "result": formatted_result,
            "metadata": {
                "run_id": self.run_id,
                "processing_time": self.get_processing_time(),
                "document_info": self.document_info,
                "stages": self.metadata["stages"],
                "errors": self.metadata["errors"],
                "warnings": self.metadata["warnings"],
                "options": self.options
            }
        }
        
        return result
    
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set a callback function for progress updates."""
        self.progress_callback = callback