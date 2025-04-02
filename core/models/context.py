# core/models/context.py
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ProcessingContext:
    """
    Context manager for multi-agent document processing with improved
    data management and monitoring.
    """

    def __init__(self, 
                 document_text: str,
                 assessment_config: Dict[str, Any],
                 options: Optional[Dict[str, Any]] = None,
                 document_metadata: Optional[Dict[str, Any]] = None):
        """Initialize the processing context with all necessary information."""
        # Core document data
        self.document_text = document_text
        self.assessment_config = assessment_config
        self.options = options or {}
        
        # Assessment identifiers
        self.assessment_id = assessment_config.get("assessment_id", "unknown")
        self.assessment_type = assessment_config.get("assessment_type", "unknown")
        self.display_name = assessment_config.get("display_name", self.assessment_id)
        
        # Run metadata
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"
        self.start_time = time.time()
        self.metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "assessment_id": self.assessment_id,
            "assessment_type": self.assessment_type,
            "current_stage": None
        }
        
        # Document info
        self.document_info = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "word_count": len(document_text.split()),
            "character_count": len(document_text),
            "filename": self.options.get("filename", "unknown"),
            "processed_by": []  # Agent names involved
        }
        
        # Update with provided metadata if any
        if document_metadata:
            self.document_info.update(document_metadata)
        
        # Chunks data
        self.chunks = []
        
        # Pipeline state tracking
        self.pipeline_state = {
            "current_stage": None,
            "stages": {},      # Status, timing per stage
            "progress": 0.0,
            "progress_message": "Initializing",
            "warnings": [],
            "errors": []
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_tokens": 0,
            "stage_tokens": {},
            "llm_calls": 0,
            "stage_durations": {},
        }
        
        # Centralized data store - single source of truth
        # IMPORTANT: These categories must match the agent roles
        self.data = {
            "planning": {},    # Used by PlannerAgent (role="planner")
            "extracted": {},   # Used by ExtractorAgent (role="extractor")
            "aggregated": {},  # Used by AggregatorAgent (role="aggregator")
            "evaluated": {},   # Used by EvaluatorAgent (role="evaluator")
            "formatted": {},   # Used by FormatterAgent (role="formatter")
            "review": {}       # Used by ReviewerAgent (role="reviewer")
        }
        
        # Evidence tracking
        self.evidence_store = {
            "references": {},  # item_id -> list of source text references
            "sources": {}      # source_id -> source text and metadata
        }
        
        # Progress callback
        self.progress_callback = None
        
        logger.info(f"ProcessingContext initialized: {self.run_id} - {self.display_name}")
        
    # --- Data Storage & Retrieval ---
    
    def store_data(self, category: str, data_type: str = None, data: Any = None) -> None:
        """
        Store data in the unified data store.
        
        Args:
            category: Data category (planning, extracted, aggregated, evaluated, formatted, review)
            data_type: Optional type within category (e.g., action_items, issues)
            data: The data to store
        """
        # Validate category exists
        if category not in self.data:
            error_msg = f"Invalid data category: '{category}'. Valid categories: {list(self.data.keys())}"
            logger.error(error_msg)
            self.add_warning(error_msg)
            # Create the category to avoid errors but log a warning
            self.data[category] = {}
        
        # Special case handling for planning and formatted categories
        if category in ["planning", "formatted", "review"]:
            if data_type:
                logger.debug(f"Data type '{data_type}' not used for category '{category}', using direct storage")
            self.data[category] = data
            logger.debug(f"Stored data directly in '{category}' category")
        else:
            # For other categories, store by data_type 
            if not data_type:
                error_msg = f"Missing data_type for category '{category}', using 'default'"
                logger.warning(error_msg)
                self.add_warning(error_msg)
                data_type = "default"
            
            # Initialize category as dict if it's currently None or not a dict
            if not isinstance(self.data[category], dict):
                self.data[category] = {}
                
            self.data[category][data_type] = data
            logger.debug(f"Stored data with type '{data_type}' in '{category}' category")
    
    def get_data(self, category: str, data_type: str = None, default: Any = None) -> Any:
        """
        Get data from the unified data store.
        
        Args:
            category: Data category (planning, extracted, aggregated, evaluated, formatted, review)
            data_type: Optional type within category (e.g., action_items, issues)
            default: Default value if not found
            
        Returns:
            The requested data or default if not found
        """
        # Validate category exists
        if category not in self.data:
            logger.warning(f"Requested data from unknown category: '{category}'")
            return default
        
        # Handle special categories
        if category in ["planning", "formatted", "review"]:
            if data_type:
                logger.debug(f"Data type '{data_type}' ignored for category '{category}'")
            return self.data[category] or default
        else:
            # For other categories, retrieve by data_type
            if not isinstance(self.data[category], dict):
                logger.warning(f"Category '{category}' does not contain a dictionary as expected")
                return default
                
            if data_type:
                return self.data[category].get(data_type, default)
            else:
                return self.data[category] or default
    
    def store_agent_data(self, agent_role: str, data_type: str, data: Any) -> None:
        """
        Standardized method to store data from an agent (compatible with BaseAgent).
        
        Args:
            agent_role: The role of the agent (e.g., "extractor", "aggregator")
            data_type: The type of data (e.g., "action_items", "issues") 
            data: The data to store
        """
        # Map agent roles to data categories
        agent_to_category = {
            "extractor": "extracted",
            "aggregator": "aggregated", 
            "evaluator": "evaluated",
            "formatter": "formatted",
            "planner": "planning",
            "reviewer": "review"
        }
        
        category = agent_to_category.get(agent_role, "")
        if not category:
            logger.warning(f"Unknown agent role: {agent_role}. Valid roles: {list(agent_to_category.keys())}")
            return
            
        # Store data using the appropriate method
        self.store_data(category, data_type, data)
    
    def get_data_for_agent(self, agent_role: str, data_type: str = None, default: Any = None) -> Any:
        """
        Standardized method to get data for an agent (compatible with BaseAgent).
        
        Args:
            agent_role: The role of the agent (e.g., "extractor", "aggregator")
            data_type: The type of data (e.g., "action_items", "issues")
            default: Default value if not found
            
        Returns:
            The requested data or default
        """
        # Map agent roles to data categories
        agent_to_category = {
            "extractor": "extracted",
            "aggregator": "aggregated", 
            "evaluator": "evaluated",
            "formatter": "formatted",
            "planner": "planning",
            "reviewer": "review"
        }
        
        category = agent_to_category.get(agent_role, "")
        if not category:
            logger.warning(f"Unknown agent role: {agent_role}. Valid roles: {list(agent_to_category.keys())}")
            return default
            
        # Get data using the appropriate method
        return self.get_data(category, data_type, default)
    
    # --- Evidence Tracking ---
    
    def add_evidence(self, item_id: str, evidence_text: str, 
                    source_info: Optional[Dict[str, Any]] = None,
                    confidence: Optional[float] = None) -> str:
        """
        Add evidence for an item.
        
        Args:
            item_id: ID of the item this evidence supports
            evidence_text: The evidence text snippet
            source_info: Additional metadata (chunk_index, etc.)
            confidence: Optional confidence score
            
        Returns:
            Evidence reference ID
        """
        # Generate reference ID
        ref_id = f"ref-{uuid.uuid4().hex[:8]}"
        
        # Store the source
        self.evidence_store["sources"][ref_id] = {
            "text": evidence_text,
            "metadata": source_info or {},
            "created_at": time.time()
        }
        
        # Link it to the item
        if item_id not in self.evidence_store["references"]:
            self.evidence_store["references"][item_id] = []
            
        # Add confidence if provided
        evidence_entry = {"ref_id": ref_id}
        if confidence is not None:
            evidence_entry["confidence"] = confidence
            
        self.evidence_store["references"][item_id].append(evidence_entry)
        
        logger.debug(f"Added evidence (ref_id: {ref_id}) for item {item_id}")
        return ref_id
    
    def get_evidence_for_item(self, item_id: str) -> List[Dict[str, Any]]:
        """
        Get all evidence for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of evidence dictionaries with text and metadata
        """
        evidence_list = []
        
        # Check if we have any evidence for this item
        if item_id not in self.evidence_store["references"]:
            return evidence_list
            
        # Get all references for this item
        for evidence_entry in self.evidence_store["references"][item_id]:
            ref_id = evidence_entry.get("ref_id")
            if not ref_id or ref_id not in self.evidence_store["sources"]:
                continue
                
            source_data = self.evidence_store["sources"][ref_id]
            evidence_item = {
                "source_id": ref_id,
                "text": source_data.get("text", ""),
                "metadata": source_data.get("metadata", {}),
                "confidence": evidence_entry.get("confidence")
            }
            evidence_list.append(evidence_item)
                
        return evidence_list
    
    # --- Chunking Management ---
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Set document chunks.
        
        Args:
            chunks: List of document chunks
        """
        self.chunks = chunks
        logger.info(f"[{self.run_id}] Set {len(chunks)} document chunks")
    
    # --- Configuration Access Methods ---
    
    def get_config(self, key: str, default=None):
        """Get a specific configuration value from assessment_config."""
        return self.assessment_config.get(key, default)
    
    def get_assessment_config(self) -> Dict[str, Any]:
        """Get the complete assessment configuration."""
        return self.assessment_config
    
    def get_target_definition(self) -> Optional[Dict[str, Any]]:
        """Get the primary target definition for the current assessment type."""
        key_map = {
            "distill": "output_definition",
            "extract": "entity_definition",
            "assess": "entity_definition",
            "analyze": "framework_definition"
        }
        key = key_map.get(self.assessment_type)
        return self.assessment_config.get(key) if key else None
    
    def get_extraction_criteria(self) -> Optional[Dict[str, Any]]:
        """Get extraction criteria if available."""
        return self.assessment_config.get("extraction_criteria")
    
    def get_framework_dimensions(self) -> Optional[List[Dict[str, Any]]]:
        """Get framework dimensions for 'analyze' assessments."""
        if self.assessment_type == "analyze":
            framework_def = self.get_target_definition()
            return framework_def.get("dimensions") if framework_def else None
        return None
    
    def get_rating_scale(self) -> Optional[Dict[str, Any]]:
        """Get rating scale for 'analyze' assessments."""
        if self.assessment_type == "analyze":
            framework_def = self.get_target_definition()
            return framework_def.get("rating_scale") if framework_def else None
        return None
    
    def get_workflow_instructions(self, agent_role: str) -> Optional[str]:
        """Get instructions for a specific agent role."""
        return self.assessment_config.get("workflow", {}).get("agent_instructions", {}).get(agent_role)
    
    def get_output_schema(self) -> Optional[Dict[str, Any]]:
        """Get the output schema definition."""
        return self.assessment_config.get("output_schema")
    
    def get_output_format_config(self) -> Optional[Dict[str, Any]]:
        """Get the output format configuration."""
        return self.assessment_config.get("output_format")
    
    # --- Stage & Progress Management ---
    
    def set_stage(self, stage_name: str) -> None:
        """Begin a processing stage."""
        self.pipeline_state["current_stage"] = stage_name
        self.metadata["current_stage"] = stage_name
        
        # Create stage record
        self.pipeline_state["stages"][stage_name] = {
            "status": "running",
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "progress": 0.0,
            "message": "Starting...",
            "agent": None,
            "error": None
        }
        
        logger.info(f"[{self.run_id}] Starting stage: {stage_name}")
        self.update_progress(self.pipeline_state["progress"], f"Starting {stage_name}")
    
    def complete_stage(self, stage_name: str, stage_result: Any = None) -> None:
        """Mark a stage as completed."""
        if stage_name not in self.pipeline_state["stages"]:
            logger.warning(f"Attempting to complete unknown stage: {stage_name}")
            self.set_stage(stage_name)
            
        # Update stage info
        stage = self.pipeline_state["stages"][stage_name]
        end_time = time.time()
        duration = end_time - stage.get("start_time", end_time)
        
        stage.update({
            "status": "completed",
            "end_time": end_time,
            "duration": duration,
            "progress": 1.0,
            "message": "Completed successfully",
            "result": stage_result
        })
        
        # Store performance metrics
        self.performance_metrics["stage_durations"][stage_name] = duration
        
        logger.info(f"[{self.run_id}] Completed stage: {stage_name} in {duration:.2f}s")
        
        # Update overall progress
        self._update_overall_progress()
    
    def fail_stage(self, stage_name: str, error_msg: str) -> None:
        """Mark a stage as failed."""
        if stage_name not in self.pipeline_state["stages"]:
            logger.warning(f"Attempting to fail unknown stage: {stage_name}")
            self.set_stage(stage_name)
            
        # Update stage info
        stage = self.pipeline_state["stages"][stage_name]
        end_time = time.time()
        duration = end_time - stage.get("start_time", end_time)
        
        stage.update({
            "status": "failed",
            "end_time": end_time,
            "duration": duration,
            "error": error_msg,
            "message": f"Failed: {error_msg[:100]}..."
        })
        
        # Add to central error list
        error_entry = {
            "stage": stage_name,
            "message": error_msg,
            "time": end_time
        }
        self.pipeline_state["errors"].append(error_entry)
        
        # Store performance metrics despite failure
        self.performance_metrics["stage_durations"][stage_name] = duration
        
        logger.error(f"[{self.run_id}] Failed stage: {stage_name} - {error_msg}")
        
        # Update overall progress
        self._update_overall_progress()
    
    def update_stage_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update progress for the current stage."""
        current_stage = self.pipeline_state.get("current_stage")
        if not current_stage or current_stage not in self.pipeline_state["stages"]:
            logger.warning(f"Cannot update progress for invalid stage: {current_stage}")
            return
            
        stage = self.pipeline_state["stages"][current_stage]
        if stage["status"] != "running":
            return
            
        progress = max(0.0, min(1.0, progress))
        stage["progress"] = progress
        
        # Update message if provided
        if message:
            stage["message"] = message
            
        # Recalculate overall progress
        self._update_overall_progress()
    
    def update_progress(self, progress: float, message: str) -> None:
        """Update overall progress and invoke the callback."""
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Ensure message is a string
        if message is None:
            message = "Processing..."
        
        # Update state
        self.pipeline_state["progress"] = progress
        self.pipeline_state["progress_message"] = message
        
        # Call progress callback if registered
        if self.progress_callback:
            try:
                # Build data for callback
                callback_data = {
                    "progress": progress,
                    "message": message,
                    "current_stage": self.pipeline_state.get("current_stage"),
                    "stages": {
                        k: {
                            "status": v.get("status"),
                            "progress": v.get("progress", 0),
                            "message": v.get("message", "")
                        } for k, v in self.pipeline_state.get("stages", {}).items()
                    }
                }
                self.progress_callback(progress, callback_data)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def _update_overall_progress(self) -> None:
        """Recalculate overall progress based on stage weights and status."""
        stages = self.pipeline_state["stages"]
        if not stages:
            return
            
        # Get stage weights from config or use equal weights
        stage_weights = self.assessment_config.get("workflow", {}).get("stage_weights", {})
        enabled_stages = self.assessment_config.get("workflow", {}).get("enabled_stages", list(stages.keys()))
        
        total_weight = 0
        weighted_progress = 0.0
        
        for stage_name, stage in stages.items():
            if stage_name not in enabled_stages:
                continue
                
            # Get weight (default to equal weighting)
            weight = stage_weights.get(stage_name, 1.0 / len(enabled_stages))
            total_weight += weight
            
            # Calculate progress contribution
            if stage["status"] == "completed":
                stage_progress = 1.0
            elif stage["status"] == "failed":
                stage_progress = stage.get("progress", 0.0)
            else:
                stage_progress = stage.get("progress", 0.0)
                
            weighted_progress += weight * stage_progress
            
        # Update overall progress
        if total_weight > 0:
            overall_progress = min(1.0, weighted_progress / total_weight)
            self.pipeline_state["progress"] = overall_progress
            
            # Update progress message
            current_stage = self.pipeline_state.get("current_stage")
            if current_stage and current_stage in stages:
                stage_message = stages[current_stage].get("message", "Processing...")
                self.pipeline_state["progress_message"] = stage_message
    
    # --- Callbacks ---
    
    def set_progress_callback(self, callback: Callable[[float, Dict[str, Any]], None]) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    # --- Performance Tracking ---
    
    def track_token_usage(self, tokens: int, stage: str = None) -> None:
        """Track token usage."""
        if tokens <= 0:
            return
            
        self.performance_metrics["total_tokens"] += tokens
        self.performance_metrics["llm_calls"] += 1
        
        # Track by stage
        stage_name = stage or self.pipeline_state.get("current_stage")
        if stage_name:
            stage_tokens = self.performance_metrics["stage_tokens"].get(stage_name, 0)
            self.performance_metrics["stage_tokens"][stage_name] = stage_tokens + tokens
    
    # --- Warning/Error Tracking ---
    
    def log_agent_action(self, agent_name: str, action: str, details: Dict[str, Any] = None) -> None:
        """Compatibility method for BaseAgent to log actions."""
        # Just log the action - we don't need to store these in our simplified approach
        logger.debug(f"Agent {agent_name} performed {action}: {details or {}}")

    def register_agent(self, stage_name: str, agent_name: str) -> None:
        """Register an agent with a stage - compatibility method."""
        if stage_name in self.pipeline_state["stages"]:
            self.pipeline_state["stages"][stage_name]["agent"] = agent_name
            
        # Also add to document_info
        if agent_name not in self.document_info["processed_by"]:
            self.document_info["processed_by"].append(agent_name)
            
    def add_warning(self, message: str, stage: str = None) -> None:
        """Add a warning message."""
        warning = {
            "message": message,
            "stage": stage or self.pipeline_state.get("current_stage"),
            "time": time.time()
        }
        self.pipeline_state["warnings"].append(warning)
        logger.warning(f"[{self.run_id}] {message}")
    
    # --- Result Generation ---
    
    def get_final_result(self) -> Dict[str, Any]:
        """Generate the final result dictionary."""
        # Start with formatted output as the main result
        formatted_data = self.get_data("formatted")
        
        final_result = {
            "result": formatted_data or {},
            "metadata": {
                "assessment_id": self.assessment_id,
                "assessment_type": self.assessment_type,
                "assessment_display_name": self.display_name,
                "document_info": self.document_info,
                "processing_time_seconds": round(time.time() - self.start_time, 2),
                "run_id": self.run_id,
                "stages_completed": [s for s, data in self.pipeline_state["stages"].items() 
                                    if data.get("status") == "completed"],
                "stages_failed": [s for s, data in self.pipeline_state["stages"].items() 
                                if data.get("status") == "failed"],
                "errors": self.pipeline_state["errors"],
                "warnings": self.pipeline_state["warnings"],
                "options": self.options
            },
            "statistics": {
                "total_tokens": self.performance_metrics["total_tokens"],
                "stage_tokens": self.performance_metrics["stage_tokens"],
                "total_llm_calls": self.performance_metrics["llm_calls"],
                "stage_durations": {k: round(v, 2) for k, v in self.performance_metrics["stage_durations"].items()},
                "total_chunks": len(self.chunks)
            }
        }
        
        # Get items specific to assessment type for backward compatibility
        items_key = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }.get(self.assessment_type, "items")
        
        # Try to get the items in priority order
        items = (
            self.get_data("evaluated", items_key) or 
            self.get_data("aggregated", items_key) or 
            self.get_data("extracted", items_key) or 
            []
        )
        
        # Add to extracted_info for compatibility with utils.result_accessor
        final_result["extracted_info"] = {items_key: items}
        
        # Include the overall assessment if available
        overall_assessment = self.get_data("evaluated", "overall_assessment")
        if overall_assessment:
            final_result["overall_assessment"] = overall_assessment
        
        # Include the review data if available
        review_data = self.get_data("review")
        if review_data:
            final_result["review"] = review_data
        
        return final_result