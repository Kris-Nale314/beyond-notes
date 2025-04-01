# core/models/context.py
import time
import uuid
import json
from datetime import datetime, timezone
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ProcessingContext:
    """
    Enhanced context manager for multi-agent document processing.
    
    Provides a streamlined interface for agents to store and retrieve data,
    track document chunks, entities, relationships, and evidence in a 
    consistent and predictable structure.
    """

    def __init__(self, 
                 document_text: str,
                 assessment_config: Dict[str, Any],
                 options: Optional[Dict[str, Any]] = None):
        """Initialize the processing context."""
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
        
        # Document info
        self.document_info = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "word_count": len(document_text.split()),
            "character_count": len(document_text),
            "filename": options.get("filename", "unknown"),
            "processed_by": []  # Agent names involved
        }
        
        # Chunking data
        self.chunks = []  # List of document chunks
        self.chunk_mapping = {}  # Maps text spans (start, end) to chunk indices
        
        # Pipeline state tracking
        self.pipeline_state = {
            "current_stage": None,
            "stages": {},      # Status, timing per stage
            "progress": 0.0,
            "progress_message": "Initializing",
            "warnings": [],
            "errors": []
        }
        
        # Token usage tracking
        self.usage_metrics = {
            "total_tokens": 0,
            "stage_tokens": {},
            "llm_calls": 0,
            "stage_durations": {}
        }
        
        # Entity and relationship tracking
        self.entities = {}  # Central registry of entities
        self.relationships = []  # Relationships between items/entities
        
        # Main data store with clear, predictable paths
        self.data = {
            # Planning phase results
            "planning": {},
            
            # Extraction phase results by assessment type
            "extracted": {
                "action_items": [],  # extract
                "issues": [],        # assess
                "key_points": [],    # distill
                "evidence": [],      # analyze
                "entities": []       # All types
            },
            
            # Aggregation phase results
            "aggregated": {
                "action_items": [],  # extract
                "issues": [],        # assess
                "key_points": [],    # distill
                "evidence": [],      # analyze
                "entities": []       # All types
            },
            
            # Evaluation phase results
            "evaluated": {
                "action_items": [],  # extract
                "issues": [],        # assess
                "key_points": [],    # distill
                "evidence": [],      # analyze
                "overall_assessment": {}  # For all types
            },
            
            # Formatting phase results (final output)
            "formatted": {}
        }
        
        # Evidence tracking
        self.evidence_store = {
            "references": {},  # item_id -> list of source text references
            "sources": {}      # source_id -> source text and metadata
        }
        
        # Temporary compatibility layer
        self.results = {}  # Legacy results storage
        
        # Progress callback
        self.progress_callback = None
        
        # Initialize metadata sharing
        self.metadata = {
            "start_time": self.start_time,
            "run_id": self.run_id,
            "current_stage": None,
            "stages": {},
            "errors": [],
            "warnings": [],
            "progress": 0.0,
            "progress_message": "Initializing"
        }
        
        logger.info(f"ProcessingContext initialized: {self.run_id} - {self.display_name}")

    # ---------- Stage & Progress Management ----------
    
    def set_stage(self, stage_name: str) -> None:
        """Begin a processing stage."""
        self.pipeline_state["current_stage"] = stage_name
        self.metadata["current_stage"] = stage_name
        
        # Create stage record
        stage_info = {
            "status": "running",
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "progress": 0.0,
            "message": "Starting...",
            "agent": None,
            "error": None
        }
        
        # Store in both places for compatibility
        self.pipeline_state["stages"][stage_name] = stage_info
        self.metadata["stages"][stage_name] = stage_info
        
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
        
        stage_update = {
            "status": "completed",
            "end_time": end_time,
            "duration": duration,
            "progress": 1.0,
            "message": "Completed successfully"
        }
        
        # Update in both places
        stage.update(stage_update)
        if stage_name in self.metadata["stages"]:
            self.metadata["stages"][stage_name].update(stage_update)
        
        # Store duration for metrics
        self.usage_metrics["stage_durations"][stage_name] = duration
        
        # Store result in legacy location if provided
        if stage_result is not None:
            self.results[stage_name] = stage_result
        
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
        
        stage_update = {
            "status": "failed",
            "end_time": end_time,
            "duration": duration,
            "error": error_msg,
            "message": f"Failed: {error_msg[:100]}..."
        }
        
        # Update in both places
        stage.update(stage_update)
        if stage_name in self.metadata["stages"]:
            self.metadata["stages"][stage_name].update(stage_update)
        
        # Add to central error list in both places
        error_entry = {
            "stage": stage_name,
            "message": error_msg,
            "time": end_time
        }
        self.pipeline_state["errors"].append(error_entry)
        self.metadata["errors"].append(error_entry)
        
        # Store duration for metrics
        self.usage_metrics["stage_durations"][stage_name] = duration
        
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
        
        # Update in both places
        if message:
            stage["message"] = message
            
        if current_stage in self.metadata["stages"]:
            self.metadata["stages"][current_stage]["progress"] = progress
            if message:
                self.metadata["stages"][current_stage]["message"] = message
            
        self._update_overall_progress()
        
    def _update_overall_progress(self) -> None:
        """Recalculate overall progress based on stages."""
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
            self.metadata["progress"] = overall_progress
            
        # Call progress callback if set
        if self.progress_callback:
            try:
                self.progress_callback(
                    self.pipeline_state["progress"], 
                    self.pipeline_state.get("progress_message", "Processing...")
                )
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
                
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set the progress callback function."""
        self.progress_callback = callback
        
    # Fix for the progress callback in ProcessingContext
    # Add this to the update_progress method in ProcessingContext
    def update_progress(self, progress: float, message: str) -> None:
        """Update overall progress and invoke the callback."""
        # Ensure progress is within bounds
        progress = max(0.0, min(1.0, progress))
        
        # Ensure message is a string
        if message is None:
            message = "Processing..."
        
        # Convert any object to string safely
        try:
            message = str(message)
        except:
            message = "Processing..."
        
        # Update state
        self.pipeline_state["progress"] = progress
        self.pipeline_state["progress_message"] = message
        
        # Update metadata too (for backward compatibility)
        self.metadata["progress"] = progress
        self.metadata["progress_message"] = message
        
        # Add to history if the method exists
        try:
            self._add_history("progress_update", {"progress": progress, "message": message})
        except AttributeError:
            # If _add_history doesn't exist, just continue
            pass
        
        # Call progress callback if provided
        if self.progress_callback:
            try:
                # Pass progress and message to callback
                self.progress_callback(progress, message)
            except Exception as e:
                # Log error but don't throw or add new warnings which would call this again
                logger.error(f"Error in progress callback: {str(e)}", exc_info=False)
                # Don't try to add a warning here as it might cause infinite recursion    
        
    # ---------- Warnings & Errors ----------
    
    def add_warning(self, message: str, stage: Optional[str] = None) -> None:
        """Add a warning message."""
        stage = stage or self.pipeline_state.get("current_stage")
        warning = {
            "stage": stage,
            "message": message,
            "time": time.time()
        }
        
        # Store in both places
        self.pipeline_state["warnings"].append(warning)
        self.metadata["warnings"].append(warning)
        
        logger.warning(f"[{self.run_id}] Warning ({stage}): {message}")
        
    # ---------- Agent Registration ----------
    
    def register_agent(self, stage_name: str, agent_name: str) -> None:
        """Register which agent is handling a stage."""
        if stage_name in self.pipeline_state["stages"]:
            self.pipeline_state["stages"][stage_name]["agent"] = agent_name
            
            # Update in metadata too
            if stage_name in self.metadata["stages"]:
                self.metadata["stages"][stage_name]["agent"] = agent_name
            
            # Add to document processors
            if agent_name not in self.document_info["processed_by"]:
                self.document_info["processed_by"].append(agent_name)
                
    # ---------- Token Usage Tracking ----------
    
    def track_token_usage(self, tokens: int) -> None:
        """Track token usage for current stage and overall."""
        if tokens <= 0:
            return
            
        self.usage_metrics["total_tokens"] += tokens
        
        # Track by stage if available
        current_stage = self.pipeline_state.get("current_stage")
        if current_stage:
            stage_tokens = self.usage_metrics["stage_tokens"].get(current_stage, 0)
            self.usage_metrics["stage_tokens"][current_stage] = stage_tokens + tokens
            
        # Increment LLM call counter
        self.usage_metrics["llm_calls"] += 1
        
    # ---------- Document Chunking ----------
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Set document chunks and build chunk mapping."""
        self.chunks = chunks
        
        # Rebuild chunk mapping
        self.chunk_mapping = {}
        for i, chunk in enumerate(chunks):
            start_pos = chunk.get("start_position", 0)
            end_pos = chunk.get("end_position", 0)
            self.chunk_mapping[(start_pos, end_pos)] = i
            
            # Ensure chunk has an index
            if "chunk_index" not in chunk:
                chunk["chunk_index"] = i
        
        logger.info(f"[{self.run_id}] Set {len(chunks)} document chunks")
        
    def get_chunk_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a chunk by its index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None
        
    def get_chunk_for_position(self, position: int) -> Optional[int]:
        """Find which chunk contains a specific position."""
        for i, chunk in enumerate(self.chunks):
            start = chunk.get("start_position", 0)
            end = chunk.get("end_position", 0)
            if start <= position <= end:
                return i
        return None
        
    # ---------- Assessment Configuration Access ----------
    
    def get_assessment_config(self) -> Dict[str, Any]:
        """Get the full assessment configuration."""
        return self.assessment_config
        
    def get_target_definition(self) -> Optional[Dict[str, Any]]:
        """Get primary target definition for the assessment type."""
        definition_map = {
            "distill": "output_definition",
            "extract": "entity_definition",
            "assess": "entity_definition",
            "analyze": "framework_definition"
        }
        
        key = definition_map.get(self.assessment_type)
        if key:
            return self.assessment_config.get(key, {})
        return None
        
    def get_workflow_instructions(self, agent_role: str) -> Optional[str]:
        """Get instructions for a specific agent role."""
        return self.assessment_config.get("workflow", {}).get("agent_instructions", {}).get(agent_role)
        
    def get_output_schema(self) -> Optional[Dict[str, Any]]:
        """Get the output schema definition."""
        return self.assessment_config.get("output_schema")
        
    def get_extraction_criteria(self) -> Optional[Dict[str, Any]]:
        """Get the extraction criteria from the config, if present."""
        return self.assessment_config.get("extraction_criteria")
    
    def get_entity_properties(self) -> Optional[Dict[str, Any]]:
        """Helper to get entity properties if assessment type is extract/assess."""
        if self.assessment_type in ["extract", "assess"]:
            entity_def = self.get_target_definition()
            return entity_def.get("properties") if isinstance(entity_def, dict) else None
        return None
    
    def get_framework_dimensions(self) -> Optional[List[Dict[str, Any]]]:
        """Helper to get framework dimensions if assessment type is analyze."""
        if self.assessment_type == "analyze":
            framework_def = self.get_target_definition()
            return framework_def.get("dimensions") if isinstance(framework_def, dict) else None
        return None

    def get_rating_scale(self) -> Optional[Dict[str, Any]]:
        """Helper to get rating scale if assessment type is analyze."""
        if self.assessment_type == "analyze":
            framework_def = self.get_target_definition()
            return framework_def.get("rating_scale") if isinstance(framework_def, dict) else None
        return None
    
    def get_output_format_config(self) -> Optional[Dict[str, Any]]:
        """Get the output format configuration (presentation hints)."""
        return self.assessment_config.get("output_format")
        
    # ---------- Data Storage & Retrieval ----------
    
    def get_data_for_agent(self, agent_role: str, data_type: str) -> Any:
        """
        Standardized method to get data for an agent by role and type.
        
        Args:
            agent_role: The role of the agent (e.g., "extractor", "aggregator")
            data_type: The type of data (e.g., "action_items", "issues")
            
        Returns:
            The requested data or an empty list/dict
        """
        # Map agent roles to data categories
        agent_to_category = {
            "extractor": "extracted",
            "aggregator": "aggregated", 
            "evaluator": "evaluated",
            "formatter": "formatted",
            "planner": "planning"
        }
        
        category = agent_to_category.get(agent_role, "")
        
        if category == "planning":
            return self.data.get("planning", {})
        elif category == "formatted":
            return self.data.get("formatted", {})
        elif category in ["extracted", "aggregated", "evaluated"]:
            if data_type == "overall_assessment" and category == "evaluated":
                return self.data.get(category, {}).get("overall_assessment", {})
            return self.data.get(category, {}).get(data_type, [])
        else:
            logger.warning(f"Unknown agent role or data type: {agent_role}/{data_type}")
            return []
            
    def store_agent_data(self, agent_role: str, data_type: str, data: Any) -> None:
        """
        Standardized method to store data from an agent.
        
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
            "planner": "planning"
        }
        
        category = agent_to_category.get(agent_role, "")
        
        if category == "planning":
            self.data["planning"] = data
            # Store in legacy location too
            self.results["planning_output"] = data
        elif category == "formatted":
            self.data["formatted"] = data
            # Store in legacy location too
            self.results["formatting"] = data
        elif category in ["extracted", "aggregated", "evaluated"]:
            # Handle special case for overall assessment from evaluator
            if category == "evaluated" and data_type == "overall_assessment":
                self.data["evaluated"]["overall_assessment"] = data
                # Store in legacy location too
                self.results["overall_assessment"] = data
            else:
                # Ensure data_type exists in the category
                if data_type not in self.data.get(category, {}):
                    logger.warning(f"Unknown data type for {category}: {data_type}")
                    # Create it anyway
                    self.data.setdefault(category, {})[data_type] = data
                else:
                    self.data[category][data_type] = data
                
                # Store in results too for legacy support
                legacy_key = f"{category}_{data_type}"
                self.results[legacy_key] = data
        else:
            logger.warning(f"Unknown agent role or data type: {agent_role}/{data_type}")
            
    # ---------- Entity Management ----------
    
    def add_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Add an entity to the registry, with optional deduplication.
        
        Args:
            entity_data: Entity data including type, name, etc.
            
        Returns:
            Entity ID (new or existing if merged)
        """
        entity_id = f"entity-{uuid.uuid4().hex[:8]}"
        entity_type = entity_data.get("type", "unknown")
        entity_name = entity_data.get("name", "")
        normalized_name = entity_name.lower().strip()
        
        # Check for existing entity (simple deduplication)
        for existing_id, existing_entity in self.entities.items():
            if (existing_entity.get("name", "").lower().strip() == normalized_name and
                existing_entity.get("type") == entity_type):
                # Found a match - could implement merging logic here if needed
                return existing_id  # Return existing ID
        
        # Store the new entity
        self.entities[entity_id] = {
            "id": entity_id,
            "created_at": time.time(),
            "stage": self.pipeline_state.get("current_stage"),
            **entity_data
        }
        
        # Add to extracted entities list
        self.data["extracted"]["entities"].append(entity_id)
        
        return entity_id
    
    def add_relationship(self, source_id: str, relationship_type: str, target_id: str, 
                         attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship between two entities or items.
        
        Args:
            source_id: ID of the source entity/item
            relationship_type: Type of relationship
            target_id: ID of the target entity/item
            attributes: Optional attributes for the relationship
            
        Returns:
            Relationship ID
        """
        rel_id = f"rel-{uuid.uuid4().hex[:8]}"
        
        relationship = {
            "id": rel_id,
            "source_id": source_id,
            "type": relationship_type,
            "target_id": target_id,
            "attributes": attributes or {},
            "created_at": time.time(),
            "stage": self.pipeline_state.get("current_stage")
        }
        
        self.relationships.append(relationship)
        return rel_id
        
    # ---------- Evidence Management ----------
    
    def add_source_reference(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a snippet of source text with its origin info.
        
        Args:
            text: The source text snippet
            source_info: Additional info like chunk_index
            
        Returns:
            Reference ID
        """
        ref_id = f"ref-{uuid.uuid4().hex[:8]}"
        source_info = source_info or {}
        
        # Auto-detect chunk if not provided
        if "chunk_index" not in source_info and len(text) < 500:
            for i, chunk in enumerate(self.chunks):
                if text in chunk.get("text", ""):
                    source_info["chunk_index"] = i
                    break
                    
        # Store the reference
        self.evidence_store["sources"][ref_id] = {
            "text": text,
            "metadata": source_info,
            "created_at": time.time()
        }
        
        return ref_id
    
    def add_evidence(self, item_id: str, evidence_text: str, source_info: Optional[Dict[str, Any]] = None, 
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
        # Create source reference
        ref_id = self.add_source_reference(evidence_text, source_info)
        
        # Link it to the item
        if item_id not in self.evidence_store["references"]:
            self.evidence_store["references"][item_id] = []
            
        # Add confidence if provided
        evidence_entry = {"ref_id": ref_id}
        if confidence is not None:
            evidence_entry["confidence"] = confidence
            
        self.evidence_store["references"][item_id].append(evidence_entry)
        
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
        
        for evidence_entry in self.evidence_store["references"].get(item_id, []):
            ref_id = evidence_entry.get("ref_id")
            if ref_id and ref_id in self.evidence_store["sources"]:
                source_data = self.evidence_store["sources"][ref_id]
                evidence_item = {
                    "source_id": ref_id,
                    "text": source_data.get("text", ""),
                    "metadata": source_data.get("metadata", {}),
                    "confidence": evidence_entry.get("confidence")
                }
                evidence_list.append(evidence_item)
                
        return evidence_list
    
    # ---------- Legacy Support Methods ----------
    
    def add_action_item(self, item_data: Dict[str, Any]) -> str:
        """Add an action item (compatibility method)."""
        item_id = f"action-{uuid.uuid4().hex[:8]}"
        
        # Create item with metadata
        action_item = {
            "id": item_id,
            "created_at": time.time(),
            "stage": self.pipeline_state.get("current_stage"),
            **item_data
        }
        
        # Store in new structure
        if "action_items" not in self.data["extracted"]:
            self.data["extracted"]["action_items"] = []
            
        self.data["extracted"]["action_items"].append(action_item)
        
        # Ensure legacy extracted_info exists
        if not hasattr(self, 'extracted_info'):
            self.extracted_info = {"action_items": []}
        elif "action_items" not in self.extracted_info:
            self.extracted_info["action_items"] = []
            
        # Store in legacy location too
        self.extracted_info["action_items"].append(action_item)
        
        return item_id
    
    def add_issue(self, issue_data: Dict[str, Any]) -> str:
        """Add an issue (compatibility method)."""
        issue_id = f"issue-{uuid.uuid4().hex[:8]}"
        
        # Create issue with metadata
        issue = {
            "id": issue_id,
            "created_at": time.time(),
            "stage": self.pipeline_state.get("current_stage"),
            **issue_data
        }
        
        # Store in new structure
        if "issues" not in self.data["extracted"]:
            self.data["extracted"]["issues"] = []
            
        self.data["extracted"]["issues"].append(issue)
        
        # Ensure legacy extracted_info exists
        if not hasattr(self, 'extracted_info'):
            self.extracted_info = {"issues": []}
        elif "issues" not in self.extracted_info:
            self.extracted_info["issues"] = []
            
        # Store in legacy location too
        self.extracted_info["issues"].append(issue)
        
        return issue_id
    
    def add_key_point(self, point_data: Union[str, Dict[str, Any]]) -> str:
        """Add a key point (compatibility method)."""
        point_id = f"point-{uuid.uuid4().hex[:8]}"
        
        # Convert string to dict if needed
        if isinstance(point_data, str):
            point_data = {"text": point_data}
        
        # Create point with metadata
        key_point = {
            "id": point_id,
            "created_at": time.time(),
            "stage": self.pipeline_state.get("current_stage"),
            **point_data
        }
        
        # Store in new structure
        if "key_points" not in self.data["extracted"]:
            self.data["extracted"]["key_points"] = []
            
        self.data["extracted"]["key_points"].append(key_point)
        
        # Ensure legacy extracted_info exists
        if not hasattr(self, 'extracted_info'):
            self.extracted_info = {"key_points": []}
        elif "key_points" not in self.extracted_info:
            self.extracted_info["key_points"] = []
            
        # Store in legacy location too
        self.extracted_info["key_points"].append(key_point)
        
        return point_id

    def _add_history(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Add an event to the processing history log.
        
        Args:
            event_type: Type of event (e.g., "stage_start", "progress_update")
            details: Additional details about the event
        """
        if not hasattr(self, 'history'):
            self.history = []
            
        event = {
            "timestamp": time.time(),
            "event": event_type,
            **details
        }
        
        self.history.append(event)
        
    # ---------- Final Result Generation ----------
    
    def get_final_result(self) -> Dict[str, Any]:
        """Generate the final result dictionary."""
        # Start with formatted output as the main result
        final_result = {
            "result": self.data.get("formatted", {}),
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
                "total_tokens": self.usage_metrics["total_tokens"],
                "stage_tokens": self.usage_metrics["stage_tokens"],
                "total_llm_calls": self.usage_metrics["llm_calls"],
                "stage_durations": self.usage_metrics["stage_durations"],
                "total_chunks": len(self.chunks)
            }
        }
        
        # Include summary of extracted items if not in formatted output
        if "extracted_info" not in final_result:
            # Create a clean copy without internal IDs and fields
            extracted_info = {}
            
            for item_type in ["action_items", "issues", "key_points"]:
                if item_type in self.data["evaluated"]:
                    # Prioritize evaluated items
                    items = self.data["evaluated"][item_type]
                elif item_type in self.data["aggregated"]:
                    # Fall back to aggregated items
                    items = self.data["aggregated"][item_type]
                elif item_type in self.data["extracted"]:
                    # Last resort: extracted items
                    items = self.data["extracted"][item_type]
                else:
                    items = []
                    
                if items:
                    # Clean up internal fields starting with underscore
                    extracted_info[item_type] = [{k: v for k, v in item.items() 
                                               if not k.startswith("_")} 
                                             for item in items]
            
            if extracted_info:
                final_result["extracted_info"] = extracted_info
                
        # Include agent access to intermediate results if needed
        if self.options.get("include_intermediate_results", False):
            final_result["intermediate_results"] = self.data
            
        return final_result
        
    # ---------- Serialization ----------
    
    def to_dict(self, include_document: bool = False, include_chunks: bool = False) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        data = {
            "run_id": self.run_id,
            "assessment_id": self.assessment_id,
            "assessment_type": self.assessment_type,
            "assessment_config": self.assessment_config,
            "options": self.options,
            "document_info": self.document_info,
            "pipeline_state": self.pipeline_state,
            "usage_metrics": self.usage_metrics,
            "data": self.data,
            "evidence_store": self.evidence_store,
            "entities": self.entities,
            "relationships": self.relationships,
            "chunks_count": len(self.chunks)
        }
        
        if include_document:
            data["document_text"] = self.document_text
            
        if include_chunks:
            data["chunks"] = self.chunks
            
        return data
        
    def to_json(self, include_document: bool = False, include_chunks: bool = False) -> str:
        """Convert context to JSON string."""
        return json.dumps(self.to_dict(include_document, include_chunks), default=str)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingContext':
        """Create context instance from dictionary."""
        doc_text = data.get("document_text", "")
        assessment_config = data.get("assessment_config", {})
        options = data.get("options", {})
        
        context = cls(doc_text, assessment_config, options)
        
        # Restore fields
        for field in ["run_id", "document_info", "pipeline_state", "usage_metrics", 
                      "data", "evidence_store", "entities", "relationships"]:
            if field in data:
                setattr(context, field, data[field])
                
        # Restore chunks if available
        if "chunks" in data:
            context.chunks = data["chunks"]
            # Rebuild chunk mapping
            context.chunk_mapping = {}
            for i, chunk in enumerate(context.chunks):
                start_pos = chunk.get("start_position", 0)
                end_pos = chunk.get("end_position", 0)
                context.chunk_mapping[(start_pos, end_pos)] = i
                
        # Ensure metadata compatibility
        context.metadata = {
            "start_time": context.pipeline_state.get("start_time", context.start_time),
            "run_id": context.run_id,
            "current_stage": context.pipeline_state.get("current_stage"),
            "stages": context.pipeline_state.get("stages", {}),
            "errors": context.pipeline_state.get("errors", []),
            "warnings": context.pipeline_state.get("warnings", []),
            "progress": context.pipeline_state.get("progress", 0.0),
            "progress_message": context.pipeline_state.get("progress_message", "")
        }
        
        # Restore results for legacy compatibility
        context.results = {}
        for stage, result in data.get("results", {}).items():
            context.results[stage] = result
            
        return context
        
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingContext':
        """Create context instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
        
    # ---------- Utility Methods ----------
    
    def log_agent_action(self, agent_name: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an action taken by an agent for debugging/transparency."""
        log_entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "action": action,
            "stage": self.pipeline_state.get("current_stage"),
            "details": details or {}
        }
        
        # Create agent_logs if it doesn't exist
        if "agent_logs" not in self.metadata:
            self.metadata["agent_logs"] = []
            
        # Log the action
        self.metadata["agent_logs"].append(log_entry)
        
        # Track specific actions for metrics
        if action in ["generate_completion", "generate_structured_output", "llm_call"]:
            self.record_agent_call(agent_name, action)
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of processing metrics."""
        summary = {
            "extraction_counts": {},
            "processing_time_seconds": round(time.time() - self.start_time, 2),
            "stage_durations_seconds": {k: round(v, 2) for k, v in self.usage_metrics["stage_durations"].items()},
            "total_llm_tokens_used": self.usage_metrics["total_tokens"],
            "total_llm_calls": self.usage_metrics["llm_calls"]
        }
        
        # Count extracted items by type
        for category, items in self.data["extracted"].items():
            if isinstance(items, list):
                summary["extraction_counts"][category] = len(items)
                
        return summary
        
    def checkpoint(self, filepath: Union[str, Path]) -> None:
        """Save the current context state to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json(include_document=True, include_chunks=True))
            logger.info(f"ProcessingContext checkpoint saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {path}: {e}", exc_info=True)
            
    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path]) -> 'ProcessingContext':
        """Load context state from a JSON checkpoint file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                context = cls.from_json(f.read())
            logger.info(f"ProcessingContext loaded from checkpoint: {path}")
            return context
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}", exc_info=True)
            raise

