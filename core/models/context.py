# core/models/context.py
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set, Union

class ProcessingContext:
    """
    Enhanced context object for document processing pipelines.
    Manages state, metadata, results, relationships, and metrics throughout processing.
    Serves as the shared brain for multi-agent document analysis.
    """
    
    def __init__(self, document_text: str, options: Optional[Dict[str, Any]] = None):
        """Initialize processing context with document and options."""
        # Core content
        self.document_text = document_text
        self.options = options or {}
        
        # Assessment configuration
        self.assessment_type = self.options.get("assessment_type", "default")
        self.assessment_config = self.options.get("assessment_config", {})
        
        # Document information
        self.document_info = {
            "created_at": datetime.now().isoformat(),
            "word_count": len(document_text.split()),
            "character_count": len(document_text),
            "processed_by": []  # List of agent names that have processed this document
        }
        
        # Chunking information
        self.chunks = []
        self.chunk_metadata = []
        self.chunk_mapping = {}  # Maps text spans to chunk indices
        
        # Entity registry - central repository of all identified entities
        self.entities = {}
        
        # Extracted information by type
        self.extracted_info = {
            "issues": [],        # Extracted issues
            "action_items": [],  # Extracted action items
            "key_points": [],    # Key points/insights
            "entities": [],      # People, organizations, etc.
            "custom": {},        # Custom extraction types
        }
        
        # Evaluation results
        self.evaluations = {}
        
        # Evidence tracking - maps conclusions to supporting text
        self.evidence = {}
        
        # Source reference tracking
        self.source_references = {}
        
        # Relationships between extracted elements
        self.relationships = []
        
        # Results storage by stage
        self.results = {}
        
        # Version history of processing stages
        self.history = []
        
        # Initialize metrics dynamically based on assessment type
        self._initialize_metrics()
        
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
            "progress_message": "Initializing",
            "agent_logs": []
        }
        
        # Progress callback
        self.progress_callback = None
    
    def _initialize_metrics(self):
        """Initialize metrics based on assessment configuration."""
        # Base metrics structure
        self.metrics = {
            "extractions": {},
            "processing": {
                "stage_durations": {},
                "agent_calls": {},
                "tokens_used": 0
            },
            "quality": {}
        }
        
        # Get extraction types from assessment config if available
        assessment_config = self.assessment_config
        if assessment_config:
            workflow = assessment_config.get("workflow", {})
            agent_roles = workflow.get("agent_roles", {})
            
            # Get extractor output schema if available
            if "extractor" in agent_roles:
                extractor_config = agent_roles["extractor"]
                output_schema = extractor_config.get("output_schema", {})
                
                # Initialize metrics for each extraction type in schema
                for key in output_schema.keys():
                    self.metrics["extractions"][key] = 0
        
        # Default extraction types if none found in config
        if not self.metrics["extractions"]:
            default_types = {
                "issues": ["issues", "entities", "key_points"],
                "action_items": ["action_items", "owners", "deadlines"],
                "default": ["items"]
            }
            
            for type_name in default_types.get(self.assessment_type, default_types["default"]):
                self.metrics["extractions"][type_name] = 0
    
    def track_extraction(self, extraction_type: str, count: int = 1) -> None:
        """
        Track extraction metrics for any extraction type.
        
        Args:
            extraction_type: Type of extraction to track
            count: Number to increment by
        """
        if extraction_type not in self.metrics["extractions"]:
            self.metrics["extractions"][extraction_type] = 0
        
        self.metrics["extractions"][extraction_type] += count
    
    def track_token_usage(self, tokens: int) -> None:
        """
        Track token usage for API calls.
        
        Args:
            tokens: Number of tokens used
        """
        self.metrics["processing"]["tokens_used"] += tokens
    
    def record_agent_call(self, agent_name: str, operation: str) -> None:
        """
        Record an agent API call.
        
        Args:
            agent_name: Name of the agent
            operation: Type of operation
        """
        if agent_name not in self.metrics["processing"]["agent_calls"]:
            self.metrics["processing"]["agent_calls"][agent_name] = {}
        
        if operation not in self.metrics["processing"]["agent_calls"][agent_name]:
            self.metrics["processing"]["agent_calls"][agent_name][operation] = 0
        
        self.metrics["processing"]["agent_calls"][agent_name][operation] += 1
    
    def set_stage(self, stage_name: str) -> None:
        """Begin a processing stage."""
        # Record start time for duration tracking
        stage_start_time = time.time()
        
        self.metadata["current_stage"] = stage_name
        self.metadata["stages"][stage_name] = {
            "status": "running",
            "start_time": stage_start_time,
            "progress": 0.0
        }
        
        # Add to history
        self.history.append({
            "timestamp": stage_start_time,
            "type": "stage_start",
            "stage": stage_name
        })
        
        # Update progress message
        self.update_progress(self.metadata["progress"], f"Starting {stage_name}")
    
    def register_agent(self, stage_name: str, agent_name: str) -> None:
        """Register an agent as handling a particular stage."""
        if stage_name in self.metadata["stages"]:
            self.metadata["stages"][stage_name]["agent"] = agent_name
            
            # Make sure processed_by exists in document_info
            if "processed_by" not in self.document_info:
                self.document_info["processed_by"] = []
                
            # Add to document processors if not already there
            if agent_name not in self.document_info["processed_by"]:
                self.document_info["processed_by"].append(agent_name)
    
    def complete_stage(self, stage_name: str, result: Any = None) -> None:
        """Complete a processing stage and store result."""
        if stage_name not in self.metadata["stages"]:
            self.set_stage(stage_name)
            
        # Calculate stage duration
        stage = self.metadata["stages"][stage_name]
        stage["status"] = "completed"
        stage["end_time"] = time.time()
        stage["duration"] = stage["end_time"] - stage["start_time"]
        
        # Record duration in metrics
        self.metrics["processing"]["stage_durations"][stage_name] = stage["duration"]
        
        # Store result
        if result is not None:
            self.results[stage_name] = result
        
        # Add to history
        self.history.append({
            "timestamp": time.time(),
            "type": "stage_complete",
            "stage": stage_name
        })
        
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
            
            # Record duration in metrics
            self.metrics["processing"]["stage_durations"][stage_name] = stage["duration"]
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
        
        # Add to history
        self.history.append({
            "timestamp": time.time(),
            "type": "stage_fail",
            "stage": stage_name,
            "error": error
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
        
        # Add to history
        self.history.append({
            "timestamp": time.time(),
            "type": "progress_update",
            "progress": progress,
            "message": message
        })
        
        # Call progress callback if provided
        if self.progress_callback:
            try:
                self.progress_callback(progress, message)
            except Exception as e:
                self.add_warning(f"Error in progress callback: {str(e)}")
    
    def _update_overall_progress(self) -> None:
        """Update overall progress based on stage progress and weights."""
        # Get stage weights from assessment config or use defaults
        stage_weights = self.assessment_config.get("workflow", {}).get("stage_weights", {
            "document_analysis": 0.05,
            "chunking": 0.05,
            "planning": 0.10,
            "extraction": 0.30,
            "aggregation": 0.20,
            "evaluation": 0.15,
            "formatting": 0.10,
            "review": 0.05
        })
        
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
            overall_progress = weighted_progress / total_weight
            self.metadata["progress"] = overall_progress
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the context."""
        warning = {
            "message": message,
            "time": time.time(),
            "stage": self.metadata.get("current_stage")
        }
        
        self.metadata["warnings"].append(warning)
        
        # Add to history
        self.history.append({
            "timestamp": time.time(),
            "type": "warning",
            "message": message,
            "stage": self.metadata.get("current_stage")
        })
    
    def log_agent_action(self, agent_name: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an agent action for transparency and debugging."""
        log_entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "action": action,
            "stage": self.metadata.get("current_stage"),
            "details": details or {}
        }
        
        if "agent_logs" not in self.metadata:
            self.metadata["agent_logs"] = []
            
        self.metadata["agent_logs"].append(log_entry)
        
        # Track agent call for metrics
        if action in ["generate_completion", "generate_structured_output"]:
            self.record_agent_call(agent_name, action)
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Set document chunks and build chunk mapping."""
        self.chunks = chunks
        self.chunk_metadata = [{
            "chunk_index": chunk.get("chunk_index", i),
            "word_count": chunk.get("word_count", len(chunk.get("text", "").split())),
            "start_position": chunk.get("start_position", 0),
            "end_position": chunk.get("end_position", 0)
        } for i, chunk in enumerate(chunks)]
        
        # Build chunk mapping for text span lookups
        for i, chunk in enumerate(chunks):
            start_pos = chunk.get("start_position", 0)
            end_pos = chunk.get("end_position", 0)
            self.chunk_mapping[(start_pos, end_pos)] = i
    
    def get_chunk_for_position(self, position: int) -> Optional[int]:
        """Get the chunk index containing a specific position in the document."""
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata.get("start_position", 0) <= position <= metadata.get("end_position", 0):
                return i
        return None
    
    def add_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Add an entity to the entity registry.
        
        Args:
            entity_data: Dictionary with entity information
            
        Returns:
            Entity ID
        """
        entity_id = f"entity-{uuid.uuid4().hex[:8]}"
        entity_type = entity_data.get("type", "unknown")
        entity_name = entity_data.get("name", "")
        
        # Normalize entity name for duplicate checking
        normalized_name = entity_name.lower().strip()
        
        # Check for existing entity with same name and type
        for existing_id, existing_entity in self.entities.items():
            if (existing_entity.get("name", "").lower().strip() == normalized_name and
                existing_entity.get("type", "") == entity_type):
                
                # Update existing entity
                for key, value in entity_data.items():
                    if key == "mentions":
                        # Combine mentions
                        existing_mentions = set(existing_entity.get("mentions", []))
                        new_mentions = set(entity_data.get("mentions", []))
                        existing_entity["mentions"] = list(existing_mentions.union(new_mentions))
                    elif key == "source_chunks":
                        # Combine source chunks
                        existing_chunks = set(existing_entity.get("source_chunks", []))
                        new_chunks = set(entity_data.get("source_chunks", []))
                        existing_entity["source_chunks"] = list(existing_chunks.union(new_chunks))
                    else:
                        # Use the most detailed value
                        if key not in existing_entity or len(str(value)) > len(str(existing_entity[key])):
                            existing_entity[key] = value
                
                # Track this extraction
                self.track_extraction("entities", 0)  # 0 because it's a duplicate
                
                return existing_id
        
        # If no existing entity, add new one
        self.entities[entity_id] = {
            "id": entity_id,
            "created_at": time.time(),
            **entity_data
        }
        
        # Also add to extracted info
        self.extracted_info["entities"].append(entity_id)
        
        # Track this extraction
        self.track_extraction("entities", 1)
        
        return entity_id
    
    def add_issue(self, issue_data: Dict[str, Any]) -> str:
        """
        Add an issue to the extracted information.
        
        Args:
            issue_data: Dictionary with issue information
            
        Returns:
            Issue ID
        """
        issue_id = f"issue-{uuid.uuid4().hex[:8]}"
        
        # Store with metadata
        issue = {
            "id": issue_id,
            "created_at": time.time(),
            "stage": self.metadata.get("current_stage"),
            **issue_data
        }
        
        self.extracted_info["issues"].append(issue)
        
        # Track this extraction
        self.track_extraction("issues", 1)
        
        return issue_id
    
    def add_action_item(self, item_data: Dict[str, Any]) -> str:
        """
        Add an action item to the extracted information.
        
        Args:
            item_data: Dictionary with action item information
            
        Returns:
            Action item ID
        """
        item_id = f"action-{uuid.uuid4().hex[:8]}"
        
        # Store with metadata
        action_item = {
            "id": item_id,
            "created_at": time.time(),
            "stage": self.metadata.get("current_stage"),
            **item_data
        }
        
        self.extracted_info["action_items"].append(action_item)
        
        # Track this extraction
        self.track_extraction("action_items", 1)
        
        return item_id
    
    def add_key_point(self, point_data: Union[str, Dict[str, Any]]) -> str:
        """
        Add a key point to the extracted information.
        
        Args:
            point_data: String or dictionary with key point information
            
        Returns:
            Key point ID
        """
        point_id = f"point-{uuid.uuid4().hex[:8]}"
        
        # Convert string to dictionary if needed
        if isinstance(point_data, str):
            point_data = {"text": point_data}
        
        # Store with metadata
        key_point = {
            "id": point_id,
            "created_at": time.time(),
            "stage": self.metadata.get("current_stage"),
            **point_data
        }
        
        self.extracted_info["key_points"].append(key_point)
        
        # Track this extraction
        self.track_extraction("key_points", 1)
        
        return point_id
    
    def add_relationship(self, source_id: str, relationship_type: str, target_id: str) -> str:
        """
        Add a relationship between two items.
        
        Args:
            source_id: ID of the source item
            relationship_type: Type of relationship (e.g., "assigned_to", "depends_on")
            target_id: ID of the target item
            
        Returns:
            Relationship ID
        """
        relationship_id = f"rel-{uuid.uuid4().hex[:8]}"
        
        relationship = {
            "id": relationship_id,
            "source_id": source_id,
            "relationship_type": relationship_type,
            "target_id": target_id,
            "created_at": time.time(),
            "stage": self.metadata.get("current_stage")
        }
        
        self.relationships.append(relationship)
        return relationship_id
    
    def add_source_reference(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a source reference to track text back to the document.
        
        Args:
            text: The text that's being referenced
            source_info: Dictionary with source information (chunk_index, span, etc.)
            
        Returns:
            Reference ID
        """
        ref_id = f"ref-{uuid.uuid4().hex[:8]}"
        
        # Try to find the chunk if not provided
        if source_info is None or "chunk_index" not in source_info:
            source_info = source_info or {}
            # Simple approach - find first chunk containing this text
            for i, chunk in enumerate(self.chunks):
                if text in chunk.get("text", ""):
                    source_info["chunk_index"] = i
                    break
        
        reference = {
            "id": ref_id,
            "text": text,
            "created_at": time.time(),
            **source_info
        }
        
        self.source_references[ref_id] = reference
        return ref_id
    
    def add_evidence(self, conclusion_id: str, evidence_text: str, source_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Add evidence supporting a conclusion.
        
        Args:
            conclusion_id: ID of the conclusion (issue, action item, etc.)
            evidence_text: The text providing evidence
            source_info: Dictionary with source information
            
        Returns:
            Evidence ID
        """
        # First add as a source reference
        ref_id = self.add_source_reference(evidence_text, source_info)
        
        # Then link to the conclusion
        if conclusion_id not in self.evidence:
            self.evidence[conclusion_id] = []
            
        self.evidence[conclusion_id].append(ref_id)
        return ref_id
    
    def add_custom_extraction(self, extraction_type: str, data: Any) -> None:
        """
        Add custom extraction data.
        
        Args:
            extraction_type: Type of extraction
            data: The extracted data
        """
        if extraction_type not in self.extracted_info["custom"]:
            self.extracted_info["custom"][extraction_type] = []
            
        self.extracted_info["custom"][extraction_type].append(data)
        
        # Track this extraction
        self.track_extraction(extraction_type, 1)
    
    def add_evaluation(self, evaluation_type: str, data: Dict[str, Any]) -> None:
        """
        Add evaluation results.
        
        Args:
            evaluation_type: Type of evaluation
            data: The evaluation data
        """
        self.evaluations[evaluation_type] = data
    
    def query_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query entities by type.
        
        Args:
            entity_type: Optional type to filter by
            
        Returns:
            List of matching entities
        """
        if entity_type:
            return [e for e_id, e in self.entities.items() if e.get("type") == entity_type]
        else:
            return list(self.entities.values())
    
    def query_items_by_entity(self, entity_id: str, item_type: str = "action_items") -> List[Dict[str, Any]]:
        """
        Query items related to a specific entity.
        
        Args:
            entity_id: Entity ID to find related items
            item_type: Type of items to return
            
        Returns:
            List of matching items
        """
        # Find relationships where this entity is the target
        related_relationships = [r for r in self.relationships if r["target_id"] == entity_id]
        source_ids = [r["source_id"] for r in related_relationships]
        
        # Return items of the specified type that match these source IDs
        if item_type in self.extracted_info and isinstance(self.extracted_info[item_type], list):
            return [item for item in self.extracted_info[item_type] if item["id"] in source_ids]
        return []
    
    def get_processing_time(self) -> float:
        """Get total processing time so far."""
        return time.time() - self.metadata["start_time"]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of processing metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        summary = {
            "extraction_counts": self.metrics["extractions"],
            "processing_time": self.get_processing_time(),
            "stage_durations": self.metrics["processing"]["stage_durations"],
            "tokens_used": self.metrics["processing"]["tokens_used"],
            "agent_calls": {}
        }
        
        # Calculate total calls per agent
        for agent, operations in self.metrics["processing"]["agent_calls"].items():
            summary["agent_calls"][agent] = sum(operations.values())
        
        return summary
    
    def to_dict(self, include_document: bool = False) -> Dict[str, Any]:
        """
        Serialize the processing context to a dictionary.
        
        Args:
            include_document: Whether to include the full document text
            
        Returns:
            Dictionary representation of the context
        """
        data = {
            "run_id": self.run_id,
            "assessment_type": self.assessment_type,
            "document_info": self.document_info,
            "metadata": self.metadata,
            "results": self.results,
            "extracted_info": self.extracted_info,
            "evaluations": self.evaluations,
            "entities": self.entities,
            "relationships": self.relationships,
            "evidence": self.evidence,
            "chunk_metadata": self.chunk_metadata,
            "history": self.history,
            "metrics": self.metrics
        }
        
        if include_document:
            data["document_text"] = self.document_text
            data["chunks"] = self.chunks
        
        return data
    
    def to_json(self, include_document: bool = False) -> str:
        """
        Serialize the processing context to JSON.
        
        Args:
            include_document: Whether to include the full document text
            
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(include_document), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingContext':
        """
        Create a ProcessingContext from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ProcessingContext instance
        """
        # Create a minimal context
        context = cls(
            document_text=data.get("document_text", ""),
            options={"assessment_type": data.get("assessment_type", "default")}
        )
        
        # Restore all fields
        context.run_id = data.get("run_id", context.run_id)
        context.document_info = data.get("document_info", {})
        context.metadata = data.get("metadata", {})
        context.results = data.get("results", {})
        context.extracted_info = data.get("extracted_info", {})
        context.evaluations = data.get("evaluations", {})
        context.entities = data.get("entities", {})
        context.relationships = data.get("relationships", [])
        context.evidence = data.get("evidence", {})
        context.chunks = data.get("chunks", [])
        context.chunk_metadata = data.get("chunk_metadata", [])
        context.history = data.get("history", [])
        context.metrics = data.get("metrics", {})
        
        # Rebuild chunk_mapping
        for i, chunk in enumerate(context.chunks):
            start_pos = chunk.get("start_position", 0)
            end_pos = chunk.get("end_position", 0)
            context.chunk_mapping[(start_pos, end_pos)] = i
        
        return context
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingContext':
        """
        Create a ProcessingContext from a JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            ProcessingContext instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_final_result(self) -> Dict[str, Any]:
        """Create the final result dictionary with metadata."""
        # Get the formatted result if available
        formatted_result = self.results.get("formatting", {})
        
        # Create result dictionary
        result = {
            "result": formatted_result,
            "metadata": {
                "run_id": self.run_id,
                "assessment_type": self.assessment_type,
                "processing_time": self.get_processing_time(),
                "document_info": self.document_info,
                "stages": self.metadata["stages"],
                "errors": self.metadata["errors"],
                "warnings": self.metadata["warnings"],
                "options": self.options
            },
            "extracted_info": {
                "issues": self._clean_ids(self.extracted_info.get("issues", [])),
                "action_items": self._clean_ids(self.extracted_info.get("action_items", [])),
                "key_points": self._clean_ids(self.extracted_info.get("key_points", [])),
                "entities": [self.entities[e_id] for e_id in self.extracted_info.get("entities", []) if e_id in self.entities]
            },
            "statistics": self.get_metrics_summary()
        }
        
        # Add evaluations to result
        if self.evaluations:
            result["evaluations"] = self.evaluations
        
        return result
    
    def _clean_ids(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal IDs to more user-friendly formats."""
        cleaned = []
        for item in items:
            # Create a copy without internal fields
            cleaned_item = {k: v for k, v in item.items() if not k.startswith("_")}
            cleaned.append(cleaned_item)
        return cleaned
    
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def checkpoint(self, filepath: str) -> None:
        """
        Save a checkpoint of the current processing state.
        
        Args:
            filepath: Path to save the checkpoint
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json(include_document=True))
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'ProcessingContext':
        """
        Load a processing context from a checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            ProcessingContext instance
        """
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())