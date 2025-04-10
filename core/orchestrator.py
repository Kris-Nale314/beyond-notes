# core/orchestrator.py
import time
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime, timezone

from core.models.document import Document
from core.models.context import ProcessingContext
from core.llm.customllm import CustomLLM
from utils.chunking import chunk_document
from assessments.loader import AssessmentLoader
from core.agents.summarizer import SummarizerAgent


logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates the multi-agent document processing pipeline with improved error handling,
    progress tracking, and performance monitoring.
    
    The orchestrator:
    1. Loads assessment configurations
    2. Initializes the ProcessingContext
    3. Coordinates the agent pipeline
    4. Manages document chunking and analysis
    5. Tracks progress and performance
    """

    def __init__(self,
                 assessment_id: str,
                 options: Optional[Dict[str, Any]] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            assessment_id: The unique ID of the assessment configuration
            options: Runtime configuration options
            api_key: API key for the LLM provider (uses env var if not provided)
        """
        self.assessment_id = assessment_id
        self.options = options or {}
        self.context = None
        
        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set OPENAI_API_KEY env var.")
        
        # Initialize Assessment Loader
        self.assessment_loader = AssessmentLoader()

        # Load assessment configuration
        logger.info(f"Orchestrator loading configuration for assessment_id: '{self.assessment_id}'")
        self.assessment_config = self.assessment_loader.load_config(self.assessment_id)
        if not self.assessment_config:
            logger.error(f"Failed to load assessment configuration for id: {self.assessment_id}")
            raise ValueError(f"Assessment configuration not found or invalid for id: {self.assessment_id}")
            
        # Extract key info
        self.assessment_type = self.assessment_config.get("assessment_type", "unknown")
        self.workflow_config = self.assessment_config.get("workflow", {})
        self.display_name = self.assessment_config.get("display_name", self.assessment_id)
        
        logger.info(f"Successfully loaded configuration for '{self.display_name}' (type: {self.assessment_type})")
        
        # Initialize LLM
        model_from_options = self.options.get("model")
        model_from_config = self.assessment_config.get("llm_settings", {}).get("model", "gpt-3.5-turbo")
        model = model_from_options or model_from_config
        self.llm = CustomLLM(self.api_key, model=model)
        logger.info(f"Orchestrator using LLM model: {self.llm.model}")
        
        # Initialize agent registry
        self.agents = {}
        self._load_agents()
        
        # Progress callback
        self.progress_callback = None

    def _load_agents(self) -> None:
        """Load and initialize the agents needed for the pipeline."""
        try:
            # Dynamic imports to avoid circular dependencies
            from core.agents.planner import PlannerAgent
            from core.agents.extractor import ExtractorAgent
            from core.agents.aggregator import AggregatorAgent
            from core.agents.evaluator import EvaluatorAgent
            from core.agents.formatter import FormatterAgent
            from core.agents.reviewer import ReviewerAgent
            # Import SummarizerAgent
            from core.agents.summarizer import SummarizerAgent
            
            # Get agent options from runtime options
            agent_options = self.options.get("agent_options", {})
            
            # Initialize agents with the LLM instance
            self.agents["planner"] = PlannerAgent(self.llm, agent_options.get("planner", {}))
            self.agents["extractor"] = ExtractorAgent(self.llm, agent_options.get("extractor", {}))
            self.agents["summarizer"] = SummarizerAgent(self.llm, agent_options.get("summarizer", {}))
            self.agents["aggregator"] = AggregatorAgent(self.llm, agent_options.get("aggregator", {}))
            self.agents["evaluator"] = EvaluatorAgent(self.llm, agent_options.get("evaluator", {}))
            self.agents["formatter"] = FormatterAgent(self.llm, agent_options.get("formatter", {}))
            self.agents["reviewer"] = ReviewerAgent(self.llm, agent_options.get("reviewer", {}))
            
            # Verify agent roles
            for role, agent in self.agents.items():
                # Skip role check for summarizer since it uses extractor role
                if role == "summarizer" and agent.role == "extractor":
                    continue
                    
                if agent.role != role:
                    logger.warning(f"Agent '{agent.name}' has role '{agent.role}' but was registered as '{role}'. This may cause data flow issues.")
                    agent.role = role  # Set the correct role to match the registry key
                    logger.info(f"Corrected '{agent.name}' role to '{role}'")
            
            logger.info(f"Loaded agents for assessment type: {self.assessment_type}")
            
        except ImportError as e:
            logger.error(f"Error importing agent class: {e}", exc_info=True)
            raise RuntimeError(f"Failed to import agent classes: {e}") from e
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize agents: {e}") from e

    # Update the _execute_stage method to use SummarizerAgent for distill assessments
    async def _execute_stage(self, stage_name: str) -> Any:
        """
        Execute a processing stage using the appropriate agent.
        
        Args:
            stage_name: Name of the stage to execute
            
        Returns:
            Stage result or None if execution failed
        """
        # Use summarizer for extraction stage in distill assessments
        if stage_name == "extraction" and self.assessment_type == "distill":
            agent_role = "summarizer"
            logger.info(f"Using SummarizerAgent for extraction stage in distill assessment")
        else:
            agent_role = self._get_agent_role_for_stage(stage_name)
            
        if not agent_role or agent_role not in self.agents:
            error_msg = f"No agent configured or available for stage '{stage_name}' (role: {agent_role})."
            self.context.fail_stage(stage_name, error_msg)
            logger.error(error_msg)
            return None

        agent = self.agents[agent_role]
        self.context.set_stage(stage_name)
        
        # Register agent with the context
        self.context.register_agent(stage_name, agent.name)

        logger.info(f"Executing stage: {stage_name} with agent: {agent_role}")
        
        try:
            # Process with the agent
            result = await agent.process(self.context)
            
            # Complete the stage
            self.context.complete_stage(stage_name, result)
            return result
            
        except Exception as e:
            error_message = f"Error executing agent '{agent_role}' for stage '{stage_name}': {str(e)}"
            self.context.fail_stage(stage_name, error_message)
            logger.error(error_message, exc_info=True)
            
            # Decide whether to re-raise based on config
            halt_on_error = self.workflow_config.get("halt_on_error", False)
            if halt_on_error:
                raise
                
            return None

    async def process_document(self, document: Document) -> Dict[str, Any]:
        """
        Process a document through the agent pipeline.
        
        Args:
            document: Document object to process
            
        Returns:
            Processing results as a dictionary
        """
        start_time = time.time()
        logger.info(f"Starting processing for document: {document.filename}, assessment: '{self.display_name}' ({self.assessment_id})")
        
        # Initialize context with all necessary information in one step
        try:
            self.context = ProcessingContext(
                document_text=document.text,
                assessment_config=self.assessment_config,
                options=self.options,
                document_metadata=document.get_summary()
            )
            
            # Set progress callback if available
            if self.progress_callback:
                self.context.set_progress_callback(self.progress_callback)
                
        except Exception as e:
            logger.error(f"Error initializing context: {e}", exc_info=True)
            return {
                "result": None,
                "metadata": {"error": f"Failed to initialize context: {e}"},
                "statistics": {}
            }
            
        try:
            # Execute the pipeline
            await self._execute_pipeline(document)
            
            # Get final result
            final_result = self.context.get_final_result()
            
            # Ensure metadata exists and includes processing time
            if isinstance(final_result, dict) and "metadata" in final_result:
                final_result["metadata"]["total_processing_time"] = time.time() - start_time
                
            logger.info(f"Processing complete for assessment '{self.assessment_id}' in {time.time() - start_time:.2f}s.")
            return final_result
            
        except Exception as e:
            logger.error(f"Error during document processing: {e}", exc_info=True)
            
            # Try to get partial results if possible
            if self.context:
                return self.context.get_final_result()
            else:
                return {"result": None, "metadata": {"error": str(e)}, "statistics": {}}

    async def _execute_pipeline(self, document: Document) -> None:
        """
        Execute the complete processing pipeline based on the workflow configuration,
        with special handling for summarization (distill) assessment type.
        
        Args:
            document: Document to process
        """
        # 1. Document Analysis
        try:
            await self._analyze_document(document)
        except Exception as e:
            logger.error(f"Error in document analysis: {e}", exc_info=True)
            self.context.fail_stage("document_analysis", str(e))
            # Continue despite error
            
        # 2. Document Chunking
        try:
            await self._chunk_document(document)
        except Exception as e:
            logger.error(f"Error in document chunking: {e}", exc_info=True)
            self.context.fail_stage("chunking", str(e))
            # If chunking fails completely, we need to stop
            if not self.context.chunks:
                logger.error("No chunks created. Cannot proceed with processing.")
                return
        
        # Special handling for distill/summarization type
        if self.assessment_type == "distill":
            logger.info("Using streamlined pipeline for summarization assessment")
            
            # Simplified stages for summarization
            streamlined_stages = ["planning", "extraction", "aggregation"]
            
            # Get user-specified format from options
            format_type = self.options.get("user_options", {}).get("format", "executive")
            logger.info(f"Summarization format type: {format_type}")
            
            # 3. Execute streamlined stages
            for stage_name in streamlined_stages:
                # Check if previous required stages completed successfully
                prerequisites = self._get_stage_prerequisites(stage_name)
                prerequisites_ok = True
                
                for prereq in prerequisites:
                    prereq_stage = self.context.pipeline_state.get("stages", {}).get(prereq, {"status": "not_started"})
                    if prereq_stage.get("status") != "completed":
                        logger.warning(f"Prerequisite stage '{prereq}' for '{stage_name}' did not complete successfully. Status: {prereq_stage.get('status', 'unknown')}")
                        prerequisites_ok = False
                        
                if not prerequisites_ok:
                    logger.error(f"Skipping stage '{stage_name}' due to failed prerequisites.")
                    self.context.fail_stage(stage_name, "Skipped due to failed prerequisites")
                    continue
                    
                # Execute the stage
                try:
                    stage_start_time = time.time()
                    await self._execute_stage(stage_name)
                    stage_duration = time.time() - stage_start_time
                    logger.info(f"Completed stage '{stage_name}' in {stage_duration:.2f}s")
                    
                    # Debug: Log data state after stage completion
                    self._log_data_state_after_stage(stage_name)
                    
                except Exception as e:
                    logger.error(f"Error in stage '{stage_name}': {str(e)}", exc_info=True)
                    self.context.fail_stage(stage_name, str(e))
                    
                    # Check if we should halt on error
                    halt_on_error = self.workflow_config.get("halt_on_error", False)
                    if halt_on_error:
                        logger.warning(f"Halting workflow due to error in '{stage_name}'")
                        break
            
            # 4. Direct formatting based on format type (implementing this method separately)
            try:
                logger.info(f"Executing direct summary formatting in format: {format_type}")
                self.context.set_stage("formatting")
                await self._generate_formatted_summary(format_type)
                self.context.complete_stage("formatting", {"message": f"Generated {format_type} summary format"})
                
                # Skip review stage for streamlined pipeline
                logger.info("Skipping review stage in streamlined pipeline")
                
            except Exception as e:
                logger.error(f"Error in direct formatting: {str(e)}", exc_info=True)
                self.context.fail_stage("formatting", str(e))
        
        else:
            # Standard pipeline for other assessment types
            logger.info(f"Using standard pipeline for {self.assessment_type} assessment")
            
            # 3. Execute enabled stages from the workflow configuration
            enabled_stages = self.workflow_config.get("enabled_stages", [])
            logger.info(f"Executing enabled stages: {enabled_stages}")
            
            for stage_name in enabled_stages:
                # Skip stages already done in steps 1-2
                if stage_name in ["document_analysis", "chunking"]:
                    continue
                    
                # Check if previous required stages completed successfully
                prerequisites = self._get_stage_prerequisites(stage_name)
                prerequisites_ok = True
                
                for prereq in prerequisites:
                    prereq_stage = self.context.pipeline_state.get("stages", {}).get(prereq, {"status": "not_started"})
                    if prereq_stage.get("status") != "completed":
                        logger.warning(f"Prerequisite stage '{prereq}' for '{stage_name}' did not complete successfully. Status: {prereq_stage.get('status', 'unknown')}")
                        prerequisites_ok = False
                        
                if not prerequisites_ok:
                    logger.error(f"Skipping stage '{stage_name}' due to failed prerequisites.")
                    self.context.fail_stage(stage_name, "Skipped due to failed prerequisites")
                    continue
                    
                # Execute the stage
                try:
                    stage_start_time = time.time()
                    await self._execute_stage(stage_name)
                    stage_duration = time.time() - stage_start_time
                    logger.info(f"Completed stage '{stage_name}' in {stage_duration:.2f}s")
                    
                    # Debug: Log data state after stage completion
                    self._log_data_state_after_stage(stage_name)
                    
                except Exception as e:
                    logger.error(f"Error in stage '{stage_name}': {str(e)}", exc_info=True)
                    self.context.fail_stage(stage_name, str(e))
                    
                    # Check if we should halt on error
                    halt_on_error = self.workflow_config.get("halt_on_error", False)
                    if halt_on_error:
                        logger.warning(f"Halting workflow due to error in '{stage_name}'")
                        break

    async def _generate_formatted_summary(self, format_type: str) -> None:
        """
        Generate a formatted summary from aggregated key points.
        
        Args:
            format_type: Type of summary format to generate (executive, comprehensive, bullet_points, narrative)
        """
        logger.info(f"Generating formatted summary in '{format_type}' format")
        
        try:
            # Get aggregated key points
            key_points = self.context.get_data_for_agent("aggregator", "key_points", default=[])
            topics = self.context.get_data_for_agent("aggregator", "topics", default=[])
            
            if not key_points and not topics:
                logger.warning("No key points or topics found for summary generation")
                # Create empty result structure
                summary_result = {
                    "summary": "No key information was found to generate a summary.",
                    "metadata": {
                        "word_count": self.context.document_info.get("word_count", 0),
                        "format_type": format_type,
                        "generated_at": datetime.now(timezone.utc).isoformat()
                    }
                }
                self.context.store_agent_data("formatter", None, summary_result)
                return
            
            # Build a prompt for generating a summary
            prompt = f"""
    You are an expert document summarizer. Create a {format_type} format summary based on the key points
    and topics extracted from a document.

    # Document Information
    - Type: {self.context.document_info.get('filename', 'document')}
    - Word Count: {self.context.document_info.get('word_count', 0)} words

    # Key Points ({len(key_points)} extracted):
    """
            
            # Add key points to the prompt (max 30 for context length)
            for i, point in enumerate(key_points[:30]):
                if isinstance(point, dict):
                    point_text = point.get("text", "")
                    prompt += f"{i+1}. {point_text}\n"
                elif isinstance(point, str):
                    prompt += f"{i+1}. {point}\n"
            
            if len(key_points) > 30:
                prompt += f"... and {len(key_points) - 30} more points\n"
            
            # Add topics if available
            if topics:
                prompt += "\n# Topics:\n"
                for topic in topics:
                    if isinstance(topic, dict):
                        topic_name = topic.get("topic", "")
                        if topic_name:
                            prompt += f"- {topic_name}\n"
            
            # Add format-specific instructions
            prompt += f"\n# Format Instructions - {format_type.title()}:\n"
            
            if format_type == "executive":
                prompt += "Create a concise executive summary (250-500 words) that highlights the most important information."
            elif format_type == "comprehensive":
                prompt += "Create a detailed summary that covers all significant aspects of the document (800-1500 words)."
            elif format_type == "bullet_points":
                prompt += "Transform the content into clear, concise bullet points organized by topic."
            elif format_type == "narrative":
                prompt += "Create a flowing narrative that captures the document's content in a readable story form."
            
            # Add final output instructions
            prompt += "\n\nYour response should be a detailed, well-written summary that accurately represents the content."
            
            # Generate the summary using the LLM
            system_prompt = "You are an expert summarizer that creates accurate, well-structured summaries of documents."
            
            # Call LLM for summary generation
            summary_text, usage = await self.llm.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Track token usage
            if usage and "total_tokens" in usage:
                self.context.track_token_usage(usage["total_tokens"])
            
            # Calculate some basic statistics
            doc_word_count = self.context.document_info.get("word_count", 0)
            summary_word_count = len(summary_text.split())
            
            # Create a proper summary result
            summary_result = {
                "summary": summary_text,
                "metadata": {
                    "word_count": doc_word_count,
                    "format_type": format_type,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                "statistics": {
                    "original_word_count": doc_word_count,
                    "summary_word_count": summary_word_count,
                    "compression_ratio": round((summary_word_count / max(1, doc_word_count)) * 100, 1)
                }
            }
            
            # Store the result
            self.context.store_agent_data("formatter", None, summary_result)
            logger.info(f"Successfully generated {format_type} summary: {summary_word_count} words")
            
        except Exception as e:
            logger.error(f"Error generating formatted summary: {e}", exc_info=True)
            
            # Create an error result
            error_result = {
                "error": f"Failed to generate summary: {str(e)}",
                "metadata": {
                    "word_count": self.context.document_info.get("word_count", 0),
                    "format_type": format_type,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            }
            self.context.store_agent_data("formatter", None, error_result)
            raise

    def _log_data_state_after_stage(self, stage_name: str) -> None:
        """Log the state of the data store after a stage completes for debugging."""
        if not self.context:
            return
            
        agent_role = self._get_agent_role_for_stage(stage_name)
        if not agent_role:
            return
            
        # Map roles to expected data categories
        role_to_category = {
            "planner": "planning",
            "extractor": "extracted",
            "aggregator": "aggregated",
            "evaluator": "evaluated",
            "formatter": "formatted",
            "reviewer": "review"
        }
        
        category = role_to_category.get(agent_role)
        if not category:
            return
            
        # Check if category exists in data store
        if category not in self.context.data:
            logger.warning(f"After stage '{stage_name}', expected data category '{category}' not found in context.data")
            return
            
        # Log data presence
        data = self.context.data[category]
        if data:
            if isinstance(data, dict):
                logger.debug(f"After stage '{stage_name}', data category '{category}' contains keys: {list(data.keys())}")
            elif isinstance(data, list):
                logger.debug(f"After stage '{stage_name}', data category '{category}' contains a list with {len(data)} items")
            else:
                logger.debug(f"After stage '{stage_name}', data category '{category}' contains data of type: {type(data).__name__}")
        else:
            logger.warning(f"After stage '{stage_name}', data category '{category}' is empty")

    async def _analyze_document(self, document: Document) -> None:
        """Analyze document to extract basic metadata."""
        self.context.set_stage("document_analysis")
        try:
            # Document info already fully initialized during context creation
            doc_info_summary = {"message": "Basic document metadata extracted"}
            await asyncio.sleep(0.01)  # Minimal work since metadata is already set
            self.context.complete_stage("document_analysis", doc_info_summary)
        except Exception as e:
            self.context.fail_stage("document_analysis", str(e))
            logger.error(f"Error in document analysis stage: {e}", exc_info=True)
            raise

    async def _chunk_document(self, document: Document) -> None:
        """Split document into processable chunks."""
        self.context.set_stage("chunking")
        try:
            # Get chunking parameters from runtime options with defaults
            chunk_size = self.options.get("chunk_size", 10000)
            chunk_overlap = self.options.get("chunk_overlap", 500)
            logger.info(f"Chunking document with size={chunk_size}, overlap={chunk_overlap}")

            # Chunk the document
            chunks = chunk_document(
                document,
                target_chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            # Store chunks in context
            self.context.set_chunks(chunks)

            # Complete the stage with statistics
            chunk_info = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(c.get("word_count", 0) for c in chunks) / max(len(chunks), 1),
                "chunking_strategy": f"size-{chunk_size}-overlap-{chunk_overlap}"
            }
            self.context.complete_stage("chunking", chunk_info)
            logger.info(f"Document chunking complete: {chunk_info['total_chunks']} chunks created.")

        except Exception as e:
            error_msg = f"Error during document chunking: {str(e)}"
            self.context.fail_stage("chunking", error_msg)
            logger.error(error_msg, exc_info=True)
            raise

    
    def _get_agent_role_for_stage(self, stage_name: str) -> Optional[str]:
        """Map stage name to agent role."""
        # Standard mapping between stage names and agent roles
        stage_to_role = {
            "planning": "planner",
            "extraction": "extractor",
            "aggregation": "aggregator",
            "evaluation": "evaluator",
            "formatting": "formatter",
            "review": "reviewer"
        }
        
        # Get role from mapping or try stage name directly
        role = stage_to_role.get(stage_name)
        if not role and stage_name in self.agents:
            # If stage name is a valid agent key, use it
            role = stage_name
            
        if not role:
            logger.warning(f"No agent role mapping found for stage: '{stage_name}'")
            
        return role

    def _get_stage_prerequisites(self, stage_name: str) -> List[str]:
        """
        Get prerequisite stages that must complete before this one.
        
        Args:
            stage_name: Stage to get prerequisites for
            
        Returns:
            List of prerequisite stage names
        """
        # Standard prerequisite chain
        standard_chain = {
            "planning": ["document_analysis", "chunking"],
            "extraction": ["planning"],
            "aggregation": ["extraction"],
            "evaluation": ["aggregation"],
            "formatting": ["evaluation"],
            "review": ["formatting"]
        }
        
        # Get prerequisites from standard chain or config
        custom_prereqs = self.workflow_config.get("stage_dependencies", {}).get(stage_name, [])
        standard_prereqs = standard_chain.get(stage_name, [])
        
        # Combine and deduplicate
        return list(set(custom_prereqs + standard_prereqs))

    # --- Public API ---
    
    def set_progress_callback(self, callback: Callable[[float, Dict[str, Any]], None]) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback
        
        # Update context if already initialized
        if self.context:
            self.context.set_progress_callback(callback)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        if not self.context:
            return {
                "progress": 0.0, 
                "message": "Not started", 
                "current_stage": None, 
                "stages": {}
            }
            
        return {
            "progress": self.context.pipeline_state.get("progress", 0.0),
            "message": self.context.pipeline_state.get("progress_message", "N/A"),
            "current_stage": self.context.pipeline_state.get("current_stage"),
            "stages": self.context.pipeline_state.get("stages", {})
        }
        
    def get_stage_info(self, stage_name: str) -> Dict[str, Any]:
        """Get status information for a specific stage."""
        if not self.context:
            return {"status": "not_started"}
            
        return self.context.pipeline_state.get("stages", {}).get(stage_name, {"status": "not_started"})
        
    def reset(self) -> None:
        """Reset the orchestrator for processing a new document."""
        logger.info("Resetting orchestrator.")
        self.context = None
        
    def get_assessment_info(self) -> Dict[str, Any]:
        """Get information about the current assessment."""
        return {
            "assessment_id": self.assessment_id,
            "assessment_type": self.assessment_type,
            "display_name": self.display_name,
            "enabled_stages": self.workflow_config.get("enabled_stages", [])
        }
        
    def get_context_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the data in the context for debugging."""
        if not self.context:
            return {"status": "Context not initialized"}
            
        result = {
            "categories": {},
            "evidence_count": len(self.context.evidence_store.get("references", {})),
            "chunks_count": len(self.context.chunks)
        }
        
        # Summarize each data category
        for category, data in self.context.data.items():
            if isinstance(data, dict):
                result["categories"][category] = {
                    "type": "dict",
                    "keys": list(data.keys()),
                    "items_count": {k: len(v) if isinstance(v, list) else 1 for k, v in data.items() if v}
                }
            elif isinstance(data, list):
                result["categories"][category] = {
                    "type": "list",
                    "length": len(data)
                }
            else:
                result["categories"][category] = {
                    "type": type(data).__name__,
                    "empty": not bool(data)
                }
                
        return result

    # --- Async Convenience Method ---
    
    @classmethod
    async def process_document_with_assessment(cls,
                                            document: Document,
                                            assessment_id: str,
                                            options: Dict[str, Any] = None,
                                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Static convenience method to process a document with a specified assessment.
        
        Args:
            document: Document object to process
            assessment_id: ID of the assessment to use
            options: Runtime options
            progress_callback: Callback for progress updates
            
        Returns:
            Processing results
        """
        try:
            orchestrator = cls(assessment_id, options)
            
            if progress_callback:
                orchestrator.set_progress_callback(progress_callback)
                
            return await orchestrator.process_document(document)
            
        except ValueError as e:
            logger.error(f"Failed to initialize orchestrator for assessment '{assessment_id}': {e}")
            return {"result": None, "metadata": {"error": str(e)}, "statistics": {}}
        except Exception as e:
            logger.error(f"Unexpected error during processing for assessment '{assessment_id}': {e}", exc_info=True)
            return {"result": None, "metadata": {"error": f"Unexpected error: {e}"}, "statistics": {}}
        
    def save_processing_context(self, output_path: Path) -> Path:
        """
        Save the processing context to disk alongside the results.
        
        Args:
            output_path: Path to the results file (will use same base name with .pkl extension)
            
        Returns:
            Path to the saved context file
        """
        if not self.context:
            logger.warning("No context available to save")
            return None
            
        # Create a context-safe copy for serialization
        try:
            # Determine the context file path based on the results path
            context_path = output_path.with_suffix('.pkl')
            
            logger.info(f"Saving processing context to: {context_path}")
            
            # Create a serialization-safe copy of the context
            # (removes non-serializable items like functions)
            import pickle
            import copy
            
            # Create a deep copy to avoid modifying the original
            context_copy = copy.deepcopy(self.context)
            
            # Remove potentially problematic attributes for serialization
            if hasattr(context_copy, '_progress_callback'):
                context_copy._progress_callback = None
                
            # Add additional metadata that might be useful for chat
            context_copy.serialized_at = datetime.now(timezone.utc).isoformat()
            context_copy.orchestrator_info = {
                "assessment_id": self.assessment_id,
                "assessment_type": self.assessment_type,
                "display_name": self.display_name
            }
            
            # Save using pickle
            with open(context_path, 'wb') as f:
                pickle.dump(context_copy, f)
                
            logger.info(f"Successfully saved context to {context_path}")
            return context_path
            
        except Exception as e:
            logger.error(f"Error saving context: {str(e)}", exc_info=True)
            return None