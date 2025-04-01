"""
Orchestrator coordinates the multi-agent document processing pipeline.

It manages the flow between document analysis, chunking, and the various agent stages
defined in the assessment configuration's workflow section. The Orchestrator is
responsible for initializing the Processing Context, running each stage in sequence,
and collating the final results.
"""

import time
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path

from core.models.document import Document
from core.models.context import ProcessingContext
from core.llm.customllm import CustomLLM
from utils.chunking import chunk_document
from assessments.loader import AssessmentLoader

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates the multi-agent document processing pipeline.
    Loads assessment configuration by ID and passes it to the ProcessingContext.
    Executes agents based on the workflow defined in the configuration.
    """

    def __init__(self,
                 assessment_id: str,
                 options: Optional[Dict[str, Any]] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            assessment_id: The unique ID of the assessment configuration (base type or template).
            options: Runtime configuration options (e.g., chunk size, user selections).
            api_key: API key for the LLM provider (uses env var if not provided).
        """
        self.assessment_id = assessment_id
        self.options = options or {}  # Runtime options provided by user/caller
        self.context: Optional[ProcessingContext] = None  # Initialize context as None

        # Get API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set OPENAI_API_KEY env var.")

        # Initialize Assessment Loader
        self.assessment_loader = AssessmentLoader()

        # Load the specific assessment configuration using its ID
        logger.info(f"Orchestrator loading configuration for assessment_id: '{self.assessment_id}'")
        self.assessment_config = self.assessment_loader.load_config(self.assessment_id)
        if not self.assessment_config:
            logger.error(f"Failed to load assessment configuration for id: {self.assessment_id}")
            raise ValueError(f"Assessment configuration not found or invalid for id: {self.assessment_id}")
        logger.info(f"Successfully loaded configuration for '{self.assessment_config.get('display_name', self.assessment_id)}'")

        # Extract key info from the loaded config for orchestrator use
        self.assessment_type = self.assessment_config.get("assessment_type", "unknown")
        self.workflow_config = self.assessment_config.get("workflow", {})
        self.display_name = self.assessment_config.get("display_name", self.assessment_id)

        # Initialize LLM (potentially using model from config or options)
        # Prioritize runtime options over config default for model selection
        model_from_options = self.options.get("model")
        model_from_config = self.assessment_config.get("llm_settings", {}).get("model", "gpt-3.5-turbo")
        model = model_from_options or model_from_config
        self.llm = CustomLLM(self.api_key, model=model)
        logger.info(f"Orchestrator using LLM model: {model}")

        # Initialize agent registry
        self.agents = {}
        self._load_agents()  # Agents will get config via context

    def _load_agents(self) -> None:
        """Load and initialize the agents needed for the pipeline."""
        try:
            # Dynamic imports to avoid circular dependencies
            from core.agents.planner import PlannerAgent
            from core.agents.extractor import ExtractorAgent
            from core.agents.aggregator import AggregatorAgent
            from core.agents.evaluator import EvaluatorAgent
            from core.agents.formatter import FormatterAgent
            from core.agents.reviewer import ReviewerAgent  # Optional

            # Create agent instances - they'll access config via context
            agent_options = self.options.get("agent_options", {})
            
            self.agents["planner"] = PlannerAgent(self.llm, agent_options.get("planner", {}))
            self.agents["extractor"] = ExtractorAgent(self.llm, agent_options.get("extractor", {}))
            self.agents["aggregator"] = AggregatorAgent(self.llm, agent_options.get("aggregator", {}))
            self.agents["evaluator"] = EvaluatorAgent(self.llm, agent_options.get("evaluator", {}))
            self.agents["formatter"] = FormatterAgent(self.llm, agent_options.get("formatter", {}))
            self.agents["reviewer"] = ReviewerAgent(self.llm, agent_options.get("reviewer", {}))

            logger.info(f"Loaded agents for assessment type: {self.assessment_type}")

        except ImportError as e:
            logger.error(f"Error importing agent class: {e}. Check agent file paths and names.", exc_info=True)
            raise RuntimeError(f"Failed to import agent classes: {e}") from e
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize agents: {e}") from e

    async def process_document(self,
                             document: Document,
                             progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Process a document through the agent pipeline using the loaded configuration.
        
        This method is more resilient to failures in individual stages, continuing
        the pipeline when possible rather than halting completely on errors.

        Args:
            document: Document object to process.
            progress_callback: Optional callback for progress updates (percentage, message).

        Returns:
            Processing results as a dictionary.
        """
        start_time = time.time()
        logger.info(f"Starting processing for document: {document.filename}, assessment: '{self.display_name}' ({self.assessment_id})")

        # Create context with the full assessment configuration
        try:
            self.context = ProcessingContext(
                document_text=document.text,
                assessment_config=self.assessment_config,
                options=self.options  # Pass runtime options separately
            )
            # Store document metadata in context
            self.context.document_info.update(document.get_summary())  # Merge doc metadata
            self.context.document_info["assessment_id"] = self.assessment_id  # Add assessment ID used
            self.context.document_info["assessment_display_name"] = self.display_name

        except Exception as e:
            logger.error(f"Fatal error initializing ProcessingContext: {e}", exc_info=True)
            # Return an error structure immediately
            return {
                "result": None,
                "metadata": {"error": f"Failed to initialize context: {e}"},
                "statistics": {}
            }

        if progress_callback:
            self.context.set_progress_callback(progress_callback)

        try:
            # 1. Document Analysis (basic info extraction)
            try:
                await self._analyze_document(document)  # Uses self.context
            except Exception as e:
                logger.error(f"Error in document analysis stage: {e}", exc_info=True)
                # Continue despite error

            # 2. Chunking
            try:
                await self._chunk_document(document)  # Uses self.context and self.options
            except Exception as e:
                logger.error(f"Error in chunking stage: {e}", exc_info=True)
                # If chunking fails completely, we might need to stop
                if not hasattr(self.context, 'chunks') or not self.context.chunks:
                    logger.error("No chunks created. Cannot proceed with processing.")
                    if hasattr(self.context, 'fail_stage'):
                        self.context.fail_stage("chunking", f"No chunks created: {str(e)}")
                    return self.context.get_final_result()

            # 3. Execute Workflow Stages defined in the config
            enabled_stages = self.workflow_config.get("enabled_stages", [])
            logger.info(f"Executing workflow stages: {enabled_stages}")

            # Execute each enabled stage *after* analysis and chunking
            for stage_name in enabled_stages:
                # Skip stages already completed
                if stage_name in ["document_analysis", "chunking"]:
                    continue

                # Execute the current stage
                stage_start_time = time.time()
                logger.info(f"Executing stage: {stage_name}")
                
                try:
                    await self._execute_stage(stage_name)  # Pass document implicitly via context
                    stage_duration = time.time() - stage_start_time
                    logger.info(f"Completed stage: {stage_name} in {stage_duration:.2f}s")
                except Exception as e:
                    logger.error(f"Unhandled error in stage '{stage_name}': {str(e)}", exc_info=True)
                    # Continue to next stage despite errors
                    
                # Check if we should halt on error
                # First try pipeline_state, then fall back to metadata for compatibility
                stage_info = None
                if hasattr(self.context, 'pipeline_state'):
                    stage_info = self.context.pipeline_state.get("stages", {}).get(stage_name, {})
                else:
                    stage_info = self.context.metadata.get("stages", {}).get(stage_name, {})
                    
                if stage_info and stage_info.get("status") == "failed":
                    logger.error(f"Stage '{stage_name}' failed.")
                    error_msg = stage_info.get("error", "Unknown error")
                    # Check if we should halt on error (can be configured)
                    halt_on_error = self.workflow_config.get("halt_on_error", False)  # Default to FALSE for more resilient processing
                    if halt_on_error:
                        logger.warning(f"Halting workflow due to error in '{stage_name}': {error_msg}")
                        break  # Stop processing further stages

            # Return the final result assembled by the context
            final_result = self.context.get_final_result()
            processing_time = time.time() - start_time
            
            # Ensure metadata exists
            if isinstance(final_result, dict) and "metadata" not in final_result:
                final_result["metadata"] = {}
                
            final_result["metadata"]["total_processing_time"] = processing_time  # Add total time
            logger.info(f"Processing complete for assessment '{self.assessment_id}' in {processing_time:.2f}s.")
            return final_result

        except Exception as e:
            # Handle any uncaught exceptions during processing stages
            current_stage = "unknown"
            if hasattr(self.context, 'pipeline_state'):
                current_stage = self.context.pipeline_state.get("current_stage", "unknown")
            elif hasattr(self.context, 'metadata'):
                current_stage = self.context.metadata.get("current_stage", "unknown")
                
            error_message = f"Unhandled error during processing stage '{current_stage}': {str(e)}"
            logger.error(error_message, exc_info=True)
            
            if self.context:
                # Try to mark the stage as failed if possible
                try:
                    self.context.fail_stage(current_stage, error_message)
                except:
                    pass  # Ignore if this fails too
                    
                # Return partial results with error info
                return self.context.get_final_result()
            else:
                # If context failed to init, return minimal error
                return {"result": None, "metadata": {"error": error_message}, "statistics": {}}

    async def _analyze_document(self, document: Document) -> None:
        """Analyze document to extract basic metadata."""
        if not self.context: 
            return  # Should not happen if called after context init
            
        self.context.set_stage("document_analysis")
        try:
            # Document info already added during context creation
            doc_info_summary = {"message": "Basic document metadata extracted"}
            await asyncio.sleep(0.01)  # Simulate minimal work
            self.context.complete_stage("document_analysis", doc_info_summary)
        except Exception as e:
            self.context.fail_stage("document_analysis", str(e))
            logger.error(f"Error in document analysis stage: {e}", exc_info=True)
            # Don't re-raise to allow pipeline to continue

    async def _chunk_document(self, document: Document) -> None:
        """Split document into processable chunks based on runtime options."""
        if not self.context: 
            return
            
        self.context.set_stage("chunking")
        try:
            # Get chunking parameters from runtime options with defaults
            chunk_size = self.options.get("chunk_size", 10000)
            chunk_overlap = self.options.get("chunk_overlap", 500)
            logger.info(f"Chunking document with size={chunk_size}, overlap={chunk_overlap}")

            # Pass the Document object to chunk_document, not just the text
            chunks = chunk_document(
                document,  # Pass the whole Document object, not document.text
                target_chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            self.context.set_chunks(chunks)  # Store chunks in context

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
            # Don't re-raise to allow pipeline to continue if possible

    async def _execute_stage(self, stage_name: str) -> None:
        """
        Execute a processing stage using the appropriate agent.
        
        This version is more resilient to errors, logging failures but
        allowing the pipeline to continue when possible.
        """
        if not self.context: 
            return

        agent_role = self._get_agent_role_for_stage(stage_name)
        if not agent_role or agent_role not in self.agents:
            error_msg = f"No agent configured or available for stage '{stage_name}' (role: {agent_role})."
            self.context.fail_stage(stage_name, error_msg)
            logger.error(error_msg)
            return  # Stop if no agent

        agent = self.agents[agent_role]
        self.context.set_stage(stage_name)
        self.context.register_agent(stage_name, agent.name if hasattr(agent, 'name') else agent_role)

        try:
            # Agent instructions are now accessed by the agent via context helpers
            # The orchestrator doesn't need to pass instructions explicitly
            result = await agent.process(self.context)  # Pass the context object
            self.context.complete_stage(stage_name, result)

        except Exception as e:
            error_message = f"Error executing agent '{agent_role}' for stage '{stage_name}': {str(e)}"
            self.context.fail_stage(stage_name, error_message)
            logger.error(error_message, exc_info=True)
            # No longer re-raise the exception to avoid halting the pipeline
            # Instead, log the error and continue to the next stage
            logger.warning(f"Continuing to next stage despite error in '{stage_name}'")

    def _get_agent_role_for_stage(self, stage_name: str) -> Optional[str]:
        """Determine which agent role is responsible for a stage."""
        # This mapping is useful for the orchestrator to select the agent instance
        stage_to_role = {
            "planning": "planner",
            "extraction": "extractor",
            "aggregation": "aggregator",
            "evaluation": "evaluator",
            "formatting": "formatter",
            "review": "reviewer"
            # Add other custom stages/roles if needed
        }
        role = stage_to_role.get(stage_name)
        if not role:
            logger.warning(f"No standard agent role mapping found for stage: '{stage_name}'")
            # Could check custom stage mappings in config if needed
        return role

    async def process_with_progress(self,
                                 document: Document,
                                 progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Process a document with detailed progress reporting wrapper.
        
        Args:
            document: Document object to process.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Processing results as a dictionary.
        """
        start_time = time.time()
        
        # Use a safe string value for the initial message
        start_message = f"Starting {self.display_name} assessment..."
        if progress_callback:
            try:
                progress_callback(0.0, start_message)
            except Exception as e:
                logger.error(f"Error in initial progress callback: {e}")
                # Continue anyway

        result = await self.process_document(document, progress_callback)

        processing_time = time.time() - start_time
        if progress_callback:
            final_status = "completed" if "error" not in result.get("metadata", {}) else "failed"
            try:
                progress_callback(1.0, f"Assessment {final_status} in {processing_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Error in final progress callback: {e}")
                # Continue anyway

        # Ensure processing time is recorded in metadata
        if isinstance(result, dict) and "metadata" in result:
            result["metadata"]["total_processing_time"] = processing_time

        # Ensure assessment identifiers are included
        if isinstance(result, dict) and "metadata" in result:
            result["metadata"]["assessment_id"] = self.assessment_id
            result["metadata"]["assessment_display_name"] = self.display_name
            result["metadata"]["assessment_type"] = self.assessment_type

        return result

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information from the context."""
        if not self.context:
            return {
                "progress": 0.0, 
                "message": "Not started", 
                "current_stage": None, 
                "stages": {}
            }
            
        # Try pipeline_state first, then fall back to metadata for compatibility
        if hasattr(self.context, 'pipeline_state'):
            return {
                "progress": self.context.pipeline_state.get("progress", 0.0),
                "message": self.context.pipeline_state.get("progress_message", "N/A"),
                "current_stage": self.context.pipeline_state.get("current_stage"),
                "stages": self.context.pipeline_state.get("stages", {})
            }
        else:
            return {
                "progress": self.context.metadata.get("progress", 0.0),
                "message": self.context.metadata.get("progress_message", "N/A"),
                "current_stage": self.context.metadata.get("current_stage"),
                "stages": self.context.metadata.get("stages", {})
            }

    def get_stage_info(self, stage_name: str) -> Dict[str, Any]:
        """Get status information for a specific processing stage from the context."""
        if not self.context:
            return {"status": "not_started"}
            
        # Try pipeline_state first, then fall back to metadata for compatibility
        if hasattr(self.context, 'pipeline_state') and "stages" in self.context.pipeline_state:
            return self.context.pipeline_state.get("stages", {}).get(stage_name, {"status": "not_started"})
        else:
            return self.context.metadata.get("stages", {}).get(stage_name, {"status": "not_started"})

    def reset(self) -> None:
        """Reset the orchestrator's context for processing a new document."""
        logger.info("Resetting orchestrator context.")
        self.context = None
        # Note: Doesn't reset assessment_id or config. Re-init orchestrator for new assessment.

    def get_assessment_name(self) -> str:
        """Get the display name of the current assessment configuration."""
        return self.display_name

    def get_user_options_schema(self) -> Dict[str, Any]:
        """Get the user options schema from the loaded assessment configuration."""
        return self.assessment_config.get("user_options", {})

    def get_output_schema(self) -> Dict[str, Any]:
        """Get the output schema from the loaded assessment configuration."""
        return self.assessment_config.get("output_schema", {})

    @classmethod
    async def process_document_with_assessment(cls,
                                            document: Document,
                                            assessment_id: str,
                                            options: Dict[str, Any] = None,
                                            progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Static convenience method to process a document with a specified assessment.
        
        Args:
            document: Document object to process.
            assessment_id: ID of the assessment configuration to use.
            options: Runtime options (e.g., model, chunk_size).
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Processing results as a dictionary.
        """
        try:
            orchestrator = cls(assessment_id, options)
            return await orchestrator.process_with_progress(document, progress_callback)
        except ValueError as e:  # Catch init errors like config not found
            logger.error(f"Failed to initialize orchestrator for assessment '{assessment_id}': {e}")
            return {"result": None, "metadata": {"error": str(e)}, "statistics": {}}
        except Exception as e:
            logger.error(f"Unexpected error during processing for assessment '{assessment_id}': {e}", exc_info=True)
            return {"result": None, "metadata": {"error": f"Unexpected error: {e}"}, "statistics": {}}