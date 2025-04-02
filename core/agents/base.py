# core/agents/base.py
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

# Import CustomLLM
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class BaseAgent(ABC):
    """
    Abstract base class for agents that work with the ProcessingContext.
    
    Provides standard interfaces for:
    1. LLM interaction (generate text and structured output)
    2. Data storage and retrieval from context
    3. Error handling and logging
    4. Evidence tracking
    """

    def __init__(self,
                 llm: CustomLLM,
                 options: Optional[Dict[str, Any]] = None):
        """Initialize BaseAgent with LLM instance and optional agent-specific options."""
        if llm is None:
            raise ValueError("LLM instance cannot be None.")

        self.llm = llm
        self.options = options or {}
        self.role = "base"  # IMPORTANT: Must be overridden by subclasses to match orchestrator role mapping
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"core.agents.{self.name}")

        self.logger.info(f"Agent '{self.name}' initialized for role '{self.role}'.")

    @abstractmethod
    async def process(self, context: ProcessingContext) -> Any:
        """
        Main processing method to be implemented by subclasses.
        
        Args:
            context: The shared ProcessingContext object.
            
        Returns:
            Processing result that will be passed to the next stage.
        """
        pass

    # --- Data Storage & Retrieval Methods ---

    def _store_data(self, context: ProcessingContext, data_type: str, data: Any) -> None:
        """
        Store agent results in the ProcessingContext.
        
        Args:
            context: The ProcessingContext
            data_type: Type of data (e.g., "action_items", "issues", "overall_assessment")
                Note: Should be None for 'planning' and 'formatted' categories
            data: The data to store
        """
        # Special case for planning and formatting which don't use data_type
        if self.role in ["planner", "formatter", "reviewer"]:
            context.store_agent_data(self.role, None, data)
            self._log_debug(f"Stored {self.role} data in context", context)
        else:
            if not data_type:
                self._log_warning(f"No data_type specified for {self.role}. Using 'default'", context)
                data_type = "default"
                
            context.store_agent_data(self.role, data_type, data)
            self._log_debug(f"Stored {data_type} data in context for {self.role}", context)
        
    def _get_data(self, context: ProcessingContext, agent_role: str, data_type: str = None, default=None) -> Any:
        """
        Retrieve data from the ProcessingContext.
        
        Args:
            context: The ProcessingContext
            agent_role: Role of the agent that stored the data (e.g., "extractor", "aggregator")
            data_type: Type of data to retrieve
            default: Default value if not found
            
        Returns:
            The retrieved data or default value
        """
        result = context.get_data_for_agent(agent_role, data_type, default)
        if result is None or result == default:
            self._log_debug(f"No data found from {agent_role}" + (f" with type {data_type}" if data_type else ""), context)
            return default
        
        self._log_debug(f"Retrieved data from {agent_role}" + (f" with type {data_type}" if data_type else ""), context)
        return result
    
    def _add_evidence(self, context: ProcessingContext, item_id: str, 
                     evidence_text: str, chunk_index: Optional[int] = None,
                     confidence: Optional[float] = None) -> str:
        """
        Add evidence to the context for an item.
        
        Args:
            context: The ProcessingContext
            item_id: The ID of the item being evidenced
            evidence_text: The text supporting the item
            chunk_index: Optional chunk index for source tracking
            confidence: Optional confidence score
            
        Returns:
            The evidence reference ID
        """
        source_info = {"chunk_index": chunk_index} if chunk_index is not None else None
        ref_id = context.add_evidence(item_id, evidence_text, source_info, confidence)
        self._log_debug(f"Added evidence (ref_id: {ref_id}) for item {item_id}", context)
        return ref_id
        
    def _get_evidence(self, context: ProcessingContext, item_id: str) -> List[Dict[str, Any]]:
        """
        Get all evidence for an item.
        
        Args:
            context: The ProcessingContext
            item_id: The ID of the item
            
        Returns:
            List of evidence dictionaries
        """
        evidence = context.get_evidence_for_item(item_id)
        self._log_debug(f"Retrieved {len(evidence)} evidence items for item {item_id}", context)
        return evidence

    # --- Logging Helpers ---
    
    def _log_info(self, message: str, context: Optional[ProcessingContext] = None) -> None:
        """Log an info message, optionally including context run_id."""
        prefix = f"[{context.run_id}] " if context and hasattr(context, 'run_id') else ""
        self.logger.info(f"{prefix}{message}")

    def _log_debug(self, message: str, context: Optional[ProcessingContext] = None) -> None:
        """Log a debug message, optionally including context run_id."""
        prefix = f"[{context.run_id}] " if context and hasattr(context, 'run_id') else ""
        self.logger.debug(f"{prefix}{message}")

    def _log_warning(self, message: str, context: Optional[ProcessingContext] = None) -> None:
        """Log a warning message, optionally including context run_id."""
        prefix = f"[{context.run_id}] " if context and hasattr(context, 'run_id') else ""
        self.logger.warning(f"{prefix}{message}")
        if context and hasattr(context, 'add_warning'):
            context.add_warning(message, stage=context.pipeline_state.get("current_stage"))

    def _log_error(self, message: str, context: Optional[ProcessingContext] = None, exc_info=False) -> None:
        """Log an error message, optionally including context run_id and exception info."""
        prefix = f"[{context.run_id}] " if context and hasattr(context, 'run_id') else ""
        self.logger.error(f"{prefix}{message}", exc_info=exc_info)

    # --- LLM Interaction Methods ---

    async def _generate(self, 
                       prompt: str, 
                       context: ProcessingContext, 
                       system_prompt: Optional[str]=None, 
                       **kwargs) -> str:
        """
        Generate text completion using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            context: The ProcessingContext for tracking tokens and logging
            system_prompt: Optional system message
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            The generated text string
        """
        self._log_debug(f"Generating LLM completion. Prompt length: {len(prompt)}", context)
        
        try:
            # Default LLM parameters from agent options or method kwargs
            llm_params = {
                "max_tokens": self.options.get("max_tokens", 1500),
                "temperature": self.options.get("temperature", 0.5),
                **kwargs  # Allow overriding via direct call args
            }
            
            # Call the LLM method which returns (result_str, usage_dict)
            result_str, usage_dict = await self.llm.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                **llm_params
            )

            # Track token usage using info from the returned usage_dict
            total_tokens = usage_dict.get("total_tokens")
            if total_tokens is not None:
                context.track_token_usage(total_tokens)
            
            self._log_debug(f"LLM call successful. Response length: {len(result_str)}, Tokens: {total_tokens or 'unknown'}", context)

            return result_str

        except Exception as e:
            self._log_error(f"LLM completion failed: {e}", context, exc_info=True)
            raise RuntimeError(f"LLM call failed for agent {self.name}") from e

    async def _generate_structured(self, 
                                prompt: str, 
                                output_schema: Dict, 
                                context: ProcessingContext, 
                                system_prompt: Optional[str]=None, 
                                **kwargs) -> Dict[str, Any]:
        """
        Generate structured output (JSON) using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            output_schema: JSON schema defining the expected output structure
            context: The ProcessingContext for tracking tokens and logging
            system_prompt: Optional system message
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            The generated structured output as a dictionary
        """
        schema_properties = list(output_schema.get("properties", {}).keys())
        self._log_debug(f"Generating structured LLM output with schema keys: {schema_properties}", context)
        
        try:
            llm_params = {
                "max_tokens": self.options.get("max_structured_tokens", 2000),
                "temperature": self.options.get("structured_temperature", 0.2),
                **kwargs
            }
            
            # Call the LLM method which returns (result_dict, usage_dict)
            result_dict, usage_dict = await self.llm.generate_structured_output(
                prompt=prompt,
                output_schema=output_schema,
                system_prompt=system_prompt,
                **llm_params
            )

            # Track token usage
            total_tokens = usage_dict.get("total_tokens")
            if total_tokens is not None:
                context.track_token_usage(total_tokens)
            
            self._log_debug(f"Structured LLM call successful. Tokens: {total_tokens or 'unknown'}", context)

            # Basic validation: Ensure we got a dictionary
            if not isinstance(result_dict, dict):
                raise TypeError(f"Structured LLM call did not return a dictionary as expected. Type: {type(result_dict)}")

            return result_dict

        except Exception as e:
            self._log_error(f"Structured LLM generation failed: {e}", context, exc_info=True)
            
            # Return an empty dictionary as a fallback
            self._log_warning("Returning empty dict as fallback from failed structured LLM call", context)
            return {}