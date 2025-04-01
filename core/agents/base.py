# core/agents/base.py
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

# Import CustomLLM and ProcessingContext
from core.llm.customllm import CustomLLM, UsageDict
from core.models.context import ProcessingContext

class BaseAgent(ABC):
    """
    Abstract base class for agents, handling LLM interaction, options, logging,
    and defining the standard 'process' interface.
    
    Features standardized data storage and retrieval methods for the enhanced
    ProcessingContext, token tracking, and consistent logging.
    """

    def __init__(self,
                 llm: CustomLLM,
                 options: Optional[Dict[str, Any]] = None):
        """Initialize BaseAgent with LLM instance and optional agent-specific options."""
        if llm is None:
            raise ValueError("LLM instance cannot be None.")

        self.llm = llm
        self.options = options or {}
        self.role = "base"  # Should be overridden by subclasses
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"core.agents.{self.name}")

        self.logger.info(f"Agent '{self.name}' initialized for role '{self.role}'.")
        if self.options:
            self.logger.debug(f"Agent '{self.name}' options: {self.options}")

    @abstractmethod
    async def process(self, context: ProcessingContext) -> Any:
        """Main processing method to be implemented by subclasses."""
        pass

    # --- Enhanced Context Interaction Methods ---

    def _store_data(self, context: ProcessingContext, data_type: str, data: Any) -> None:
        """
        Store agent results in the ProcessingContext using the standardized method.
        
        Args:
            context: The ProcessingContext
            data_type: Type of data (e.g., "action_items", "issues", "overall_assessment")
            data: The data to store
        """
        context.store_agent_data(self.role, data_type, data)
        self._log_debug(f"Stored {data_type} data in context", context)
        
    def _get_data(self, context: ProcessingContext, agent_role: str, data_type: str, default=None) -> Any:
        """
        Retrieve data from the ProcessingContext using the standardized method.
        
        Args:
            context: The ProcessingContext
            agent_role: Role of the agent that stored the data (e.g., "extractor", "aggregator")
            data_type: Type of data to retrieve
            default: Default value if not found
            
        Returns:
            The retrieved data or default value
        """
        result = context.get_data_for_agent(agent_role, data_type)
        if result is None or (isinstance(result, list) and len(result) == 0):
            self._log_debug(f"No {data_type} data found from {agent_role}", context)
            return default
        
        self._log_debug(f"Retrieved {data_type} data from {agent_role}", context)
        return result
    
    def _add_evidence(self, context: ProcessingContext, item_id: str, 
                     evidence_text: str, chunk_index: Optional[int] = None) -> str:
        """
        Add evidence to the context for an item.
        
        Args:
            context: The ProcessingContext
            item_id: The ID of the item being evidenced
            evidence_text: The text supporting the item
            chunk_index: Optional chunk index for source tracking
            
        Returns:
            The evidence reference ID
        """
        source_info = {"chunk_index": chunk_index} if chunk_index is not None else None
        return context.add_evidence(item_id, evidence_text, source_info)

    # --- Logging Helpers ---
    
    def _log_info(self, message: str, context: Optional[ProcessingContext] = None) -> None:
        """Log an info message, optionally including context run_id."""
        prefix = f"[{context.run_id}] " if context else ""
        self.logger.info(f"{prefix}{message}")

    def _log_debug(self, message: str, context: Optional[ProcessingContext] = None) -> None:
        """Log a debug message, optionally including context run_id."""
        prefix = f"[{context.run_id}] " if context else ""
        self.logger.debug(f"{prefix}{message}")

    def _log_warning(self, message: str, context: Optional[ProcessingContext] = None) -> None:
        """Log a warning message, optionally including context run_id."""
        prefix = f"[{context.run_id}] " if context else ""
        self.logger.warning(f"{prefix}{message}")
        if context:
            context.add_warning(message, stage=context.metadata.get("current_stage"))

    def _log_error(self, message: str, context: Optional[ProcessingContext] = None, exc_info=False) -> None:
        """Log an error message, optionally including context run_id and exception info."""
        prefix = f"[{context.run_id}] " if context else ""
        self.logger.error(f"{prefix}{message}", exc_info=exc_info)

    # --- LLM Interaction Wrappers ---

    async def _generate(self, 
                       prompt: str, 
                       context: ProcessingContext, 
                       system_prompt: Optional[str]=None, 
                       **kwargs) -> str:
        """
        Wrapper for LLM completion call with logging, error handling, and token tracking.
        
        Args:
            prompt: The prompt to send to the LLM
            context: The ProcessingContext for tracking tokens and logging
            system_prompt: Optional system message
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            The generated text string
        """
        self._log_debug(f"Generating LLM completion. Prompt length: {len(prompt)}", context)
        context.log_agent_action(self.name, "llm_call_start", {"prompt_length": len(prompt), **kwargs})
        
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
            else:
                self._log_warning("Token usage information missing from LLM response.", context)
                total_tokens = 0  # Assume 0 if missing

            context.log_agent_action(self.name, "llm_call_success", 
                                    {"response_length": len(result_str), "usage": usage_dict})
            self._log_debug(f"LLM call successful. Response length: {len(result_str)}, Tokens: {total_tokens}", context)

            return result_str  # Return only the string result

        except Exception as e:
            self._log_error(f"LLM completion failed: {e}", context, exc_info=True)
            context.log_agent_action(self.name, "llm_call_error", {"error": str(e)})
            raise RuntimeError(f"LLM call failed for agent {self.name}") from e


    async def _generate_structured(self, 
                                prompt: str, 
                                output_schema: Dict, 
                                context: ProcessingContext, 
                                system_prompt: Optional[str]=None, 
                                **kwargs) -> Dict[str, Any]:
        """
        Wrapper for structured LLM call with logging, error handling, and token tracking.
        
        Args:
            prompt: The prompt to send to the LLM
            output_schema: JSON schema defining the expected output structure
            context: The ProcessingContext for tracking tokens and logging
            system_prompt: Optional system message
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            The generated structured output as a dictionary
        """
        self._log_debug(f"Generating structured LLM output. Schema keys: {list(output_schema.get('properties', {}).keys())}", context)
        context.log_agent_action(self.name, "structured_llm_call_start", {"prompt_length": len(prompt), **kwargs})
        
        try:
            # Verify schema structure has required elements
            if "type" not in output_schema:
                output_schema["type"] = "object"
                self._log_debug("Added missing 'type': 'object' to output schema", context)
                
            if output_schema.get("type") == "object" and "properties" not in output_schema:
                output_schema["properties"] = {}
                self._log_debug("Added missing 'properties' to output schema", context)
            
            # Log important details for debugging
            self._log_debug(f"Schema structure: {json.dumps(output_schema, indent=2)}", context)
            
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
            else:
                self._log_warning("Token usage information missing from structured LLM response.", context)
                total_tokens = 0

            context.log_agent_action(self.name, "structured_llm_call_success", {"usage": usage_dict})
            self._log_debug(f"Structured LLM call successful. Tokens: {total_tokens}", context)

            # Basic validation: Ensure we got a dictionary
            if not isinstance(result_dict, dict):
                raise TypeError(f"Structured LLM call did not return a dictionary as expected. Type: {type(result_dict)}")

            return result_dict  # Return only the dictionary result

        except Exception as e:
            # Catch potential TypeErrors from validation above too
            self._log_error(f"Structured LLM generation failed: {e}", context, exc_info=True)
            context.log_agent_action(self.name, "structured_llm_call_error", {"error": str(e)})
            
            # Return an empty dictionary as a fallback
            self._log_warning("Returning empty dict as fallback from failed structured LLM call", context)
            return {}