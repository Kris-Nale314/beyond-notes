# core/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
import json
import time

from core.models.context import ProcessingContext
from core.llm.customllm import CustomLLM

class Agent(ABC):
    """
    Base class for all agents in the beyond-notes system.
    
    Agents are specialized components that perform specific tasks
    within the document processing pipeline.
    """
    
    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """
        Initialize an agent.
        
        Args:
            llm: Language model for agent operations
            options: Configuration options for the agent
        """
        self.llm = llm
        self.options = options or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Process the document using this agent's specific logic.
        
        Args:
            context: The shared processing context
            
        Returns:
            Processing results specific to this agent
        """
        pass
    
    async def generate_completion(self, 
                                prompt: str, 
                                max_tokens: int = 1000,
                                temperature: float = 0.7) -> str:
        """
        Generate text using the agent's LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        return await self.llm.generate_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    async def generate_structured_output(self,
                                       prompt: str,
                                       json_schema: Dict[str, Any],
                                       max_tokens: int = 1000,
                                       temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a structured output in JSON format from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            json_schema: JSON schema defining the expected response structure
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated JSON response
        """
        return await self.llm.generate_structured_output(
            prompt=prompt,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def log_info(self, context: ProcessingContext, message: str) -> None:
        """Add an info message to the context."""
        context.log_agent_action(self.name, "info", {"message": message})
    
    def log_error(self, context: ProcessingContext, message: str) -> None:
        """Add an error message to the context."""
        context.log_agent_action(self.name, "error", {"message": message})
    
    def log_warning(self, context: ProcessingContext, message: str) -> None:
        """Add a warning message to the context."""
        context.log_agent_action(self.name, "warning", {"message": message})
    
    def log_action(self, context: ProcessingContext, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an agent action."""
        context.log_agent_action(self.name, action, details or {})
    
    def get_agent_instructions(self, context: ProcessingContext) -> str:
        """
        Get instructions for this agent from the assessment configuration.
        
        Args:
            context: The processing context
            
        Returns:
            Instructions string or empty string if not found
        """
        assessment_config = context.assessment_config
        if not assessment_config:
            return ""
        
        workflow = assessment_config.get("workflow", {})
        agent_roles = workflow.get("agent_roles", {})
        
        # Extract the role name from the class name
        role = self.name.lower().replace("agent", "")
        
        if role in agent_roles:
            return agent_roles[role].get("instructions", "")
        
        return ""
    
    async def process_in_batches(self, 
                               context: ProcessingContext,
                               items: List[Any],
                               process_func,
                               batch_size: int = 3,
                               delay: float = 0.1) -> List[Any]:
        """
        Process a list of items in batches to avoid rate limits.
        
        Args:
            context: The processing context
            items: List of items to process
            process_func: Async function to process each item
            batch_size: Size of each batch
            delay: Delay between batches in seconds
            
        Returns:
            List of processing results
        """
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(items))
            
            # Create batch tasks
            batch_tasks = [
                process_func(item, i) 
                for i, item in enumerate(items[start_idx:end_idx], start=start_idx)
            ]
            
            # Process batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.log_error(context, f"Error processing item {start_idx + i}: {str(result)}")
                    results.append({"error": str(result), "index": start_idx + i})
                else:
                    results.append(result)
            
            # Update progress
            progress = (batch_idx + 1) / total_batches
            context.update_stage_progress(
                progress, f"Processed batch {batch_idx+1}/{total_batches} ({end_idx}/{len(items)} items)"
            )
            
            # Delay between batches to avoid rate limits
            if batch_idx < total_batches - 1:
                await asyncio.sleep(delay)
        
        return results