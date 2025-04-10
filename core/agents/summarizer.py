import logging
from typing import Dict, Any, List, Optional

from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__)

class SummarizerAgent(BaseAgent):
    """
    A simple agent for summarizing document chunks.
    Extracts key points from each chunk and sends them to the aggregator.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the SummarizerAgent."""
        super().__init__(llm, options or {})
        # Keep role as 'extractor' for orchestrator compatibility
        self.role = "extractor"
        self.logger = logging.getLogger(f"core.agents.{self.__class__.__name__}")
        
        # Simple configuration
        self.temperature = self.options.get("temperature", 0.3)
        self.max_tokens = self.options.get("max_tokens", 1500)
        self.logger.info(f"SummarizerAgent initialized")

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract key points from document chunks.
        
        Args:
            context: The shared ProcessingContext object
            
        Returns:
            Dictionary containing key points and other extracted information
        """
        self._log_info(f"Starting summarization for document: {context.document_info.get('filename', 'Unknown')}", context)
        
        try:
            # Get document chunks
            chunks = context.chunks
            if not chunks:
                raise ValueError("No document chunks available for summarization")
                
            # Get format type
            format_type = context.options.get("user_options", {}).get("format", "executive")
            
            # Process chunks
            all_key_points = []
            total_chunks = len(chunks)
            
            # Process each chunk
            for chunk_index, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                if not chunk_text.strip():
                    continue
                
                # Update progress
                progress = (chunk_index + 1) / total_chunks
                context.update_stage_progress(
                    progress, 
                    f"Extracting key points from chunk {chunk_index + 1}/{total_chunks}"
                )
                
                # Process this chunk
                chunk_points = await self._extract_key_points(
                    context, chunk_text, chunk_index, format_type
                )
                
                # Add evidence and store key points
                for point in chunk_points:
                    # Add an ID for tracking
                    point_id = f"kp-{len(all_key_points) + 1}"
                    point["id"] = point_id
                    
                    # Add evidence linking to chunk
                    self._add_evidence(
                        context=context,
                        item_id=point_id,
                        evidence_text=chunk_text,
                        chunk_index=chunk_index
                    )
                    
                    # Add to our collection
                    all_key_points.append(point)
            
            # Store extracted key points for the aggregator
            self._store_data(context, "key_points", all_key_points)
            
            # Return extraction results
            return {
                "key_points": all_key_points,
                "format_type": format_type,
                "chunks_processed": total_chunks
            }
            
        except Exception as e:
            error_msg = f"Error during key point extraction: {str(e)}"
            self._log_error(error_msg, context, exc_info=True)
            
            # Return error
            return {"error": error_msg}

    async def _extract_key_points(self, 
                               context: ProcessingContext,
                               chunk_text: str,
                               chunk_index: int,
                               format_type: str) -> List[Dict[str, Any]]:
        """
        Extract key points from a single chunk.
        
        Args:
            context: The ProcessingContext
            chunk_text: The text content of the chunk
            chunk_index: The index of the chunk
            format_type: The format type (executive, comprehensive, etc.)
            
        Returns:
            List of key point dictionaries
        """
        # Skip empty chunks
        if not chunk_text or len(chunk_text.strip()) < 50:
            return []
        
        # Create a simple prompt
        prompt = f"""
Extract 5-8 key points from the following text. Focus on the most important information.

FORMAT TYPE: {format_type}

TEXT:
{chunk_text}

For each key point:
1. Identify the main point
2. Assign it to a relevant topic
3. Rate its importance (high, medium, or low)

RESPOND IN JSON FORMAT:
{{
  "key_points": [
    {{ "text": "First key point", "topic": "Topic1", "importance": "high" }},
    {{ "text": "Second key point", "topic": "Topic2", "importance": "medium" }},
    ...
  ]
}}
"""

        # Define schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "topic": {"type": "string"},
                            "importance": {"type": "string", "enum": ["high", "medium", "low"]}
                        },
                        "required": ["text"]
                    }
                }
            },
            "required": ["key_points"]
        }
        
        try:
            # Call LLM for structured output
            result, usage = await self.llm.generate_structured_output(
                prompt=prompt,
                output_schema=schema,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Track token usage
            if usage and usage.get("total_tokens"):
                context.track_token_usage(usage["total_tokens"])
            
            # Extract key points from result
            key_points = result.get("key_points", [])
            
            # If no key points were extracted, try a fallback approach
            if not key_points:
                key_points = await self._fallback_extraction(context, chunk_text, chunk_index)
                
            return key_points
            
        except Exception as e:
            self._log_error(f"Error extracting key points from chunk {chunk_index}: {str(e)}", context)
            
            # Try fallback approach
            return await self._fallback_extraction(context, chunk_text, chunk_index)

    async def _fallback_extraction(self, 
                                context: ProcessingContext,
                                chunk_text: str,
                                chunk_index: int) -> List[Dict[str, Any]]:
        """
        Fallback method when structured extraction fails.
        
        Args:
            context: The ProcessingContext
            chunk_text: The text content of the chunk
            chunk_index: The index of the chunk
            
        Returns:
            List of key point dictionaries
        """
        # Create a simpler prompt that just asks for bullet points
        prompt = f"""
Extract important points from this text as a simple bullet list.
Start each point with a dash (-) on a new line.

TEXT:
{chunk_text}
"""

        try:
            # Simple text completion
            bullet_text, _ = await self.llm.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse bullet points
            key_points = []
            lines = bullet_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('- ') or line.startswith('* ')):
                    point_text = line[2:].strip()
                    if point_text:
                        key_points.append({
                            "text": point_text,
                            "topic": "General",
                            "importance": "medium"
                        })
            
            # If we still have no points, create one generic point
            if not key_points and chunk_text.strip():
                # Take first 100 chars as a fallback point
                preview = chunk_text.strip()[:100] + "..."
                key_points.append({
                    "text": f"Content from document section: {preview}",
                    "topic": "Content",
                    "importance": "low"
                })
                
            return key_points
            
        except Exception as e:
            self._log_error(f"Fallback extraction also failed: {str(e)}", context)
            
            # Create a minimal fallback point
            return [{
                "text": f"Failed to extract content from chunk {chunk_index+1}",
                "topic": "Error",
                "importance": "low"
            }]