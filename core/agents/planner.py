# core/agents/planner.py
from typing import Dict, Any, List, Optional
import json
import asyncio

from core.agents.base import Agent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class PlannerAgent(Agent):
    """
    The Planner Agent studies the document and creates tailored
    instructions for other agents in the workflow.
    
    Responsibilities:
    - Analyze document type and content
    - Determine optimal processing strategy
    - Create custom instructions for downstream agents
    - Identify key areas to focus on
    """
    
    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the planner agent.
        
        Args:
            llm: Language model for agent operations
            options: Configuration options for the agent
        """
        super().__init__(llm, options)
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Analyze document and create a processing plan.
        
        Args:
            context: The shared processing context
            
        Returns:
            Processing plan with agent instructions
        """
        self.log_info(context, "Starting document planning")
        
        try:
            # Get document information
            document_text = context.document_text[:10000]  # Use first 10K chars for planning
            document_info = context.document_info
            assessment_type = context.assessment_type
            assessment_config = context.assessment_config
            
            # Get assessment details
            assessment_desc = assessment_config.get("description", "")
            issue_definition = assessment_config.get("issue_definition", {})
            
            # Create a prompt for document analysis
            prompt = f"""
            You are an expert document planner for an AI analysis system focusing on identifying {assessment_type}.
            
            Assessment Type: {assessment_type}
            Assessment Description: {assessment_desc}
            
            I need you to analyze the beginning of this document and create a processing plan.
            
            Document Information:
            - Filename: {document_info.get('filename', 'Unknown')}
            - Word count: {document_info.get('word_count', 0)}
            - Character count: {document_info.get('character_count', 0)}
            
            Here's the beginning of the document:
            
            {document_text[:3000]}
            
            Please analyze this document and provide:
            1. Document type (meeting transcript, report, article, etc.)
            2. Key focus areas to analyze
            3. Specific instructions for extraction (what to look for)
            4. Suggested evaluation framework
            5. Any special considerations for this document
            
            Format your response as a JSON object with the following keys:
            - document_type: string
            - focus_areas: list of strings
            - extraction_instructions: object with instructions for the extractor
            - evaluation_framework: object with evaluation criteria
            - special_considerations: any special handling needed
            """
            
            # Define the expected schema for the response
            json_schema = {
                "type": "object",
                "properties": {
                    "document_type": {"type": "string"},
                    "focus_areas": {"type": "array", "items": {"type": "string"}},
                    "extraction_instructions": {
                        "type": "object",
                        "properties": {
                            "key_entities": {"type": "array", "items": {"type": "string"}},
                            "important_sections": {"type": "array", "items": {"type": "string"}},
                            "data_points": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "evaluation_framework": {
                        "type": "object",
                        "properties": {
                            "criteria": {"type": "array", "items": {"type": "string"}},
                            "rating_scales": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "special_considerations": {"type": "string"}
                },
                "required": ["document_type", "focus_areas", "extraction_instructions"]
            }
            
            # Generate the planning output
            planning_result = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=1500,
                temperature=0.4
            )
            
            # Log the plan
            self.log_info(context, f"Planning complete. Document type: {planning_result.get('document_type', 'Unknown')}")
            
            return planning_result
            
        except Exception as e:
            error_message = f"Planning failed: {str(e)}"
            self.log_error(context, error_message)
            raise
    
    # These methods are now inherited from the base Agent class