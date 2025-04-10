import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__)

class FormatterAgent(BaseAgent):
    """
    Formats the aggregated information into the final output structure.
    
    For summarization tasks, focuses on combining extracted key points and topics
    from the SummarizerAgent into a cohesive final summary.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the FormatterAgent."""
        super().__init__(llm, options or {})
        self.role = "formatter"
        self.logger = logging.getLogger(f"core.agents.{self.__class__.__name__}")
        self.logger.info(f"FormatterAgent initialized")

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Generate the final formatted output based on assessment type.
        
        Args:
            context: The shared ProcessingContext object.
            
        Returns:
            The final formatted output dictionary.
        """
        self._log_info(f"Starting formatting phase for assessment: '{context.display_name}'", context)
        assessment_type = context.assessment_type
        
        try:
            # Select the appropriate formatter method based on assessment type
            if assessment_type == "distill":
                formatted_result = await self._format_summary(context)
            elif assessment_type == "assess":
                formatted_result = await self._format_issues(context)
            elif assessment_type == "extract":
                formatted_result = await self._format_action_items(context)
            elif assessment_type == "analyze":
                formatted_result = await self._format_analysis(context)
            else:
                formatted_result = await self._format_generic(context)
            
            # Add metadata to result
            formatted_result = self._add_metadata(formatted_result, context)
            
            # Store the formatted result
            self._store_data(context, None, formatted_result)
            self._log_info("Formatting complete. Final output structure generated.", context)
            
            return formatted_result
            
        except Exception as e:
            self._log_error(f"Formatting failed: {str(e)}", context, exc_info=True)
            error_output = {
                "error": f"Formatting Agent Failed: {str(e)}",
                "message": "Could not generate final output"
            }
            self._store_data(context, None, error_output)
            return error_output

    async def _format_summary(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Format summary output from SummarizerAgent results.
        
        Args:
            context: The ProcessingContext containing SummarizerAgent results
            
        Returns:
            Formatted summary dictionary
        """
        # Get data from SummarizerAgent (via aggregator)
        key_points = self._get_data(context, "aggregator", "key_points", [])
        topics = self._get_data(context, "aggregator", "topics", [])
        quotes = self._get_data(context, "aggregator", "quotes", [])
        
        # Get document info
        document_info = context.document_info
        word_count = document_info.get("word_count", 0)
        
        # Get summary format type and options
        format_type = context.options.get("user_options", {}).get("format", "executive")
        include_quotes = context.options.get("user_options", {}).get("include_quotes", True)
        
        # Build the prompt for the final summary
        prompt = self._build_summary_prompt(
            key_points=key_points,
            topics=topics,
            quotes=quotes,
            format_type=format_type,
            word_count=word_count,
            context=context
        )
        
        # Define the schema for the summary
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The main summary text"
                },
                "executive_summary": {
                    "type": "string",
                    "description": "A concise executive summary"
                },
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "details": {"type": "string"}
                        },
                        "required": ["topic"]
                    },
                    "description": "Topics covered in the document"
                }
            },
            "required": ["summary"]
        }
        
        # Generate the final summary
        summary_output = await self._generate_structured(
            prompt=prompt,
            output_schema=summary_schema,
            context=context,
            temperature=0.3
        )
        
        # Calculate word counts
        summary_text = summary_output.get("summary", "")
        summary_word_count = len(summary_text.split())
        
        # Prepare statistics
        statistics = {
            "original_word_count": word_count,
            "summary_word_count": summary_word_count,
            "compression_ratio": round((summary_word_count / max(1, word_count)) * 100, 1),
            "key_points_count": len(key_points),
            "topics_count": len(topics)
        }
        
        # Assemble the final result
        result = {
            "summary": summary_text,
            "executive_summary": summary_output.get("executive_summary", ""),
            "key_points": key_points,
            "topics": summary_output.get("topics", topics),
            "statistics": statistics
        }
        
        # Include quotes if requested and available
        if include_quotes and quotes:
            result["quotes"] = quotes
            
        return result
        
    def _build_summary_prompt(self, key_points, topics, quotes, format_type, word_count, context):
        """Build the prompt for generating the final summary."""
        # Base prompt with document info
        prompt = f"""
You are an expert AI summarizer. Your task is to create a final {format_type} summary based on the key points and topics extracted from a document.

# DOCUMENT INFORMATION
- Word Count: {word_count} words
- Key Points: {len(key_points)} extracted
- Topics: {len(topics)} identified

# FORMAT TYPE: {format_type.upper()}
"""

        # Add format-specific instructions
        if format_type == "executive":
            prompt += """
Create a concise executive summary that:
- Highlights the most important information (250-500 words)
- Focuses on decisions, outcomes, and business implications
- Uses direct, clear language appropriate for executives
- Provides both a main "summary" and a shorter "executive_summary" (2-3 sentences)
"""
        elif format_type == "comprehensive":
            prompt += """
Create a detailed comprehensive summary that:
- Covers all significant aspects of the document (800-1500 words)
- Preserves important details and nuances
- Organizes information by topic or theme
- Provides a complete understanding of the content
- Includes a main "summary" and detailed "topics" section
"""
        elif format_type == "bullet_points":
            prompt += """
Create a bullet-point style summary that:
- Transforms content into easily scannable points
- Organizes information by topic
- Uses concise, action-oriented language
- Includes a brief narrative "summary" and organized topics
"""
        elif format_type == "narrative":
            prompt += """
Create a narrative-style summary that:
- Presents a flowing, readable story of the document
- Maintains the voice and tone of the original
- Preserves the logical flow of information
- Provides a cohesive "summary" that reads like a short article
"""

        # Add key points (limit to 20 for prompt size)
        prompt += "\n# KEY POINTS EXTRACTED:\n"
        for i, point in enumerate(key_points[:20]):
            if isinstance(point, dict):
                point_text = point.get("text", "")
                point_topic = point.get("topic", "")
                point_importance = point.get("importance", "medium")
                
                prompt += f"{i+1}. [{point_importance.upper()}] {point_text}"
                if point_topic:
                    prompt += f" (Topic: {point_topic})"
                prompt += "\n"
            elif isinstance(point, str):
                prompt += f"{i+1}. {point}\n"
                
        if len(key_points) > 20:
            prompt += f"... and {len(key_points) - 20} more points\n"
            
        # Add topics section
        prompt += "\n# TOPICS IDENTIFIED:\n"
        for i, topic in enumerate(topics[:10]):
            if isinstance(topic, dict):
                topic_name = topic.get("topic", "")
                if not topic_name:
                    continue
                    
                prompt += f"{i+1}. {topic_name}"
                
                # Add point count if available
                points_count = topic.get("points_count", 0)
                if points_count:
                    prompt += f" ({points_count} points)"
                    
                prompt += "\n"
                
                # Add a few example points for each topic
                topic_points = topic.get("key_points", [])
                for j, point in enumerate(topic_points[:3]):
                    prompt += f"   - {point}\n"
                if len(topic_points) > 3:
                    prompt += f"   - ... and {len(topic_points) - 3} more points\n"
        
        # Add quotes if available
        if quotes:
            prompt += "\n# NOTABLE QUOTES:\n"
            for i, quote in enumerate(quotes[:5]):
                if isinstance(quote, dict):
                    quote_text = quote.get("text", "")
                    speaker = quote.get("speaker", "")
                    
                    prompt += f"{i+1}. \"{quote_text}\""
                    if speaker:
                        prompt += f" - {speaker}"
                    prompt += "\n"
            
            if len(quotes) > 5:
                prompt += f"... and {len(quotes) - 5} more quotes\n"
        
        # Final instructions
        prompt += """
# OUTPUT REQUIREMENTS
1. Create a "summary" that synthesizes the key information according to the format requirements
2. For executive and comprehensive formats, include an "executive_summary" field with 2-3 sentence overview
3. For the topics section, preserve the main topics but enhance with coherent details

Focus on accuracy and clarity. Don't add information not supported by the key points.
"""
        
        return prompt

    async def _format_issues(self, context: ProcessingContext) -> Dict[str, Any]:
        """Format issues output for assess assessments."""
        # Basic implementation for issues
        issues = self._get_data(context, "evaluator", "issues", [])
        if not issues:
            issues = self._get_data(context, "aggregator", "issues", [])
            
        overall_assessment = self._get_data(context, "evaluator", "overall_assessment", {})
        
        # Create a simple result structure
        result = {
            "issues": issues,
            "overall_assessment": overall_assessment,
            "statistics": {
                "total_issues": len(issues),
                "document_word_count": context.document_info.get("word_count", 0)
            }
        }
        
        return result

    async def _format_action_items(self, context: ProcessingContext) -> Dict[str, Any]:
        """Format action items output for extract assessments."""
        # Basic implementation for action items
        action_items = self._get_data(context, "evaluator", "action_items", [])
        if not action_items:
            action_items = self._get_data(context, "aggregator", "action_items", [])
            
        overall_assessment = self._get_data(context, "evaluator", "overall_assessment", {})
        
        # Create a simple result structure
        result = {
            "action_items": action_items,
            "overall_assessment": overall_assessment,
            "statistics": {
                "total_action_items": len(action_items),
                "document_word_count": context.document_info.get("word_count", 0)
            }
        }
        
        return result

    async def _format_analysis(self, context: ProcessingContext) -> Dict[str, Any]:
        """Format analysis output for analyze assessments."""
        # Basic implementation for framework analysis
        framework_data = self._get_data(context, "evaluator", "dimension_ratings", [])
        overall_assessment = self._get_data(context, "evaluator", "overall_assessment", {})
        
        # Create a simple result structure
        result = {
            "dimension_ratings": framework_data,
            "overall_assessment": overall_assessment,
            "statistics": {
                "dimensions_count": len(framework_data),
                "document_word_count": context.document_info.get("word_count", 0)
            }
        }
        
        return result

    async def _format_generic(self, context: ProcessingContext) -> Dict[str, Any]:
        """Generic formatter for other assessment types."""
        assessment_type = context.assessment_type
        data_type = self._get_data_type_for_assessment(assessment_type)
        
        # Get items
        items = self._get_data(context, "evaluator", data_type, [])
        if not items:
            items = self._get_data(context, "aggregator", data_type, [])
        
        overall_assessment = self._get_data(context, "evaluator", "overall_assessment", {})
        
        # Simple result structure
        result = {
            data_type: items,
            "overall_assessment": overall_assessment,
            "statistics": {
                "item_count": len(items),
                "document_word_count": context.document_info.get("word_count", 0)
            }
        }
        
        return result

    def _add_metadata(self, result: Dict[str, Any], context: ProcessingContext) -> Dict[str, Any]:
        """Add metadata to the result."""
        metadata = {
            "assessment_id": context.assessment_id,
            "assessment_type": context.assessment_type,
            "document_name": context.document_info.get("filename", "Unknown"),
            "word_count": context.document_info.get("word_count", 0),
            "processing_time": time.time() - context.start_time if hasattr(context, "start_time") else 0,
            "date_analyzed": datetime.now(timezone.utc).isoformat(),
            "user_options": context.options.get("user_options", {})
        }
        
        # Include the metadata in the result
        if "metadata" not in result:
            result["metadata"] = metadata
        else:
            # Merge with existing metadata
            result["metadata"].update(metadata)
        
        return result

    def _get_data_type_for_assessment(self, assessment_type: str) -> str:
        """Determine the key/type of data being processed based on assessment type."""
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type, "items")