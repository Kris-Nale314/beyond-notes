# core/agents/reviewer.py
from typing import Dict, Any, List, Optional
import json
import asyncio

from core.agents.base import Agent
from core.models.context import ProcessingContext

class ReviewerAgent(Agent):
    """
    The Reviewer Agent performs quality control to ensure consistency and completeness.
    
    Responsibilities:
    - Check for missing information
    - Ensure consistency across the report
    - Verify alignment with user needs
    - Suggest improvements
    """
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Review the generated report for quality and consistency.
        
        Args:
            context: The shared processing context
            
        Returns:
            Review results with suggestions
        """
        self.log_info(context, "Starting report review")
        
        try:
            # Get formatting results
            formatting_results = context.results.get("formatting", {})
            report = formatting_results.get("report", {})
            statistics = formatting_results.get("statistics", {})
            
            # Get assessment config
            assessment_config = context.assessment_config
            report_format = assessment_config.get("report_format", {})
            
            # Get custom instructions from assessment config
            agent_instructions = self.get_agent_instructions(context)
            
            # Review the report
            review_results = await self._review_report(
                context, report, statistics, assessment_config, agent_instructions
            )
            
            # Apply suggestions if possible
            if review_results.get("apply_suggestions", False):
                updated_report = await self._apply_suggestions(
                    context, report, review_results.get("suggestions", [])
                )
                review_results["updated_report"] = updated_report
            
            self.log_info(context, "Review complete")
            
            return review_results
            
        except Exception as e:
            error_message = f"Review failed: {str(e)}"
            self.log_error(context, error_message)
            raise
    
    async def _review_report(self,
                           context: ProcessingContext,
                           report: Dict[str, Any],
                           statistics: Dict[str, Any],
                           assessment_config: Dict[str, Any],
                           agent_instructions: str) -> Dict[str, Any]:
        """
        Review the report for quality, completeness, and consistency.
        
        Args:
            context: The processing context
            report: The generated report
            statistics: Report statistics
            assessment_config: Assessment configuration
            agent_instructions: Instructions from the assessment config
            
        Returns:
            Review results with suggestions
        """
        # Define schema for review results
        json_schema = {
            "type": "object",
            "properties": {
                "quality_score": {"type": "number"},
                "completeness_score": {"type": "number"},
                "consistency_score": {"type": "number"},
                "overall_score": {"type": "number"},
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "improvement_areas": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {"type": "string"},
                            "issue": {"type": "string"},
                            "suggestion": {"type": "string"},
                            "severity": {"type": "string"}
                        }
                    }
                },
                "missing_information": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "review_summary": {"type": "string"},
                "apply_suggestions": {"type": "boolean"}
            },
            "required": ["quality_score", "completeness_score", "consistency_score", "overall_score", "review_summary"]
        }
        
        # Create a prompt for reviewing the report
        prompt = f"""
        You are an expert report reviewer for an AI document analysis system.
        
        Please review the following report for quality, completeness, consistency, and alignment with the assessment requirements.
        
        Report:
        ```
        {json.dumps(report, indent=2)}
        ```
        
        Statistics:
        ```
        {json.dumps(statistics, indent=2)}
        ```
        
        Assessment Configuration:
        ```
        {json.dumps(assessment_config, indent=2)}
        ```
        
        {agent_instructions}
        
        Please conduct a thorough review and provide:
        1. Quality score (0-10): Evaluate the overall quality of the report
        2. Completeness score (0-10): Assess if all required information is included
        3. Consistency score (0-10): Check for consistency across sections
        4. Overall score (0-10): Overall assessment of the report
        5. Strengths: List 2-3 strengths of the report
        6. Improvement areas: List 2-3 areas that could be improved
        7. Specific suggestions: For each issue, provide a section, description, suggestion, and severity
        8. Missing information: Note any important information that's missing
        9. Review summary: A concise summary of your review
        10. Apply suggestions flag: Should the suggestions be automatically applied (true/false)
        
        Format your response as a structured JSON object.
        """
        
        try:
            # Generate review results
            review = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=3000,
                temperature=0.4
            )
            
            return review
            
        except Exception as e:
            # If review fails, create a basic review
            self.log_error(context, f"Error reviewing report: {str(e)}")
            
            return {
                "quality_score": 5,
                "completeness_score": 5,
                "consistency_score": 5,
                "overall_score": 5,
                "strengths": ["Report was generated successfully"],
                "improvement_areas": ["Could not perform detailed review due to an error"],
                "suggestions": [],
                "missing_information": ["Could not identify missing information due to an error"],
                "review_summary": "Review process encountered an error. Basic report is available but detailed review could not be completed.",
                "apply_suggestions": False,
                "error": str(e)
            }
    
    async def _apply_suggestions(self,
                               context: ProcessingContext,
                               report: Dict[str, Any],
                               suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply suggestions to improve the report.
        
        Args:
            context: The processing context
            report: The original report
            suggestions: List of suggestions
            
        Returns:
            Updated report
        """
        # If no suggestions or report is empty, return original
        if not suggestions or not report:
            return report
        
        # Define schema for updated report
        json_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "executive_summary": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "issues": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "severity": {"type": "string"},
                                        "category": {"type": "string"},
                                        "impact": {"type": "string"},
                                        "priority": {"type": "string"},
                                        "recommendations": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "conclusion": {"type": "string"},
                "appendix": {
                    "type": "object",
                    "properties": {
                        "methodology": {"type": "string"},
                        "definitions": {"type": "string"}
                    }
                }
            },
            "required": ["title", "executive_summary", "sections"]
        }
        
        # Create a prompt for updating the report
        prompt = f"""
        You are an expert report editor for an AI document analysis system.
        
        Please update the following report by applying these suggested improvements:
        
        Original Report:
        ```
        {json.dumps(report, indent=2)}
        ```
        
        Suggestions to Apply:
        ```
        {json.dumps(suggestions, indent=2)}
        ```
        
        Please:
        1. Apply all the suggested improvements
        2. Maintain the original structure of the report
        3. Ensure consistent formatting and style
        4. Return the complete updated report
        
        Format your response as a structured JSON object matching the original report structure.
        """
        
        try:
            # Generate updated report
            updated_report = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=4000,
                temperature=0.3
            )
            
            self.log_info(context, f"Applied {len(suggestions)} suggestions to improve the report")
            
            return updated_report
            
        except Exception as e:
            # If update fails, return original report
            self.log_error(context, f"Error applying suggestions: {str(e)}")
            return report