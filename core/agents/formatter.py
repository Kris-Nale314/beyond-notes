# core/agents/formatter.py
from typing import Dict, Any, List, Optional
import json
import asyncio

from core.agents.base import Agent
from core.models.context import ProcessingContext

class FormatterAgent(Agent):
    """
    The Formatter Agent transforms the aggregated, evaluated information 
    into structured outputs.
    
    Responsibilities:
    - Create well-structured final reports
    - Format information according to the assessment type
    - Apply consistent styling and organization
    - Generate visual indicators for severity/priority
    """
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Format evaluation results into the final output.
        
        Args:
            context: The shared processing context
            
        Returns:
            Formatted results
        """
        self.log_info(context, "Starting result formatting")
        
        try:
            # Get evaluation results
            evaluation_results = context.results.get("evaluation", {})
            issues = evaluation_results.get("issues", [])
            overall_assessment = evaluation_results.get("overall_assessment", {})
            
            # Get assessment config
            assessment_config = context.assessment_config
            report_format = assessment_config.get("report_format", {})
            sections = report_format.get("sections", [])
            issue_presentation = report_format.get("issue_presentation", {})
            
            # Get custom instructions from assessment config
            agent_instructions = self.get_agent_instructions(context)
            
            # Create the report
            report = await self._create_report(
                context, issues, overall_assessment, sections, issue_presentation, agent_instructions
            )
            
            # Generate report statistics
            stats = {
                "total_issues": len(issues),
                "by_severity": self._count_by_field(issues, "severity"),
                "by_category": self._count_by_field(issues, "category"),
                "by_priority": self._count_by_field(issues, "priority")
            }
            
            # Create formatted output
            formatted_output = {
                "report": report,
                "statistics": stats,
                "metadata": {
                    "document_info": context.document_info,
                    "assessment_type": context.assessment_type,
                    "processing_time": context.get_processing_time()
                }
            }
            
            self.log_info(context, "Formatting complete")
            
            return formatted_output
            
        except Exception as e:
            error_message = f"Formatting failed: {str(e)}"
            self.log_error(context, error_message)
            raise
    
    async def _create_report(self,
                           context: ProcessingContext,
                           issues: List[Dict[str, Any]],
                           overall_assessment: Dict[str, Any],
                           sections: List[str],
                           issue_presentation: Dict[str, str],
                           agent_instructions: str) -> Dict[str, Any]:
        """
        Create a formatted report based on the evaluation results.
        
        Args:
            context: The processing context
            issues: List of evaluated issues
            overall_assessment: Overall assessment
            sections: Report sections from the assessment config
            issue_presentation: Issue presentation format
            agent_instructions: Instructions from the assessment config
            
        Returns:
            Formatted report
        """
        # Define schema for the report
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
                        },
                        "required": ["title", "content"]
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
        
        # Get document info and additional metadata
        document_info = context.document_info
        document_name = document_info.get("filename", "Document")
        
        # Create a prompt for report generation
        prompt = f"""
        You are an expert report formatter for an AI document analysis system.
        
        Please create a well-structured report based on the issues and overall assessment from the document analysis.
        
        Document Information:
        ```
        {json.dumps(document_info, indent=2)}
        ```
        
        Overall Assessment:
        ```
        {json.dumps(overall_assessment, indent=2)}
        ```
        
        Issues:
        ```
        {json.dumps(issues, indent=2)}
        ```
        
        Report Sections:
        {json.dumps(sections, indent=2)}
        
        Issue Presentation Format:
        {json.dumps(issue_presentation, indent=2)}
        
        {agent_instructions}
        
        Format Requirements:
        1. Create a professional, well-organized report
        2. Group issues by severity (Critical, High, Medium, Low)
        3. Include all the required sections
        4. Provide clear, actionable information
        5. Use markdown formatting for better readability
        
        Format your response as a structured JSON object.
        """
        
        try:
            # Generate the report
            report = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=4000,
                temperature=0.4
            )
            
            return report
            
        except Exception as e:
            # If report generation fails, create a basic report
            self.log_error(context, f"Error generating report: {str(e)}")
            
            # Create a basic report structure
            basic_report = {
                "title": f"Issues Assessment - {document_name}",
                "executive_summary": overall_assessment.get("executive_summary", "Executive summary not available."),
                "sections": [
                    {
                        "title": "Critical Issues",
                        "content": "Issues with critical severity.",
                        "issues": [i for i in issues if i.get("severity") == "critical"]
                    },
                    {
                        "title": "High-Priority Issues",
                        "content": "Issues with high severity.",
                        "issues": [i for i in issues if i.get("severity") == "high"]
                    },
                    {
                        "title": "Medium-Priority Issues",
                        "content": "Issues with medium severity.",
                        "issues": [i for i in issues if i.get("severity") == "medium"]
                    },
                    {
                        "title": "Low-Priority Issues",
                        "content": "Issues with low severity.",
                        "issues": [i for i in issues if i.get("severity") == "low"]
                    }
                ],
                "conclusion": "Report generation encountered an error. Please refer to the individual issues for details.",
                "error": str(e)
            }
            
            return basic_report
    
    def _count_by_field(self, items: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        """
        Count items by a specific field value.
        
        Args:
            items: List of items to count
            field: Field to count by
            
        Returns:
            Dictionary with counts by field value
        """
        counts = {}
        
        for item in items:
            value = item.get(field, "unknown")
            if value:
                counts[value] = counts.get(value, 0) + 1
        
        return counts
    
    def _group_by_field(self, items: List[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group items by a specific field value.
        
        Args:
            items: List of items to group
            field: Field to group by
            
        Returns:
            Dictionary with groups by field value
        """
        groups = {}
        
        for item in items:
            value = item.get(field, "unknown")
            if value not in groups:
                groups[value] = []
            groups[value].append(item)
        
        return groups