# pages/04_Chat.py
import streamlit as st
import os
import json
import logging
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-chat")

# Import components
from core.llm.customllm import CustomLLM
from utils.paths import AppPaths
from core.models.context import ProcessingContext  # Import the context model
from utils.ui.styles import get_base_styles
from utils.ui.components import page_header, section_header

# Ensure directories exist
AppPaths.ensure_dirs()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Chat with Documents",
    page_icon="💬",
    layout="wide",
)

# Apply shared styles
st.markdown(get_base_styles(), unsafe_allow_html=True)

# Add chat-specific styles
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
    }
    .chat-message.assistant {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
    }
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    .chat-content {
        flex-grow: 1;
    }
    .chat-timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        text-align: right;
    }
    .suggestion-button {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_chat_page():
    """Initialize the session state variables for the chat page."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_document" not in st.session_state:
        st.session_state.current_document = None
    
    if "current_assessment" not in st.session_state:
        st.session_state.current_assessment = None
    
    if "current_context" not in st.session_state:
        st.session_state.current_context = None
    
    if "llm" not in st.session_state:
        # Initialize LLM if not already done
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            # Use GPT-4 for better reasoning if available, otherwise fall back to GPT-3.5
            model = "gpt-4" if os.environ.get("USE_GPT4", "false").lower() == "true" else "gpt-3.5-turbo"
            st.session_state.llm = CustomLLM(api_key, model=model)
            logger.info(f"Initialized LLM with model: {model}")
        else:
            st.session_state.llm = None
            logger.warning("No API key found. LLM will not be available.")

def load_recent_assessments():
    """Load a list of recent assessments with improved data extraction."""
    assessment_types = ["distill", "extract", "assess", "analyze"]
    recent_assessments = []
    
    # Look for assessment results in output directories
    for assessment_type in assessment_types:
        output_dir = AppPaths.get_assessment_output_dir(assessment_type)
        
        if output_dir.exists():
            # Look for both JSON result files and context files
            for file_path in output_dir.glob(f"{assessment_type}_*.*"):
                try:
                    # Filter for json and pickle files
                    if file_path.suffix not in ['.json', '.pkl', '.md']:
                        continue
                        
                    # Get file stats
                    stat = file_path.stat()
                    
                    # For JSON files, load basic metadata
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                                metadata = data.get("metadata", {})
                                
                                # Create an ID for this assessment
                                assessment_id = file_path.stem
                                
                                # Check if we've already added this assessment
                                if any(a["id"] == assessment_id for a in recent_assessments):
                                    continue
                                
                                # Create assessment entry
                                assessment = {
                                    "id": assessment_id,
                                    "result_path": file_path,
                                    "context_path": file_path.with_suffix('.pkl'),  # Look for matching context file
                                    "md_path": file_path.with_suffix('.md'),  # Look for matching markdown file
                                    "type": assessment_type,
                                    "display_name": metadata.get("assessment_display_name", assessment_type.title()),
                                    "document_name": metadata.get("document_info", {}).get("filename", "Unknown Document"),
                                    "modified": datetime.fromtimestamp(stat.st_mtime),
                                    "has_context": file_path.with_suffix('.pkl').exists(),
                                    "has_markdown": file_path.with_suffix('.md').exists(),
                                    "metadata": metadata
                                }
                                
                                recent_assessments.append(assessment)
                            except json.JSONDecodeError:
                                # Skip invalid JSON files
                                logger.warning(f"Invalid JSON in file: {file_path}")
                                continue
                except Exception as e:
                    logger.warning(f"Error loading assessment file {file_path}: {str(e)}")
                    continue
    
    # Sort by modification time (most recent first)
    recent_assessments.sort(key=lambda x: x["modified"], reverse=True)
    
    return recent_assessments

def load_assessment_data(assessment):
    """Load assessment data from files with context when available."""
    try:
        # Load result data
        result_data = None
        if assessment["result_path"].exists():
            with open(assessment["result_path"], 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        
        # Try to load context (if available)
        context_data = None
        if assessment["has_context"] and assessment["context_path"].exists():
            try:
                import pickle
                with open(assessment["context_path"], 'rb') as f:
                    context_data = pickle.load(f)
                logger.info(f"Successfully loaded context data from {assessment['context_path']}")
            except Exception as e:
                logger.error(f"Error loading context data: {str(e)}")
        
        # Load markdown report if available
        markdown_report = None
        if assessment["has_markdown"] and assessment["md_path"].exists():
            try:
                with open(assessment["md_path"], 'r', encoding='utf-8') as f:
                    markdown_report = f.read()
                logger.info(f"Successfully loaded markdown report from {assessment['md_path']}")
            except Exception as e:
                logger.error(f"Error loading markdown report: {str(e)}")
        
        return {
            "result": result_data,
            "context": context_data,
            "markdown": markdown_report,
            "metadata": assessment.get("metadata", {})
        }
    except Exception as e:
        logger.error(f"Error loading assessment data: {str(e)}")
        return None

def get_document_content(assessment_data):
    """Extract document content with improved fallbacks."""
    try:
        # Try different paths to get document text
        
        # 1. First check if context has document_text
        if assessment_data.get("context") and hasattr(assessment_data["context"], "document_text"):
            return assessment_data["context"].document_text
        
        # 2. Try to get from result metadata
        metadata = assessment_data.get("result", {}).get("metadata", {})
        document_info = metadata.get("document_info", {})
        
        if "document_text" in document_info:
            return document_info.get("document_text")
        
        # 3. Check for text in original document
        filename = document_info.get("filename")
        if filename:
            # Try to find the file in uploads dir
            upload_path = AppPaths.get_temp_path("uploads") / filename
            if upload_path.exists():
                try:
                    # Try with utf-8 encoding first
                    with open(upload_path, 'r', encoding='utf-8') as f:
                        doc_text = f.read()
                        logger.info(f"Successfully loaded document text from upload: {upload_path}")
                        return doc_text
                except UnicodeDecodeError:
                    # If utf-8 fails, try with latin-1 which should handle any byte sequence
                    with open(upload_path, 'r', encoding='latin-1') as f:
                        doc_text = f.read()
                        logger.info(f"Successfully loaded document text from upload (latin-1): {upload_path}")
                        return doc_text
        
        logger.warning("Document content not available through standard methods")
        return "Document content not available. Please provide context in your questions."
    except Exception as e:
        logger.error(f"Error getting document content: {str(e)}")
        return "Error retrieving document content."

def prepare_context_for_chat(assessment_data):
    """Prepare structured context information for better chat responses."""
    context_dict = {"document_info": {}, "assessment_info": {}, "key_data": {}}
    
    try:
        # Get metadata from result
        result = assessment_data.get("result", {})
        metadata = result.get("metadata", {}) or assessment_data.get("metadata", {})
        
        # Basic document info
        document_info = metadata.get("document_info", {})
        context_dict["document_info"] = {
            "filename": document_info.get("filename", "Unknown"),
            "word_count": document_info.get("word_count", 0),
            "character_count": document_info.get("character_count", 0),
            "line_count": document_info.get("line_count", 0)
        }
        
        # Assessment info
        context_dict["assessment_info"] = {
            "type": metadata.get("assessment_type", "unknown"),
            "id": metadata.get("assessment_id", "unknown"),
            "display_name": metadata.get("assessment_display_name", "Unknown"),
            "processing_time": metadata.get("processing_time_seconds", 0),
            "stages_completed": metadata.get("stages_completed", []),
            "total_tokens": assessment_data.get("result", {}).get("statistics", {}).get("total_tokens", 0)
        }
        
        # Get key data based on assessment type
        assessment_type = metadata.get("assessment_type", "unknown")
        
        # Extract key data based on type
        if assessment_type == "distill":
            # For summary, get key points and executive summary
            key_points = []
            
            # Check extracted_info first (most likely location)
            if "extracted_info" in result and "key_points" in result["extracted_info"]:
                key_points = result["extracted_info"]["key_points"]
            elif "key_points" in result:
                key_points = result["key_points"]
                
            # Get executive summary if available
            executive_summary = None
            if "overall_assessment" in result and "executive_summary" in result["overall_assessment"]:
                executive_summary = result["overall_assessment"]["executive_summary"]
                
            context_dict["key_data"] = {
                "key_points": key_points,
                "executive_summary": executive_summary,
                "topics": result.get("topics", [])
            }
            
        elif assessment_type == "assess":
            # For issues assessment, get issues array
            issues = []
            if "issues" in result:
                issues = result["issues"]
            
            executive_summary = result.get("executive_summary")
            
            context_dict["key_data"] = {
                "issues": issues,
                "executive_summary": executive_summary
            }
            
        elif assessment_type == "extract":
            # For action items, get action items array
            action_items = []
            if "action_items" in result:
                action_items = result["action_items"]
                
            context_dict["key_data"] = {
                "action_items": action_items
            }
            
        elif assessment_type == "analyze":
            # For framework analysis, structure depends on the framework
            # This is a generic approach
            context_dict["key_data"] = {
                "framework_results": result
            }
            
        # If we have context object, extract additional insights
        if assessment_data.get("context"):
            context_obj = assessment_data["context"]
            
            # Add chunk information if available
            if hasattr(context_obj, "chunks") and context_obj.chunks:
                context_dict["document_info"]["chunks"] = len(context_obj.chunks)
                
            # Add token tracking if available
            if hasattr(context_obj, "token_usage") and context_obj.token_usage:
                context_dict["assessment_info"]["token_usage"] = context_obj.token_usage
                
            # Add evidence store if available
            if hasattr(context_obj, "evidence_store") and context_obj.evidence_store:
                # Get a count of evidence items
                evidence_count = 0
                if "references" in context_obj.evidence_store:
                    references = context_obj.evidence_store["references"]
                    evidence_count = sum(len(refs) for refs in references.values())
                
                context_dict["assessment_info"]["evidence_count"] = evidence_count
        
        # Add report data if available
        if assessment_data.get("markdown"):
            context_dict["report"] = assessment_data["markdown"]
            
        return context_dict
        
    except Exception as e:
        logger.error(f"Error preparing context for chat: {str(e)}")
        return {"error": str(e)}

def extract_agent_insights(assessment_data):
    """Extract agent-specific insights and organize them for the LLM."""
    insights = {}
    
    try:
        # Check if we have context object with agent data
        if assessment_data.get("context") and hasattr(assessment_data["context"], "data"):
            context_obj = assessment_data["context"]
            
            # Extract data from each agent
            insights["planner"] = context_obj.get_data_for_agent("planner")
            insights["extractor"] = {}
            insights["aggregator"] = {}
            insights["evaluator"] = {}
            
            # Get assessment type
            assessment_type = assessment_data.get("metadata", {}).get("assessment_type", "unknown")
            data_type = {
                "extract": "action_items",
                "assess": "issues",
                "distill": "key_points",
                "analyze": "evidence"
            }.get(assessment_type, "items")
            
            # Extract type-specific data
            if data_type:
                insights["extractor"][data_type] = context_obj.get_data_for_agent("extractor", data_type)
                insights["aggregator"][data_type] = context_obj.get_data_for_agent("aggregator", data_type)
                insights["evaluator"][data_type] = context_obj.get_data_for_agent("evaluator", data_type)
                
            # Get overall assessment if available
            insights["evaluator"]["overall_assessment"] = context_obj.get_data_for_agent("evaluator", "overall_assessment")
            
        # Fallback to result data if context not available
        elif assessment_data.get("result"):
            result = assessment_data["result"]
            
            # Try to infer agent outputs from result structure
            if "extracted_info" in result:
                insights["extractor"] = result["extracted_info"]
                
            if "overall_assessment" in result:
                insights["evaluator"] = {"overall_assessment": result["overall_assessment"]}
                
        return insights
        
    except Exception as e:
        logger.error(f"Error extracting agent insights: {str(e)}")
        return {"error": str(e)}

async def generate_chat_response(user_message, assessment_data, document_content, chat_history=None):
    """Generate a response using the LLM with enhanced context awareness."""
    try:
        if "llm" not in st.session_state or not st.session_state.llm:
            return "Error: LLM not initialized. Please check your API key."
        
        # Prepare context information in a structured way
        context_dict = prepare_context_for_chat(assessment_data)
        
        # Extract agent insights
        agent_insights = extract_agent_insights(assessment_data)
        
        # Get assessment type and metadata
        metadata = assessment_data.get("result", {}).get("metadata", {}) or assessment_data.get("metadata", {})
        assessment_type = metadata.get("assessment_type", "unknown")
        
        # Create system prompt with context about the document and assessment
        system_prompt = f"""
You are ChatBeyond, an AI assistant that helps users explore documents processed by Beyond Notes, a multi-agent document analysis system.

You have access to:
1. The document content
2. The assessment results produced by multiple specialized AI agents
3. Structured context and insights extracted from the document

Assessment Details:
- Type: {assessment_type.title()}
- Document: {context_dict.get("document_info", {}).get("filename", "Unknown Document")}
- Words: {context_dict.get("document_info", {}).get("word_count", "Unknown")}
- Processing: {context_dict.get("assessment_info", {}).get("stages_completed", [])}

Your capabilities:
- Answer questions about the document's content
- Explain the assessment findings in detail
- Highlight key information based on the assessment type
- Refer to specific evidence that supports your answers
- Acknowledge when information is not available in the assessment

When answering:
1. Primarily use the extracted insights and assessment results
2. Reference the document content when needed for clarification or additional context
3. Format responses with Markdown for readability
4. Be concise but thorough, focusing on the user's specific question
5. Provide specific references to evidence when available
6. Make it clear when you're inferring information vs. stating explicit findings

You have access to data from multiple analysis agents: Planner, Extractor, Aggregator, and Evaluator. Each provides different perspectives on the document.
"""

        # Prepare chat history context (last 5 messages maximum)
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            last_messages = chat_history[-5:] if len(chat_history) > 5 else chat_history
            chat_context = "Previous conversation:\n"
            for msg in last_messages:
                prefix = "User" if msg["is_user"] else "Assistant"
                chat_context += f"{prefix}: {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}\n\n"
        
        # Create a summary of key data based on assessment type
        key_data_summary = ""
        key_data = context_dict.get("key_data", {})
        
        if assessment_type == "distill":
            # For summary assessment, highlight key points
            key_points = key_data.get("key_points", [])
            key_data_summary += f"Key Points ({len(key_points)}):\n"
            
            # Include up to 5 key points with high importance
            high_importance_points = [p for p in key_points if p.get("importance", "").lower() == "high"][:3]
            for point in high_importance_points:
                key_data_summary += f"- {point.get('text', '')}\n"
                
            # Add executive summary if available
            if key_data.get("executive_summary"):
                key_data_summary += f"\nExecutive Summary:\n{key_data.get('executive_summary')[:300]}...\n"
                
        elif assessment_type == "assess":
            # For issues assessment, highlight top issues
            issues = key_data.get("issues", [])
            key_data_summary += f"Issues ({len(issues)}):\n"
            
            # Include up to 3 high severity issues
            high_severity_issues = [i for i in issues if i.get("severity", "").lower() in ["critical", "high"]][:3]
            for issue in high_severity_issues:
                key_data_summary += f"- {issue.get('title', '')}: {issue.get('severity', '').upper()}\n"
                
            # Add executive summary if available
            if key_data.get("executive_summary"):
                key_data_summary += f"\nExecutive Summary:\n{key_data.get('executive_summary')[:300]}...\n"
                
        elif assessment_type == "extract":
            # For action items, highlight top action items
            action_items = key_data.get("action_items", [])
            key_data_summary += f"Action Items ({len(action_items)}):\n"
            
            # Include up to 3 high priority action items
            high_priority_items = [a for a in action_items if a.get("priority", "").lower() == "high"][:3]
            for item in high_priority_items:
                key_data_summary += f"- {item.get('description', '')}\n"
        
        # Add user's query with context
        prompt = f"""
{chat_context}

Document Context Summary:
{key_data_summary}

User Question: {user_message}

Relevant Document Content (preview): 
{document_content[:2000]}...

Remember to answer based primarily on the assessment results and insights, referring to the document content only as needed for context.
"""
        
        # Call the LLM
        response, usage = await st.session_state.llm.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=2000,
            temperature=0.3
        )
        
        logger.info(f"Generated response with {usage.get('total_tokens', 0)} tokens")
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

def format_timestamp(timestamp):
    """Format a timestamp for display in chat."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%I:%M %p")

def display_chat_message(message, is_user=False):
    """Display a chat message with appropriate styling."""
    if is_user:
        avatar = "👤"
        message_class = "user"
    else:
        avatar = "🤖"
        message_class = "assistant"
    
    timestamp = format_timestamp(message.get("timestamp", time.time()))
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="chat-avatar">{avatar}</div>
        <div class="chat-content">
            {message["content"]}
            <div class="chat-timestamp">{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def get_suggested_questions(assessment_type):
    """Get suggested questions based on assessment type."""
    # Base questions for all types
    base_questions = [
        "What is the most important information in this document?",
        "Can you summarize the key findings?",
        "What should I focus on in this document?"
    ]
    
    # Type-specific questions
    if assessment_type == "distill":
        return base_questions + [
            "What are the main key points of the document?",
            "What topics does this document cover?",
            "Can you extract important quotes from the document?",
            "What are the most significant insights from this document?"
        ]
    elif assessment_type == "extract":
        return base_questions + [
            "What are the high priority action items?",
            "Who is responsible for the most action items?",
            "When are the action items due?",
            "Are there any overdue or urgent action items?"
        ]
    elif assessment_type == "assess":
        return base_questions + [
            "What are the critical issues identified?",
            "What are the highest severity risks?",
            "What are the recommended solutions for the issues?",
            "Are there any recurring themes among the issues?"
        ]
    elif assessment_type == "analyze":
        return base_questions + [
            "What is the overall maturity rating?",
            "Which dimension has the highest score?",
            "What are the main areas for improvement?",
            "What evidence supports the ratings?"
        ]
    else:
        return base_questions

def main():
    """Main function for the chat page."""
    # Initialize page
    initialize_chat_page()
    
    # Page header
    page_header("💬 Chat with Documents", "Ask questions about your analyzed documents to explore insights and findings")
    
    # Check for API key
    if "llm" not in st.session_state or not st.session_state.llm:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to use the chat feature.")
        return
    
    # Sidebar - Document Selection
    with st.sidebar:
        section_header("Select Document")
        
        # Load recent assessments
        recent_assessments = load_recent_assessments()
        
        if not recent_assessments:
            st.warning("No assessed documents found. Process a document in one of the assessment tabs first.")
        else:
            # Group by document name
            documents = {}
            for assessment in recent_assessments:
                doc_name = assessment.get("document_name")
                if doc_name not in documents:
                    documents[doc_name] = []
                documents[doc_name].append(assessment)
            
            # Display document selection
            selected_document = st.selectbox(
                "Document",
                options=list(documents.keys()),
                format_func=lambda x: x
            )
            
            if selected_document:
                # Display assessments for this document
                doc_assessments = documents.get(selected_document, [])
                
                assessment_options = {
                    f"{a['type'].title()} - {a['modified'].strftime('%Y-%m-%d %H:%M')}": i
                    for i, a in enumerate(doc_assessments)
                }
                
                selected_assessment_key = st.selectbox(
                    "Assessment",
                    options=list(assessment_options.keys()),
                    format_func=lambda x: x
                )
                
                if selected_assessment_key:
                    assessment_idx = assessment_options.get(selected_assessment_key)
                    selected_assessment = doc_assessments[assessment_idx]
                    
                    # Load assessment data if not already loaded
                    if (not st.session_state.current_assessment or 
                        st.session_state.current_assessment.get("id") != selected_assessment.get("id")):
                        
                        st.info("Loading assessment data...")
                        
                        # Load full assessment data with context
                        assessment_data = load_assessment_data(selected_assessment)
                        
                        if assessment_data:
                            # Extract document content
                            document_content = get_document_content(assessment_data)
                            
                            # Store in session state
                            st.session_state.current_document = document_content
                            st.session_state.current_assessment = {
                                "id": selected_assessment.get("id"),
                                "data": assessment_data,
                                "type": selected_assessment.get("type"),
                                "display_name": selected_assessment.get("display_name"),
                                "document_name": selected_assessment.get("document_name")
                            }
                            
                            # Clear chat history for new document
                            st.session_state.chat_history = []
                            
                            # Show success message
                            st.success(f"Loaded assessment: {selected_assessment.get('display_name')}")
                            st.rerun()  # Refresh to update UI
                        else:
                            st.error("Failed to load assessment data.")
        
        # Display data sources
        if st.session_state.current_assessment:
            assessment_data = st.session_state.current_assessment.get("data", {})
            
            with st.expander("Data Sources", expanded=False):
                st.markdown("### Available Information")
                
                # Check for result data
                if assessment_data.get("result"):
                    st.markdown("✅ **Assessment Results**")
                else:
                    st.markdown("❌ Assessment Results")
                
                # Check for context data
                if assessment_data.get("context"):
                    st.markdown("✅ **Processing Context**")
                    
                    # Show context details
                    if hasattr(assessment_data["context"], "chunks"):
                        st.markdown(f"- Chunks: {len(assessment_data['context'].chunks)}")
                    
                    if hasattr(assessment_data["context"], "token_usage"):
                        st.markdown(f"- Tokens: {assessment_data['context'].token_usage:,}")
                else:
                    st.markdown("❌ Processing Context")
                
                # Check for document content
                if st.session_state.current_document and len(st.session_state.current_document) > 100:
                    st.markdown("✅ **Document Content**")
                    st.markdown(f"- Length: {len(st.session_state.current_document):,} characters")
                else:
                    st.markdown("❌ Document Content")
                    
                # Check for report
                if assessment_data.get("markdown"):
                    st.markdown("✅ **Markdown Report**")
                    report_length = len(assessment_data["markdown"])
                    st.markdown(f"- Length: {report_length:,} characters")
                else:
                    st.markdown("❌ Markdown Report")
    
    # Main content - Chat interface
    if not st.session_state.current_assessment:
        st.info("Select an assessment from the sidebar to start chatting.")
    else:
        # Display current document info
        current_assessment = st.session_state.current_assessment
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### Chatting with: {current_assessment.get('document_name')}")
            st.caption(f"Assessment Type: {current_assessment.get('type').title()} | ID: {current_assessment.get('id')}")
        
        with col2:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        st.markdown("---")
        for message in st.session_state.chat_history:
            display_chat_message(message, message["is_user"])
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("Your message:", key="chat_input", height=100, placeholder="Ask me anything about this document...")
            col1, col2 = st.columns([5, 1])
            
            with col1:
                submit_button = st.form_submit_button("Send Message", use_container_width=True, type="primary")
            
            with col2:
                # Add a setting for response length
                response_detail = st.selectbox(
                    "Detail",
                    options=["Concise", "Detailed"],
                    index=0,
                    key="response_detail"
                )
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "content": user_input,
                    "is_user": True,
                    "timestamp": time.time()
                })
                
                # Add temporary message while generating response
                with st.spinner("Generating response..."):
                    # Get current assessment data and document content
                    assessment_data = st.session_state.current_assessment.get("data", {})
                    document_content = st.session_state.current_document
                    
                    # Pass response detail preference
                    is_detailed = response_detail == "Detailed"
                    
                    # Generate response
                    response = asyncio.run(generate_chat_response(
                        user_input, 
                        assessment_data, 
                        document_content,
                        st.session_state.chat_history
                    ))
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "content": response,
                        "is_user": False,
                        "timestamp": time.time()
                    })
                
                # Rerun to update UI with new messages
                st.rerun()
        
        # Show helpful suggestions if no chat history
        if not st.session_state.chat_history:
            st.markdown("### Suggested Questions")
            st.caption("Click on any question to start the conversation")
            
            # Get suggested questions based on assessment type
            assessment_type = st.session_state.current_assessment.get("type")
            suggestions = get_suggested_questions(assessment_type)
            
            # Display suggestions in a grid
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(
                        suggestion, 
                        key=f"suggestion_{i}",
                        use_container_width=True,
                        type="secondary",
                        help="Click to ask this question"
                    ):
                        # Add suggestion to chat history
                        st.session_state.chat_history.append({
                            "content": suggestion,
                            "is_user": True,
                            "timestamp": time.time()
                        })
                        
                        # Generate response
                        with st.spinner("Generating response..."):
                            assessment_data = st.session_state.current_assessment.get("data", {})
                            document_content = st.session_state.current_document
                            
                            response = asyncio.run(generate_chat_response(
                                suggestion, 
                                assessment_data, 
                                document_content
                            ))
                            
                            st.session_state.chat_history.append({
                                "content": response,
                                "is_user": False,
                                "timestamp": time.time()
                            })
                        
                        st.rerun()

if __name__ == "__main__":
    main()