# pages/03_Chat.py
import streamlit as st
import os
import asyncio
import logging
import json
import time
import pickle
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-chat")

# Import components with clear error handling
try:
    logger.info("Importing core components...")
    from core.llm.customllm import CustomLLM
    from utils.paths import AppPaths
    from utils.ui.styles import get_base_styles
    from utils.ui.components import page_header, section_header
    from utils.accessor import DataAccessor
    logger.info("Successfully imported all components")
except ImportError as e:
    logger.error(f"Failed to import components: {e}", exc_info=True)
    st.error(f"Failed to load necessary components: {e}")
    st.stop()

# Ensure directories exist
AppPaths.ensure_dirs()

# Page configuration
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
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 200px);
        min-height: 400px;
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        overflow: hidden;
    }
    
    /* Chat messages area */
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding: 1rem;
    }
    
    /* Chat input area */
    .chat-input {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        background-color: rgba(0, 0, 0, 0.05);
    }
    
    /* Message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
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
    
    /* Document selector styling */
    .document-card {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .document-card:hover {
        background-color: rgba(33, 150, 243, 0.1);
        transform: translateY(-2px);
    }
    
    .document-card.selected {
        border-color: #2196F3;
        background-color: rgba(33, 150, 243, 0.15);
    }
    
    .document-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .document-meta {
        font-size: 0.8rem;
        opacity: 0.7;
    }
    
    /* Suggestion button styling */
    .suggestion-button {
        background-color: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-block;
    }
    
    .suggestion-button:hover {
        background-color: rgba(33, 150, 243, 0.2);
        transform: translateY(-2px);
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

def load_recent_results():
    """Load a list of recent assessment results."""
    assessment_types = ["distill", "extract", "assess", "analyze"]
    recent_results = []
    
    # Look for assessment results in output directories
    for assessment_type in assessment_types:
        output_dir = AppPaths.get_assessment_output_dir(assessment_type)
        
        if output_dir.exists():
            # Look for json result files that have matching context files
            for json_path in output_dir.glob(f"{assessment_type}_result_*.json"):
                try:
                    # Check if a corresponding context file exists
                    context_path = json_path.with_suffix('.pkl')
                    if not context_path.exists():
                        continue
                        
                    # Get file stats
                    stat = json_path.stat()
                    
                    # Load basic metadata from JSON
                    with open(json_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            metadata = data.get("metadata", {})
                            
                            # Create a result entry
                            result_entry = {
                                "id": json_path.stem,
                                "result_path": json_path,
                                "context_path": context_path,
                                "type": assessment_type,
                                "display_name": metadata.get("assessment_display_name", assessment_type.title()),
                                "document_name": metadata.get("document_info", {}).get("filename", "Unknown Document"),
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "metadata": metadata
                            }
                            
                            recent_results.append(result_entry)
                        except json.JSONDecodeError:
                            # Skip invalid JSON files
                            logger.warning(f"Invalid JSON in file: {json_path}")
                            continue
                except Exception as e:
                    logger.warning(f"Error loading result file {json_path}: {str(e)}")
                    continue
    
    # Sort by modification time (most recent first)
    recent_results.sort(key=lambda x: x["modified"], reverse=True)
    
    return recent_results

def load_result_data(result_path, context_path):
    """Load assessment result and context data from files."""
    try:
        # Load result data
        result_data = None
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        
        # Load context data
        context_data = None
        if context_path.exists():
            try:
                with open(context_path, 'rb') as f:
                    context_data = pickle.load(f)
                logger.info(f"Successfully loaded context data from {context_path}")
            except Exception as e:
                logger.error(f"Error loading context data: {str(e)}")
        
        return {
            "result": result_data,
            "context": context_data
        }
    except Exception as e:
        logger.error(f"Error loading result data: {str(e)}")
        return None

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
            "Can you extract important quotes from the document?"
        ]
    elif assessment_type == "extract":
        return base_questions + [
            "What are the high priority action items?",
            "Who is responsible for the most action items?",
            "When are the action items due?"
        ]
    elif assessment_type == "assess":
        return base_questions + [
            "What are the critical issues identified?",
            "What are the highest severity risks?",
            "What are the recommended solutions for the issues?"
        ]
    elif assessment_type == "analyze":
        return base_questions + [
            "What is the overall maturity rating?",
            "Which dimension has the highest score?",
            "What are the main areas for improvement?"
        ]
    else:
        return base_questions

async def generate_chat_response(message, assessment_data):
    """Generate a response using the LLM."""
    try:
        if "llm" not in st.session_state or not st.session_state.llm:
            return "Error: LLM not initialized. Please check your API key."
        
        # Extract standardized data using our accessor
        result = assessment_data.get("result", {})
        context = assessment_data.get("context")
        assessment_type = result.get("metadata", {}).get("assessment_type", "unknown")
        
        # Use DataAccessor to get structured data based on assessment type
        if assessment_type == "distill":
            data = DataAccessor.get_summary_data(result, context)
        elif assessment_type == "assess":
            data = DataAccessor.get_issues_data(result, context)
        elif assessment_type == "extract":
            data = DataAccessor.get_action_items_data(result, context)
        elif assessment_type == "analyze":
            data = DataAccessor.get_analysis_data(result, context)
        else:
            data = {}
        
        # Create a system prompt with information about the document and assessment
        system_prompt = f"""
You are ChatBeyond, an AI assistant that helps users explore documents processed by Beyond Notes.

Document Information:
- Document: {data.get("metadata", {}).get("document_info", {}).get("filename", "Unknown")}
- Assessment Type: {assessment_type.title()}
- Word Count: {data.get("metadata", {}).get("document_info", {}).get("word_count", "Unknown")}

Your task is to answer questions about the document based on the assessment results.
Be clear, concise, and focus on information contained in the assessment.
Format your responses with Markdown for readability when appropriate.
"""

        # Create a user prompt with assessment data
        prompt = f"""
Here is the information from the {assessment_type.title()} assessment:

"""

        # Add assessment-specific data to the prompt
        if assessment_type == "distill":
            # Add summary content
            if data.get("summary_content"):
                prompt += f"Summary:\n{data['summary_content'][:1500]}...\n\n"
            
            # Add key points
            key_points = data.get("key_points", [])
            if key_points:
                prompt += f"Key Points ({len(key_points)}):\n"
                for i, point in enumerate(key_points[:10]):
                    if isinstance(point, dict):
                        prompt += f"- {point.get('text', '')}\n"
                    elif isinstance(point, str):
                        prompt += f"- {point}\n"
                if len(key_points) > 10:
                    prompt += f"... and {len(key_points) - 10} more key points\n"
                prompt += "\n"
            
            # Add topics
            topics = data.get("topics", [])
            if topics:
                prompt += f"Topics ({len(topics)}):\n"
                for topic in topics[:5]:
                    if isinstance(topic, dict):
                        prompt += f"- {topic.get('topic', '')}\n"
                prompt += "\n"
                
        elif assessment_type == "assess":
            # Add executive summary
            if data.get("executive_summary"):
                prompt += f"Executive Summary:\n{data['executive_summary']}\n\n"
            
            # Add issues
            issues = data.get("issues", [])
            if issues:
                prompt += f"Issues ({len(issues)}):\n"
                for i, issue in enumerate(issues[:5]):
                    if isinstance(issue, dict):
                        severity = issue.get("evaluated_severity", issue.get("severity", "medium")).upper()
                        prompt += f"- [{severity}] {issue.get('title', '')}: {issue.get('description', '')[:100]}...\n"
                if len(issues) > 5:
                    prompt += f"... and {len(issues) - 5} more issues\n"
                prompt += "\n"
                
        elif assessment_type == "extract":
            # Add action items
            action_items = data.get("action_items", [])
            if action_items:
                prompt += f"Action Items ({len(action_items)}):\n"
                for i, item in enumerate(action_items[:5]):
                    if isinstance(item, dict):
                        priority = item.get("evaluated_priority", item.get("priority", "medium")).upper()
                        prompt += f"- [{priority}] {item.get('description', '')}; Owner: {item.get('owner', 'Unassigned')}\n"
                if len(action_items) > 5:
                    prompt += f"... and {len(action_items) - 5} more action items\n"
                prompt += "\n"
                
        elif assessment_type == "analyze":
            # Add overall rating
            if data.get("overall_rating"):
                prompt += f"Overall Rating: {data['overall_rating']}\n\n"
            
            # Add dimension assessments
            dimensions = data.get("dimension_assessments", [])
            if dimensions:
                prompt += f"Dimensions ({len(dimensions)}):\n"
                for dimension in dimensions[:5]:
                    if isinstance(dimension, dict):
                        rating = dimension.get("dimension_rating", "")
                        rating_text = f" - Rating: {rating}" if rating else ""
                        prompt += f"- {dimension.get('dimension_name', '')}{rating_text}\n"
                prompt += "\n"
            
            # Add recommendations
            recommendations = data.get("strategic_recommendations", [])
            if recommendations:
                prompt += f"Recommendations ({len(recommendations)}):\n"
                for rec in recommendations[:3]:
                    if isinstance(rec, dict):
                        prompt += f"- {rec.get('recommendation', '')}\n"
                prompt += "\n"
        
        # Add user's question
        prompt += f"User's Question: {message}\n\nPlease answer based on the above information."
        
        # Call the LLM
        response, usage = await st.session_state.llm.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1500,
            temperature=0.3
        )
        
        logger.info(f"Generated response with {usage.get('total_tokens', 0)} tokens")
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

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
    
    # Layout with sidebar for document selection and main area for chat
    with st.sidebar:
        st.header("Select Document")
        
        # Load recent assessment results
        recent_results = load_recent_results()
        
        if not recent_results:
            st.warning("No processed documents found. Process a document in one of the assessment tabs first.")
        else:
            # Group by document name for easier navigation
            documents_by_name = {}
            for result in recent_results:
                doc_name = result.get("document_name")
                if doc_name not in documents_by_name:
                    documents_by_name[doc_name] = []
                documents_by_name[doc_name].append(result)
            
            # Let user select document
            selected_document = st.selectbox(
                "Document",
                options=list(documents_by_name.keys()),
                format_func=lambda x: x
            )
            
            # If document selected, show available assessments
            if selected_document:
                st.subheader("Available Assessments")
                
                doc_assessments = documents_by_name.get(selected_document, [])
                
                # Use radio buttons for better UX
                assessment_options = {}
                for i, assessment in enumerate(doc_assessments):
                    timestamp = assessment["modified"].strftime("%Y-%m-%d %H:%M")
                    label = f"{assessment['type'].title()} ({timestamp})"
                    assessment_options[label] = i
                
                if assessment_options:
                    selected_label = st.radio(
                        "Select Assessment",
                        options=list(assessment_options.keys())
                    )
                    
                    if selected_label:
                        assessment_idx = assessment_options[selected_label]
                        selected_assessment = doc_assessments[assessment_idx]
                        
                        # Load assessment data if not already loaded
                        current_id = st.session_state.current_assessment.get("id") if st.session_state.current_assessment else None
                        
                        if current_id != selected_assessment.get("id"):
                            with st.spinner("Loading assessment data..."):
                                result_path = selected_assessment["result_path"]
                                context_path = selected_assessment["context_path"]
                                
                                assessment_data = load_result_data(result_path, context_path)
                                
                                if assessment_data:
                                    # Store in session state
                                    st.session_state.current_assessment = {
                                        "id": selected_assessment.get("id"),
                                        "data": assessment_data,
                                        "type": selected_assessment.get("type"),
                                        "display_name": selected_assessment.get("display_name"),
                                        "document_name": selected_assessment.get("document_name")
                                    }
                                    
                                    # Clear chat history for new document
                                    st.session_state.chat_history = []
                                    
                                    st.success(f"Loaded assessment: {selected_assessment.get('display_name')}")
                                    st.rerun()
                                else:
                                    st.error("Failed to load assessment data.")
                
                # Button to clear chat history
                if st.session_state.current_assessment:
                    if st.button("Clear Chat History", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
    
    # Main content - Chat interface
    if not st.session_state.current_assessment:
        st.info("Select an assessment from the sidebar to start chatting.")
    else:
        # Show current document info
        assessment = st.session_state.current_assessment
        assessment_data = assessment.get("data", {})
        assessment_type = assessment.get("type", "unknown")
        
        st.subheader(f"Chatting with: {assessment.get('document_name')}")
        st.caption(f"Assessment Type: {assessment_type.title()}")
        
        # Chat interface
        chat_container = st.container(height=500)
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                display_chat_message(message, message["is_user"])
            
            # If no chat history, show suggested questions
            if not st.session_state.chat_history:
                st.markdown("### Suggested Questions")
                suggestions = get_suggested_questions(assessment_type)
                
                # Display suggestions as buttons
                cols = st.columns(2)
                for i, suggestion in enumerate(suggestions):
                    col = cols[i % 2]
                    with col:
                        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                            # Add suggestion to chat history
                            st.session_state.chat_history.append({
                                "content": suggestion,
                                "is_user": True,
                                "timestamp": time.time()
                            })
                            
                            # Generate response
                            with st.spinner("Generating response..."):
                                response = asyncio.run(generate_chat_response(
                                    suggestion, 
                                    assessment_data
                                ))
                                
                                st.session_state.chat_history.append({
                                    "content": response,
                                    "is_user": False,
                                    "timestamp": time.time()
                                })
                            
                            st.rerun()
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("Your message:", key="chat_input", height=100, placeholder="Ask me anything about this document...")
            submit_button = st.form_submit_button("Send Message", use_container_width=True, type="primary")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "content": user_input,
                    "is_user": True,
                    "timestamp": time.time()
                })
                
                # Add temporary message while generating response
                with st.spinner("Generating response..."):
                    response = asyncio.run(generate_chat_response(
                        user_input, 
                        assessment_data
                    ))
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "content": response,
                        "is_user": False,
                        "timestamp": time.time()
                    })
                
                # Rerun to update UI with new messages
                st.rerun()

if __name__ == "__main__":
    try:
        logger.info("Starting 03_Chat.py main execution")
        main()
        logger.info("03_Chat.py execution completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")