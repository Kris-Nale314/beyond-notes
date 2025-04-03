# pages/03_Chat.py
import streamlit as st
import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-chat")

# Import components
from core.llm.customllm import CustomLLM
from utils.paths import AppPaths

# Ensure directories exist
AppPaths.ensure_dirs()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Chat with Documents",
    page_icon="💬",
    layout="wide",
)

# Define accent colors
PRIMARY_COLOR = "#4CAF50"  # Green
SECONDARY_COLOR = "#2196F3"  # Blue
ACCENT_COLOR = "#FF9800"  # Orange
ERROR_COLOR = "#F44336"  # Red

# CSS to customize the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
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
    .chat-input {
        display: flex;
        margin-top: 2rem;
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
    
    if "llm" not in st.session_state:
        # Initialize LLM if not already done
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            st.session_state.llm = CustomLLM(api_key, model="gpt-3.5-turbo")
        else:
            st.session_state.llm = None

def load_recent_assessments():
    """Load a list of recent assessments."""
    assessment_types = ["distill", "extract", "assess", "analyze"]
    recent_assessments = []
    
    # Look for assessment results in output directories
    for assessment_type in assessment_types:
        output_dir = AppPaths.get_assessment_output_dir(assessment_type)
        
        if output_dir.exists():
            # Look for JSON result files
            for file_path in output_dir.glob(f"{assessment_type}_result_*.json"):
                try:
                    # Get file stats
                    stat = file_path.stat()
                    
                    # Try to load basic metadata
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            metadata = data.get("metadata", {})
                            
                            # Create assessment entry
                            assessment = {
                                "id": file_path.stem,
                                "path": file_path,
                                "type": assessment_type,
                                "display_name": metadata.get("assessment_display_name", assessment_type.title()),
                                "document_name": metadata.get("document_info", {}).get("filename", "Unknown Document"),
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "data": data
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

def load_assessment(assessment_path):
    """Load assessment data from a file."""
    try:
        with open(assessment_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading assessment data: {str(e)}")
        return None

def get_document_content(assessment_data):
    """Extract document content from assessment data if available."""
    try:
        # Try to get document text from metadata
        metadata = assessment_data.get("metadata", {})
        document_info = metadata.get("document_info", {})
        
        # Some assessments might store the document text
        if "document_text" in document_info:
            return document_info.get("document_text")
        
        # If not available directly, try to infer from file path
        if "filename" in document_info:
            filename = document_info.get("filename")
            # Try to find the file in uploads dir
            upload_path = AppPaths.get_temp_path("uploads") / filename
            if upload_path.exists():
                try:
                    # Try with utf-8 encoding first
                    with open(upload_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    # If utf-8 fails, try with latin-1 which should handle any byte sequence
                    with open(upload_path, 'r', encoding='latin-1') as f:
                        return f.read()
        
        return "Document content not available. Please provide context in your questions."
    except Exception as e:
        logger.error(f"Error getting document content: {str(e)}")
        return "Error retrieving document content."

def get_assessment_summary(assessment_data):
    """Generate a summary of the assessment data."""
    try:
        metadata = assessment_data.get("metadata", {})
        assessment_type = metadata.get("assessment_type", "unknown")
        document_name = metadata.get("document_info", {}).get("filename", "Unknown Document")
        
        result = assessment_data.get("result", {})
        
        summary = f"Assessment of document: {document_name}\n"
        summary += f"Assessment type: {assessment_type.title()}\n\n"
        
        # Type-specific summaries
        if assessment_type == "distill":
            summary += "**Key Points:**\n"
            key_points = result.get("key_points", [])
            for i, point in enumerate(key_points[:5], 1):
                summary += f"{i}. {point.get('text', '')}\n"
            
            if len(key_points) > 5:
                summary += f"... and {len(key_points) - 5} more points\n"
                
        elif assessment_type == "extract":
            summary += "**Action Items:**\n"
            action_items = result.get("action_items", [])
            for i, item in enumerate(action_items[:5], 1):
                desc = item.get("description", "")
                owner = item.get("owner", "Unassigned")
                summary += f"{i}. {desc} (Owner: {owner})\n"
                
            if len(action_items) > 5:
                summary += f"... and {len(action_items) - 5} more action items\n"
                
        elif assessment_type == "assess":
            summary += "**Issues:**\n"
            issues = result.get("issues", [])
            for i, issue in enumerate(issues[:5], 1):
                title = issue.get("title", "")
                severity = issue.get("severity", "Unknown")
                summary += f"{i}. {title} (Severity: {severity})\n"
                
            if len(issues) > 5:
                summary += f"... and {len(issues) - 5} more issues\n"
                
        elif assessment_type == "analyze":
            summary += "**Analysis Results:**\n"
            # This will depend on your framework structure
            if "overall_maturity" in result:
                overall = result.get("overall_maturity", {})
                rating = overall.get("overall_rating", "N/A")
                summary += f"Overall Rating: {rating}\n"
                summary += f"Summary: {overall.get('summary_statement', '')}\n"
        
        return summary
    except Exception as e:
        logger.error(f"Error generating assessment summary: {str(e)}")
        return "Error generating assessment summary."

async def generate_response(user_message, assessment_data, document_content):
    """Generate a response using the LLM."""
    try:
        if "llm" not in st.session_state or not st.session_state.llm:
            return "Error: LLM not initialized. Please check your API key."
        
        # Create a system prompt with context about the document and assessment
        metadata = assessment_data.get("metadata", {})
        assessment_type = metadata.get("assessment_type", "unknown")
        
        # Get assessment summary
        assessment_summary = get_assessment_summary(assessment_data)
        
        # Create system prompt
        system_prompt = f"""
You are a helpful AI assistant that answers questions about documents that have been analyzed by Beyond Notes.
You have access to the assessment results and document content.

Assessment type: {assessment_type}
Document: {metadata.get("document_info", {}).get("filename", "Unknown Document")}

Assessment Summary:
{assessment_summary}

Result Data: {json.dumps(assessment_data.get("result", {}), indent=2)}

When answering questions:
1. Use the assessment data and document content to provide accurate information
2. If the answer is not in the data, say so clearly
3. Format your responses with Markdown for readability
4. Be concise but thorough
        """
        
        # Add user's query and document context
        prompt = f"""
Question: {user_message}

Document Content (for reference): {document_content[:5000]}...
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

def display_chat_message(message, is_user=False):
    """Display a chat message with appropriate styling."""
    if is_user:
        avatar = "👤"
        message_class = "user"
    else:
        avatar = "🤖"
        message_class = "assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="chat-avatar">{avatar}</div>
        <div class="chat-content">{message}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function for the chat page."""
    # Initialize page
    initialize_chat_page()
    
    # Header
    st.markdown('<div class="main-header">Chat with Assessed Documents</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your analyzed documents to explore their content and assessment results.")
    
    # Sidebar - Document Selection
    with st.sidebar:
        st.markdown("### Select Assessment")
        
        # Load recent assessments
        recent_assessments = load_recent_assessments()
        
        if not recent_assessments:
            st.warning("No assessed documents found. Process a document in the Assess tab first.")
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
                        # Load full assessment data
                        assessment_data = load_assessment(selected_assessment.get("path"))
                        
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
                        else:
                            st.error("Failed to load assessment data.")
        
        # Display assessment summary
        if st.session_state.current_assessment:
            with st.expander("Assessment Summary", expanded=True):
                assessment_summary = get_assessment_summary(
                    st.session_state.current_assessment.get("data", {})
                )
                st.markdown(assessment_summary)
    
    # Main content - Chat interface
    if not st.session_state.current_assessment:
        st.info("Select an assessment from the sidebar to start chatting.")
    else:
        # Display current document info
        st.markdown(f"### Chatting with: {st.session_state.current_assessment.get('document_name')}")
        st.caption(f"Assessment: {st.session_state.current_assessment.get('display_name')}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message["content"], message["is_user"])
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("Your message:", key="chat_input", height=100)
            submit_button = st.form_submit_button("Send", use_container_width=True)
            
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
                    
                    # Generate response
                    response = asyncio.run(generate_response(
                        user_input, assessment_data, document_content
                    ))
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "content": response,
                        "is_user": False,
                        "timestamp": time.time()
                    })
                
                # Rerun to update UI with new messages
                st.rerun()
        
        # Show helpful suggestions
        if not st.session_state.chat_history:
            st.markdown("### Suggested Questions")
            assessment_type = st.session_state.current_assessment.get("type")
            
            # Type-specific suggestions
            if assessment_type == "distill":
                suggestions = [
                    "What are the main key points of the document?",
                    "What's the most important information in this document?",
                    "Can you summarize the document in one paragraph?",
                    "What topics does this document cover?"
                ]
            elif assessment_type == "extract":
                suggestions = [
                    "What are the action items from this document?",
                    "Who is responsible for the most action items?",
                    "What are the high priority action items?",
                    "When are the action items due?"
                ]
            elif assessment_type == "assess":
                suggestions = [
                    "What are the critical issues identified in this document?",
                    "What are the main risks mentioned?",
                    "Which issues have the highest severity?",
                    "What are the recommended solutions for the issues?"
                ]
            elif assessment_type == "analyze":
                suggestions = [
                    "What is the overall maturity rating?",
                    "Which dimension has the highest score?",
                    "What are the main areas for improvement?",
                    "What evidence supports the ratings?"
                ]
            else:
                suggestions = [
                    "What is this document about?",
                    "What are the main findings?",
                    "Can you summarize the key information?"
                ]
            
            # Display suggestions as clickable buttons
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(suggestion, key=f"suggestion_{i}"):
                        # Add suggestion to chat history
                        st.session_state.chat_history.append({
                            "content": suggestion,
                            "is_user": True,
                            "timestamp": time.time()
                        })
                        
                        # Generate response (same code as above)
                        with st.spinner("Generating response..."):
                            assessment_data = st.session_state.current_assessment.get("data", {})
                            document_content = st.session_state.current_document
                            
                            response = asyncio.run(generate_response(
                                suggestion, assessment_data, document_content
                            ))
                            
                            st.session_state.chat_history.append({
                                "content": response,
                                "is_user": False,
                                "timestamp": time.time()
                            })
                        
                        st.rerun()

# Import asyncio for async/await support
import asyncio

if __name__ == "__main__":
    main()