# pages/03_Chat_with_Document.py
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
    from utils.accessor import DataAccessor
    
    # Import enhanced UI components
    from utils.ui.enhanced import (
        enhanced_page_header,
        enhanced_section_header,
        apply_enhanced_theme
    )
    
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
apply_enhanced_theme()

# Add dark-themed chat-specific styles
st.markdown("""
<style>
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 250px);
        min-height: 500px;
        border-radius: 10px;
        background-color: rgba(30, 30, 30, 0.6);
        margin-bottom: 1rem;
        overflow: hidden;
        border: 1px solid rgba(60, 60, 60, 0.6);
    }
    
    /* Chat messages area */
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        padding: 1rem;
        scrollbar-width: thin;
        scrollbar-color: rgba(100, 100, 100, 0.5) rgba(30, 30, 30, 0.3);
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.3);
        border-radius: 10px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background-color: rgba(100, 100, 100, 0.5);
        border-radius: 10px;
    }
    
    /* Chat input area */
    .chat-input {
        border-top: 1px solid rgba(80, 80, 80, 0.5);
        padding: 1rem;
        background-color: rgba(40, 40, 40, 0.7);
    }
    
    /* Message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        animation: fadeIn 0.3s ease;
        color: rgba(255, 255, 255, 0.9);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message.user {
        background-color: rgba(33, 150, 243, 0.15);
        border-left: 4px solid #2196F3;
    }
    
    .chat-message.assistant {
        background-color: rgba(76, 175, 80, 0.15);
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
        flex-shrink: 0;
    }
    
    .user .chat-avatar {
        background-color: rgba(33, 150, 243, 0.2);
        color: #2196F3;
    }
    
    .assistant .chat-avatar {
        background-color: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
    }
    
    .chat-content {
        flex-grow: 1;
        line-height: 1.5;
    }
    
    .chat-content p {
        margin-bottom: 0.8rem;
    }
    
    .chat-content a {
        color: #2196F3;
        text-decoration: underline;
    }
    
    .chat-content code {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    .chat-content pre {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 0.8rem;
        border-radius: 6px;
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    .chat-content blockquote {
        border-left: 3px solid rgba(255, 255, 255, 0.2);
        padding-left: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .chat-timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        text-align: right;
    }
    
    /* Document selector styling */
    .document-card {
        background-color: rgba(30, 30, 30, 0.7);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(80, 80, 80, 0.5);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .document-card:hover {
        background-color: rgba(33, 150, 243, 0.15);
        transform: translateY(-2px);
        border-color: rgba(33, 150, 243, 0.5);
    }
    
    .document-card.selected {
        border-color: #2196F3;
        background-color: rgba(33, 150, 243, 0.15);
    }
    
    .document-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .document-meta {
        font-size: 0.8rem;
        opacity: 0.7;
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Suggestion button styling */
    .suggestion-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 15px 0;
    }
    
    .suggestion-button {
        background-color: rgba(30, 30, 30, 0.7);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        text-align: left;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .suggestion-button:hover {
        background-color: rgba(33, 150, 243, 0.15);
        transform: translateY(-2px);
        border-color: rgba(33, 150, 243, 0.5);
    }
    
    /* File selection panel */
    .file-selector {
        background: linear-gradient(145deg, rgba(30, 30, 30, 0.7), rgba(50, 50, 50, 0.7));
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(70, 70, 70, 0.7);
    }
    
    /* Current document display */
    .current-document {
        background: linear-gradient(145deg, rgba(20, 55, 97, 0.3), rgba(42, 36, 83, 0.3));
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid rgba(33, 150, 243, 0.3);
        display: flex;
        align-items: center;
    }
    
    .document-icon {
        font-size: 24px;
        margin-right: 15px;
        background-color: rgba(33, 150, 243, 0.2);
        color: #2196F3;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .document-info {
        flex-grow: 1;
    }
    
    .document-info h3 {
        margin: 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    
    .document-info .meta {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-top: 3px;
    }
    
    /* Welcome screen */
    .welcome-screen {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(145deg, rgba(30, 30, 30, 0.7), rgba(50, 50, 50, 0.7));
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid rgba(70, 70, 70, 0.7);
    }
    
    .welcome-icon {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .welcome-title {
        font-size: 24px;
        margin-bottom: 15px;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .welcome-text {
        font-size: 16px;
        max-width: 600px;
        margin: 0 auto 20px auto;
        color: rgba(255, 255, 255, 0.7);
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
                    # Try multiple potential context file patterns
                    context_path = None
                    
                    # Option 1: Direct suffix replacement (.json -> .pkl)
                    potential_path = json_path.with_suffix('.pkl')
                    if potential_path.exists():
                        context_path = potential_path
                    
                    # Option 2: Replace "result" with "context" in filename
                    if not context_path:
                        context_name = json_path.stem.replace('result', 'context')
                        potential_path = output_dir / f"{context_name}.pkl"
                        if potential_path.exists():
                            context_path = potential_path
                    
                    # Skip if no context file found
                    if not context_path:
                        logger.warning(f"No context file found for {json_path}")
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
                            logger.info(f"Found result: {json_path} with context: {context_path}")
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
                logger.info(f"Successfully loaded result data from {result_path}")
        
        # Load context data
        context_data = None
        if context_path.exists():
            try:
                with open(context_path, 'rb') as f:
                    # Use a more robust pickle loading approach
                    try:
                        context_data = pickle.load(f)
                        logger.info(f"Successfully loaded context data from {context_path}")
                    except EOFError:
                        logger.error(f"EOF error when loading context from {context_path}")
                    except Exception as e:
                        logger.error(f"Error during pickle loading: {str(e)}")
            except Exception as e:
                logger.error(f"Error opening context file: {str(e)}")
        
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

def process_markdown(text):
    """Process markdown in text for display in chat."""
    if not text:
        return ""
    
    # Process bold formatting
    text = text.replace("**", "<strong>", 1)
    while "**" in text:
        text = text.replace("**", "</strong>", 1)
        if "**" in text:
            text = text.replace("**", "<strong>", 1)
    
    # Process italic formatting
    text = text.replace("*", "<em>", 1)
    while "*" in text:
        text = text.replace("*", "</em>", 1)
        if "*" in text:
            text = text.replace("*", "<em>", 1)
    
    # Process code blocks
    import re
    # Replace ```language\ncode\n``` with <pre><code>code</code></pre>
    code_block_pattern = r'```(?:\w+)?\n(.*?)\n```'
    text = re.sub(code_block_pattern, r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    
    # Replace inline `code` with <code>code</code>
    inline_code_pattern = r'`([^`]+)`'
    text = re.sub(inline_code_pattern, r'<code>\1</code>', text)
    
    # Process links
    # Find markdown links [text](url)
    link_pattern = r'\[(.*?)\]\((.*?)\)'
    text = re.sub(link_pattern, r'<a href="\2" target="_blank">\1</a>', text)
    
    # Process bullet points
    # Replace lines starting with "- " with bullet points
    text = re.sub(r'(?m)^- (.*?)$', r'<li>\1</li>', text)
    # Wrap consecutive list items in <ul> tags
    text = re.sub(r'(<li>.*?</li>)(?:\n*<li>)', r'\1\n<li>', text)
    # Add opening <ul> for first list item
    text = re.sub(r'(?<!\n<li>)(<li>)', r'<ul>\1', text)
    # Add closing </ul> after last list item
    text = re.sub(r'(</li>)(?!\n*<li>)', r'\1</ul>', text)
    
    # Process paragraphs (blank lines)
    text = text.replace("\n\n", "</p><p>")
    
    # Process line breaks
    text = text.replace("\n", "<br>")
    
    return f"<p>{text}</p>"

def display_chat_message(message, is_user=False):
    """Display a chat message with appropriate styling and markdown processing."""
    if is_user:
        avatar = "👤"
        message_class = "user"
    else:
        avatar = "🤖"
        message_class = "assistant"
    
    timestamp = format_timestamp(message.get("timestamp", time.time()))
    
    # Process markdown in content
    content_html = process_markdown(message["content"])
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="chat-avatar">{avatar}</div>
        <div class="chat-content">
            {content_html}
            <div class="chat-timestamp">{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_document_info(assessment):
    """Display current document information in a nice card."""
    assessment_type = assessment.get("type", "unknown")
    document_name = assessment.get("document_name", "Unknown Document")
    display_name = assessment.get("display_name", assessment_type.title())
    modified = assessment.get("modified", datetime.now()).strftime("%b %d, %Y at %I:%M %p")
    
    # Select icon based on assessment type
    icons = {
        "distill": "📝",
        "assess": "⚠️",
        "extract": "📋",
        "analyze": "📊"
    }
    icon = icons.get(assessment_type, "📄")
    
    st.markdown(f"""
    <div class="current-document">
        <div class="document-icon">{icon}</div>
        <div class="document-info">
            <h3>{document_name}</h3>
            <div class="meta">{display_name} • Processed {modified}</div>
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

def display_suggestions(suggestions, assessment):
    """Display suggestion buttons with enhanced styling."""
    st.markdown("<h3 style='color: rgba(255, 255, 255, 0.9);'>Suggested Questions</h3>", unsafe_allow_html=True)
    
    # Use a grid layout with st.columns instead of custom HTML for better compatibility
    col1, col2 = st.columns(2)
    
    for i, suggestion in enumerate(suggestions):
        # Alternate between columns
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            # Use a standard button with the suggestion text
            if st.button(suggestion, key=f"suggestion_{i}"):
                # Add suggestion to chat history
                st.session_state.chat_history.append({
                    "content": suggestion,
                    "is_user": True,
                    "timestamp": time.time()
                })
                
                # Generate response
                with st.spinner("Thinking..."):
                    response = asyncio.run(generate_chat_response(
                        suggestion, 
                        assessment.get("data", {})
                    ))
                    
                    st.session_state.chat_history.append({
                        "content": response,
                        "is_user": False,
                        "timestamp": time.time()
                    })
                
                st.rerun()

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
Use bullet points for lists and structure your answers for easy scanning.
Keep your responses focused and to the point.
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

def display_welcome_screen():
    """Display a welcome screen when no document is selected."""
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-icon">💬</div>
        <div class="welcome-title">Chat with your analyzed documents</div>
        <div class="welcome-text">
            Select a processed document from the sidebar to start chatting.
            You can ask questions about the document's content, summary, issues, or any insights discovered during analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function for the chat page."""
    # Initialize page
    initialize_chat_page()
    
    # Enhanced page header
    enhanced_page_header(
        "Chat with Documents", 
        "💬",
        "Ask questions about your analyzed documents to explore insights and findings",
        ("#2196F3", "#4CAF50")  # Blue to green gradient
    )
    
    # Check for API key
    if "llm" not in st.session_state or not st.session_state.llm:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to use the chat feature.")
        return
    
    # Layout with sidebar for document selection and main area for chat
    with st.sidebar:
        enhanced_section_header("Document Selection", icon="📄")
        
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
                "Select Document",
                options=list(documents_by_name.keys()),
                format_func=lambda x: x
            )
            
            # If document selected, show available assessments
            if selected_document:
                st.markdown("### Available Assessments")
                
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
                                result_path = Path(str(selected_assessment["result_path"]))
                                context_path = Path(str(selected_assessment["context_path"]))
                                
                                # Validate paths
                                if not result_path.exists():
                                    st.error(f"Result file not found: {result_path}")
                                else:
                                    if not context_path.exists():
                                        st.warning(f"Context file not found: {context_path}")
                                    
                                    assessment_data = load_result_data(result_path, context_path)
                                    
                                    if assessment_data and assessment_data.get("result"):
                                        # Store in session state
                                        st.session_state.current_assessment = {
                                            "id": selected_assessment.get("id"),
                                            "data": assessment_data,
                                            "type": selected_assessment.get("type"),
                                            "display_name": selected_assessment.get("display_name"),
                                            "document_name": selected_assessment.get("document_name"),
                                            "modified": selected_assessment.get("modified")
                                        }
                                        
                                        # Clear chat history for new document
                                        st.session_state.chat_history = []
                                        
                                        # Show success but don't rerun to avoid state reset issues
                                        st.success(f"Loaded assessment: {selected_assessment.get('display_name')}")
                                    else:
                                        st.error("Failed to load assessment data. Ensure files are complete.")
                
                # Button to clear chat history
                if st.session_state.current_assessment:
                    if st.button("Clear Chat History", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
    
    # Main content - Chat interface
    if not st.session_state.current_assessment:
        display_welcome_screen()
    else:
        # Show current document info
        assessment = st.session_state.current_assessment
        display_document_info(assessment)
        
        # Chat container with proper styling
        st.markdown("<div style='background-color: rgba(40, 40, 40, 0.7); padding: 15px; border-radius: 10px; border: 1px solid rgba(60, 60, 60, 0.6);'>", unsafe_allow_html=True)
        
        # Message display area
        #st.markdown("<div style='height: 400px; overflow-y: auto; padding: 0.5rem; margin-bottom: 1rem; background-color: rgba(30, 30, 30, 0.6); border-radius: 8px;'>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message, message["is_user"])
        
        # If no chat history, show suggested questions
        if not st.session_state.chat_history:
            suggestions = get_suggested_questions(assessment.get("type", "unknown"))
            display_suggestions(suggestions, assessment)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat input form
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
                with st.spinner("Thinking..."):
                    response = asyncio.run(generate_chat_response(
                        user_input, 
                        assessment.get("data", {})
                    ))
                    
                    # Add response to chat history
                    st.session_state.chat_history.append({
                        "content": response,
                        "is_user": False,
                        "timestamp": time.time()
                    })
                
                # Rerun to update UI with new messages
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)    

if __name__ == "__main__":
    try:
        logger.info("Starting 03_Chat_with_Document.py main execution")
        main()
        logger.info("03_Chat_with_Document.py execution completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")