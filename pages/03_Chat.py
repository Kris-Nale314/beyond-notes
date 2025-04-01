import os
import sys
import asyncio
import streamlit as st
import logging
import json
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project components
from core.llm.customllm import CustomLLM
from assessments.loader import AssessmentLoader

# Page config
st.set_page_config(
    page_title="Chat with Document - Beyond Notes",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

async def process_chat_query(query, document_text, assessment_type, assessment_result, model="gpt-3.5-turbo"):
    """Process a chat query about the analyzed document."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    
    # Initialize LLM
    llm = CustomLLM(api_key, model=model)
    
    # Create a prompt based on the assessment type and result
    if assessment_type == "distill":
        system_prompt = "You are an assistant helping with document analysis. You have access to a summarized document and can answer questions about its content."
    elif assessment_type == "extract":
        system_prompt = "You are an assistant helping with action item tracking. You have access to extracted action items from a document and can answer questions about tasks, owners, and timelines."
    elif assessment_type == "assess":
        system_prompt = "You are an assistant helping with issue analysis. You have access to identified issues from a document and can answer questions about problems, risks, and challenges."
    elif assessment_type == "analyze":
        system_prompt = "You are an assistant helping with framework analysis. You have access to a readiness assessment and can answer questions about capabilities, maturity levels, and recommendations."
    else:
        system_prompt = "You are an assistant helping with document analysis. You have access to an analyzed document and can answer questions about its content."
    
    # Extract key information based on assessment type
    context_info = ""
    
    if assessment_type == "distill":
        if "summary" in assessment_result:
            context_info += f"Document Summary:\n{assessment_result['summary']}\n\n"
        if "topics" in assessment_result and isinstance(assessment_result["topics"], list):
            context_info += "Key Topics:\n"
            for topic in assessment_result["topics"]:
                context_info += f"- {topic.get('topic', '')}\n"
    
    elif assessment_type == "extract":
        if "action_items" in assessment_result and isinstance(assessment_result["action_items"], list):
            context_info += "Action Items:\n"
            for item in assessment_result["action_items"]:
                owner = item.get("owner", "Unassigned")
                description = item.get("description", "")
                due_date = item.get("due_date", "")
                context_info += f"- {description} (Owner: {owner}, Due: {due_date})\n"
    
    elif assessment_type == "assess":
        if "issues" in assessment_result and isinstance(assessment_result["issues"], list):
            context_info += "Identified Issues:\n"
            for issue in assessment_result["issues"]:
                title = issue.get("title", "")
                severity = issue.get("severity", "")
                category = issue.get("category", "")
                context_info += f"- {title} (Severity: {severity}, Category: {category})\n"
    
    elif assessment_type == "analyze":
        if "overall_readiness" in assessment_result:
            overall = assessment_result["overall_readiness"]
            rating = overall.get("overall_rating", "")
            context_info += f"Overall Readiness Rating: {rating}\n\n"
            context_info += f"Summary: {overall.get('summary_statement', '')}\n\n"
        
        if "dimension_assessments" in assessment_result:
            context_info += "Dimension Ratings:\n"
            for dim_name, dim_data in assessment_result["dimension_assessments"].items():
                rating = dim_data.get("dimension_rating", "")
                context_info += f"- {dim_name}: {rating}\n"
    
    # Create the chat prompt
    prompt = f"""
    {system_prompt}
    
    DOCUMENT ANALYSIS INFORMATION:
    Assessment Type: {assessment_type}
    
    {context_info}
    
    USER QUESTION: 
    {query}
    
    Please answer the question based on the analyzed document information. If you cannot answer the question with the available information, please say so and suggest what additional information might be needed.
    """
    
    # Get the response
    try:
        response = await llm.generate_completion(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3
        )
        return response
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        return f"Error: {str(e)}"

def format_message(role, content):
    """Format a chat message with appropriate styling."""
    if role == "user":
        return f'<div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;">{content}</div>'
    else:
        return f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px;">{content}</div>'

def main():
    st.title("Chat with Your Document")
    
    # Check if we have a processed document
    if "assessment_result" not in st.session_state:
        st.warning("No analyzed document found. Please analyze a document first.")
        
        # Return button
        if st.button("Go to Analyze Page"):
            st.switch_page("pages/02_Analyze.py")
        
        return
    
    # Get assessment information
    document = st.session_state.document
    assessment_result = st.session_state.assessment_result
    assessment_type = st.session_state.assessment_type
    
    # Sidebar with document info
    with st.sidebar:
        st.header("Document Information")
        st.write(f"**Filename:** {document.filename}")
        st.write(f"**Assessment Type:** {assessment_type.title()}")
        
        # Model selection for chat
        st.subheader("Chat Settings")
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Select the model to use for chat"
        )
    
    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        # Welcome message if no history
        if not st.session_state.chat_history:
            st.info(f"Welcome! You can now chat about your {assessment_type} assessment of '{document.filename}'")
        
        # Format and display chat history
        for msg in st.session_state.chat_history:
            st.markdown(format_message(msg["role"], msg["content"]), unsafe_allow_html=True)
    
    # Input for new messages
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question:", height=100)
        submit_button = st.form_submit_button("Send")
    
    # Process new message
    if submit_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show "thinking" message
        with chat_container:
            st.markdown(format_message("user", user_input), unsafe_allow_html=True)
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(format_message("assistant", "Thinking..."), unsafe_allow_html=True)
        
        # Process query
        with st.spinner(""):
            response = asyncio.run(process_chat_query(
                user_input, 
                document.text, 
                assessment_type, 
                assessment_result,
                model=model
            ))
        
        # Replace thinking message with actual response
        thinking_placeholder.empty()
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display the new message
        with chat_container:
            st.markdown(format_message("assistant", response), unsafe_allow_html=True)
    
    # Option to clear chat history
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()