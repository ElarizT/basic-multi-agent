import os
import re
import tiktoken
import logging
from typing import List, Dict, Any, Optional
from docx import Document
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        # Fallback to approximate counting
        return len(text) // 4  # Rough approximation

def check_text_for_issues(text: str) -> Dict[str, Any]:
    """Check text for common issues like grammatical errors, spelling mistakes, etc."""
    issues = {
        "spelling": [],
        "grammar": [],
        "style": [],
        "total_count": 0
    }
    
    # Placeholder for actual implementation
    # In a real app, you would integrate with a proper grammar checking library
    
    # Simple patterns to demonstrate functionality
    # Check for doubled words
    doubled_words = re.finditer(r'\b(\w+)\s+\1\b', text, re.IGNORECASE)
    for match in doubled_words:
        issues["style"].append({
            "type": "doubled_word",
            "text": match.group(),
            "position": match.span()
        })
        issues["total_count"] += 1
    
    # Check for common spelling mistakes (very basic)
    common_misspellings = {
        "teh": "the",
        "alot": "a lot",
        "recieve": "receive",
        "wierd": "weird",
        "occured": "occurred"
    }
    
    for misspelled, correct in common_misspellings.items():
        for match in re.finditer(r'\b' + misspelled + r'\b', text, re.IGNORECASE):
            issues["spelling"].append({
                "type": "misspelling",
                "text": match.group(),
                "suggestion": correct,
                "position": match.span()
            })
            issues["total_count"] += 1
    
    return issues

def extract_text_from_file(file) -> str:
    """Extract text from various file formats."""
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == '.txt':
            return file.getvalue().decode('utf-8')
        
        elif file_extension == '.pdf':
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif file_extension in ['.docx', '.doc']:
            doc = Document(file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        
        else:
            return f"Unsupported file format: {file_extension}"
    
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        return f"Error processing file: {str(e)}"

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format chat history for display or processing."""
    formatted_chat = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            formatted_chat += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_chat += f"Assistant: {content}\n\n"
        elif role == "system":
            # System messages are typically not displayed
            pass
    
    return formatted_chat

def generate_system_prompt(agent_type: str) -> str:
    """Generate an appropriate system prompt based on the agent type."""
    prompts = {
        "general": (
            "You are a helpful assistant. Provide clear and concise responses to the user's questions. "
            "Be friendly and conversational."
        ),
        "proofreader": (
            "You are a professional proofreader and editor. Analyze texts for grammar, spelling, "
            "punctuation, style, and clarity issues. Provide specific, actionable feedback to "
            "improve writing quality."
        ),
        "reasoning": (
            "You are an expert reasoning assistant specializing in complex tasks like mathematics, "
            "physics, and other scientific disciplines. Show your work step-by-step, explain your "
            "reasoning clearly, and cite relevant formulas or principles when appropriate."
        )
    }
    
    return prompts.get(agent_type, prompts["general"])
