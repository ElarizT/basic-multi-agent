import os
import gradio as gr
from dotenv import load_dotenv
from utils import generate_system_prompt

# Load environment variables from .env file
load_dotenv()

# Define model constants - moved from groq_utils.py for simplicity
GENERAL_CHATBOT_MODEL = "qwen-2.5-32b"
PROOFREADING_MODEL = "deepseek-r1-distill-llama-70b"
REASONING_MODEL = "deepseek-r1-distill-qwen-32b"

# Initialize conversation histories
general_messages = [
    {"role": "system", "content": generate_system_prompt("general")},
    {"role": "assistant", "content": "Hello! How can I help you today?"}
]

reasoning_messages = [
    {"role": "system", "content": generate_system_prompt("reasoning")},
    {"role": "assistant", "content": "Hello! I'm your scientific reasoning assistant. I can help you solve complex problems in mathematics, physics, and other scientific fields. What problem would you like help with today?"}
]

# Groq API utilities
def check_api_key_configured():
    """Check if the Groq API key is configured in environment variables."""
    return bool(os.environ.get("GROQ_API_KEY"))

def get_groq_client():
    """Get a Groq client using the API key from environment variables."""
    import groq
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("Warning: Groq API key not found in .env file")
        return None
        
    return groq.Client(api_key=api_key)

def generate_groq_response(messages, model, temperature=0.7, max_tokens=1000):
    """Generate a response using the Groq API."""
    if not check_api_key_configured():
        return "Error: Groq API key not configured in the .env file. Please add your API key to the .env file."
    
    client = get_groq_client()
    if not client:
        return "Error: Could not initialize Groq client."
    
    try:
        # Make the chat completion request
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract the content from the response
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

# Main components converted from Streamlit to Gradio
def general_chatbot_interface(message, history):
    """General chatbot interface using Qwen 2.5 (32B) model."""
    global general_messages
    
    # Add user message to history
    general_messages.append({"role": "user", "content": message})
    
    # Generate response using Groq API
    response = generate_groq_response(
        messages=general_messages,
        model=GENERAL_CHATBOT_MODEL,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Add assistant response to history
    general_messages.append({"role": "assistant", "content": response})
    
    return response

def clear_general_chat_history():
    """Clear the general chat history."""
    global general_messages
    general_messages = [
        {"role": "system", "content": generate_system_prompt("general")},
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]
    return None, []

def proofread_text(text):
    """Proofread text using Deepseek-R1 Distill Llama (70B) model."""
    if not text:
        return "Please enter some text to proofread."
    
    # Create messages for the API request
    messages = [
        {"role": "system", "content": generate_system_prompt("proofreader")},
        {"role": "user", "content": f"""Please analyze the following text and provide detailed feedback:
        
Text to analyze:
```
{text[:3000]}  # Limiting text length to avoid token limits
```

Please provide:
1. A summary of the main grammatical and spelling issues
2. Style and clarity suggestions
3. An overall assessment of the writing quality
4. Specific corrections for the most important issues
5. A revised version of a portion of the text addressing the most critical issues
        """}
    ]
    
    # Generate analysis using Groq API
    analysis = generate_groq_response(
        messages=messages,
        model=PROOFREADING_MODEL,
        temperature=0.4,
        max_tokens=2000
    )
    
    return analysis

def reasoning_chatbot_interface(message, history):
    """Scientific reasoning chatbot interface using Deepseek-R1 Distill Qwen (32B) model."""
    global reasoning_messages
    
    # Add user message to history
    reasoning_messages.append({"role": "user", "content": message})
    
    # Generate response using Groq API
    response = generate_groq_response(
        messages=reasoning_messages,
        model=REASONING_MODEL,
        temperature=0.3,
        max_tokens=2000
    )
    
    # Add assistant response to history
    reasoning_messages.append({"role": "assistant", "content": response})
    
    return response

def clear_reasoning_chat_history():
    """Clear the reasoning chat history."""
    global reasoning_messages
    reasoning_messages = [
        {"role": "system", "content": generate_system_prompt("reasoning")},
        {"role": "assistant", "content": "Hello! I'm your scientific reasoning assistant. I can help you solve complex problems in mathematics, physics, and other scientific fields. What problem would you like help with today?"}
    ]
    return None, []

def solve_equation(equation, variable):
    """Solve an equation for a variable."""
    if not equation or not variable:
        return "Please provide both an equation and a variable to solve for."
    
    # Prepare prompt for the model
    messages = [
        {"role": "system", "content": generate_system_prompt("reasoning")},
        {"role": "user", "content": f"""Solve the following equation for {variable}: {equation}. 
        
Please show a step-by-step solution, explaining each step clearly. First set up the equation, then solve it, and finally verify your answer.

If there are multiple solutions, list them all. If there's no solution or infinitely many solutions, explain why.
        """}
    ]
    
    # Generate solution using Groq API
    solution = generate_groq_response(
        messages=messages,
        model=REASONING_MODEL,
        temperature=0.2,
        max_tokens=1000
    )
    
    return solution

def calculate_calculus(operation, expression, variable, lower_bound=None, upper_bound=None, approach_value=None):
    """Calculate calculus operations (derivative, integral, limit)."""
    if not expression or not variable:
        return "Please provide both an expression and a variable."
    
    # Prepare the prompt based on operation type
    if operation == "Derivative":
        user_prompt = f"""Find the derivative of {expression} with respect to {variable}.
        
Please show a step-by-step solution, applying the appropriate differentiation rules and explaining each step."""
    
    elif operation == "Integral":
        if lower_bound and upper_bound:
            user_prompt = f"""Calculate the definite integral of {expression} with respect to {variable} from {lower_bound} to {upper_bound}.
            
Please show a step-by-step solution, including the antiderivative and the evaluation of the bounds."""
        else:
            user_prompt = f"""Find the indefinite integral of {expression} with respect to {variable}.
            
Please show a step-by-step solution, explaining the integration techniques you use."""
    
    elif operation == "Limit":
        approach_value = approach_value or "0"
        user_prompt = f"""Calculate the limit of {expression} as {variable} approaches {approach_value}.
        
Please show a step-by-step solution, explaining your approach and any relevant limit laws or techniques."""
    
    else:
        return "Invalid operation. Please select Derivative, Integral, or Limit."
    
    # Prepare messages for the API request
    messages = [
        {"role": "system", "content": generate_system_prompt("reasoning")},
        {"role": "user", "content": user_prompt}
    ]
    
    # Generate solution using Groq API
    solution = generate_groq_response(
        messages=messages,
        model=REASONING_MODEL,
        temperature=0.2,
        max_tokens=1000
    )
    
    return solution

# Create the Gradio interface
with gr.Blocks(title="Multi-Purpose AI Agent", theme=gr.themes.Soft()) as app:
    # Header
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 1rem">
        <h1>Multi-Purpose AI Agent</h1>
        <p>Powered by Groq LLM API</p>
    </div>
    """)
    
    # API Key Status
    with gr.Row():
        if check_api_key_configured():
            gr.HTML("<div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>✅ Groq API key loaded successfully from .env file</div>")
        else:
            gr.HTML("<div style='background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>⚠️ Groq API key not found in .env file! Please add your Groq API key to the .env file in the format: GROQ_API_KEY=your_api_key_here</div>")
    
    # Model information
    with gr.Accordion("Models Used", open=False):
        gr.HTML(f"""
        <div style="margin: 10px;">
            <h3>Qwen 2.5 (32B)</h3>
            <p><i>Used for: General Chatbot</i></p>
            <p>General-purpose chatbot model with broad knowledge and conversational ability.</p>
            <hr>
            <h3>Deepseek-R1 Distill Llama (70B)</h3>
            <p><i>Used for: Proofreader</i></p>
            <p>Optimized for detailed text analysis and proofreading with advanced language understanding.</p>
            <hr>
            <h3>Deepseek-R1 Distill Qwen (32B)</h3>
            <p><i>Used for: Reasoning Assistant</i></p>
            <p>Specialized for complex reasoning tasks in mathematics, physics, and other scientific domains.</p>
        </div>
        """)
    
    # Main tabs
    with gr.Tabs() as tabs:
        # General Chatbot Tab
        with gr.Tab("General Chatbot"):
            gr.Markdown("""
            This is a general-purpose chatbot powered by Qwen 2.5 (32B) that can answer questions and engage in conversation.
            Type your message below to start chatting!
            """)
            
            # Create chat components manually to avoid ChatInterface compatibility issues
            general_chatbot_messages = gr.Chatbot()
            general_chat_input = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False
            )
            general_chat_clear = gr.Button("Clear Conversation")
            
            # Add example queries
            with gr.Accordion("Examples"):
                gr.Examples(
                    examples=[
                        "Tell me about quantum computing",
                        "What's the difference between machine learning and deep learning?",
                        "Can you help me understand how blockchain works?"
                    ],
                    inputs=general_chat_input
                )
            
            def respond_to_user(message, chat_history):
                response = general_chatbot_interface(message, chat_history)
                chat_history.append((message, response))
                return "", chat_history
            
            def clear_general_chat():
                clear_general_chat_history()
                return [], None
            
            # Set up event handlers
            general_chat_input.submit(
                respond_to_user,
                [general_chat_input, general_chatbot_messages],
                [general_chat_input, general_chatbot_messages]
            )
            
            general_chat_clear.click(
                clear_general_chat,
                None,
                [general_chatbot_messages, general_chat_input]
            )
        
        # Proofreader Tab
        with gr.Tab("Proofreader"):
            gr.Markdown("""
            Upload a document or paste text to proofread. This tool uses Deepseek-R1 Distill Llama (70B) to check for issues and provide suggestions for improvement.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.TextArea(
                        label="Enter text to proofread",
                        placeholder="Paste or type the text you want to proofread here...",
                        lines=15
                    )
                    proofread_button = gr.Button("Proofread Text")
                
                with gr.Column(scale=1):
                    output = gr.TextArea(
                        label="Proofreading Results",
                        lines=15,
                        interactive=False
                    )
            
            proofread_button.click(
                fn=proofread_text,
                inputs=text_input,
                outputs=output
            )
            
            gr.Examples(
                [
                    "I'm planing to go too the store tomorrow to buy some grocerys, but I'm not sure weather I should go in the morning or afternoon.",
                    "The CEO made a statement saying that the company have been doing well and they expect to rise the revenue in comming quarter."
                ],
                inputs=text_input
            )
        
        # Reasoning Assistant Tab
        with gr.Tab("Reasoning Assistant"):
            with gr.Tabs() as reasoning_tabs:
                # General reasoning tab
                with gr.Tab("General Reasoning"):
                    gr.Markdown("""
                    This assistant can help you solve complex problems in mathematics, physics, and other scientific fields.
                    It uses Deepseek-R1 Distill Qwen (32B) to provide step-by-step reasoning to help you understand the solution process.
                    """)
                    
                    # Create chat components manually to avoid ChatInterface compatibility issues
                    reasoning_chatbot_messages = gr.Chatbot()
                    reasoning_chat_input = gr.Textbox(
                        placeholder="Describe your problem...",
                        show_label=False
                    )
                    reasoning_chat_clear = gr.Button("Clear Conversation")
                    
                    # Add example queries
                    with gr.Accordion("Examples"):
                        gr.Examples(
                            examples=[
                                "If a ball is thrown upward with an initial velocity of 20 m/s, how high will it go?",
                                "Prove that the sum of the first n odd numbers equals n²",
                                "What is the derivative of ln(x²+1) with respect to x?"
                            ],
                            inputs=reasoning_chat_input
                        )
                    
                    def respond_to_reasoning(message, chat_history):
                        response = reasoning_chatbot_interface(message, chat_history)
                        chat_history.append((message, response))
                        return "", chat_history
                    
                    def clear_reasoning_chat():
                        clear_reasoning_chat_history()
                        return [], None
                    
                    # Set up event handlers
                    reasoning_chat_input.submit(
                        respond_to_reasoning,
                        [reasoning_chat_input, reasoning_chatbot_messages],
                        [reasoning_chat_input, reasoning_chatbot_messages]
                    )
                    
                    reasoning_chat_clear.click(
                        clear_reasoning_chat,
                        None,
                        [reasoning_chatbot_messages, reasoning_chat_input]
                    )
                
                # Equation Solver tab
                with gr.Tab("Equation Solver"):
                    gr.Markdown("### Equation Solver")
                    gr.Markdown("Enter an equation to solve.")
                    
                    with gr.Row():
                        equation_input = gr.Textbox(
                            label="Enter an equation (e.g., x^2 - 5*x + 6 = 0)",
                            placeholder="Use * for multiplication, ^ for exponents, and = for equality"
                        )
                        variable_input = gr.Textbox(
                            label="Variable to solve for",
                            placeholder="x",
                            value="x"
                        )
                    
                    solve_btn = gr.Button("Solve Equation")
                    equation_output = gr.Textbox(
                        label="Solution",
                        lines=10,
                        interactive=False
                    )
                    
                    solve_btn.click(
                        fn=solve_equation,
                        inputs=[equation_input, variable_input],
                        outputs=equation_output
                    )
                    
                    gr.Examples(
                        [
                            ["x^2 - 5*x + 6 = 0", "x"],
                            ["2*x + 3*y = 10", "y"],
                            ["sin(x) = 0.5", "x"]
                        ],
                        inputs=[equation_input, variable_input]
                    )
                
                # Calculus tab
                with gr.Tab("Calculus"):
                    gr.Markdown("### Calculus Solver")
                    
                    with gr.Row():
                        calculus_op = gr.Radio(
                            ["Derivative", "Integral", "Limit"],
                            label="Select operation",
                            value="Derivative"
                        )
                    
                    with gr.Row():
                        calc_expression = gr.Textbox(
                            label="Enter an expression",
                            placeholder="e.g., x^3 + 2*x^2 - 5*x + 3"
                        )
                        calc_variable = gr.Textbox(
                            label="Variable",
                            value="x"
                        )
                    
                    # Dynamic components for different calculus operations
                    with gr.Group() as integral_group:
                        with gr.Row():
                            lower_bound = gr.Textbox(label="Lower bound (optional)")
                            upper_bound = gr.Textbox(label="Upper bound (optional)")
                    
                    with gr.Group() as limit_group:
                        approach_value = gr.Textbox(
                            label="As variable approaches",
                            value="0"
                        )
                    
                    # Initially hide specific operation inputs
                    integral_group.visible = False
                    limit_group.visible = False
                    
                    # Show/hide components based on selected operation
                    def update_calculus_ui(operation):
                        return {
                            integral_group: operation == "Integral",
                            limit_group: operation == "Limit"
                        }
                    
                    calculus_op.change(
                        fn=update_calculus_ui,
                        inputs=calculus_op,
                        outputs=[integral_group, limit_group]
                    )
                    
                    calc_btn = gr.Button(f"Calculate")
                    calculus_output = gr.Textbox(
                        label="Result",
                        lines=10,
                        interactive=False
                    )
                    
                    def calculus_handler(operation, expression, variable, lower=None, upper=None, approach=None):
                        return calculate_calculus(operation, expression, variable, lower, upper, approach)
                    
                    calc_btn.click(
                        fn=calculus_handler,
                        inputs=[calculus_op, calc_expression, calc_variable, lower_bound, upper_bound, approach_value],
                        outputs=calculus_output
                    )
                    
                    gr.Examples(
                        [
                            ["Derivative", "x^3 + 2*x^2 - 5*x + 3", "x", "", "", ""],
                            ["Integral", "x^2 + 1", "x", "", "", ""],
                            ["Integral", "x^2 + 1", "x", "0", "1", ""],
                            ["Limit", "(sin(x))/x", "x", "", "", "0"]
                        ],
                        inputs=[calculus_op, calc_expression, calc_variable, lower_bound, upper_bound, approach_value]
                    )

# Launch the app
if __name__ == "__main__":
    app.launch()