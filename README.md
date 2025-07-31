# Basic Multi-Agent

A multi-agent Python application providing several AI-powered assistants for different tasks, built with Gradio. This project is intended for educational and prototype purposes.

## Features

### 1. General Chatbot
- Powered by Qwen 2.5 (32B)
- Engages in general conversation and answers questions across a wide range of topics
- Example queries:
  - "Tell me about quantum computing"
  - "What's the difference between machine learning and deep learning?"
  - "Can you help me understand how blockchain works?"

### 2. Proofreader
- Uses Deepseek-R1 Distill Llama (70B)
- Upload or paste text to receive grammar, spelling, style, and clarity feedback
- Provides overall assessment and suggested corrections
- Example text:
  - "I'm planing to go too the store tomorrow to buy some grocerys..."
  - "The CEO made a statement saying that the company have been doing well..."

### 3. Reasoning Assistant
- Uses Deepseek-R1 Distill Qwen (32B)
- Specialized for complex scientific reasoning (math, physics, etc.)
- Step-by-step solutions for problems, including calculus
- Example queries:
  - "If a ball is thrown upward with an initial velocity of 20 m/s, how high will it go?"
  - "Prove that the sum of the first n odd numbers equals n²"
  - "What is the derivative of ln(x²+1) with respect to x?"

### 4. Equation Solver & Calculus Tools
- Solve equations symbolically given a variable (e.g., `x^2 - 5*x + 6 = 0`)
- Calculate derivatives, integrals, and limits for mathematical expressions

## Models Used

- **Qwen 2.5 (32B)** — General chatbot with broad conversational knowledge
- **Deepseek-R1 Distill Llama (70B)** — Advanced language model for proofreading and text analysis
- **Deepseek-R1 Distill Qwen (32B)** — Scientific reasoning assistant for step-by-step problem solving

## Requirements

- Python 3.8+
- [Gradio](https://gradio.app/) for the UI
- Groq API key (add to `.env` as `GROQ_API_KEY=your_api_key_here`)
- Model dependencies as specified in `requirements.txt`

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/ElarizT/basic-multi-agent.git
    cd basic-multi-agent
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up your `.env` file with the Groq API key:
    ```
    GROQ_API_KEY=your_api_key_here
    ```
4. Run the application:
    ```bash
    python multi_agent_app_gradio.py
    ```
5. Access the Gradio web interface as instructed in the terminal.

## File Support

- Proofreader supports `.txt`, `.pdf`, `.docx`, and `.doc` files.

## License

This project is private and intended for educational or prototype use.

## Author

[ElarizT](https://github.com/ElarizT)
