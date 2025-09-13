# Basic Multi-Agent (Research Paper Agent)

A unified research pipeline that performs: web research (Gemini 2.5 Flash) → reasoning synthesis (DeepSeek V3.1 via OpenRouter) → proofreading (Gemini 2.5 Flash) → PDF export. Built with Streamlit.

## Features

### Flow
1) Web research: searches the web and fetches top sources
2) Research brief: Gemini 2.5 Flash synthesizes a factual brief with inline citations [n]
3) Reasoning: DeepSeek V3.1 (via OpenRouter) drafts a rigorous research paper in Markdown, keeping [n] markers
4) Proofreading: Gemini 2.5 Flash polishes the draft while preserving structure and citations
5) Export: Download the final paper as a PDF

## Models Used

- **Gemini 2.5 Flash** — Research and proofreading (Google AI Studio)
- **DeepSeek V3.1** — Reasoning/synthesis via OpenRouter

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- API Keys:
  - `GEMINI_API_KEY` from Google AI Studio
  - `OPENROUTER_API_KEY` from openrouter.ai
- Optional overrides: `GEMINI_MODEL` (default `gemini-2.5-flash`), `DEEPSEEK_MODEL` (default `deepseek/deepseek-chat-v3.1:free`)
- Model dependencies in `requirements.txt`

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
3. Create `.env` with keys:
  ```
  GEMINI_API_KEY=your_google_ai_studio_key
  OPENROUTER_API_KEY=your_openrouter_key
  # Optional
  # GEMINI_MODEL=gemini-2.5-flash
  # DEEPSEEK_MODEL=deepseek/deepseek-chat-v3.1:free
  # OPENROUTER_SITE_URL=https://your-site
  # OPENROUTER_SITE_TITLE=YourSiteTitle
  ```
4. Run the application:
  ```bash
  streamlit run app.py
  ```
5. Streamlit will open in your browser (or print a local URL in the terminal).

## File Support

- The app performs web research; you provide a topic (and optional custom search query). No file upload needed.

## License

This project is private and intended for educational or prototype use.

---

Note: The legacy Gradio app (`multi_agent_app_gradio.py`) is kept for reference but is no longer the primary entry point.

## Author

[ElarizT](https://github.com/ElarizT)
