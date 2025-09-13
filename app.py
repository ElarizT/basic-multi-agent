import os
import io
import textwrap
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import requests
import google.generativeai as genai
from openai import OpenAI
from fpdf import FPDF


# ===== Environment =====
load_dotenv()


# ===== Models & API Keys =====
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat-v3.1:free")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://localhost")
OPENROUTER_SITE_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "ResearchAgent")


# ===== Helpers: Web Search & Fetch =====
def web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                # r keys: title, href, body
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        results.append({"title": "Search error", "url": "", "snippet": str(e)})
    return results


def fetch_page_text(url: str, timeout: int = 10, max_chars: int = 8000) -> str:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        # Normalize whitespace
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text[:max_chars]
    except Exception as e:
        return f"[ERROR fetching {url}: {e}]"


# ===== Helpers: Gemini (Research & Proofread) =====
def get_gemini_model():
    if not GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        return None


def gemini_research(query: str, pages: List[Dict[str, str]]) -> str:
    model = get_gemini_model()
    if not model:
        return "Error: Gemini not configured (missing or invalid GEMINI_API_KEY)."

    # Prepare context
    sources_block = []
    for i, p in enumerate(pages, start=1):
        url = p.get("url", "")
        title = p.get("title", "")
        text = p.get("text", "")
        sources_block.append(f"[{i}] {title} — {url}\n{textwrap.shorten(text, width=1200, placeholder='…')}")
    sources_str = "\n\n".join(sources_block)

    prompt = f"""
Act as a web research agent. Using the provided web content, create a concise, factual research brief about the topic:

Topic: {query}

Requirements:
- Extract key facts and findings from the sources.
- Include inline citation markers like [1], [2] matching the numbered sources below.
- Note consensus vs disagreement; be cautious with claims, avoid hallucinations.
- Return Markdown with sections: Overview, Key Points (bulleted), Notable Quotes (if any), Sources (list with [n] and URL).

Sources (numbered):
{sources_str}
"""

    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or str(resp)
    except Exception as e:
        return f"Error from Gemini research: {e}"


def gemini_proofread(markdown_text: str) -> str:
    model = get_gemini_model()
    if not model:
        return "Error: Gemini not configured (missing or invalid GEMINI_API_KEY)."
    prompt = f"""
You are a professional scientific editor. Proofread and lightly edit the following research paper for grammar, clarity, and flow while preserving meaning and citations. Keep Markdown structure and citation markers like [1], [2].

Return only the improved Markdown.

---
{markdown_text}
"""
    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or str(resp)
    except Exception as e:
        return f"Error from Gemini proofreading: {e}"


# ===== Helpers: DeepSeek via OpenRouter (Reasoning) =====
def get_openrouter_client():
    if not OPENROUTER_API_KEY:
        return None
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        return client
    except Exception:
        return None


def deepseek_reason(research_markdown: str, topic: str, audience: str, length_words: int, citation_style: str) -> str:
    client = get_openrouter_client()
    if not client:
        return "Error: OpenRouter not configured (missing or invalid OPENROUTER_API_KEY)."

    system = (
        "You are a meticulous scientific reasoning agent. Synthesize a rigorous research paper based strictly "
        "on the provided research brief. Avoid hallucinations; if evidence is insufficient, say so. Maintain "
        "inline citation markers [n] referring to the research brief's numbered sources. Return Markdown with "
        "sections: Title, Abstract, Introduction, Method/Approach (if applicable), Main Discussion, Limitations, Conclusion, References."
    )

    user = f"""
Topic: {topic}
Audience: {audience}
Target length: ~{length_words} words
Citation style: {citation_style} (format references but keep inline markers like [n])

Research Brief (Markdown with [n] citations):
---
{research_markdown}
"""

    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_SITE_TITLE,
            },
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error from DeepSeek reasoning: {e}"


# ===== PDF generation =====
def _ensure_unicode_fonts() -> Optional[dict]:
    """Ensure Unicode TTF fonts are available locally; download if missing.

    Returns a dict with paths for 'regular' and 'bold' if successful; otherwise None.
    """
    try:
        import os
        import pathlib
        import requests

        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        pathlib.Path(fonts_dir).mkdir(parents=True, exist_ok=True)

        dejavu_regular = os.path.join(fonts_dir, "DejaVuSans.ttf")
        dejavu_bold = os.path.join(fonts_dir, "DejaVuSans-Bold.ttf")

        url_regular = "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans.ttf"
        url_bold = "https://github.com/dejavu-fonts/dejavu-fonts/raw/version_2_37/ttf/DejaVuSans-Bold.ttf"

        def download_if_missing(path: str, url: str):
            if not os.path.exists(path):
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)

        download_if_missing(dejavu_regular, url_regular)
        download_if_missing(dejavu_bold, url_bold)
        return {"regular": dejavu_regular, "bold": dejavu_bold}
    except Exception:
        return None


def _ascii_fallback(text: str) -> str:
    # Replace common Unicode punctuation with ASCII fallbacks
    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "•": "*",
        "→": "->",
        "←": "<-",
        "×": "x",
        "≥": ">=",
        "≤": "<=",
        "±": "+/-",
        "®": "(R)",
        "©": "(C)",
        "™": "(TM)",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_pdf(markdown_text: str, title: str) -> bytes:
    # Unicode-capable PDF rendering using fpdf2 and DejaVu fonts
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    fonts = _ensure_unicode_fonts()
    if fonts:
        # Register Unicode fonts
        pdf.add_font("DejaVu", "", fonts["regular"], uni=True)
        pdf.add_font("DejaVu", "B", fonts["bold"], uni=True)
        title_text = title
        body_text = markdown_text
        header_font = ("DejaVu", "B", 16)
        meta_font = ("DejaVu", "", 10)
        body_font = ("DejaVu", "", 12)
    else:
        # Fallback to core fonts with ASCII-only replacement
        title_text = _ascii_fallback(title)
        body_text = _ascii_fallback(markdown_text)
        header_font = ("Helvetica", "B", 16)
        meta_font = ("Helvetica", "", 10)
        body_font = ("Times", "", 12)

    pdf.set_font(*header_font)
    pdf.multi_cell(0, 10, title_text)
    pdf.ln(2)
    pdf.set_font(*meta_font)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.ln(2)
    pdf.set_font(*body_font)
    for para in body_text.split("\n\n"):
        for line in textwrap.wrap(para, width=110):
            pdf.multi_cell(0, 6, line)
        pdf.ln(2)
    out = pdf.output(dest="S").encode("latin1", errors="ignore")
    return out


# ===== Streamlit UI =====
st.set_page_config(page_title="Research Paper Agent", layout="wide")

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 0.75rem;">
        <h1>Research Paper Agent</h1>
        <p>Web research (Gemini 2.5 Flash) → Reasoning (DeepSeek V3.1) → Proofreading (Gemini 2.5 Flash) → PDF</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Models: Gemini={GEMINI_MODEL} · DeepSeek={DEEPSEEK_MODEL}")

with st.form("research_form"):
    topic = st.text_input("Research topic", placeholder="e.g., Quantum error correction for NISQ devices", key="topic")
    audience = st.text_input("Audience (e.g., graduate students, general technical)", value="graduate students")
    query_override = st.text_input("Optional: custom web search query", placeholder="Leave blank to auto-use the topic")
    seed_urls_text = st.text_area("Optional: seed URLs (one per line)", placeholder="https://example.com/article\nhttps://arxiv.org/abs/1234.5678")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_sources = st.slider("Max sources", 3, 12, 6)
    with col2:
        length_words = st.slider("Target length (words)", 600, 3000, 1200, step=100)
    with col3:
        citation_style = st.selectbox("Citation style", ["APA", "MLA", "IEEE", "Chicago"], index=0)
    run = st.form_submit_button("Generate Research Paper", use_container_width=True)

if run:
    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    # Step 1: Web search
    with st.status("Step 1/4: Searching the web…", expanded=False) as status:
        q = query_override.strip() or topic
        results = web_search(q, max_results=max_sources)
        st.write({"query": q, "results_found": len(results)})

        # Collect URLs from search + seeds
        urls: List[str] = []
        for r in results:
            u = (r.get("url") or r.get("href") or "").strip()
            if u:
                urls.append(u)
        # Seed URLs from textarea
        if seed_urls_text.strip():
            for line in seed_urls_text.splitlines():
                u = line.strip()
                if u:
                    urls.append(u)
        # Deduplicate, preserve order
        seen = set()
        dedup_urls = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                dedup_urls.append(u)

        # Fetch pages
        pages: List[Dict[str, str]] = []
        for url in dedup_urls[:max_sources]:
            text = fetch_page_text(url)
            pages.append({"title": url, "url": url, "text": text})

        # Fallback: Wikipedia if nothing fetched
        if not pages:
            slug = topic.strip().replace(" ", "_")
            wiki_urls = [
                f"https://en.wikipedia.org/wiki/{slug}",
                f"https://simple.wikipedia.org/wiki/{slug}",
            ]
            for url in wiki_urls:
                text = fetch_page_text(url)
                if text and not text.startswith("[ERROR"):
                    pages.append({"title": url, "url": url, "text": text})
                    break

        st.write({"pages_collected": len(pages)})
        status.update(label="Step 1 complete", state="complete")

    # Step 2: Gemini research brief
    with st.status("Step 2/4: Building research brief (Gemini)…", expanded=False) as status:
        if not pages:
            st.error("No sources could be fetched. Please add seed URLs or refine your search query, then try again.")
            st.stop()
        research_md = gemini_research(topic, pages)
        st.markdown("### Research Brief")
        st.markdown(research_md)
        status.update(label="Step 2 complete", state="complete")

    # Step 3: DeepSeek reasoning to synthesize paper
    with st.status("Step 3/4: Reasoning and drafting (DeepSeek)…", expanded=False) as status:
        draft_md = deepseek_reason(research_md, topic, audience, length_words, citation_style)
        st.markdown("### Draft Paper (pre-proofread)")
        st.markdown(draft_md)
        status.update(label="Step 3 complete", state="complete")

    # Step 4: Gemini proofreading
    with st.status("Step 4/4: Proofreading (Gemini)…", expanded=False) as status:
        final_md = gemini_proofread(draft_md)
        st.markdown("## Final Paper")
        st.markdown(final_md)
        status.update(label="Step 4 complete", state="complete")

    # PDF download
    pdf_bytes = generate_pdf(final_md, title=f"{topic} — Research Paper")
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"research_paper_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
