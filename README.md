# RAG AI Document Assistant

A PDF-based question-answering application built using Python, LangChain, Qdrant and NVIDIA Nemotron. Upload PDF documents and ask questions based on their content.

## Live Demo

**[ Live Application](https://pushpendra8668-agentic-rag-ai-assistant-app-lvrvqr.streamlit.app/)**

## Features
- PDF document upload and processing.
- AI-powered question answering.
- Semantic search using vector embeddings.
- Relevant source page references.
- Interactive Streamlit interface.

## Tech Stack
- Python
- Streamlit
- LangChain
- Qdrant
- FastEmbed
- NVIDIA Nemotron
- PyPDF

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/Agentic-RAG-AI-Assistant.git
cd Agentic-RAG-AI-Assistant
pip install -r requirements.txt
```

Create a `.env` file and add your NVIDIA API key:

```env
NVIDIA_API_KEY=your_api_key
```

Start the application:

```bash
python -m streamlit run app.py
```

## Author

**Pushpendra Maurya**  
B.Tech Student | AI & Data Science
