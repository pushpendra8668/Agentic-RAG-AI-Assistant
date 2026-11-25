import streamlit as st
import tempfile
import os

from src.document_loader import load_pdf
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, get_retriever
from src.rag_pipeline import create_rag_chain, generate_answer


st.set_page_config(
    page_title="Intelligent RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Intelligent RAG AI Assistant")

st.write(
    "Upload your PDF documents and ask questions using AI."
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar
with st.sidebar:
    st.header("📂 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("Process Document"):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")

        else:
            with st.spinner("Processing your document..."):

                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    documents = load_pdf(temp_path)

                    embeddings = get_embeddings()

                    vectorstore = create_vector_store(
                        documents,
                        embeddings
                    )

                    st.session_state.vectorstore = vectorstore

                    st.session_state.rag_chain = create_rag_chain()

                    st.session_state.messages = []

                    st.success("Document processed successfully!")

                except Exception as e:
                    st.error(f"Processing failed: {e}")

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by LangChain, Qdrant and NVIDIA AI")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat interface
question = st.chat_input("Ask something about your PDF...")

if question:

    if st.session_state.vectorstore is None:
        st.warning("Please upload and process a PDF first.")

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Finding answer..."):

                try:
                    retriever = get_retriever(
                        st.session_state.vectorstore
                    )

                    response, sources = generate_answer(
                        retriever,
                        st.session_state.rag_chain,
                        question
                    )

                    st.markdown(response)

                    st.markdown("**Sources:**")

                    st.write(
                        ", ".join(
                            f"Page {page}" for page in sources
                        )
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
st.markdown(
    """
    <div style="text-align:center; color:gray; padding:15px;">
        Created by <b>Pushpendra Maurya</b> | © 2026
    </div>
    """,
    unsafe_allow_html=True
)