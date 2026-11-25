from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore


def create_vector_store(documents, embeddings):

    # Create in-memory Qdrant database
    client = QdrantClient(location=":memory:")

    # Calculate embedding dimensions
    vector = embeddings.embed_query("Test embedding")

    vector_size = len(vector)

    # Create collection
    client.create_collection(
        collection_name="rag_documents",
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    # Initialize vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_documents",
        embedding=embeddings
    )

    # Store document chunks
    vectorstore.add_documents(documents)

    return vectorstore


def get_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever
