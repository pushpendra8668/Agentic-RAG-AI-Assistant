import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_llm():

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in .env file")

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500
    )

    return llm


def create_rag_chain():

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an intelligent AI document assistant.

            Answer questions using ONLY the provided context.

            If the answer is not available in the context,
            clearly say that you don't know.

            Context:
            {context}
            """
        ),
        ("human", "{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain


def generate_answer(retriever, rag_chain, question):

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "No relevant information found."

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    sources = [
        doc.metadata.get("page", "Unknown") + 1
        if isinstance(doc.metadata.get("page"), int)
        else "Unknown"
        for doc in retrieved_docs
    ]

    return answer, sources