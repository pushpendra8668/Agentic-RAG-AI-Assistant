from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def get_embeddings():

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embeddings