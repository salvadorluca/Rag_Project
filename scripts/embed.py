import llama_index
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from typing import List
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def embed(model, path, ds):
    """
    Embed the documents in the dataset using the specified model.
    model: "BGE" or "BGE_Finetuned"
    path: path to the chroma database
    ds: dataset object
    """

    # Initialize the Chroma client and collection
    chroma_client = chromadb.PersistentClient(path=path)
    chroma_collection = chroma_client.get_or_create_collection("mydocs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Choose the embedding model
    if model == "BGE_Finetuned":
        embed_model = "local:../models/BGE_Finetuned"
    elif model == "BGE":
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    else:
        raise ValueError("The model specified is not valid.")

    # Prepare the documents
    documents = []
    for row in ds["train"]:
        # doc_content = f"Title: {row['title']}\nContent: {row['text']}\nurl: {row['url']}"
        doc_content = [
            f"Title: {row['title']}\nContent: {row['text']}",
            {"url": row["url"]},
        ]
        documents.append(doc_content)
    llama_documents = [Document(text=doc[0], metadata=doc[1]) for doc in documents]

    # Embed the documents
    index = VectorStoreIndex.from_documents(
        llama_documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print("Finished embedding!")
    return None
