from typing import List
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.core import PromptTemplate
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def data_to_df(nodes: List[TextNode]):
    """Convert a list of TextNode objects to a pandas DataFrame."""
    return pd.DataFrame([node.dict() for node in nodes])


def choose_embed_model(model_embedding):
    if model_embedding == "BGE_Finetuned" or model_embedding == "Hybrid_Finetuned":
        embed_model = "local:../models/BGE_Finetuned"
    elif model_embedding == "BGE" or model_embedding == "Hybrid_BGE":
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    else:
        raise ValueError("The model specified is not valid.")
    return embed_model


def create_document_dictionary(nodes):
    """
    Converts a list of nodes to a list of dictionaries in a specific format.

    Args:
        nodes (list): List of nodes, where each node contains keys such as 'text', 'metadata', etc.

    Returns:
        list: A list of dictionaries with formatted 'text' fields.
    """
    document_list = []
    for node in nodes:

        text_content = getattr(node, "text", "")

        document_dict = {"text": text_content}

        # Append the dictionary to the final list
        document_list.append(document_dict)

    return document_list


def create_document_dictionary_hybrid(nodes):
    """
    Converts a list of nodes to a list of dictionaries in a specific format.

    Args:
        nodes (list): List of nodes, where each node contains keys such as 'text', 'metadata', etc.

    Returns:
        list: A list of dictionaries with formatted 'text' fields.
    """
    document_list = []
    for node in nodes:

        text_content = node.get("text", "")

        document_dict = {"text": text_content}

        # Append the dictionary to the final list
        document_list.append(document_dict)

    return document_list


def return_results(results, document_list):
    indices = []
    for result in results.results:
        indices.append(result.index)
    return indices


def query_rewriter(query: str, openai_api_key: str):
    """
    Rewrite a given query using OpenAI's GPT-4 model.
    args:
        query (str): The input query to be rewritten.
        openai_api_key (str): The OpenAI API key.
    returns:
        str: The rewritten query.
    """
    client = OpenAI(api_key=openai_api_key)
    query_gen_str2 = """You are a helpful assistant generating an efficient search query based on a \ 
    input query. Keep in mind this is a finance context, use the appropriate terminology. Do not include any introductory text. \ 
    Output ONLY the search query directly. 
    Query: {query} Final Query: """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional financial research query optimization assistant.",
            },
            {"role": "user", "content": query_gen_str2.format(query=query)},
        ],
        max_tokens=250,
        temperature=0.7,
        top_p=0.9,
    )

    # Extract and return the generated query
    generated_query = response.choices[0].message.content.strip()

    return generated_query
