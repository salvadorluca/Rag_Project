import llama_index
from typing import List
from llama_index.retrievers.bm25 import BM25Retriever
import pandas as pd
from llama_index.core.schema import TextNode, NodeWithScore
from Stemmer import Stemmer
from utils import *
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
import cohere


def bm25_retriever(query, nodes, top_k=10, reranking=False, cohere_api_key=None):
    """
    BM25 retriever that uses BM25 algorithm.
    """
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer("english"),
        language="english",
    )
    retrieved_nodes = bm25_retriever.retrieve(query)
    if reranking:
        co = cohere.ClientV2(api_key=cohere_api_key)
    nodes_ = [node.node for node in retrieved_nodes]

    if reranking:
        document_list = create_document_dictionary(nodes_)
        results = co.rerank(
            query=query,
            documents=document_list,
            top_n=top_k,
            model="rerank-english-v3.0",
        )
        indices = return_results(results, document_list)

        nodes_n = [nodes_[i] for i in indices]
    else:
        nodes_n = nodes_
    new_nodes = [NodeWithScore(node=n) for n in nodes_n]
    return new_nodes


def emb_retriever(
    index,
    query,
    top_k=10,
    query_rewriting=False,
    openai_api_key=None,
    reranking=False,
    top_k_before_reranking=50,
    cohere_api_key=None,
):
    """
    Embedding retriever that uses BGE/BGE Finetuned embeddings.
    """
    if reranking:
        co = cohere.ClientV2(api_key=cohere_api_key)
        retriever = index.as_retriever(similarity_top_k=top_k_before_reranking)
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)

    if query_rewriting:
        query = query_rewriter(query, openai_api_key)

    retrieved_nodes = retriever.retrieve(query)
    nodes_ = [node.node for node in retrieved_nodes]

    if reranking:
        document_list = create_document_dictionary(nodes_)
        results = co.rerank(
            query=query,
            documents=document_list,
            top_n=top_k,
            model="rerank-english-v3.0",
        )
        indices = return_results(results, document_list)

        nodes_n = [nodes_[i] for i in indices]
    else:
        nodes_n = nodes_
    new_nodes = [NodeWithScore(node=n) for n in nodes_n]
    return new_nodes


def hybrid_retriever(
    index,
    query,
    nodes,
    top_k=10,
    alpha=0.9,
    query_rewriting=False,
    openai_api_key=None,
    reranking=False,
    top_k_pool=300,
    top_k_before_reranking=50,
    cohere_api_key=None,
):
    """
    Hybrid retriever that combines BM25 and BGE/BGE Finetuned retrievers.
    """

    retriever = index.as_retriever(similarity_top_k=top_k_pool)

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k_pool,
        stemmer=Stemmer("english"),
        language="english",
    )

    if query_rewriting:
        query = query_rewriter(query, openai_api_key)
    nodes_emb = retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    df_emb = data_to_df(nodes_emb)
    df_bm25 = data_to_df(bm25_results)

    df_emb["id"] = df_emb["node"].apply(lambda x: x["text"])
    df_bm25["id"] = df_bm25["node"].apply(lambda x: x["text"])
    df_new = pd.merge(df_bm25, df_emb, on="id", how="outer").fillna(0)
    df_new["score_x"] = (df_new["score_x"] - df_new["score_x"].min()) / (
        df_new["score_x"].max() - df_new["score_x"].min()
    )
    df_new["score_y"] = (df_new["score_y"] - df_new["score_y"].min()) / (
        df_new["score_y"].max() - df_new["score_y"].min()
    )
    alpha = alpha
    df_new["final_score"] = df_new["score_x"] * (1 - alpha) + df_new["score_y"] * alpha
    df_new.sort_values(by="final_score", ascending=False, inplace=True)
    df_new.reset_index(inplace=True)

    nodes_ = []
    if reranking:
        top_k_retrieved = top_k_before_reranking
    else:
        top_k_retrieved = top_k
    for k in range(top_k_retrieved):
        if df_new["score_y"][k] != 0:
            df_new["node_y"][k]["score"] = df_new["final_score"][k]
            nodes_.append(df_new["node_y"][k])
        else:
            df_new["node_x"][k]["score"] = df_new["final_score"][k]
            nodes_.append(df_new["node_x"][k])

    if reranking:
        co = cohere.ClientV2(api_key=cohere_api_key)
        document_list = create_document_dictionary_hybrid(nodes_)
        results = co.rerank(
            query=query,
            documents=document_list,
            top_n=top_k,
            model="rerank-english-v3.0",
        )
        indices = return_results(results, document_list)

        nodes_n = [nodes_[i] for i in indices]
    else:
        nodes_n = nodes_
    new_nodes = [
        NodeWithScore(
            node=TextNode(text=n["text"], id=n["id_"], metadata=n["metadata"]),
            score=n["score"],
        )
        for n in nodes_n
    ]
    return new_nodes


def retrieve(
    model,
    query,
    index=None,
    nodes=None,
    top_k=10,
    alpha=0.9,
    query_rewriting=False,
    openai_api_key=None,
    reranking=False,
    top_k_pool=300,
    top_k_before_reranking=50,
    cohere_api_key=None,
):
    """
    Retrieve documents based on the model specified.
    model:
        - BGE: Retrieve documents using BGE embeddings.
        - BGE_Finetuned: Retrieve documents using BGE Finetuned embeddings.
        - Hybrid_BGE: Retrieve documents using a hybrid model that combines BM25 and BGE embeddings.
        - Hybrid_Finetuned: Retrieve documents using a hybrid model that combines BM25 and BGE Finetuned embeddings.
        - BM25: Retrieve documents using BM25 algorithm.
    query: Query to retrieve documents.
    index: Index object.
    nodes: List of nodes (Needed for BM25 and hybrid retrieval).
    top_k: Number of documents to retrieve.
    alpha: Weight for the hybrid model.
    query_rewriting: Whether to rewrite the query.
    openai_api_key: OpenAI API key (Needed for query rewriting).
    reranking: Whether to rerank the documents.
    top_k_pool: Number of documents to retrieve for the hybrid model.
    top_k_before_reranking: Number of documents to retrieve before reranking.
    cohere_api_key: Cohere API key (Needed for reranking).
    """
    if model == "BGE" or model == "BGE_Finetuned":
        return emb_retriever(
            index,
            query,
            top_k,
            query_rewriting,
            openai_api_key,
            reranking,
            top_k_before_reranking,
            cohere_api_key,
        )
    elif model == "Hybrid_BGE" or model == "Hybrid_Finetuned":
        return hybrid_retriever(
            index,
            query,
            nodes,
            top_k,
            alpha,
            query_rewriting,
            openai_api_key,
            reranking,
            top_k_pool,
            top_k_before_reranking,
            cohere_api_key,
        )
    elif model == "BM25":
        return bm25_retriever(query, nodes, top_k, reranking, cohere_api_key)
    else:
        raise ValueError("The model specified is not valid.")


def answer(query, nodes):
    """
    Answer the query based on the nodes specified.
    """
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)

    response = response_synthesizer.synthesize(query, nodes=nodes)
    return response
