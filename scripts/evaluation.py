import llama_index
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from typing import List
from llama_index.retrievers.bm25 import BM25Retriever
import pandas as pd
from llama_index.core.schema import TextNode
from Stemmer import Stemmer
from tqdm.notebook import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import *
import cohere
import time
import numpy as np
import json


# RERANKING STUFF
import nest_asyncio

nest_asyncio.apply()
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# LLM
Settings.chunk_size = 512
#################


#################
## Evaluate BM ##
#################
def evaluate_BM(
    dataset,
    top_k=5,
    verbose=False,
    query_rewriting=False,
    half=5,
    reranking=False,
    openai_api_key="",
    cohere_api_key="",
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]

    # Set up the retriever and reranker if needed
    if reranking:
        co = cohere.ClientV2(api_key=cohere_api_key)
    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer("english"),
        language="english",
    )

    eval_results = []
    eval_results_half = []
    # Loop over the queries
    for query_id, query in tqdm(queries.items()):
        # Query rewriting
        if query_rewriting:
            query = query_rewriter(query, openai_api_key)
        # Retrieve the nodes
        retrieved_nodes = retriever.retrieve(query)
        nodes_ = [node.node for node in retrieved_nodes]
        # Reranking
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

        retrieved_ids = [node.node_id for node in nodes_n]
        expected_id = relevant_docs[query_id][0]
        # Calculate the metrics
        is_hit = expected_id in retrieved_ids

        dcg = 0

        for i, id_ in enumerate(retrieved_ids):
            if id_ == expected_id:
                dcg = 1 / np.log2(i + 2)
                break

        eval_result = {
            "is_hit": is_hit,
            "dcg": dcg,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
        # Calculate the metrics on the top k/2
        dcg_half = 0
        is_hit_half = expected_id in retrieved_ids[:half]
        for i, id_ in enumerate(retrieved_ids[:half]):
            if id_ == expected_id:
                dcg_half = 1 / np.log2(i + 2)
                break

        eval_result_half = {
            "is_hit": is_hit_half,
            "dcg": dcg_half,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results_half.append(eval_result_half)

    return eval_results, eval_results_half


##################
## Evaluate bge ##
##################


def evaluate_emb(
    dataset,
    embed_model,
    top_k=10,
    verbose=False,
    reranking=False,
    top_k_before_reranking=20,
    query_rewriting=False,
    half=5,
    openai_api_key="",
    cohere_api_key="",
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    # Create the index
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    # Choose the right retriever, with or without reranking
    if reranking:
        co = cohere.ClientV2(api_key=cohere_api_key)
        retriever = index.as_retriever(similarity_top_k=top_k_before_reranking)
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)

    eval_results = []
    eval_results_half = []

    # Loop over the queries
    for j, (query_id, query) in enumerate(tqdm(queries.items())):
        # Query rewriting
        if query_rewriting:
            query = query_rewriter(query, openai_api_key)

        # Retrieve the nodes
        retrieved_nodes = retriever.retrieve(query)
        nodes_ = [node.node for node in retrieved_nodes]

        # Reranking
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
        # Get the retrieved ids
        retrieved_ids = [node.node_id for node in nodes_n]
        expected_id = relevant_docs[query_id][0]
        # Calculate the metrics
        is_hit = expected_id in retrieved_ids
        dcg = 0

        for i, id_ in enumerate(retrieved_ids):
            if id_ == expected_id:
                dcg = 1 / np.log2(i + 2)
                break

        # Calculate the metrics on the top k/2
        is_hit_half = expected_id in retrieved_ids[:half]
        dcg_half = 0
        for i, id_ in enumerate(retrieved_ids[:half]):
            if id_ == expected_id:
                dcg_half = 1 / np.log2(i + 2)
                break
        # Append the results
        eval_result_half = {
            "is_hit": is_hit_half,
            "dcg": dcg_half,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results_half.append(eval_result_half)
        eval_result = {
            "is_hit": is_hit,
            "dcg": dcg,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)

    return eval_results, eval_results_half


def evaluate_hybrid(
    dataset,
    embed_model,
    top_k=5,
    verbose=False,
    top_k_pool=300,
    alpha=0.7,
    reranking=False,
    top_k_pool_before_reranking=50,
    query_rewriting=False,
    half=5,
    openai_api_key="",
    cohere_api_key="",
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs
    # Create the index
    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    # Choose the right retriever, with or without reranking
    retriever = index.as_retriever(similarity_top_k=top_k_pool)
    if reranking:
        co = cohere.ClientV2(api_key=cohere_api_key)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k_pool,
        stemmer=Stemmer("english"),
        language="english",
    )
    eval_results_half = []
    eval_results = []
    # Loop over the queries
    for j, (query_id, query) in enumerate(tqdm(queries.items())):
        # Query rewriting
        if query_rewriting:
            query = query_rewriter(query, openai_api_key)
        # Retrieve the nodes
        nodes_emb = retriever.retrieve(query)
        bm25_results = bm25_retriever.retrieve(query)
        # Combine the results
        df_emb = data_to_df(nodes_emb)
        df_bm25 = data_to_df(bm25_results)
        df_emb["id"] = df_emb["node"].apply(lambda x: x["text"])
        df_bm25["id"] = df_bm25["node"].apply(lambda x: x["text"])
        df_new = pd.merge(df_bm25, df_emb, on="id", how="outer").fillna(0)
        # Normalize the scores
        df_new["score_x"] = (df_new["score_x"] - df_new["score_x"].min()) / (
            df_new["score_x"].max() - df_new["score_x"].min()
        )
        df_new["score_y"] = (df_new["score_y"] - df_new["score_y"].min()) / (
            df_new["score_y"].max() - df_new["score_y"].min()
        )
        # Combine the scores
        alpha = alpha
        df_new["final_score"] = (
            df_new["score_x"] * (1 - alpha) + df_new["score_y"] * alpha
        )
        df_new.sort_values(by="final_score", ascending=False, inplace=True)
        df_new.reset_index(inplace=True)

        nodes_ = []
        # Choose the top k nodes, if reranking is True choose the amount of nodes before reranking
        if reranking:
            top_k_retrieved = top_k_pool_before_reranking
        else:
            top_k_retrieved = top_k

        for k in range(top_k_retrieved):
            if df_new["score_y"][k] != 0:

                nodes_.append(df_new["node_y"][k])
            else:
                nodes_.append(df_new["node_x"][k])
        # Reranking
        if reranking:
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

        retrieved_ids = [node["id_"] for node in nodes_n]

        expected_id = relevant_docs[query_id][0]
        # Calculate the metrics
        is_hit = expected_id in retrieved_ids
        dcg = 0

        for i, id_ in enumerate(retrieved_ids):
            if id_ == expected_id:
                dcg = 1 / np.log2(i + 2)
                break
        # Calculate the metrics on the top k/2
        dcg_half = 0
        is_hit_half = expected_id in retrieved_ids[:half]
        for i, id_ in enumerate(retrieved_ids[:half]):
            if id_ == expected_id:
                dcg_half = 1 / np.log2(i + 2)
                break

        eval_result_half = {
            "is_hit": is_hit_half,
            "dcg": dcg_half,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results_half.append(eval_result_half)
        eval_result = {
            "is_hit": is_hit,
            "dcg": dcg,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)

    return eval_results, eval_results_half


#######################
## Evaluate Function ##
#######################


def evaluate(
    dataset,
    retriever_model,
    reranking=False,
    top_k=5,
    verbose=False,
    top_k_pool=300,
    alpha=0.7,
    query_rewriting=False,
    top_k_before_reranking=50,
    half=5,
    openai_api_key="",
    cohere_api_key="",
):
    """
    Evaluate the model on the dataset.
    dataset: Question and answer pairs
    retriever_model: "BGE", "BM25", "BGE_Finetuned", "Hybrid_BGE", "Hybrid_Finetuned"
    reranking: True or False
    top_k: Number of documents to retrieve
    verbose: True or False
    top_k_pool: Number of documents to retrieve for the hybrid model before combining
    alpha: Weight for the hybrid model
    query_rewriting: True or False
    top_k_before_reranking: Number of documents to retrieve before reranking (if reranking is True)
    half: Number of documents to consider to measure also top K/2 and dcg on first k/2 elements
    openai_api_key: OpenAI API key
    cohere_api_key: Cohere API key
    """

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection("mydocs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Choose the right model
    if retriever_model == "BGE":
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        eval_results, eval_results_half = evaluate_emb(
            dataset,
            embed_model,
            top_k,
            verbose,
            reranking,
            query_rewriting=query_rewriting,
            top_k_before_reranking=top_k_before_reranking,
            half=top_k // 2,
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
        )

    elif retriever_model == "BM25":
        eval_results, eval_results_half = evaluate_BM(
            dataset,
            top_k,
            verbose,
            query_rewriting,
            half=top_k // 2,
            openai_api_key=openai_api_key,
            reranking=reranking,
            cohere_api_key=cohere_api_key,
        )

    elif retriever_model == "BGE_Finetuned":
        embed_model = "local:../models/BGE_Finetuned"

        eval_results, eval_results_half = evaluate_emb(
            dataset,
            embed_model,
            top_k,
            verbose,
            reranking,
            query_rewriting=query_rewriting,
            top_k_before_reranking=top_k_before_reranking,
            half=top_k // 2,
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
        )

    elif retriever_model == "Hybrid_BGE":
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        eval_results, eval_results_half = evaluate_hybrid(
            dataset,
            embed_model,
            top_k,
            verbose,
            top_k_pool,
            alpha,
            reranking,
            query_rewriting=query_rewriting,
            half=top_k // 2,
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
        )

    elif retriever_model == "Hybrid_Finetuned":
        embed_model = "local:../models/BGE_Finetuned"

        eval_results, eval_results_half = evaluate_hybrid(
            dataset,
            embed_model,
            top_k,
            verbose,
            top_k_pool,
            alpha,
            reranking,
            query_rewriting=query_rewriting,
            half=top_k // 2,
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
        )
    else:
        return "Invalid model"

    # Clculate the metrics on the top k/2
    count_half = 0
    count_dcg_half = 0
    hit_rate_half = 0
    if eval_results_half:
        for res in eval_results_half:
            count_dcg_half += res["dcg"]
            if res["is_hit"]:
                count_half += 1
        hit_rate_half = count_half / len(eval_results_half)
        dcg_half = count_dcg_half / len(eval_results_half)

    # Calculate the metrics on the top k
    count = 0
    count_dcg = 0
    for res in eval_results:
        count_dcg += res["dcg"]
        if res["is_hit"]:
            count += 1
    hit_rate = count / len(eval_results)
    dcg = count_dcg / len(eval_results)
    return {
        "Hit rate on top k": hit_rate,
        "Hit rate on top (k/2)": hit_rate_half,
        "DCG on top k": dcg,
        "DCG on top (k/2)": dcg_half,
    }
