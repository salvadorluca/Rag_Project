from llama_index.core import VectorStoreIndex
import chromadb
import llama_index
import json
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import List
from llama_index.core import Document
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm.notebook import tqdm
import pandas as pd
from llama_index.core import VectorStoreIndex
from Stemmer import Stemmer
from llama_index.retrievers.bm25 import BM25Retriever
import numpy as np
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from utils import data_to_df


def linear_search_alpha(val_dataset, retriever_model, alphas):
    """
    This function is used to find the optimal alpha value for the hybrid retriever.
    It uses a linear search to find the optimal alpha value.
    val_dataset: The validation dataset to be used for evaluation
    retriever_model: The model for which to find the optimal alpha value
    alphas: The list of alpha values to be used for the search
    """

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.get_or_create_collection("mydocs")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Choose the right embedding model based on the retriever model
    if retriever_model == "Hybrid_Finetuned":
        embed_model = "local:../models/BGE_Finetuned"
    elif retriever_model == "Hybrid_BGE":
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    else:
        raise ValueError("The model specified is not valid.")

    # This function is used to evaluate the hybrid retriever, similar to the one used in evaluation.py
    def evaluate_hybrid(
        dataset,
        embed_model,
        top_k=5,
        verbose=False,
        top_k_pool=300,
        alpha=0.7,
    ):
        corpus = dataset.corpus
        queries = dataset.queries
        relevant_docs = dataset.relevant_docs

        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
        index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
        retriever = index.as_retriever(similarity_top_k=top_k_pool)

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k_pool,  # Numero di documenti da selezionare
            stemmer=Stemmer("english"),
            language="english",
        )

        eval_results = []
        for query_id, query in tqdm(queries.items()):
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
            df_new["final_score"] = (
                df_new["score_x"] * (1 - alpha) + df_new["score_y"] * alpha
            )
            df_new.sort_values(by="final_score", ascending=False, inplace=True)
            df_new.reset_index(inplace=True)

            retrieved_ids = []

            for k in range(top_k):
                if df_new["score_y"][k] != 0:

                    retrieved_ids.append(df_new["node_y"][k]["id_"])
                else:
                    retrieved_ids.append(df_new["node_x"][k]["id_"])

            expected_id = relevant_docs[query_id][0]
            is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

            eval_result = {
                "is_hit": is_hit,
                "retrieved": retrieved_ids,
                "expected": expected_id,
                "query": query_id,
            }
            eval_results.append(eval_result)
        return eval_results

    hit_rates = {}

    print("Starting Cycle")
    for alpha in alphas:
        eval_results = evaluate_hybrid(
            val_dataset, embed_model, top_k=5, verbose=True, alpha=alpha
        )

        count = 0
        for res in eval_results:
            if res["is_hit"]:
                count += 1
        hit_rates[alpha] = count / len(eval_results[0])
        print(f"Finished {alpha} with hit rate of {hit_rates[alpha]}")
        # with open(f"alphas_{model_embedding}.json", "w") as json_file:
        #     json.dump(hit_rates, json_file)
    print("Finished")
    return alphas
