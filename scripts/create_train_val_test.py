from llama_index.core import Document
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.llms.openai import OpenAI


prompt_to_generate_queries = "You are given a financial news document. Your task is to generate one search query that a user might type when looking for information related to the main argument or central discussion of this document. The query should be: \
    General enough that it could apply to multiple documents on the same topic, not just this one.\
    Reflective of the central issue or debate presented, rather than overly focusing on specific details.\
    Formulated as a natural, open-ended question or topic query that someone unfamiliar with the original text might use.\
    \
    Context information is below.\
    \
    ---------------------\
    {context_str}\
    ---------------------\
    "


# IF YOU WANT TO USE OLLAMA, UNCOMMENT THE FOLLOWING LINES
# from llama_index.llms.ollama import Ollama
# llm = Ollama(model="llama3", request_timeout=120.0)


def create_train_val_test(ds, split="test", openai_key=""):
    """
    Create train, val, or test dataset for the Llama model.
    ds: dataset object
    split: "train", "val", or "test"
    openai_key: OpenAI API key
    """

    llm = OpenAI(
        model="gpt-4o-mini",
        api_key=openai_key,
    )

    documents = []
    # Choose the split based on the input
    if split == "test":
        n = range(len(ds["train"]) - 1000, len(ds["train"]))
    elif split == "train":
        n = range(10000)
    elif split == "val":
        n = range(len(ds["train"]) - 2000, len(ds["train"]) - 1000)
    else:
        raise ValueError("split must be either 'train', 'val' or 'test'")
    # Create the documents
    for row in ds["train"].select(n):
        doc_content = [
            f"Title: {row['title']}\nContent: {row['text']}",
            {"url": row["url"]},
        ]
        documents.append(doc_content)
    train_documents = [Document(text=doc[0], metadata=doc[1]) for doc in documents]
    # Generate the question and answer pairs
    train_dataset = generate_qa_embedding_pairs(
        qa_generate_prompt_tmpl=prompt_to_generate_queries,
        llm=llm,
        nodes=train_documents,
        output_path=f"../{split}_dataset_new.json",
        num_questions_per_chunk=1,
    )
    # The return is something more, the data is already saved in the main directory
    return train_dataset
