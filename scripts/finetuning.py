from llama_index.finetuning import SentenceTransformersFinetuneEngine


def finetune_BGE(train, val, path="BGE_Finetuned"):
    """
    Finetune the BGE model on the given dataset
    """
    finetune_engine = SentenceTransformersFinetuneEngine(
        train,
        model_id="BAAI/bge-small-en",
        model_output_path=path,
        val_dataset=val,
    )

    finetune_engine.finetune()
