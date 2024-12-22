# Enhancing Financial News Retrieval with Embedding Fine-Tuning and Advanced Retrieval-Augmented Generation Techniques

This repository implements a **Retrieval-Augmented Generation** (RAG) pipeline testing different models for retrieval including BM25, BGE and a finetuned BGE model on a financial news dataset. The system supports hybrid, embedding-based, and fine-tuned retrieval techniques, integrating evaluation methods to measure retrieval performance.

## Project Structure

The repository is organized as follows:

├── scripts/\
│   ├── models.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Model definitions and utilities for retrieval.\
│   ├── finetuning.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Fine-tuning the BGE model.\
│   ├── utils.py          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Utility functions (data conversion, query processing, etc.).\
│   ├── evaluation.py     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      # Evaluation of retrieval models.\
│   ├── find_alpha.py   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        # Alpha Linear search for hybrid retrieval optimization.\
│   ├── create_train_val_test.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Generate train, validation, and test datasets.\
│   ├── embed.py             &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # Embedding documents into the vector store.\
├── embeddings/            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     # Directory containing embeddings data downloadable [here](https://epflch-my.sharepoint.com/:f:/g/personal/marco_giuliano_epfl_ch/EomT1E_2ZaFCkB7zpx-jDAMBxLz6UP8NGtdBNvK10RaYHQ?e=w8s5QY).\
│   ├── save    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Directory containing BGE embeddings data.\
│   ├── save_finetuned   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  # Directory containing finetuned BGE embeddings data.\
├── models/           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          # Contains the fine-tuned BGE model downloadable [here](https://epflch-my.sharepoint.com/:f:/r/personal/marco_giuliano_epfl_ch/Documents/RAG?csf=1&web=1&e=gFDKsS).\
│   ├── BGE_Finetuned   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        # Fine-tuned BGE model saved here.\
├── main.ipynb          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        # Notebook for end-to-end implementation and experimentation.\
├── RAG.ipynb          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;         # Notebook for querying the RAG model.\
├── train_dataset_new.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Train dataset with question and answer pairs.\
├── val_dataset_new.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Validation dataset with question and answer pairs.\
├── test_dataset_new.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Test dataset with question and answer pairs.\
├── requirements.txt         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # Python dependencies.\
├── req_clean.txt         &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   # Python dependencies for windows, explained in point 2.\
└── README.md           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        # Project documentation.



## Main Files

1. **`main.ipynb`**  
   - This notebook provides an end-to-end implementation of the pipeline, including:
     - Evaluation of the different retrieval methods.
     - Creation of train and evaluation set.
     - Finetuning the BGE model.

2. **`RAG.ipynb`**  
   - Implements the **RAG pipeline** using
   - Includes query answering using the retrieved documents and a language model.

## Data preparation and finetuning of the embedding model

- The generation of the train val and test datasets can be done in the **`main.ipynb`** notebook, through the function create_train_val_test available in **`create_train_val_test.py`**.
- The finetuning of the model can be done in  **`main.ipynb`** with the function finetune_BGE available in **`finetuning.py`**.
- The embedding of the dataset can be done in the **`RAG.ipynb`** notebook through the function embed from **`embed.py`**. We advise you to download them though, as explained in point 5.


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/CS-433/ml-project-2-iragazzi
```
### 2. Install dependencies
If you are on Mac use
```bash
pip install -r requirements.txt
```
If you are on Windows the library streamlit-chromadb-connection has a dependency not available on windows (uv-loop).\
Please install streamlit-chromadb-connection separately without its dependencies 
```bash
pip install streamlit-chromadb-connection --no-deps
```
Afterwards install its dependencies and the other requirements that are all present in the file req-clean.txt
```bash
pip install -r req-clean.txt
```

### 3. Download train val and test data

Download the **`train_dataset_new.json`** **`val_dataset_new.json`** and **`test_dataset_new.json`** data from [here](https://epflch-my.sharepoint.com/:f:/g/personal/marco_giuliano_epfl_ch/EomT1E_2ZaFCkB7zpx-jDAMBxLz6UP8NGtdBNvK10RaYHQ?e=w8s5QY) and put them in your directory.
### 4. Download the Finetuned model
Download the folder **`models/`** from [here](https://epflch-my.sharepoint.com/:f:/g/personal/marco_giuliano_epfl_ch/EomT1E_2ZaFCkB7zpx-jDAMBxLz6UP8NGtdBNvK10RaYHQ?e=w8s5QY) and put it in your directory.

### 5. Download the embeddings
From the same [link](https://epflch-my.sharepoint.com/:f:/g/personal/marco_giuliano_epfl_ch/EomT1E_2ZaFCkB7zpx-jDAMBxLz6UP8NGtdBNvK10RaYHQ?e=w8s5QY) download the embeddings and put it in your directory.\
This is needed if you want to do RAG in the **`RAG.ipynb`**  notebook without embedding the data yourself, but you can still go trough all the **`main.ipynb`**  without them. \
If you want to try the RAG with a method using BGE, you can just download **`embeddings/save/`**.\
If you want to use BGE Finetuned you need **`embeddings/save_finetuned/`**.\
You can also embed the data on your own as explained in **`RAG.ipynb`** .

### 6. Try out the notebooks
You can now reproduce the evaluations we have done with the **`main.ipynb`**, and you can also ask the RAG model some questions in the **`RAG.ipynb`** notebook!

## Requirements

- Python 3.8+
- Jupyter Notebook
- OpenAI API key (optional for query rewriting, and for asking the model question in the **`RAG.ipynb`** notebook)
- Cohere API key (optional for reranking)
