from pathlib import Path
from argparse import ArgumentParser

import faiss
import numpy as np
from ollama import embed, generate
from pypdf import PdfReader

def main(N: int = 5):
    model_name = "llama3.2"
    knowledge_base = Path("./openui-docs")

    documents = []
    for p in knowledge_base.rglob("*"):
        if p.is_file() & (p.suffix == ".md"):
            documents.append(p.read_text())
            print(p.name)
    print(f"found {len(documents)} documents")

    print(f"start embedding")
    embeddings: np.ndarray = [
        embed(model=model_name, input=doc)["embeddings"] for doc in documents
    ]
    # print(embeddings)
    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = "How do I setup a RAG model ?"
    # retrieve documents
    print("start retrieving")
    N = 5
    query_embeddings = embed(model=model_name, input=query)["embeddings"]
    _, indices = index.search(np.array((query_embeddings)), N)
    print(indices)
    retrieved_documents = [documents[i] for i in indices[0]]

    print("infer")
    augmented_query = f"Query: {query},\nDocuments: {'; '.join(retrieved_documents)}"
    response = generate(model=model_name,prompt=augmented_query)
    print(f"used {retrieved_documents}")
    print(response["response"])


if (__name__) == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-N", type=int, default=5)
    args = parser.parse_args()
    main(args.N)
