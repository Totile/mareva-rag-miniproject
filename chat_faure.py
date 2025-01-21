from pathlib import Path
from argparse import ArgumentParser

import faiss
import numpy as np
import ollama
from pypdf import PdfReader

def main(knowledge_base_idx: str = "o", N: int = 5):
    model_name = "llama3.2"

    documents = []
    knowledge_base = Path("./Gabriel Fauré.pdf")
    for page in PdfReader(knowledge_base).pages:
        text = page.extract_text()
        if text:
            documents.append(text)
        # print(page.extract_text())


    print(f"start embedding")
    embeddings: np.ndarray = [
        ollama.embed(model=model_name, input=doc)["embeddings"] for doc in documents
    ]
    print([len(emb) for emb in embeddings])
    # print(embeddings)
    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = "Quand es tu né ?"
    # retrieve documents
    print("start retrieving")
    query_embeddings = ollama.embed(model=model_name, input=query)["embeddings"]
    _, indices = index.search(np.array((query_embeddings)), N)
    print(indices)
    retrieved_documents = [documents[i] for i in indices[0]]

    print("infer")
    ollama.create(model="faure", system="Tu es Gabriel Fauré. 'Query' est la question à laquelle tu dois répondre, 'Documents' est les informations sur lesquelles tu peux te fonder", from_=model_name)
    augmented_query = f"Query: {query},\nDocuments: {'; '.join(retrieved_documents)}"
    response = ollama.generate(model=model_name,prompt=augmented_query)
    print(f"used {retrieved_documents}")
    print(response["response"])


if (__name__) == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--knowledge-base", "-k", default="o", help="o: open-ui\nf: Gabriel Fauré")
    parser.add_argument("-N", type=int, default=5)
    args = parser.parse_args()
    main(args.knowledge_base, args.N)
