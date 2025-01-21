from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
from ollama import embed, generate

def main():
    model_name = "llama3.2"

    with open(
        "../openui-docs/docs/getting-started/advanced-topics/env-configuration.md", "r"
    ) as f:
        documents = [f.read()]

    print(f"start embedding")
    embeddings: np.ndarray = [
        embed(model=model_name, input=doc)["embeddings"] for doc in documents
    ]
    # print(embeddings)
    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = "How do I setup the environment ?"

    # retrieve documents
    print("start retrieving")
    N = 1
    query_embeddings = embed(model=model_name, input=query)["embeddings"]
    _, indices = index.search(np.array((query_embeddings)), N)
    print(indices)
    retrieved_documents = [documents[i] for i in indices[0]]

    print("infer")
    augmented_query = f"Query: {query},\nDocuments: {'; '.join(retrieved_documents)}"
    response = generate(model=model_name,prompt=augmented_query)
    print(response["response"])


if (__name__) == "__main__":
    main()
