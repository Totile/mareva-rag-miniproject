#%%
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import ConversationalRetrievalChain


from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph

from langchain import hub
import os

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
#define llm
llm = OllamaLLM(model="llama3.2")
#define prompt
prompt = hub.pull("rlm/rag-prompt")

#load pdf
loader = PyPDFLoader("./gabriel01.pdf")
documents = loader.load()

#split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,add_start_index=True)
chunks = text_splitter.split_documents(documents)
print(f"Book split into {len(chunks)} sub-documents.")


#embeddings
embeddings_model = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings_model)
document_ids = vector_store.add_documents(documents=chunks)
print("embedded")
#%%

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "C'est qui Gabriel Faure"})
print(response["answer"])
#%%
print("Chatbot is ready! Type 'exit' to end the conversation.")

while True:
    question = input("You: ")
    if question.lower() == 'exit':
        print("Goodbye!")
        break

    response = graph.invoke({"question": question})
    answer = response["answer"]

    print(f"Bot: {answer}")
# %%
