import os
import pathlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever(data_dir: str = "../data", persist_directory: str = "./chroma_db"):
    """
    Initializes or loads a local Chroma vector store from the markdown corpus.
    Returns a retriever object for semantic search.
    """
    # Use HuggingFace local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if the vector database already exists to avoid re-indexing
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing vector store from disk...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 6})
        
    print("Building new vector store from markdown corpus. This may take a moment...")
    
    # Load all markdown files recursively
    loader = DirectoryLoader(
        data_dir, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} markdown files.")
    
    # Chunk the documents to fit into context windows better
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks.")
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 6})

if __name__ == "__main__":
    # Quick test if run directly
    retriever = get_retriever()
    results = retriever.invoke("How do I fix a billing issue?")
    print("\n--- Test Results ---")
    for r in results:
        print(f"Source: {r.metadata['source']}")
        print(r.page_content[:150] + "...\n")
