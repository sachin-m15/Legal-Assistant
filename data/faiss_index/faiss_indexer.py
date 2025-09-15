import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
DOCS_PATH = "data/indian_law_docs"
FAISS_INDEX_PATH = "data/faiss_index"


def create_faiss_index():
    """
    Processes legal documents, creates embeddings, and saves a FAISS index.
    """
    print("Starting the FAISS index creation process...")
    documents = []

    # 1. Load documents from the specified directory
    if not os.path.exists(DOCS_PATH):
        print(
            f"Error: Directory '{DOCS_PATH}' not found. Please add legal documents to this folder."
        )
        return

    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCS_PATH, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print(
            f"No text documents found in '{DOCS_PATH}'. Please add your legal documents."
        )
        return

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")

    # 2. Split documents into smaller, manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} chunks. Generating embeddings...")

    # 3. Create embeddings using a pre-trained model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create the FAISS index and save it locally
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)

    print("FAISS index created successfully!")
    print(f"Index saved at: {FAISS_INDEX_PATH}")
    print("You can now run main_app.py to use the application.")


if __name__ == "__main__":
    create_faiss_index()
