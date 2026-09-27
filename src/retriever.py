from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==============================
# LOAD PDF
# ==============================

PDF_PATH = "data/documents/machine_learning_notes.pdf"

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"PDF loaded successfully!")
print(f"Number of pages: {len(documents)}")


# ==============================
# SPLIT INTO CHUNKS
# ==============================

print("\nSplitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks created: {len(chunks)}")


# ==============================
# CREATE EMBEDDING MODEL
# ==============================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# ==============================
# CREATE FAISS VECTOR STORE
# ==============================

print("\nCreating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector database created successfully!")


# ==============================
# TEST RETRIEVAL
# ==============================

question = "What is Machine Learning?"

print("\n========================================")
print("QUESTION:")
print(question)
print("========================================")

results = vectorstore.similarity_search(question, k=3)

print("\n========== RETRIEVED CHUNKS ==========")

for i, result in enumerate(results, start=1):

    print(f"\n----- CHUNK {i} -----")
    print(result.page_content)

    print("\nMetadata:")
    print(result.metadata)


# ==============================
# SAVE VECTOR DATABASE
# ==============================

VECTORSTORE_PATH = "vectorstore/faiss_index"

vectorstore.save_local(VECTORSTORE_PATH)

print("\n========================================")
print("Vector database saved successfully!")
print("========================================")