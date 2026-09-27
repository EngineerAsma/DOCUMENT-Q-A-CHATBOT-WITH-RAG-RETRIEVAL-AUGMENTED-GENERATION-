from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 1. PDF PATH
# ==========================================

PDF_PATH = "data/documents/machine_learning_notes.pdf"


# ==========================================
# 2. LOAD PDF
# ==========================================

print("Loading PDF...")

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"PDF loaded successfully!")
print(f"Number of pages: {len(documents)}")


# ==========================================
# 3. SPLIT DOCUMENT INTO CHUNKS
# ==========================================

print("\nSplitting document into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks created: {len(chunks)}")


# ==========================================
# 4. DISPLAY SAMPLE CHUNK
# ==========================================

print("\n========== SAMPLE CHUNK ==========")

print(chunks[0].page_content)

print("\n========== METADATA ==========")

print(chunks[0].metadata)