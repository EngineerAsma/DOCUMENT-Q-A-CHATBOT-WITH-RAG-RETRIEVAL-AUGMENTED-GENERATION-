from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from chains import generate_answer

VECTORSTORE_PATH = "vectorstore/faiss_index"

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

print("\nLoading saved FAISS vector database...")

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS vector database loaded successfully!")


def ask_question(question):
    print("\n========================================")
    print("QUESTION:")
    print(question)
    print("========================================")

    results = vectorstore.similarity_search(
        question,
        k=2
    )

    print("\n========== RETRIEVED DOCUMENTS ==========")

    for i, result in enumerate(results, start=1):
        print(f"\n----- CHUNK {i} -----")
        print(result.page_content)
        print("\nMetadata:")
        print(result.metadata)

    context = "\n\n".join(
        result.page_content
        for result in results
    )

    answer = generate_answer(
        question,
        context
    )

    print("\n========== FINAL ANSWER ==========")
    print(answer)

    return answer