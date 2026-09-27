from transformers import AutoTokenizer, AutoModelForCausalLM


print("Loading Qwen model...")


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_PATH = "models/qwen2.5-0.5b"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_PATH
)


print("Qwen model loaded successfully!")


# ==========================================
# TEST QUESTION
# ==========================================

question = "What is Machine Learning?"

context = """
Machine Learning is a discipline of Artificial Intelligence
that enables machines to automatically learn from data and
past experiences, identify patterns, and make predictions
with minimal human intervention.
"""


prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Answer:
"""


messages = [
    {
        "role": "user",
        "content": prompt
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    text,
    return_tensors="pt"
)


outputs = model.generate(
    **inputs,
    max_new_tokens=100
)


answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)


print("\n========================================")
print("QUESTION:")
print(question)
print("========================================")

print("\n========== GENERATED ANSWER ==========")
print(answer)