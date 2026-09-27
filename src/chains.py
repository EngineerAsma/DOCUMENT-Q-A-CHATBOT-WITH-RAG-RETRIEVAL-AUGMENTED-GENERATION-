
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================
# LOAD QWEN LANGUAGE MODEL
# ==========================================

print("Loading Qwen language model...")

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

print("Qwen language model loaded successfully!")


# ==========================================
# GENERATE ANSWER FROM CONTEXT
# ==========================================

def generate_answer(question, context):

    prompt = f"""
You are a document question-answering assistant.

IMPORTANT INSTRUCTIONS:

1. Answer the question using the CONTEXT provided below.
2. If the answer is present in the CONTEXT, use the information from
   the CONTEXT faithfully.
3. Do NOT replace the document's information with your own knowledge.
4. Do NOT invent facts.
5. Do NOT add information that is not supported by the CONTEXT.
6. You may combine information from multiple parts of the CONTEXT
   when necessary.
7. If the CONTEXT does not contain information that answers the
   question, say:
   "The answer is not available in the document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
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
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=650,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()
