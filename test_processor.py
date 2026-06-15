from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-e2b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
try:
    print("Testing with None")
    out = tokenizer(text=["Hello", "World"], images=[None, None], return_tensors="pt")
    print("None worked")
except Exception as e:
    print("None failed:", e)

try:
    print("Testing with empty list")
    out = tokenizer(text=["Hello", "World"], images=[[], []], return_tensors="pt")
    print("Empty list worked")
except Exception as e:
    print("Empty list failed:", e)

