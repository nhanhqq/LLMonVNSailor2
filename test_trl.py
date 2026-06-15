from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-e2b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

ds = Dataset.from_dict({"text": ["Hello world", "Test this out"]})

try:
    print("Testing with tokenizer.tokenizer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer.tokenizer,  # Extract pure tokenizer
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=128,
            output_dir="tmp"
        )
    )
    for batch in trainer.get_train_dataloader():
        print(batch)
        break
    print("tokenizer.tokenizer worked!")
except Exception as e:
    print("Failed:", e)

