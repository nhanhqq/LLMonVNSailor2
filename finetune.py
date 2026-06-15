import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch._dynamo.config.suppress_errors = True
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import setproctitle

max_seq_length = 1024
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-e2b-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma",
)

def formatting_prompts_func(examples):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answer"]
    texts = []
    for q, c, a in zip(questions, contexts, answers):
        messages = [
            {"role": "user", "content": f"Dựa vào thông tin sau:\n{c}\n\nCâu hỏi: {q}"},
            {"role": "assistant", "content": a}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

subsets = ["BKAI_RAG", "LegalRAG", "expert", "viQuAD"]
dataset_list = []
for subset in subsets:
    ds = load_dataset("sailor2/Vietnamese_RAG", subset, split="train")
    dataset_list.append(ds)

dataset = concatenate_datasets(dataset_list)
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

class ProcessTitleCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            epoch = state.epoch if state.epoch else 0
            total_epochs = args.num_train_epochs
            setproctitle.setproctitle(f"Train - Ep: {epoch:.2f}/{total_epochs} Step: {step} Loss: {loss:.4f} LR: {lr:.2e}")

from trl import SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=getattr(tokenizer, "tokenizer", tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=10,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        disable_tqdm=False,
        report_to="none",
        remove_unused_columns=False,
    ),
    callbacks=[ProcessTitleCallback()],
)

trainer.train()

# Log full metrics to CSV
log_history = trainer.state.log_history
df = pd.DataFrame(log_history)
df.to_csv("training_logs.csv", index=False)

# Plot training loss
loss_history = [log for log in log_history if "loss" in log]
if loss_history:
    steps = [log["step"] for log in loss_history]
    losses = [log["loss"] for log in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker="o", linestyle="-", color="b")
    plt.title("Training Loss over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_metrics.png")
    plt.close()

model.save_pretrained("gemma-rag-lora")
tokenizer.save_pretrained("gemma-rag-lora")
