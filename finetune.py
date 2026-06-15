from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import setproctitle

max_seq_length = 2048
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
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
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
            {"role": "model", "content": a}
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
dataset = dataset.map(formatting_prompts_func, batched=True)

class ProcessTitleCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            loss = logs.get("loss", 0.0)
            setproctitle.setproctitle(f"Train - Step: {step} Loss: {loss:.4f}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        disable_tqdm=False,
        report_to="none",
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
