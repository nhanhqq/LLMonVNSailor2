# Vietnamese RAG Finetuning with Gemma 4 & Ollama

This repository provides a complete, end-to-end pipeline to fine-tune the **Gemma 4** language model on the `sailor2/Vietnamese_RAG` dataset using **Unsloth**, export the trained weights to the **GGUF** format, and deploy it locally using **Ollama** for Chatbot and RAG inference.

## Features
- 🚀 **Fast Fine-Tuning**: Utilizes Unsloth and 4-bit quantization (LoRA) for highly optimized and memory-efficient training.
- 📚 **Comprehensive Dataset**: Automatically loads and concatenates all 4 subsets of `sailor2/Vietnamese_RAG` (`BKAI_RAG`, `LegalRAG`, `expert`, `viQuAD`).
- 📊 **Automated Logging**: Tracks training metrics (loss, learning rate) and exports them to a CSV file (`training_logs.csv`), while generating a visual loss curve (`training_metrics.png`).
- ⚡ **Ollama Integration**: Scripts are included to easily convert the fine-tuned model into GGUF format and import it into Ollama for fast, local inference.

## Prerequisites
- A Linux environment with an NVIDIA GPU (CUDA supported).
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running in the background.

## Installation
First, install all required Python dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Usage Workflow

Follow these steps in order to fine-tune and run the model:

### Step 1: Fine-Tuning the Model
Run the fine-tuning script. This will download the base model and dataset, train the LoRA adapters, and save the final weights to the `gemma-rag-lora` directory.
*(Note: You can monitor the training progress and loss directly in your system's task manager/htop as the process title is updated in real-time).*
```bash
python3 finetune.py
```
> **Outputs**: `gemma-rag-lora/` (weights), `training_logs.csv`, `training_metrics.png`

### Step 2: Export to GGUF
Convert the PyTorch fine-tuned model into a quantized GGUF format that Ollama can understand.
```bash
python3 export_gguf.py
```
> **Outputs**: A `.gguf` file (e.g., `gemma-rag-model-unsloth.Q4_K_M.gguf`)

### Step 3: Create the Ollama Model
Use the provided bash script to read the `Modelfile` and import the newly generated GGUF file into your local Ollama library.
```bash
chmod +x import_ollama.sh
./import_ollama.sh
```
> This creates a local model named `gemma_rag_vn` in Ollama.

### Step 4: Run Inference
Test the fine-tuned model by running the inference script. It will send a sample Vietnamese context and question to the Ollama API and print the generated answer.
```bash
python3 inference.py
```

## Customization
- **Base Model**: You can change the `model_name` inside `finetune.py` to point to any other Gemma version (e.g., `unsloth/gemma-4-9b-it-bnb-4bit`).
- **Hyperparameters**: Feel free to adjust `max_steps`, `learning_rate`, or `batch_size` within the `TrainingArguments` block in `finetune.py` to better fit your hardware capabilities.