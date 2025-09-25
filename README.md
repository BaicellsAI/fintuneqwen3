# AI Agent Instructions for Traditional Chinese Medicine (TCM) LLM Fine-tuning Project

## Project Overview
This repository contains a complete workflow for fine-tuning Qwen3 0.6B language models for Traditional Chinese Medicine (TCM) applications. The project demonstrates three fine-tuning approaches:

1. Standard Supervised Fine-Tuning (SFT)
2. Low-Rank Adaptation (LoRA) fine-tuning
3. Direct Preference Optimization (DPO) fine-tuning

## Data Flow

```
data_prepare/ → model_finetune/ → trainer_output/
```

- Data preparation scripts download and convert TCM datasets into Qwen3 format
- Fine-tuning scripts apply different training approaches (SFT, LoRA, DPO)
- Models are saved to respective output directories with checkpoints

## Key Components

### Data Preparation
- `data_prepare/download_tcm_data.py`: Downloads TCM dataset from Hugging Face
- `data_prepare/convert_to_qwen3_format.py`: Converts data to Qwen3-compatible format
- Data format: `{"instruction": "question", "input": "additional context", "output": "answer"}`

### Model Fine-tuning
- `model_finetune/finetune_sft.py`: Standard supervised fine-tuning
- `model_finetune/finetune_lora.py`: Parameter-efficient fine-tuning using LoRA
- `model_finetune/finetune_dpo.py`: Direct Preference Optimization with positive/negative samples

### Model Loading & Inference
- `model_download/download_model.py`: Downloads base model from Hugging Face
- `model_download/load_local_model.py`: Loads locally fine-tuned model for inference

## Common Workflows

### Preparing TCM Dataset
```python
# Download dataset from Hugging Face
python data_prepare/download_tcm_data.py

# Convert to Qwen3 format if needed
python data_prepare/convert_to_qwen3_format.py
```

### Fine-tuning Models
```python
# For standard SFT fine-tuning
python model_finetune/finetune_sft.py

# For LoRA fine-tuning (parameter-efficient)
python model_finetune/finetune_lora.py

# For DPO fine-tuning (preference optimization)
python model_finetune/finetune_dpo.py
```

### Hardware Considerations
- Models detect and use MPS (Apple Silicon) or CPU automatically
- Batch sizes need adjustment based on available memory (2-4 for Mac)
- For memory issues, reduce batch size or gradient accumulation steps

## Project Conventions
- Output directories follow naming patterns: `./sft_finetuned`, `./lora_finetuned`, `./dpo_finetuned`
- Checkpoints are saved every epoch for resumable training
- Loss curves are automatically generated as PNG files after training
- All scripts use UTF-8 encoding for proper Chinese character handling

## Example Pattern
Fine-tuning script structure typically follows:
1. Load model and tokenizer
2. Prepare dataset in appropriate format
3. Configure training arguments
4. Initialize trainer
5. Run training loop
6. Save model and visualize loss curve
