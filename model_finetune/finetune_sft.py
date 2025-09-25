# Qwen3 0.6B SFT微调脚本
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import json
import matplotlib.pyplot as plt
import os
import sys

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "../data_prepare/tcm_shizhen_for_qwen.json"
OUTPUT_DIR = "./sft_finetuned"

# 加载模型和分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"模型或分词器加载失败: {e}")
    sys.exit(1)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = get_device()

# 加载训练数据
def load_data(path, tokenizer):
    if not os.path.exists(path):
        print(f"数据文件不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = []
    for idx, item in enumerate(data):
        if not all(k in item for k in ("instruction", "input", "output")):
            print(f"警告: 第{idx}条数据缺少字段，已跳过")
            continue
        try:
            input_ids = tokenizer(
                item["instruction"] + item["input"],
                return_tensors="pt",
                truncation=True,
                max_length=256
            )["input_ids"].squeeze()
            labels = tokenizer(
                item["output"],
                return_tensors="pt",
                truncation=True,
                max_length=256
            )["input_ids"].squeeze()
            dataset.append({"input_ids": input_ids, "labels": labels})
        except Exception as e:
            print(f"警告: 第{idx}条数据编码失败，已跳过: {e}", file=sys.stderr)
    return dataset

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def plot_loss_curve(trainer, output_dir):
    loss_history = trainer.state.log_history
    steps = [x["step"] for x in loss_history if "loss" in x]
    losses = [x["loss"] for x in loss_history if "loss" in x]
    loss_curve_path = os.path.join(output_dir, "sft_loss_curve.png")
    if steps and losses:
        if not os.access(os.path.dirname(loss_curve_path) or ".", os.W_OK):
            print(f"警告: Loss曲线保存路径不可写: {loss_curve_path}", file=sys.stderr)
        else:
            plt.plot(steps, losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("SFT Training Loss Curve")
            plt.savefig(loss_curve_path)
            plt.close()
            print(f"Loss曲线已保存为 {loss_curve_path}")
    else:
        print("警告: 未找到loss数据，未生成loss曲线", file=sys.stderr)

def main():
    train_dataset = load_data(DATA_PATH, tokenizer)
    if not train_dataset:
        print("错误: 训练数据为空，请检查数据文件内容。", file=sys.stderr)
        sys.exit(1)
    train_ds = SimpleDataset(train_dataset)
    if len(train_ds) < 1:
        print("错误: 训练集样本数不足。", file=sys.stderr)
        sys.exit(1)
    if not os.access(os.path.dirname(OUTPUT_DIR) or ".", os.W_OK):
        print(f"错误: 输出目录不可写: {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to=[]
    )

    trainer = Trainer(
        model=model.to(DEVICE),
        args=training_args,
        train_dataset=train_ds,
    )

    try:
        train_output = trainer.train()
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"SFT微调模型已保存到 {OUTPUT_DIR}")
    except Exception as e:
        print(f"训练或保存模型失败: {e}", file=sys.stderr)
        sys.exit(1)

    plot_loss_curve(trainer, OUTPUT_DIR)

if __name__ == "__main__":
    main()
