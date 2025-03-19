import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
import warnings
warnings.filterwarnings("ignore", message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", message="`use_cache=True` is incompatible")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

# 设置设备和环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = load_dataset("json", data_files="./datasets/ruozhiba/train_dataset_2.json", split="train")
# 划分训练集和验证集
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)  # 10% 的数据作为验证集
# 创建 DatasetDict
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# 加载模型和分词器
model_id = "./models/Qwen2.5-0.5B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right", truncation=True)

# 配置LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=256,
    bias="none",
    task_type="CAUSAL_LM",
)

# 定义训练参数
args = TrainingArguments(
    output_dir=os.path.join(os.getcwd(), "results/qwen2.5-0.5b-lora"),
    num_train_epochs=0.5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=200,
    save_strategy="steps",
    save_steps=200,
    evaluation_strategy="steps",        # 添加评估策略
    eval_steps=50,                    # 每 1000 步评估一次
    learning_rate=1e-4,
    fp16=True,                          # 启用 fp16 混合精度训练，提升效率
    max_grad_norm=0.3,
    warmup_ratio=0.3,
    lr_scheduler_type="linear",
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,        # 确保加载最佳模型
    metric_for_best_model="loss",       # 监控验证集损失
    greater_is_better=False,            # 损失越小越好
)

# 创建SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],  # 添加验证集
    peft_config=peft_config,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
trainer.save_model()

# 清理内存
del model
del trainer
torch.cuda.empty_cache()