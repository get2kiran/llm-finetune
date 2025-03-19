from datasets import load_from_disk, load_dataset
import os, json

# 配置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7880'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7880'
# 系统 prompt
system_message = "回答问题"


# 转换为 messages
def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]}
        ]
    }


# 从 hub 加载数据集
dataset = load_dataset("LooksJuicy/ruozhiba", split="train")

# 转换 dataset 为 OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

print(dataset[345]["messages"])

# 保存到磁盘
# dataset.to_json("train_dataset.json", orient="records")
# 保存到磁盘
output_file = "./datasets/ruozhiba/train_dataset_2.json"
with open(output_file, "w", encoding="utf-8") as f:
    # 将数据集转换为列表格式并保存为 JSON 文件
    json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=4)

print(f"数据已保存到 {output_file}")