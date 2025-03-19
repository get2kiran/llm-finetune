from transformers import pipeline

# 加载模型
# pipe = pipeline("text-generation", model="models/Qwen2.5-0.5B-Instruct")
pipe = pipeline("text-generation", model="results/qwen2.5-0.5b-lora")
# 提供输入
input_text = "只剩一个心脏了还能活吗？"

# 调整生成参数
output = pipe(
    input_text,
    max_length=1024,  # 增加生成的最大长度
    num_return_sequences=1,  # 生成多个序列
    temperature=0.01,
)

print(output)