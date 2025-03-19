# 概述
- 使用trl、peft、transformers等库，实现对huggingface上模型的微调。
- [blog](https://legendleochen.top/2025/02/06/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%EF%BC%9ASFT%EF%BC%88LoRA%E3%80%81Prefix%EF%BC%89%E5%92%8CRLHF/)

# 文件
- datasets：数据集
- models：大模型
- results：微调后的模型
- chat.py：仅聊天
- convert.py：获取数据集并转为json文件保存
- download_data.py：仅下载原数据集
- main_GRPO.py：RLHF微调模型，使用GRPO算法
- main_lora.py：SFT微调模型，lora方法
- main_prefix.py：SFT微调模型，prefix方法