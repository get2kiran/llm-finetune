import torch
from datasets import Dataset, load_dataset, load_from_disk
import os

# 配置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


dataset = load_dataset("hfl/ruozhiba_gpt4", split='train')
dataset.save_to_disk("./datasets/ruozhiba")  # 保存到该目录下