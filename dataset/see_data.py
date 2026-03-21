from datasets import load_dataset
import glob
import os
from PIL import Image
from io import BytesIO

# 1. 收集 tar 文件
data_files = glob.glob('/inspire/hdd/project/exploration-topic/public/ent/download_dataset/BLIP3o-60k/*.tar')
print(f"num tar files: {len(data_files)}")

# 2. 加载 webdataset
train_dataset = load_dataset(
    "webdataset",
    data_files=data_files,
    cache_dir='/inspire/hdd/project/exploration-topic/public/ent/download_dataset/cache/blip3o',
    split="train",
    num_proc=64
)

# 3. 查看数据集基本信息
print(train_dataset)
print("num rows:", len(train_dataset))

# 4. 查看字段名
print("column names:", train_dataset.column_names)

# 5. 看前3个样本的内容结构
for i in range(3):
    sample = train_dataset[i]
    print(f"\n========== sample {i} ==========")
    print("keys:", sample.keys())
    for k, v in sample.items():
        if isinstance(v, bytes):
            print(f"{k}: bytes, len={len(v)}")
        else:
            print(f"{k}: type={type(v)}, value={str(v)[:300]}")