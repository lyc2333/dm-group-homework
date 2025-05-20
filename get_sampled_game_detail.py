from tqdm import tqdm
import csv
import time
import requests
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

data = []
with open('steam_games.ndjson', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
df = pd.DataFrame(data)

# Step 1: 清洗 estimated_owners 字段
df = df[df['estimated_owners'].notna()]
df['owners_class'] = df['estimated_owners'].astype(str)


# 假设 df 是包含 owners_class 的完整 DataFrame
total_sample_size = 1000

# 计算每个区间的样本数
class_counts = df['owners_class'].value_counts()
class_proportions = class_counts / class_counts.sum()

# 计算每个区间要抽取的样本数（向上取整）
class_sample_sizes = np.ceil(class_proportions * total_sample_size).astype(int)

# 调整总数为恰好1000（可能由于向上取整超出）
overage = class_sample_sizes.sum() - total_sample_size
if overage > 0:
    # 找出样本最多的区间，从中减去多余的数量
    for cls in class_sample_sizes.sort_values(ascending=False).index:
        if class_sample_sizes[cls] > overage:
            class_sample_sizes[cls] -= overage
            break

# 分层抽样
sampled_df = (
    df.groupby('owners_class')
    .apply(lambda x: x.sample(n=min(len(x), class_sample_sizes[x.name]), random_state=42))
    .reset_index(drop=True)
)

print(f"最终抽取样本总数：{len(sampled_df)}")
print(sampled_df['owners_class'].value_counts())

sampled_df[['app_id', 'estimated_owners']].to_csv("sampled_games.csv", index=False)

# ---

# 开始爬虫

# 参数设置
input_file = 'sampled_games.csv'
output_file = 'sampled_games_detail.csv'
fieldnames = ['app_id', 'copiesSold', 'revenue', 'players', 'owners', 'steamPercent', 'accuracy']
fields_param = 'copiesSold,revenue,players,owners,steamPercent,accuracy'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'}

# 读取原始待采样的 app_id
df_ids = pd.read_csv(input_file)
all_app_ids = set(df_ids['app_id'].astype(str).tolist())

# 读取已完成的 app_id（如果文件已存在）
if os.path.exists(output_file):
    try:
        df_done = pd.read_csv(output_file, dtype={'app_id': str})
        finished_app_ids = set(df_done['app_id'].dropna().astype(str))
    except Exception:
        finished_app_ids = set()
else:
    # 若文件不存在则初始化写入表头
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    finished_app_ids = set()

# 剩余未处理的 app_id
pending_app_ids = sorted(all_app_ids - finished_app_ids)
print(f"待处理游戏数：{len(pending_app_ids)} / {len(all_app_ids)}")

# 遍历剩余 app_id 并请求 API
for app_id in tqdm(pending_app_ids, desc="Fetching game details (resume mode)"):
    url = f'https://api.gamalytic.com/game/{app_id}?fields={fields_param}'

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
        else:
            if response.status_code == 429:
                print("warning: 429 triggered")
            data = {}
    except Exception as e:
        print(e)
        data = {}

    # 组织数据并写入
    row = {field: data.get(field, None) for field in fieldnames}
    row['app_id'] = app_id

    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

    time.sleep(0.5)  # 限流
