import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

os.makedirs("project/results/figures", exist_ok=True)

data = []
with open("project/data/steam_games.ndjson", "r") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
df = pd.DataFrame(data)

df["release_date"] = pd.to_datetime(
    df["release_date"], format="%b %d, %Y", errors="coerce"
)
df["release_year"] = df["release_date"].dt.year
price_bins = [-1, 0, 5, 10, 20, 50, 100, float("inf")]
price_labels = ["Free", "0-5", "5-10", "10-20", "20-50", "50-100", "100+"]
df["price_group"] = pd.cut(df["price"], bins=price_bins, labels=price_labels)

# 1. 发行年份分布
plt.figure(figsize=(12, 6))
year_counts = df["release_year"].value_counts().sort_index()
sns.barplot(x=year_counts.index, y=year_counts.values)
plt.title("Game Releases by Year", fontsize=14)
plt.xticks(rotation=45)
plt.xlabel("Release Year")
plt.ylabel("Number of Games")
plt.savefig(
    "project/results/figures/release_year_distribution.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 2. 价格分布
plt.figure(figsize=(10, 6))
sns.countplot(x="price_group", data=df, order=price_labels)
plt.title("Price Distribution", fontsize=14)
plt.xlabel("Price Range ($)")
plt.ylabel("Count")
plt.savefig(
    "project/results/figures/price_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()

# 3. 操作系统支持
plt.figure(figsize=(8, 5))
os_columns = ["windows", "mac", "linux"]
os_counts = df[os_columns].sum()
sns.barplot(x=os_counts.index, y=os_counts.values)
plt.title("Supported Operating Systems", fontsize=14)
plt.ylabel("Number of Games")
plt.savefig("project/results/figures/os_support.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. Metacritic评分分布
plt.figure(figsize=(10, 6))
if df["metacritic_score"].nunique() > 1:
    bins = [0, 50, 70, 85, 100]
    labels = ["<50", "50-70", "70-85", "85+"]
    df["metacritic_group"] = pd.cut(df["metacritic_score"], bins=bins, labels=labels)
    sns.countplot(x="metacritic_group", data=df, order=labels)
else:
    plt.text(0.5, 0.5, "No Metacritic Score Data", ha="center", va="center")
    plt.axis("off")
plt.title("Metacritic Score Distribution", fontsize=14)
plt.savefig(
    "project/results/figures/metacritic_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()

# 5. 用户评价分析
plt.figure(figsize=(10, 6))
if (df["positive"] + df["negative"]).sum() > 0:
    df["positive_ratio"] = df["positive"] / (df["positive"] + df["negative"])
    sns.histplot(df["positive_ratio"].dropna(), bins=20, kde=True)
else:
    plt.text(0.5, 0.5, "No Review Data", ha="center", va="center")
    plt.axis("off")
plt.title("Positive Review Ratio Distribution", fontsize=14)
plt.savefig("project/results/figures/review_ratio.png", dpi=300, bbox_inches="tight")
plt.close()

# 6. 折扣分布
plt.figure(figsize=(10, 6))
if df["discount"].nunique() > 1:
    sns.histplot(df["discount"], bins=20, kde=True)
else:
    plt.text(0.5, 0.5, "No Discount Data", ha="center", va="center")
    plt.axis("off")
plt.title("Discount Distribution", fontsize=14)
plt.xlabel("Discount Percentage")
plt.savefig(
    "project/results/figures/discount_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()

# 7. 支持语言数量
plt.figure(figsize=(10, 6))
df["num_languages"] = df["supported_languages"].apply(len)
sns.histplot(df["num_languages"], bins=20, kde=True)
plt.title("Supported Languages Count Distribution", fontsize=14)
plt.xlabel("Number of Supported Languages")
plt.savefig("project/results/figures/language_count.png", dpi=300, bbox_inches="tight")
plt.close()

# 8. 玩家保有量分布
plt.figure(figsize=(10, 6))
try:
    df["owners"] = (
        df["estimated_owners"]
        .str.split(" - ")
        .apply(lambda x: (int(x[0]) + int(x[1])) / 2 if isinstance(x, list) else 0)
    )
    bins = [0, 10000, 50000, 100000, 500000, 1000000, float("inf")]
    labels = ["<10k", "10k-50k", "50k-100k", "100k-500k", "500k-1M", "1M+"]
    df["owner_group"] = pd.cut(df["owners"], bins=bins, labels=labels)
    sns.countplot(x="owner_group", data=df, order=labels)
except Exception as e:
    plt.text(0.5, 0.5, "No Ownership Data", ha="center", va="center")
    plt.axis("off")
plt.title("Estimated Player Ownership Distribution", fontsize=14)
plt.xlabel("Player Ownership Range")
plt.xticks(rotation=45)
plt.savefig(
    "project/results/figures/ownership_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()
