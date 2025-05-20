import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import random

# 创建保存目录
os.makedirs("project/results/models", exist_ok=True)
os.makedirs("project/results/predictions", exist_ok=True)


# 读取数据
def load_data(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)


df = load_data("project/data/steam_games.ndjson")


# 数据预处理
def preprocess_data(df):
    # 处理目标变量：使用预估玩家保有量的随机值
    df["owners"] = (
        df["estimated_owners"]
        .str.split(" - ")
        .apply(
            lambda x: (
                random.randint(int(x[0]), int(x[1]))
                if isinstance(x, list) and len(x) == 2
                else np.nan
            )
        )
    )

    # 确保列中只有数值数据
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce")

    # 移除 NaN 值
    df = df.dropna(subset=["discount"])

    # 特征工程
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["positive_ratio"] = df["positive"] / (
        df["positive"] + df["negative"] + 1e-6
    )  # 防止除以零
    df["num_languages"] = df["supported_languages"].apply(len)
    df["has_achievements"] = df["achievements"].apply(lambda x: 1 if x > 0 else 0)
    df["discount_group"] = pd.cut(
        df["discount"],
        bins=[-1, 0, 25, 50, 75, 101],
        labels=["No Discount", "0-25%", "25-50%", "50-75%", "75-100%"],
    )

    # 选择特征
    features = df[
        [
            "price",
            "release_year",
            "windows",
            "mac",
            "linux",
            "metacritic_score",
            "positive_ratio",
            "num_languages",
            "has_achievements",
            "discount_group",
            "recommendations",
        ]
    ]

    # 处理缺失值
    features["metacritic_score"] = features["metacritic_score"].fillna(
        features["metacritic_score"].median()
    )
    features["release_year"] = features["release_year"].fillna(
        features["release_year"].mode()[0]
    )

    return features, df["owners"].dropna()


# 执行预处理
X, y = preprocess_data(df)

print(y)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 定义预处理管道
numeric_features = [
    "price",
    "metacritic_score",
    "positive_ratio",
    "num_languages",
    "recommendations",
]
categorical_features = ["discount_group"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 创建模型管道
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# 网格搜索优化参数
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 模型评估
y_pred = best_model.predict(X_test)

metrics = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R2 Score": r2_score(y_test, y_pred),
}

# 保存模型和预测结果
pd.Series(metrics).to_csv("project/results/models/metrics.csv")
pd.DataFrame({"True": y_test, "Predicted": y_pred}).to_csv(
    "project/results/predictions/predictions.csv", index=False
)

# 可视化结果
plt.figure(figsize=(12, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Actual vs Predicted Player Ownership")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig(
    "project/results/predictions/actual_vs_predicted.png", dpi=300, bbox_inches="tight"
)
plt.close()

# 特征重要性可视化
feature_names = numeric_features + list(
    best_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_features)
)

importances = best_model.named_steps["regressor"].feature_importances_

plt.figure(figsize=(12, 8))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig(
    "project/results/models/feature_importance.png", dpi=300, bbox_inches="tight"
)
plt.close()

print(
    f"""
模型训练完成！
最佳参数：{grid_search.best_params_}
测试集评估指标：
- RMSE: {metrics['RMSE']:.2f}
- R² Score: {metrics['R2 Score']:.4f}
结果已保存至：
- project/results/models/ 模型和评估指标
- project/results/predictions/ 预测结果和可视化
"""
)
