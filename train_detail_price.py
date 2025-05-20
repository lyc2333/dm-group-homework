import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ------------------------------
# Step 1: 读取与合并数据
# ------------------------------


data = []
with open('steam_games.ndjson', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
df_all = pd.DataFrame(data)



sampled_ids = pd.read_csv('sampled_games.csv')['app_id'].astype(str)
df_sampled = df_all[df_all['app_id'].astype(str).isin(sampled_ids)].copy()
details = pd.read_csv('sampled_games_detail.csv')
details['app_id'] = details['app_id'].astype(str)

# 合并
df = pd.merge(df_sampled, details, on='app_id', how='left')

# 去除没有copiesSold的记录（可能是404）
df = df.dropna(subset=['copiesSold'])

# ------------------------------
# Step 2: 处理销量区间字段
# ------------------------------
df['owners_lower'] = df['estimated_owners'].str.extract(r'(\d+)').astype(float)
df['owners_upper'] = df['estimated_owners'].str.extract(r'-\s*(\d+)').astype(float)
df['owners_mid'] = (df['owners_lower'] + df['owners_upper']) / 2

# ------------------------------
# Step 3: 构造特征集
# ------------------------------
# 数值特征
numeric_features = [
    'price', 'discount', 'positive', 'negative',
    'peak_ccu', 'recommendations', 'metacritic_score',
    'average_playtime_forever', 'owners_lower', 'owners_upper', 'owners_mid'
]

# 类别特征（只用genre和category名称，假设已经处理为字符串）
# categorical_features = ['genres', 'categories']

# 填充缺失值
df[numeric_features] = df[numeric_features].fillna(0)
# df[categorical_features] = df[categorical_features].fillna('Unknown')

# ------------------------------
# Step 4: 构造训练集
# ------------------------------
# X = df[numeric_features + categorical_features]
X = df[numeric_features]
y = np.log1p(df['copiesSold'])  # 使用对数回归

# 预处理：类别特征做OneHot编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 模型 + Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 拆分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练
model.fit(X_train, y_train)

# ------------------------------
# Step 5: 模型评估
# ------------------------------
y_pred = model.predict(X_test)
# rmse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred), squared=False)
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE (真实销量单位): {rmse:.0f}')
print(f'R² Score (log-scale): {r2:.4f}')

# ------------------------------
# Step 6: 保存模型（可选）
# ------------------------------
import joblib
joblib.dump(model, 'models/detail_sales_predictor.pkl')
