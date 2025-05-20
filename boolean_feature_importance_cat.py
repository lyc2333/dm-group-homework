import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = []
with open('steam_games.ndjson', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)


# 创建DataFrame
df = pd.DataFrame(data)

# 初始化并转换
mlb = MultiLabelBinarizer()
categories_encoded = mlb.fit_transform(df['categories'])
prefixed_columns = [f"cat_{c}" for c in mlb.classes_]

# 转换为DataFrame并合并
categories_df = pd.DataFrame(categories_encoded, columns=prefixed_columns)
df = pd.concat([df, categories_df], axis=1)

# 2. 分类目标：将 estimated_owners 作为 label
df = df[df['estimated_owners'].notna()]
df['owners_class'] = df['estimated_owners'].astype(str)

# 3. 特征工程（保持之前的一致）
num_feats = ['price', 'dlc_count', 'metacritic_score',
             'achievements', 'recommendations',
             'positive', 'negative', 'discount', 'peak_ccu']

# num_feats = ['price', 'dlc_count', 'metacritic_score',
#              'achievements']
df[num_feats] = df[num_feats].fillna(0)
df['plat_support'] = df[['windows', 'mac', 'linux']].sum(axis=1)
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
df['n_languages'] = df['supported_languages'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['n_categories'] = df['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['n_genres'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['n_tags'] = df['tags'].apply(lambda x: len(x) if isinstance(x, dict) else 0)

features = num_feats + ['plat_support', 'release_year',  'n_categories', 'n_genres', 'n_tags'] +prefixed_columns


# 统计每个销量区间的出现次数
owners_class_counts = df['owners_class'].value_counts().sort_index()

# 打印出来
print("销量区间（estimated_owners）类别分布：\n")
for cls, count in owners_class_counts.items():
    print(f"{cls:<25} : {count}")


# 4. 准备训练数据
X = df[features]
y = df['owners_class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. 建立分类模型

models = {
    # 'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'MLPClassifier': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    # 'SGDClassifier': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

}

pipelines = {
    name: Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    for name, model in models.items()
}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} 准确率 Accuracy: {acc:.3f}")
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred))
    joblib.dump(pipeline, f'{name}.pkl')
    clf = pipeline.named_steps['clf']
    importances = clf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    plt.figure(figsize=(16, 8))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importances with Categories')
    plt.savefig('feature_imp_cat.png')


