import json
import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. 加载数据
data = []
with open('steam_games.ndjson', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)


# 2. 分类目标：将 estimated_owners 作为 label
df = df[df['estimated_owners'].notna()]
df['owners_class'] = df['estimated_owners'].astype(str)
df['owners_class'] = df['estimated_owners'].apply(lambda x: '0 - 20000' if x == '0 - 0' else x)

# 3. 特征工程
num_feats = ['price',  'achievements', 'discount']


df[num_feats] = df[num_feats].fillna(0)
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
df['n_categories'] = df['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['n_genres'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['n_tags'] = df['tags'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
df['is_family_sharing'] = df['categories'].apply(lambda x: 'Family Sharing' in x if isinstance(x, list) else False)

features = num_feats + ['release_year', 'n_categories', 'n_genres', 'n_tags', 'is_family_sharing']


# 统计每个销量区间的出现次数并打印
owners_class_counts = df['owners_class'].value_counts().sort_index()
print("销量区间（estimated_owners）类别分布：\n")
for cls, count in owners_class_counts.items():
    print(f"{cls:<25} : {count}")


# 4. 准备训练数据
X = df[features]
y = df['owners_class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. 建立分类模型

# 每个模型及其对应的参数搜索空间
models = {
    'LogisticRegression': (
        LogisticRegression(max_iter=1000),
        {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__solver': ['lbfgs', 'liblinear']
        }
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=42),
        {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20]
        }
    ),
    'MLPClassifier': (
        MLPClassifier(max_iter=300, random_state=42),
        {
            'clf__hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
            'clf__alpha': [0.0001, 0.001]
        }
    ),
    # 'SGDClassifier': (
    #     SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
    #     {
    #         'clf__alpha': [0.0001, 0.001, 0.01],
    #         'clf__penalty': ['l2', 'l1', 'elasticnet']
    #     }
    # )
}

# 遍历每个模型及其搜索空间
for name, (model, param_grid) in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    print(f"\n正在为 {name} 进行 GridSearchCV 超参数搜索...")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 使用最佳模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 输出结果
    print(f"\n{name} 最佳参数: {grid_search.best_params_}")
    print(f"{name} 测试集准确率: {accuracy_score(y_test, y_pred):.3f}")
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred))

    # 保存模型
    pathlib.Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, f'models/{name}_no_user_data_best.pkl')
