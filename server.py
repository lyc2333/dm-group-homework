import json
import os
import time
from flask import Flask, request, jsonify, render_template, send_file
import base64
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from get_data import get_steam_data, get_steamspy_data, unify_data, add_extra_feature
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_by_id')
def predict_by_id():
    app_id = request.args.get('id')
    model_name = request.args.get('model', 'SGD')
    result = predict_from_id(app_id, model_name)
    return jsonify(result)


@app.route('/predict_by_features', methods=['POST'])
def predict_by_features():
    data = request.json
    model_name = data.get('model', 'SGD')
    features = data.get('features', {})
    result = predict_from_features(features, model_name)
    return jsonify(result)


@app.route('/update_model', methods=['POST'])
def update_model():
    file = request.files['file']
    model_name = request.form.get('model', 'SGD')
    status = update_model_with_jsonl(file, model_name)
    return jsonify(status)


def get_image_base64(app_id):
    img_url = f"https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"
    resp = requests.get(img_url)
    if resp.status_code == 200:
        return base64.b64encode(resp.content).decode('utf-8')
    return None


def get_game_data(app_id):

    url = "https://store.steampowered.com/api/appdetails"

    while True:
        try:
            resp = requests.get(url=url, params={"appids": app_id})
            if resp.status_code == 200:
                steam_data = resp.json()
                break
        except Exception as e:
            print(e)
            time.sleep(0.2)
    url = "https://steamspy.com/api.php"

    while True:
        try:
            resp = requests.get(url=url, params={"request": "appdetails", "appid": app_id})
            if resp.status_code == 200:
                steamspy_data = resp.json()
                break
        except Exception as e:
            print(e)
            time.sleep(0.2)
    return {"steam": steam_data, "steam_spy": steamspy_data}

model_path_dict = {
        "LogisticRegression": "LogisticRegression_best.pkl",
        "RandomForest": "RandomForest_best.pkl",
        "MLP": "MLPClassifier_best.pkl",
        "LogisticRegression (No User Data)": "LogisticRegression_no_user_data_best.pkl",
        "RandomForest (No User Data)": "RandomForest_no_user_data_best.pkl",
        "MLP (No User Data)": "MLPClassifier_no_user_data_best.pkl",
        "DetailSalesPredictor": "DetailSalesPredictor.pkl",
        "RandomForest (Updated)":"RandomForest_updated.pkl",
        "MLP (Updated)":"MLPClassifier_updated.pkl"
    }
def predict_from_id(app_id, model_name):
    steam_data = get_steam_data(app_id)
    steamspy_data = get_steamspy_data(app_id)
    features = unify_data(steam_data, steamspy_data)

    model_path = os.path.join("models", f"{model_path_dict[model_name]}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。")

    # 加载模型（通常是一个包含预处理步骤的Pipeline）
    model = joblib.load(model_path)

    # 将 features 转换为 DataFrame 格式
    X = pd.DataFrame([features])  # 注意外层加 []，保持二维结构

    # 预测（支持分类或回归）
    try:
        X = add_extra_feature(X)
        valid_keys = getattr(model, "feature_names_in_", None)
        if valid_keys is not None:
            X = X[valid_keys]

        res = {"image_base64": get_image_base64(app_id)}

        y_pred = model.predict(X)

        if model_name == 'DetailSalesPredictor':
            res['prediction'] = str(np.e**y_pred[0]-1)
        else:
            res['prediction'] = str(y_pred[0])
        return res
    except Exception as e:
        return {"error": str(e)}


def predict_from_features(features, model_name):

    model_path = os.path.join("models", f"{model_path_dict[model_name]}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。")

    # 加载模型（通常是一个包含预处理步骤的Pipeline）
    model = joblib.load(model_path)

    # 将 features 转换为 DataFrame 格式
    X = pd.DataFrame([features])  # 注意外层加 []，保持二维结构

    # 预测（支持分类或回归）
    try:
        X = add_extra_feature(X)

        valid_keys = getattr(model, "feature_names_in_", None)
        if valid_keys is not None:
            X = X[valid_keys]

        y_pred = model.predict(X)
        if model_name == 'DetailSalesPredictor':
            return {"prediction": str(np.e**y_pred[0]-1)}
        else:
            return {"prediction": str(y_pred[0])}
    except Exception as e:
        return {"error": str(e)}


def update_model_with_jsonl(file, model_name):
    try:
        if model_name == "RandomForest":
            acc = train_updated_random_forest(file)
            return {"message": "ok", "accuracy": acc}
        elif model_name == "MLP":
            acc = train_updated_mlp(file)
            return {"message": "ok", "accuracy": acc}
        else:
            return {"error": "wrong model"}
    except Exception as e:
        return {"error": str(e)}


def train_updated_random_forest(new_data):
    # 1. 加载数据
    data = []
    with open('steam_games.ndjson', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    for line in new_data:
        data.append(json.loads(line))
    df = pd.DataFrame(data)

    # 2. 分类目标：将 estimated_owners 作为 label
    df = df[df['estimated_owners'].notna()]
    df['owners_class'] = df['estimated_owners'].astype(str)
    df['owners_class'] = df['estimated_owners'].apply(lambda x: '0 - 20000' if x == '0 - 0' else x)

    # 3. 特征工程
    num_feats = ['price',  'achievements', 'recommendations',
                 'positive', 'negative', 'discount']

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

    model = joblib.load(r'models\RandomForest.pkl')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, f'models/RandomForest_updated.pkl')
    return accuracy_score(y_test, y_pred)


def train_updated_mlp(new_data):
    # 1. 加载数据
    data = []
    for line in new_data:
        data.append(json.loads(line))
    df = pd.DataFrame(data)

    # 2. 分类目标：将 estimated_owners 作为 label
    df = df[df['estimated_owners'].notna()]
    df['owners_class'] = df['estimated_owners'].astype(str)
    df['owners_class'] = df['estimated_owners'].apply(lambda x: '0 - 20000' if x == '0 - 0' else x)

    # 3. 特征工程
    num_feats = ['price',  'achievements', 'recommendations',
                 'positive', 'negative', 'discount']

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
    model_path =  r'models\MLPClassifier_updated.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = MLPClassifier(max_iter=300, random_state=42,)

    # 多次 partial_fit 相当于多轮 epoch
    for epoch in range(10):
        model.partial_fit(X_train, y_train, classes=['0 - 20000', '100000 - 200000', '1000000 - 2000000', '10000000 - 20000000', '100000000 - 200000000', '20000 - 50000', '200000 - 500000', '2000000 - 5000000', '20000000 - 50000000', '200000000 - 500000000', '50000 - 100000', '500000 - 1000000', '5000000 - 10000000', '50000000 - 100000000'])

    y_pred = model.predict(X_test)
    joblib.dump(model,model_path)
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=20201, debug=True)
