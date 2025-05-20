import csv
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from get_data import unify_data,add_extra_feature
steamspy_csv = "merged_output.csv"
new_game_details = "new_game_app_details.jsonl"


def csv_to_dict_by_id(file_path):
    result = {}
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['appid']
            result[key] = row
    return result


def jsonl_to_dict_by_id(file_path):
    result = {}
    with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            item = json.loads(line)
            key = str(item['steam_appid'])
            result[key] = item
    return result


if __name__ == "__main__":
    steamspy_data_dict = csv_to_dict_by_id(steamspy_csv)
    steam_data_dict = jsonl_to_dict_by_id(new_game_details)
    game_list = []
    for key in steam_data_dict:
        new_data= unify_data(steam_data_dict[key], steamspy_data_dict[key])
        if new_data['price'] is None or new_data['price']=="":
            continue # 未发布游戏
        game_list.append(new_data)

    df = pd.DataFrame(game_list)
    # 2. 分类目标：将 estimated_owners 作为 label
    df = df[df['estimated_owners'].notna()]
    df['owners_class'] = df['estimated_owners'].astype(str)

    # 3. 特征工程（保持之前的一致）
    df = add_extra_feature(df)

    # 统计每个销量区间的出现次数
    owners_class_counts = df['owners_class'].value_counts().sort_index()

    # 打印出来
    print("销量区间（estimated_owners）类别分布：\n")
    for cls, count in owners_class_counts.items():
        print(f"{cls:<25} : {count}")

    # 4. 准备训练数据

    models = {
        "LogisticRegression": "LogisticRegression_best.pkl",
        #"SGD": "SGDClassifier_best.pkl",
        "RandomForest": "RandomForest_best.pkl",
        "MLP": "MLPClassifier_best.pkl",
        "LogisticRegression (No User Data)": "LogisticRegression_no_user_data_best.pkl",
        #"SGD (No User Data)": "SGDClassifier_no_user_data_best.pkl",
        "RandomForest (No User Data)": "RandomForest_no_user_data_best.pkl",
        "MLP (No User Data)": "MLPClassifier_no_user_data_best.pkl",
        
    }
    
    for model_name in models:
        model = joblib.load("models/"+models[model_name])
        features = getattr(model, "feature_names_in_", None)
        if features is not None:
            X = df[features]
        y = df['owners_class']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(model_name,acc)
        print(classification_report(y, y_pred))