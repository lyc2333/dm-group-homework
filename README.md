# dm-group-homework

运行代码前，请从https://www.kaggle.com/datasets/vikawenzel/scraped-steam-data-games-reviews-charts 下载 steam_games.ndjson数据集，并置于项目根目录。

## 相关代码

+ `steam_spider.py`: 从steamspy爬取元数据
+ `pre_experiment/*`：数据分析与可视化相关代码
+ `boolean_feature_importance_*.py`: 分析列表类元素的特征重要性
+ `get_sampled_game_detail.py`: 抽样并从gamalytic爬取具体销量
+ `get_data.py`: 从steam和steamspy爬取完整数据，并转换为和训练数据一致的格式
+ `train_predict_model.py`: 训练分类模型，并打印测试集结果
+ `train_no_user_data_predict_model.py`: 训练分类模型，不使用不含发售后信息，并打印测试集结果
+ `train_detail_price.py`: 利用gamalytic爬取数据预测具体销量，并打印测试集结果
+ `predict_new_data.py`: 预测爬取的2025年新增数据
+ `server.py`: 服务器后端代码
+ `templates/index.html`: 服务器前端代码

