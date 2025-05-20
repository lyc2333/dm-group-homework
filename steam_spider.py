# standard library imports
import csv
import datetime as dt
import json
import os
import statistics
import time
import pathlib

# third-party imports
import numpy as np
import pandas as pd
import requests

# customisations - ensure tables show all columns
# pd.set_option("max_columns", 100)

from ssl import SSLError


def get_request(url, parameters=None):
    """Return json-formatted response of a get request using optional parameters.
    
    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request
    
    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)
        
        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' '*10)
        
        # recusively try again
        return get_request(url, parameters)
    
    if response:
        return response.json()
    else:
        # response is none usually means too many requests. Wait and try again 
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying.')
        return get_request(url, parameters)

path = 'data/download'
path = pathlib.Path(path)
path.mkdir(parents=True,exist_ok=True)

for i in range(0,80):
    url = "https://steamspy.com/api.php"
    parameters = {"request": "all",'page':i }

    # request 'all' from steam spy and parse into dataframe
    json_data = get_request(url, parameters=parameters)
    steam_spy_all = pd.DataFrame.from_dict(json_data, orient='index')

    # generate sorted app_list from steamspy data
    app_list = steam_spy_all.sort_values('appid').reset_index(drop=True)

    # export disabled to keep consistency across download sessions
    app_list.to_csv(path/f"app_list_{i:03d}.csv", index=False)
    time.sleep(10)

# 合并CSV文件

import csv
import glob
import heapq

def merge_csv_files(input_pattern, output_file):
    # 找到所有匹配的CSV文件
    files = sorted(glob.glob(input_pattern))

    # 打开所有文件的CSV reader
    readers = []
    for file in files:
        f = open(file, 'r', newline='', encoding='utf-8')
        reader = csv.DictReader(f)
        readers.append((reader, f))  # 保存reader和对应的文件句柄

    # 准备输出文件
    with open(output_file, 'w', newline='', encoding='utf-8') as fout:
        writer = None
        
        # 初始化堆
        heap = []
        
        for idx, (reader, _) in enumerate(readers):
            try:
                row = next(reader)
                heapq.heappush(heap, (int(row['appid']), idx, row))
            except StopIteration:
                pass  # 空文件也能处理
        
        while heap:
            id_val, idx, row = heapq.heappop(heap)
            if writer is None:
                # 初始化写入器，写入表头
                writer = csv.DictWriter(fout, fieldnames=row.keys())
                writer.writeheader()
            
            writer.writerow(row)
            
            reader, _ = readers[idx]
            try:
                next_row = next(reader)
                heapq.heappush(heap, (int(next_row['appid']), idx, next_row))
            except StopIteration:
                pass

    # 关闭所有打开的文件
    for _, f in readers:
        f.close()


merge_csv_files(r'data\download\app_list_0*.csv', 'merged_output.csv')
