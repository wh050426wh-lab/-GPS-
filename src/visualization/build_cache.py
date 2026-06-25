import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from project_config import MINUTE_CACHE_DIR, DATA_CACHE_DIR

TIME_INTERVAL = 15

files = sorted(os.listdir(MINUTE_CACHE_DIR))

time_list = []
heat_data = []
buffer = []

print('正在加载所有分钟数据（不抽样）...')
for fname in tqdm(files):
    df = pd.read_csv(os.path.join(MINUTE_CACHE_DIR, fname))
    for _, row in df.iterrows():
        buffer.append([row['lat'], row['lng']])

    minute = int(fname.split('_')[1].replace('.csv', ''))
    if minute % TIME_INTERVAL == 0 and buffer:
        time_label = fname.replace('.csv', '').replace('_', ':')
        time_list.append(time_label)
        heat_data.append(buffer.copy())
        buffer = []

cache_file = os.path.join(DATA_CACHE_DIR, f'heatmap_data_{TIME_INTERVAL}min_full.pkl')
with open(cache_file, 'wb') as f:
    pickle.dump({'time_list': time_list, 'heat_data': heat_data}, f)

print(f'已保存缓存: {cache_file}')
print(f'时间片数: {len(time_list)}')
print(f'总点数: {sum(len(h) for h in heat_data)}')
print(f'文件大小: {os.path.getsize(cache_file) / 1024 / 1024:.1f} MB')