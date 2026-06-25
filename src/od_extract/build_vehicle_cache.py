import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import CLEANED_FILE, VEHICLE_CACHE_DIR  # noqa: E402

CHUNK_SIZE = 5_000_000


def write_vehicle_chunk(df_chunk):
    """把这一块数据按车辆id拆分，分别追加写入各自的缓存文件"""
    for vid, df_v in df_chunk.groupby('id'):
        file_path = os.path.join(VEHICLE_CACHE_DIR, f'{vid}.csv')
        write_header = not os.path.exists(file_path)
        df_v.to_csv(file_path, mode='a', header=write_header, index=False)


def main():
    os.makedirs(VEHICLE_CACHE_DIR, exist_ok=True)
    print(f'开始构建车辆缓存，输出目录：{VEHICLE_CACHE_DIR}')

    leftover = pd.DataFrame()
    total_rows = 0
    total_vehicles = set()

    reader = pd.read_csv(CLEANED_FILE, chunksize=CHUNK_SIZE, parse_dates=['time'])

    for i, chunk in enumerate(reader):
        chunk = pd.concat([leftover, chunk], ignore_index=True)
        chunk = chunk.sort_values(by=['id', 'time']).reset_index(drop=True)

        last_id = chunk['id'].iloc[-1]
        leftover = chunk[chunk['id'] == last_id].copy()
        chunk_complete = chunk[chunk['id'] != last_id].copy()

        if len(chunk_complete) == 0:
            print(f'第 {i+1} 块：暂无完整车辆数据，继续累积')
            continue

        write_vehicle_chunk(chunk_complete)
        total_rows += len(chunk_complete)
        total_vehicles.update(chunk_complete['id'].unique())

        print(f'第 {i+1} 块完成：写入 {len(chunk_complete)} 行，涉及车辆 {chunk_complete["id"].nunique()} 辆'
              f'（累计车辆数 {len(total_vehicles)}）')

    if len(leftover) > 0:
        write_vehicle_chunk(leftover)
        total_rows += len(leftover)
        total_vehicles.update(leftover['id'].unique())
        print(f'末尾遗留车辆写入完成：{len(leftover)} 行')

    print('=' * 60)
    print(f'车辆缓存构建完成！')
    print(f'总行数：{total_rows}')
    print(f'车辆总数：{len(total_vehicles)}')
    print(f'缓存目录：{VEHICLE_CACHE_DIR}')


if __name__ == '__main__':
    main()