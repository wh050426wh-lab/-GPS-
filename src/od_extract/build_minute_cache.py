import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import VEHICLE_CACHE_DIR, MINUTE_CACHE_DIR  # noqa: E402

# 每批处理多少辆车后，集中写一次磁盘（避免内存中无限堆积所有分钟数据）
BATCH_SIZE = 500


def resample_one_vehicle(file_path):
    """读取一辆车的缓存文件，按分钟重采样，返回 DataFrame（索引为分钟时间）"""
    df = pd.read_csv(file_path, parse_dates=['time'])
    if len(df) == 0:
        return None

    vid = df['id'].iloc[0]
    df = df.set_index('time').sort_index()

    # 按分钟重采样：取每分钟最后一条记录，缺失的分钟用前值填充
    df_resampled = df[['long', 'lati', 'status', 'speed']].resample('1min').last().ffill()
    df_resampled['id'] = vid
    df_resampled = df_resampled.reset_index()  # 把 time 从索引变回普通列

    return df_resampled


def flush_batch(minute_buffer):
    """把缓存在内存里的、按分钟分组的数据，追加写入对应的分钟文件"""
    for minute_str, rows in minute_buffer.items():
        file_path = os.path.join(MINUTE_CACHE_DIR, f'{minute_str}.csv')
        write_header = not os.path.exists(file_path)
        df_minute = pd.DataFrame(rows)
        df_minute.to_csv(file_path, mode='a', header=write_header, index=False)


def main():
    os.makedirs(MINUTE_CACHE_DIR, exist_ok=True)
    print(f'开始构建分钟缓存，输出目录：{MINUTE_CACHE_DIR}')

    vehicle_files = [f for f in os.listdir(VEHICLE_CACHE_DIR) if f.endswith('.csv')]
    print(f'车辆缓存文件总数：{len(vehicle_files)}')

    minute_buffer = defaultdict(list)
    total_minute_points = 0
    processed_vehicles = 0

    for i, fname in enumerate(vehicle_files):
        file_path = os.path.join(VEHICLE_CACHE_DIR, fname)
        df_resampled = resample_one_vehicle(file_path)

        if df_resampled is not None:
            # 按分钟分组，暂存到内存buffer
            for _, row in df_resampled.iterrows():
                minute_str = row['time'].strftime('%H-%M')  # 文件名格式：08-05.csv
                minute_buffer[minute_str].append({
                    'time': row['time'],
                    'id': row['id'],
                    'long': row['long'],
                    'lati': row['lati'],
                    'status': row['status'],
                    'speed': row['speed'],
                })
            total_minute_points += len(df_resampled)

        processed_vehicles += 1

        # 每处理 BATCH_SIZE 辆车，集中落盘一次，释放内存
        if processed_vehicles % BATCH_SIZE == 0:
            flush_batch(minute_buffer)
            minute_buffer = defaultdict(list)
            print(f'已处理 {processed_vehicles}/{len(vehicle_files)} 辆车，'
                  f'累计分钟级记录 {total_minute_points} 条')

    # 处理完所有车辆后，把buffer里剩余的数据落盘
    if minute_buffer:
        flush_batch(minute_buffer)

    print('=' * 60)
    print('分钟缓存构建完成！')
    print(f'处理车辆数：{processed_vehicles}')
    print(f'累计分钟级记录：{total_minute_points}')
    print(f'缓存目录：{MINUTE_CACHE_DIR}')


if __name__ == '__main__':
    main()