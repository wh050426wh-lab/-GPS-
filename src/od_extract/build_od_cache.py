import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import OD_ORDERS_FILE, OD_CACHE_DIR  # noqa: E402


def main():
    os.makedirs(OD_CACHE_DIR, exist_ok=True)
    print(f'开始构建OD缓存，输出目录：{OD_CACHE_DIR}')

    df = pd.read_csv(OD_ORDERS_FILE, parse_dates=['开始时间', '结束时间'])
    print(f'读取OD订单表：{len(df)} 行')

    # ---- 1. 订单表本体缓存（原样复制，供后续直接读取）----
    cache_orders_path = os.path.join(OD_CACHE_DIR, 'od_orders_cache.csv')
    df.to_csv(cache_orders_path, index=False, encoding='utf-8-sig')
    print(f'订单表缓存已保存：{cache_orders_path}')

    # ---- 2. 按小时预聚合统计 ----
    df['小时'] = df['开始时间'].dt.hour

    # 2.1 按小时订单量
    hourly_count = df.groupby('小时').size().rename('订单数').reset_index()
    hourly_count_path = os.path.join(OD_CACHE_DIR, 'hourly_order_count.csv')
    hourly_count.to_csv(hourly_count_path, index=False, encoding='utf-8-sig')
    print(f'按小时订单量已保存：{hourly_count_path}')
    print(hourly_count.to_string(index=False))

    # 2.2 按小时平均距离
    hourly_dist = df.groupby('小时')['订单距离_km'].mean().rename('平均距离_km').reset_index()
    hourly_dist_path = os.path.join(OD_CACHE_DIR, 'hourly_avg_distance.csv')
    hourly_dist.to_csv(hourly_dist_path, index=False, encoding='utf-8-sig')
    print(f'按小时平均距离已保存：{hourly_dist_path}')

    # 2.3 按小时平均时长（转换为分钟，便于阅读）
    df['订单时长_分钟'] = df['订单时长_秒'] / 60
    hourly_duration = df.groupby('小时')['订单时长_分钟'].mean().rename('平均时长_分钟').reset_index()
    hourly_duration_path = os.path.join(OD_CACHE_DIR, 'hourly_avg_duration.csv')
    hourly_duration.to_csv(hourly_duration_path, index=False, encoding='utf-8-sig')
    print(f'按小时平均时长已保存：{hourly_duration_path}')

    print('=' * 60)
    print('OD缓存构建完成！')


if __name__ == '__main__':
    main()