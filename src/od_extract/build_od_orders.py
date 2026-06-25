import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import OD_MARKERS_FILE, OD_ORDERS_FILE  # noqa: E402

# 异常订单判断阈值
MIN_ORDER_SECONDS = 60           # 订单最短60秒，太短不合理
MAX_ORDER_SECONDS = 4 * 3600     # 订单最长4小时，超过视为异常
MIN_ORDER_DIST_KM = 0.1          # 订单最短100米，太近（包括0）不合理
MAX_ORDER_DIST_KM = 100          # 订单最长100公里，超过视为异常
MIN_AVG_SPEED_KMH = 3            # 平均速度低于此值（比步行还慢）视为异常


def haversine_km(lon1, lat1, lon2, lat2):
    """计算两点间球面距离（公里），用于判断订单距离是否异常"""
    import numpy as np
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv(OD_MARKERS_FILE, parse_dates=['time'])
    df = df.sort_values(by=['id', 'time']).reset_index(drop=True)
    print(f'读取标记数据：{len(df)} 行（pickup+dropoff）')

    # 错位拼接：把下一行的时间/经纬度/类型平移到当前行
    df['next_time'] = df['time'].shift(-1)
    df['next_long'] = df['long'].shift(-1)
    df['next_lati'] = df['lati'].shift(-1)
    df['next_id'] = df['id'].shift(-1)
    df['next_type'] = df['point_type'].shift(-1)

    # 筛选：当前是pickup，下一行是同一车辆的dropoff
    cond = (df['point_type'] == 'pickup') & \
           (df['next_type'] == 'dropoff') & \
           (df['id'] == df['next_id'])

    df_order = df.loc[cond, ['id', 'time', 'long', 'lati', 'next_time', 'next_long', 'next_lati']].copy()
    df_order.columns = ['车辆id', '开始时间', '开始经度', '开始纬度', '结束时间', '结束经度', '结束纬度']

    n_raw_orders = len(df_order)
    print(f'初步配对订单数：{n_raw_orders}')

    # ---- 异常订单处理 ----
    df_order['订单时长_秒'] = (df_order['结束时间'] - df_order['开始时间']).dt.total_seconds()
    df_order['订单距离_km'] = haversine_km(
        df_order['开始经度'], df_order['开始纬度'],
        df_order['结束经度'], df_order['结束纬度']
    )

    # 1. 时间为负（结束早于开始，理论上不该出现，但做一次保险检查）
    cond_neg_time = df_order['订单时长_秒'] <= 0
    n_neg_time = int(cond_neg_time.sum())

    # 2. 时长太短（小于60秒，乘客坐车不可能这么短）
    cond_short_time = df_order['订单时长_秒'] < MIN_ORDER_SECONDS
    n_short_time = int(cond_short_time.sum())

    # 3. 时长明显异常（超过4小时）
    cond_long_time = df_order['订单时长_秒'] > MAX_ORDER_SECONDS
    n_long_time = int(cond_long_time.sum())

    # 4. 距离太近（小于100米，包括距离为0的情况）
    cond_short_dist = df_order['订单距离_km'] < MIN_ORDER_DIST_KM
    n_short_dist = int(cond_short_dist.sum())

    # 5. 距离明显异常（超过100公里）
    cond_long_dist = df_order['订单距离_km'] > MAX_ORDER_DIST_KM
    n_long_dist = int(cond_long_dist.sum())

    # 6. 平均速度异常低（防止"近距离配长时间"这种组合被前面条件漏掉）
    #    时长为0或负的行不参与速度计算，避免除零问题，用 cond_neg_time 排除
    df_order['平均速度_kmh'] = df_order['订单距离_km'] / (df_order['订单时长_秒'] / 3600)
    cond_too_slow = (~cond_neg_time) & (df_order['平均速度_kmh'] < MIN_AVG_SPEED_KMH)
    n_too_slow = int(cond_too_slow.sum())

    cond_abnormal = (cond_neg_time | cond_short_time | cond_long_time |
                      cond_short_dist | cond_long_dist | cond_too_slow)
    n_abnormal = int(cond_abnormal.sum())

    df_order_clean = df_order.loc[~cond_abnormal].reset_index(drop=True)

    print(f'异常订单明细：')
    print(f'  - 时间为负：{n_neg_time}')
    print(f'  - 时长太短(<{MIN_ORDER_SECONDS}秒)：{n_short_time}')
    print(f'  - 时长太长(>{MAX_ORDER_SECONDS/3600:.0f}小时)：{n_long_time}')
    print(f'  - 距离太近(<{MIN_ORDER_DIST_KM}km)：{n_short_dist}')
    print(f'  - 距离太远(>{MAX_ORDER_DIST_KM}km)：{n_long_dist}')
    print(f'  - 平均速度过慢(<{MIN_AVG_SPEED_KMH}km/h)：{n_too_slow}')
    print(f'异常订单总计（去重后）：{n_abnormal}')
    print(f'最终有效订单数：{len(df_order_clean)}')

    df_order_clean.to_csv(OD_ORDERS_FILE, index=False, encoding='utf-8-sig')
    print(f'OD订单表已保存到：{OD_ORDERS_FILE}')

    return df_order_clean


if __name__ == '__main__':
    main()