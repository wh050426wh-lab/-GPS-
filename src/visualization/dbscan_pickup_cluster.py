import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import OD_ORDERS_FILE, OD_CACHE_DIR  # noqa: E402

# 深圳市参考点（用于经纬度转米的近似换算）
REF_LAT = 22.52847
REF_LON = 114.05454

# DBSCAN参数（单位：米）
EPS_METERS = 150       # 从500改成200米
MIN_SAMPLES = 10        # 适当提高，避免太多零散小簇

# 时间分段颗粒度（分钟）
TIME_BIN_MINUTES = 15


def lonlat_to_meters(lon, lat):
    """将经纬度转换为以参考点为原点的平面坐标（米），用于DBSCAN距离计算"""
    x = (lon - REF_LON) * 111320 * np.cos(np.radians(REF_LAT))
    y = (lat - REF_LAT) * 110540
    return x, y


def cluster_one_timebin(df_bin):
    """对某个时间段内的上车点做DBSCAN聚类，返回每个簇的中心点和热力值"""
    if len(df_bin) < MIN_SAMPLES:
        return pd.DataFrame()

    x, y = lonlat_to_meters(df_bin['开始经度'].values, df_bin['开始纬度'].values)
    coords = np.column_stack([x, y])

    db = DBSCAN(eps=EPS_METERS, min_samples=MIN_SAMPLES).fit(coords)
    df_bin = df_bin.copy()
    df_bin['cluster_label'] = db.labels_

    # label为-1表示噪声点，不属于任何簇，要排除
    df_valid = df_bin[df_bin['cluster_label'] != -1]
    if len(df_valid) == 0:
        return pd.DataFrame()

    # 每个簇的中心点（取簇内点的经纬度均值）和热力值（簇内点数量）
    cluster_summary = df_valid.groupby('cluster_label').agg(
        lat=('开始纬度', 'mean'),
        lng=('开始经度', 'mean'),
        count=('开始纬度', 'count'),
    ).reset_index(drop=True)

    return cluster_summary


def main():
    print(f'读取OD订单表：{OD_ORDERS_FILE}')
    df = pd.read_csv(OD_ORDERS_FILE, parse_dates=['开始时间'])
    print(f'共 {len(df)} 个上车点')

    # 按15分钟分段：把开始时间向下取整到最近的15分钟
    df['time_bin'] = df['开始时间'].dt.floor(f'{TIME_BIN_MINUTES}min')

    all_clusters = []
    time_bins = sorted(df['time_bin'].unique())
    print(f'共 {len(time_bins)} 个时间段（每{TIME_BIN_MINUTES}分钟一段）')

    for i, t_bin in enumerate(time_bins):
        df_bin = df[df['time_bin'] == t_bin]
        cluster_result = cluster_one_timebin(df_bin)

        if len(cluster_result) > 0:
            cluster_result['time'] = t_bin.strftime('%H:%M:%S')
            all_clusters.append(cluster_result)

        if (i + 1) % 20 == 0 or i == len(time_bins) - 1:
            print(f'已处理 {i+1}/{len(time_bins)} 个时间段')

    if not all_clusters:
        print('没有生成任何聚类结果，请检查参数设置')
        return

    df_result = pd.concat(all_clusters, ignore_index=True)
    df_result = df_result[['lat', 'lng', 'count', 'time']]

    output_path = os.path.join(OD_CACHE_DIR, 'dbscan_pickup_clusters.csv')
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print('=' * 50)
    print(f'聚类完成！共生成 {len(df_result)} 个簇（跨所有时间段）')
    print(f'平均每个时间段产生 {len(df_result)/len(time_bins):.1f} 个簇')
    print(f'结果已保存：{output_path}')
    # 检查是否有占比异常大的簇（可能是参数设置过松导致的合并问题）
    total_points_per_bin = df.groupby('time_bin').size()
    for t_bin in time_bins:
        t_str = t_bin.strftime('%H:%M:%S')
        bin_clusters = df_result[df_result['time'] == t_str]
        if len(bin_clusters) > 0:
            max_count = bin_clusters['count'].max()
            total_in_bin = total_points_per_bin.get(t_bin, 0)
            if total_in_bin > 0 and max_count / total_in_bin > 0.4:
                print(f'⚠️ 警告：{t_str} 时段最大簇占比 {max_count / total_in_bin * 100:.1f}%，可能eps设置过大')
    print(df_result.head(10))
    print('\n簇大小（count）分布统计：')
    print(df_result['count'].describe())
    print(f'\ncount > 500 的簇数量：{(df_result["count"] > 500).sum()}')
    print(f'count > 200 的簇数量：{(df_result["count"] > 200).sum()}')


if __name__ == '__main__':
    main()