import os
import sys

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import CLEANED_FILE, OD_ORDERS_FILE, MAP_OUTPUT_DIR

SHENZHEN_CENTER = [22.52847, 114.05454]

REF_LAT = 22.52847
REF_LON = 114.05454

# ==========================
# 热力图参数（沿用你喜欢的效果）
# ==========================

HEATMAP_RADIUS = 10
HEATMAP_BLUR = 8
HEATMAP_MIN_OPACITY = 0.4

# ==========================
# DBSCAN参数
# ==========================

EPS_METERS = 100
MIN_SAMPLES = 10


def lonlat_to_meters(lon, lat):
    x = (lon - REF_LON) * 111320 * np.cos(np.radians(REF_LAT))
    y = (lat - REF_LAT) * 110540
    return x, y


def run_dbscan(df, lat_col, lon_col):
    """
    聚类后返回：
    lat
    lng
    count
    """

    x, y = lonlat_to_meters(
        df[lon_col].values,
        df[lat_col].values
    )

    coords = np.column_stack([x, y])

    db = DBSCAN(
        eps=EPS_METERS,
        min_samples=MIN_SAMPLES
    )

    labels = db.fit_predict(coords)

    result = df.copy()
    result['cluster_label'] = labels

    result = result[result['cluster_label'] != -1]

    if len(result) == 0:
        return pd.DataFrame(
            columns=['lat', 'lng', 'count']
        )

    cluster_summary = (
        result
        .groupby('cluster_label')
        .agg(
            lat=(lat_col, 'mean'),
            lng=(lon_col, 'mean'),
            count=(lat_col, 'count')
        )
        .reset_index(drop=True)
    )

    return cluster_summary


def build_heatmap(points, title_for_log):

    m = folium.Map(
        location=SHENZHEN_CENTER,
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    HeatMap(
        points,
        radius=HEATMAP_RADIUS,
        blur=HEATMAP_BLUR,
        min_opacity=HEATMAP_MIN_OPACITY
    ).add_to(m)

    print(f'{title_for_log}：共 {len(points)} 个点用于渲染')

    return m


def main():

    os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)

    # =====================================================
    # 1 车辆位置热力图（原始点）
    # =====================================================

    print('读取车辆位置数据...')

    df_vehicle = pd.read_csv(
        CLEANED_FILE,
        usecols=['lati', 'long']
    )

    sample_size = min(
        100000,
        len(df_vehicle)
    )

    df_vehicle_sample = df_vehicle.sample(
        n=sample_size,
        random_state=42
    )

    points_vehicle = (
        df_vehicle_sample[
            ['lati', 'long']
        ]
        .values
        .tolist()
    )

    m1 = build_heatmap(
        points_vehicle,
        '车辆位置-原始点'
    )

    m1.save(
        os.path.join(
            MAP_OUTPUT_DIR,
            'heatmap_vehicle_nocluster.html'
        )
    )

    # =====================================================
    # 2 车辆位置热力图（聚类版）
    # =====================================================

    print('车辆位置DBSCAN聚类...')

    cluster_vehicle = run_dbscan(
        df_vehicle_sample,
        'lati',
        'long'
    )

    print(
        f'车辆位置聚类得到 {len(cluster_vehicle)} 个热点'
    )

    cluster_vehicle.to_csv(
        os.path.join(
            MAP_OUTPUT_DIR,
            'vehicle_hotspots.csv'
        ),
        index=False,
        encoding='utf-8-sig'
    )

    # 关键：只画聚类中心，不使用count做weight
    points_vehicle_cluster = (
        cluster_vehicle[
            ['lat', 'lng']
        ]
        .values
        .tolist()
    )

    m2 = build_heatmap(
        points_vehicle_cluster,
        '车辆位置-聚类后'
    )

    m2.save(
        os.path.join(
            MAP_OUTPUT_DIR,
            'heatmap_vehicle_clustered.html'
        )
    )

    # =====================================================
    # 3 上车点热力图（原始点）
    # =====================================================

    print('读取上车点数据...')

    df_pickup = pd.read_csv(
        OD_ORDERS_FILE,
        usecols=[
            '开始经度',
            '开始纬度'
        ]
    )

    points_pickup = (
        df_pickup[
            ['开始纬度', '开始经度']
        ]
        .values
        .tolist()
    )

    m3 = build_heatmap(
        points_pickup,
        '上车点-原始点'
    )

    m3.save(
        os.path.join(
            MAP_OUTPUT_DIR,
            'heatmap_pickup_nocluster.html'
        )
    )

    # =====================================================
    # 4 上车点热力图（聚类版）
    # =====================================================

    print('上车点DBSCAN聚类...')

    cluster_pickup = run_dbscan(
        df_pickup,
        '开始纬度',
        '开始经度'
    )

    print(
        f'上车点聚类得到 {len(cluster_pickup)} 个热点'
    )

    cluster_pickup.to_csv(
        os.path.join(
            MAP_OUTPUT_DIR,
            'pickup_hotspots.csv'
        ),
        index=False,
        encoding='utf-8-sig'
    )

    # 同样只画聚类中心
    points_pickup_cluster = (
        cluster_pickup[
            ['lat', 'lng']
        ]
        .values
        .tolist()
    )

    m4 = build_heatmap(
        points_pickup_cluster,
        '上车点-聚类后'
    )

    m4.save(
        os.path.join(
            MAP_OUTPUT_DIR,
            'heatmap_pickup_clustered.html'
        )
    )

    print('=' * 60)
    print('热力图生成完成')
    print('=' * 60)

    print('heatmap_vehicle_nocluster.html')
    print('heatmap_vehicle_clustered.html')
    print('heatmap_pickup_nocluster.html')
    print('heatmap_pickup_clustered.html')

    print('vehicle_hotspots.csv')
    print('pickup_hotspots.csv')


if __name__ == '__main__':
    main()