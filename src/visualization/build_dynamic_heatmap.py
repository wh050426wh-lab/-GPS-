import os
import sys

import folium
import pandas as pd
from folium.plugins import HeatMapWithTime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import OD_CACHE_DIR, MAP_OUTPUT_DIR  # noqa: E402

SHENZHEN_CENTER = [22.52847, 114.05454]

# 权重归一化上限：count达到这个值就算满权重(1.0)，超过的也封顶在1.0
# 用全局固定值而不是"每个时间片内部的最大值"做归一化，
# 避免单个时间片里偶然出现的超大簇（比如944）把同一时段其他正常簇的权重压到接近0
GLOBAL_WEIGHT_CAP = 150


def main():
    cluster_file = os.path.join(OD_CACHE_DIR, 'dbscan_pickup_clusters.csv')
    print(f'读取DBSCAN聚类结果：{cluster_file}')
    df = pd.read_csv(cluster_file)
    print(f'共 {len(df)} 条聚类记录，覆盖 {df["time"].nunique()} 个时间片')
    print(f'count列范围：{df["count"].min()} ~ {df["count"].max()}')
    print(f'使用全局权重上限：{GLOBAL_WEIGHT_CAP}（即count>={GLOBAL_WEIGHT_CAP}的簇权重封顶为1.0）')

    time_list = sorted(df['time'].unique())

    heat_data = []
    for t in time_list:
        df_t = df[df['time'] == t]
        points = [
            [row['lat'], row['lng'], min(row['count'] / GLOBAL_WEIGHT_CAP, 1.0)]
            for _, row in df_t.iterrows()
        ]
        heat_data.append(points)

    m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap')

    HeatMapWithTime(
        heat_data,
        index=time_list,
        radius=20,
        auto_play=False,
        max_opacity=0.8,
        min_opacity=0.3,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'},
    ).add_to(m)

    output_path = os.path.join(MAP_OUTPUT_DIR, 'dynamic_heatmap_15min.html')
    m.save(output_path)
    print(f'动态热力图已生成：{output_path}')
    print('打开后，地图下方会有时间轴播放控件，可拖动滑块查看不同15分钟时段的热点变化')


if __name__ == '__main__':
    main()