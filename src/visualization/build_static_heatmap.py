import os
import sys

import folium
import pandas as pd
from folium.plugins import HeatMap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import CLEANED_FILE, OD_ORDERS_FILE, MAP_OUTPUT_DIR  # noqa: E402

SHENZHEN_CENTER = [22.52847, 114.05454]


def build_vehicle_position_heatmap():
    """车辆位置静态热力图：基于全天清洗后的GPS数据，反映车辆整体活动覆盖范围"""
    print('正在读取清洗后数据用于车辆位置热力图...')
    # 全天数据量很大（3887万行），热力图不需要每个点都用，抽样即可反映分布规律
    df = pd.read_csv(CLEANED_FILE, usecols=['lati', 'long'])
    sample_size = min(500_000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f'抽样 {sample_size} 行用于热力图（原始数据 {len(df)} 行）')

    m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap')
    heat_data = df_sample[['lati', 'long']].values.tolist()
    HeatMap(heat_data, radius=6, blur=4, min_opacity=0.3).add_to(m)

    output_path = os.path.join(MAP_OUTPUT_DIR, 'heatmap_vehicle_position.html')
    m.save(output_path)
    print(f'车辆位置热力图已保存：{output_path}')


def build_pickup_point_heatmap():
    """上车点静态热力图：基于OD订单表的开始经纬度，反映真实乘客需求热点"""
    print('正在读取OD订单数据用于上车点热力图...')
    df = pd.read_csv(OD_ORDERS_FILE, usecols=['开始经度', '开始纬度'])
    print(f'共 {len(df)} 个上车点')

    m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap')
    heat_data = df[['开始纬度', '开始经度']].values.tolist()
    HeatMap(heat_data, radius=8, blur=6, min_opacity=0.3,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}).add_to(m)

    output_path = os.path.join(MAP_OUTPUT_DIR, 'heatmap_pickup_points.html')
    m.save(output_path)
    print(f'上车点热力图已保存：{output_path}')


def main():
    os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)
    build_vehicle_position_heatmap()
    print('-' * 50)
    build_pickup_point_heatmap()


if __name__ == '__main__':
    main()