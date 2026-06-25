import os
import sys
import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMapWithTime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from project_config import OD_CACHE_DIR, MAP_OUTPUT_DIR

SHENZHEN_CENTER = [22.52847, 114.05454]

# ==================== 参数 ====================
SMOOTH_METHOD = 'ema'     # 'ema' 或 'wma'
EMA_ALPHA = 0.3           # EMA平滑系数
WMA_WINDOW = 3            # WMA窗口大小
RADIUS = 25               # 热力半径
MAX_OPACITY = 0.5         # 最大透明度
MIN_OPACITY = 0.2         # 最小透明度
ROUND_PRECISION = 3       # 坐标精度：3=100米, 4=10米
# =============================================

# 1. 读聚类数据
df = pd.read_csv(os.path.join(OD_CACHE_DIR, 'dbscan_pickup_clusters.csv'))

# 筛选高峰期
df['hour'] = df['time'].str[:2].astype(int)
df = df[(df['hour'] >= 7) & (df['hour'] <= 9) |
        (df['hour'] >= 17) & (df['hour'] <= 19)]

print(f'聚类数据: {len(df)} 条, {df["time"].nunique()} 个时间片')

# 2. 创建簇ID（round(3)=100米网格，簇更融合）
df['cluster_id'] = (df['lat'].round(ROUND_PRECISION).astype(str) + '_' +
                    df['lng'].round(ROUND_PRECISION).astype(str))

time_list = sorted(df['time'].unique())

# 同一网格内的簇 count 求和
pivot = df.pivot_table(
    index='time', columns='cluster_id',
    values='count', aggfunc='sum', fill_value=0
).reindex(time_list, fill_value=0)

print(f'矩阵: {pivot.shape[0]} 时间片 × {pivot.shape[1]} 簇')
print(f'count范围: {pivot.values.min():.0f} ~ {pivot.values.max():.0f}')

# 3. 平滑
if SMOOTH_METHOD == 'ema':
    smoothed = pivot.copy().astype(float)
    for i in range(1, len(smoothed)):
        smoothed.iloc[i] = EMA_ALPHA * pivot.iloc[i] + (1 - EMA_ALPHA) * smoothed.iloc[i-1]
    print(f'EMA平滑 (alpha={EMA_ALPHA}) 完成')

elif SMOOTH_METHOD == 'wma':
    weights = np.arange(1, WMA_WINDOW + 1, dtype=float)
    weights = weights / weights.sum()
    smoothed = pivot.copy().astype(float)
    values = smoothed.values
    for i in range(len(values)):
        start = max(0, i - WMA_WINDOW + 1)
        window = values[start:i+1]
        w = weights[-(i-start+1):]
        values[i] = np.average(window, axis=0, weights=w)
    smoothed = pd.DataFrame(values, index=pivot.index, columns=pivot.columns)
    print(f'WMA平滑 (window={WMA_WINDOW}) 完成')

print(f'平滑后范围: {smoothed.values.min():.2f} ~ {smoothed.values.max():.2f}')

# 4. 还原为长格式
result = smoothed.reset_index().melt(
    id_vars='time', var_name='cluster_id', value_name='count'
)
result = result[result['count'] > 0.1].copy()
result[['lat', 'lng']] = result['cluster_id'].str.split('_', expand=True).astype(float)

# 5. 权重映射（对数映射，让小簇也可见）
cap = result['count'].quantile(0.95)
result['weight'] = np.log1p(result['count']) / np.log1p(cap)
result['weight'] = result['weight'].clip(0, 1.0)

print(f'权重上限(95分位): {cap:.1f}')
print(f'权重范围: {result["weight"].min():.3f} ~ {result["weight"].max():.3f}')

# 打印权重分布
for thresh, color in [(0.2, '蓝'), (0.4, '青'), (0.6, '绿'), (0.8, '橙')]:
    pct = (result['weight'] >= thresh).sum() / len(result) * 100
    print(f'  权重>={thresh}: {pct:.0f}% → {color}色以上')

# 6. 构建热力数据
heat_data = []
for t in time_list:
    df_t = result[result['time'] == t]
    points = [[row['lat'], row['lng'], row['weight']] for _, row in df_t.iterrows()]
    heat_data.append(points)

print(f'生成 {len(heat_data)} 个时间片')

# 7. 画图
m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap')
HeatMapWithTime(
    heat_data,
    index=time_list,
    radius=RADIUS,
    max_opacity=MAX_OPACITY,
    min_opacity=MIN_OPACITY,
    gradient={
        0.2: 'blue',
        0.4: 'cyan',
        0.6: 'lime',
        0.75: 'yellow',
        0.85: 'orange',
        1.0: 'red',
    },
).add_to(m)

output = os.path.join(MAP_OUTPUT_DIR, f'{SMOOTH_METHOD}_heatmap.html')
m.save(output)
print(f'已保存: {output}')