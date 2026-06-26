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

#修改25行输入文件名,113行输出文件名

# 1. 读聚类数据
df = pd.read_csv(os.path.join(OD_CACHE_DIR, 'dbscan_vehicle_clusters_15min.csv'))

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
#cap = result['count'].max()
#result['weight'] = np.log1p(result['count']) / np.log1p(cap)
#result['weight'] = result['weight'].clip(0, 1.0)

# 5. 权重映射（时间片内归一化 + 全局强度系数）
time_totals = result.groupby('time')['count'].sum()
global_max_total = time_totals.max()

for t in time_list:
    mask = result['time'] == t
    local_max = result.loc[mask, 'count'].max()
    time_total = time_totals[t]

    # 帧内用对数归一化（让小值也可见）
    result.loc[mask, 'weight'] = (
            np.log1p(result.loc[mask, 'count']) / np.log1p(local_max)
    )
    # 乘全局强度系数（体现帧间高峰低谷）
    result.loc[mask, 'weight'] *= (time_total / global_max_total)

# 5. 权重映射（对数 + 全局max）
#global_max = result['count'].max()
#result['weight'] = np.log1p(result['count']) / np.log1p(global_max)


#print(f'权重上限(95分位): {cap:.1f}')
print(f'权重范围: {result["weight"].min():.3f} ~ {result["weight"].max():.3f}')

# ===== 数据分布分析 =====
print('\n--- 权重分位数分布 ---')
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f'  {int(q*100):3d}%分位: {result["weight"].quantile(q):.3f}')

print('\n--- 各时间片 count 总和（判断高峰低谷）---')
time_total = result.groupby('time')['count'].sum().sort_index()
for t, total in time_total.items():
    bar = '█' * int(total / time_total.max() * 20)
    print(f'  {t}: {total:8.1f}  {bar}')

print('\n--- 各时间片最大 weight ---')
time_max_weight = result.groupby('time')['weight'].max().sort_index()
for t, w in time_max_weight.items():
    print(f'  {t}: {w:.3f}')
# ========================

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
        0.05: 'blue',
        0.15: 'cyan',
        0.30: 'lime',
        0.52: 'yellow',
        0.80: 'orange',
        1.0: 'red',
    }
).add_to(m)

output = os.path.join(MAP_OUTPUT_DIR, f'{SMOOTH_METHOD}_heatmap_position_15min.html')
m.save(output)

# 8. 注入柱状图
time_totals_sorted = time_total.sort_index()
labels = [str(t) for t in time_totals_sorted.index]
values = [round(float(v), 1) for v in time_totals_sorted.values]

chart_js = f"""
<div id="bar-chart" style="position:fixed;bottom:80px;left:10px;width:400px;height:200px;
background:rgba(0,0,0,0.75);z-index:9999;padding:10px;box-sizing:border-box;
border-radius:8px;cursor:move;">
  <div id="drag-handle" style="color:white;font-size:12px;margin-bottom:5px;">
    📊 各时间片 &nbsp;<span style="float:right;cursor:pointer;" onclick="document.getElementById('bar-chart').style.display='none'">✕</span>
  </div>
  <canvas id="myChart"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
// 拖动逻辑
var el = document.getElementById('bar-chart');
var handle = document.getElementById('drag-handle');
var dragging = false, ox, oy;
handle.addEventListener('mousedown', function(e) {{
  dragging = true;
  ox = e.clientX - el.offsetLeft;
  oy = e.clientY - el.offsetTop;
}});
document.addEventListener('mousemove', function(e) {{
  if (dragging) {{
    el.style.left = (e.clientX - ox) + 'px';
    el.style.top  = (e.clientY - oy) + 'px';
    el.style.bottom = 'auto';
  }}
}});
document.addEventListener('mouseup', function() {{ dragging = false; }});

var labels = {labels};
var values = {values};
var colors = values.map(() => 'gray');
var ctx = document.getElementById('myChart').getContext('2d');
var chart = new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: labels,
    datasets: [{{ data: values, backgroundColor: colors, borderWidth: 0 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: 'white', font: {{ size: 9 }} }} }},
      y: {{ display: false }}
    }}
  }}
}});

// 轮询检测当前时间片
var currentIdx = -1;
setInterval(function() {{
  var dateEl = document.querySelector('.timecontrol-date');
  if (dateEl) {{
    var dateText = dateEl.innerText.trim();
    var idx = labels.indexOf(dateText);
    if (idx !== -1 && idx !== currentIdx) {{
      currentIdx = idx;
      chart.data.datasets[0].backgroundColor = values.map((_, i) => i === idx ? 'red' : 'gray');
      chart.update();
    }}
  }}
}}, 200);
</script>
"""

with open(output, 'r', encoding='utf-8') as f:
    html = f.read()
with open(output, 'w', encoding='utf-8') as f:
    f.write(html.replace('</body>', chart_js + '</body>'))

print('柱状图注入完成')

print(f'已保存: {output}')