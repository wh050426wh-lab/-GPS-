import os
import sys

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import json
import numpy as np
import streamlit.components.v1 as components
from folium.plugins import Draw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import VEHICLE_CACHE_DIR, MINUTE_CACHE_DIR, SHENZHEN_BOUNDARY_FILE  # noqa: E402

SHENZHEN_CENTER = [22.52847, 114.05454]
DATA_DATE = '2013-10-22'

MAX_VEHICLES_FOR_ANIMATION = 6
MULTI_VEHICLE_COLORS = ['#E63946', '#1d3557', '#2a9d8f', '#f4a261', '#8338ec', '#fb8500', '#3a86ff', '#06D6A0']


@st.cache_data
def list_vehicle_ids():
    files = [f.replace('.csv', '') for f in os.listdir(VEHICLE_CACHE_DIR) if f.endswith('.csv')]
    return sorted(files, key=lambda x: int(x))


@st.cache_data
def load_vehicle_data(vehicle_id):
    file_path = os.path.join(VEHICLE_CACHE_DIR, f'{vehicle_id}.csv')
    df = pd.read_csv(file_path, parse_dates=['time'])
    return df.sort_values(by='time').reset_index(drop=True)


@st.cache_data
def load_minute_snapshot(hour, minute):
    minute_str = f'{hour:02d}-{minute:02d}'
    file_path = os.path.join(MINUTE_CACHE_DIR, f'{minute_str}.csv')
    if not os.path.exists(file_path):
        return pd.DataFrame()
    return pd.read_csv(file_path, parse_dates=['time'])


@st.cache_data
def load_shenzhen_boundary():
    with open(SHENZHEN_BOUNDARY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_boundary_to_map(target, boundary_geojson):
    folium.GeoJson(
        boundary_geojson,
        name='深圳市边界',
        style_function=lambda feature: {
            'fillColor': 'transparent',
            'color': '#3388ff',
            'weight': 3.5,
            'fillOpacity': 0,
            'dashArray': '5, 5',
        },
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['区域：']),
    ).add_to(target)
    return target


def add_road_match_banner(m, enabled):
    if not enabled:
        return m
    banner_html = '''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 9999;
                background-color: #FFF3CD; color: #664d03; padding: 6px 16px;
                border: 1px solid #ffe69c; border-radius: 6px; font-size: 13px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.15);">
        ⚠️ 路网校正：功能开发中，当前显示原始GPS轨迹（未做道路匹配）
    </div>
    '''
    m.get_root().html.add_child(folium.Element(banner_html))
    return m


def extract_bbox_from_drawing(drawing):
    if not drawing:
        return None
    geometry = drawing.get('geometry', {})
    if geometry.get('type') != 'Polygon':
        return None
    coords = geometry.get('coordinates', [[]])[0]
    if len(coords) < 3:
        return None
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lats), max(lats), min(lngs), max(lngs)


def filter_by_bbox(df, bbox):
    if bbox is None or len(df) == 0:
        return df
    south, north, west, east = bbox
    return df[(df['lati'] >= south) & (df['lati'] <= north) &
              (df['long'] >= west) & (df['long'] <= east)]


def find_nearest_vehicle(df, click_lat, click_lng):
    if len(df) == 0:
        return None
    distances = np.sqrt((df['lati'] - click_lat) ** 2 + (df['long'] - click_lng) ** 2)
    nearest_idx = distances.idxmin()
    return df.loc[nearest_idx]


# =====================================================================
# 核心改动1：统一的图层面板构建函数
# 单车模式：每层带车辆ID前缀；多车模式：每辆车独立一套图层
# =====================================================================

def add_standard_layers(m, df, vehicle_id, color, show_all=True):
    """
    为单辆车添加标准6图层（与图示一致）：
    载客轨迹 / 空载轨迹 / 上车点 / 下车点 / 起点/终点 / 深圳市边界
    vehicle_id: 用于图层命名区分（多车时显示各自ID）
    color: 该车的主色（多车对比时每车不同色）
    """
    prefix = f'车辆{vehicle_id}·' if vehicle_id else ''

    df = df.copy()
    df['status_grp'] = (df['status'] != df['status'].shift()).cumsum()

    fg_busy = folium.FeatureGroup(name=f'{prefix}载客轨迹', show=show_all)
    fg_idle = folium.FeatureGroup(name=f'{prefix}空载轨迹', show=show_all)

    for _, seg in df.groupby('status_grp'):
        if len(seg) < 2:
            continue
        points = seg[['lati', 'long']].values.tolist()
        if seg['status'].iloc[0] == 1:
            folium.PolyLine(points, color=color, weight=4, opacity=0.85,
                            tooltip=f'{prefix}载客').add_to(fg_busy)
        else:
            # 空载用同色但半透明虚线，与载客实线区分
            folium.PolyLine(points, color=color, weight=2.5, opacity=0.45,
                            dash_array='6, 5', tooltip=f'{prefix}空载').add_to(fg_idle)

    fg_busy.add_to(m)
    fg_idle.add_to(m)

    fg_pickup = folium.FeatureGroup(name=f'{prefix}上车点', show=show_all)
    fg_dropoff = folium.FeatureGroup(name=f'{prefix}下车点', show=show_all)

    df['status_prev'] = df['status'].shift(1)
    for _, row in df[(df['status'] == 1) & (df['status_prev'] == 0)].iterrows():
        folium.CircleMarker(
            [row['lati'], row['long']], radius=6,
            color=color, fill=True, fill_color='#ffffff', fill_opacity=0.9, weight=2.5,
            popup=f"上车 {row['time']}", tooltip=f'{prefix}上车',
        ).add_to(fg_pickup)
    for _, row in df[(df['status'] == 0) & (df['status_prev'] == 1)].iterrows():
        folium.CircleMarker(
            [row['lati'], row['long']], radius=6,
            color=color, fill=True, fill_color=color, fill_opacity=0.7, weight=2,
            popup=f"下车 {row['time']}", tooltip=f'{prefix}下车',
        ).add_to(fg_dropoff)

    fg_pickup.add_to(m)
    fg_dropoff.add_to(m)

    fg_endpoints = folium.FeatureGroup(name=f'{prefix}起点/终点', show=show_all)
    folium.Marker(
        [df.iloc[0]['lati'], df.iloc[0]['long']],
        popup=f"起点 {df.iloc[0]['time']}",
        icon=folium.Icon(color='blue', icon='home'),
        tooltip=f'{prefix}起点',
    ).add_to(fg_endpoints)
    folium.Marker(
        [df.iloc[-1]['lati'], df.iloc[-1]['long']],
        popup=f"终点 {df.iloc[-1]['time']}",
        icon=folium.Icon(color='black', icon='flag'),
        tooltip=f'{prefix}终点',
    ).add_to(fg_endpoints)
    fg_endpoints.add_to(m)

    return m


def build_trajectory_map(df, vehicle_id, road_match=False):
    """单车静态轨迹地图，图层面板与图示一致"""
    if len(df) == 0:
        return None
    center_lat, center_lon = df['lati'].mean(), df['long'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

    # 单车用红色（载客实线）+ 绿色（空载虚线）的传统配色
    add_standard_layers(m, df, vehicle_id, color='#E63946')

    boundary = load_shenzhen_boundary()
    fg_boundary = folium.FeatureGroup(name='深圳市边界', show=True)
    add_boundary_to_map(fg_boundary, boundary)
    fg_boundary.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    add_road_match_banner(m, road_match)
    return m


def build_combined_map_html(df, vehicle_id, road_match=False):
    """单车：静态轨迹（背景）+ 动态播放叠加，图层面板统一"""
    if len(df) == 0:
        return None

    m = build_trajectory_map(df, vehicle_id, road_match=road_match)

    df = df.copy().reset_index(drop=True)
    df['time_diff_sec'] = df['time'].diff().dt.total_seconds().fillna(0)

    points = []
    for _, row in df.iterrows():
        points.append({
            'lat': row['lati'], 'lng': row['long'],
            'status': int(row['status']), 'speed': float(row['speed']),
            'time': row['time'].strftime('%H:%M:%S'),
            'time_diff': float(row['time_diff_sec']),
        })

    total_real_seconds = df['time_diff_sec'].sum()
    target_play_seconds = 20
    time_scale = target_play_seconds / total_real_seconds if total_real_seconds > 0 else 1

    points_json = json.dumps(points)
    map_var_name = m.get_name()

    animation_script = f'''
    <div id="anim-controls" style="position:fixed; bottom:30px; right:20px; z-index:9999;
         background:rgba(255,255,255,0.95); padding:12px 16px;
         border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.2);
         font-family:'Microsoft YaHei',sans-serif; min-width:180px;">
        <div style="margin-bottom:10px;">
            <button id="playBtn" style="padding:6px 14px; font-size:14px; cursor:pointer;
                background:#2196F3; color:white; border:none; border-radius:6px;">▶ 播放动画</button>
            <button id="resetBtn" style="padding:6px 14px; font-size:14px; cursor:pointer;
                background:#6c757d; color:white; border:none; border-radius:6px; margin-left:6px;">⟲ 重置</button>
        </div>
        <div style="width:100%; height:5px; background:#e9ecef; border-radius:3px; margin-bottom:8px;">
            <div id="progressFill" style="height:100%; width:0%; background:#2196F3;
                 border-radius:3px; transition:width 0.1s;"></div>
        </div>
        <div id="statusBox" style="font-size:12px; color:#555; margin-bottom:10px; min-height:40px;">
            点击播放后显示状态
        </div>
        <div style="border-top:1px solid #e9ecef; margin-bottom:8px;"></div>
        <div style="font-size:12px; color:#333; line-height:2;">
            <div><span style="display:inline-block;width:24px;height:4px;
                background:#E63946;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>载客轨迹</div>
            <div><span style="display:inline-block;width:24px;height:4px;
                background:#06D6A0;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>空载轨迹</div>
            <div>🔵 起点 &nbsp; ⚫ 终点</div>
        </div>
    </div>
    <script>
    (function() {{
        var points = {points_json};
        var timeScale = {time_scale};
        var totalPoints = points.length;
        var idx = 0, timerHandle = null;
        function waitForMap() {{
            if (typeof {map_var_name} === 'undefined') {{ setTimeout(waitForMap, 100); return; }}
            var targetMap = {map_var_name};
            var carIcon = L.divIcon({{ html: '🚕', iconSize: [30, 30], className: '' }});
            var carMarker = L.marker([points[0].lat, points[0].lng], {{icon: carIcon}}).addTo(targetMap);
            var trailLine = L.polyline([], {{color: '#1d3557', weight: 5, opacity: 0.9}}).addTo(targetMap);
            var statusBox    = document.getElementById('statusBox');
            var progressFill = document.getElementById('progressFill');
            var playBtn      = document.getElementById('playBtn');
            function speedColor(s) {{
                if (s === 0) return '#999';
                if (s < 20)  return '#06D6A0';
                if (s < 50)  return '#FFD166';
                return '#E63946';
            }}
            function step() {{
                if (idx >= totalPoints) {{
                    timerHandle = null;
                    playBtn.innerText = '✅ 播放完成';
                    progressFill.style.width = '100%';
                    return;
                }}
                var p = points[idx];
                carMarker.setLatLng([p.lat, p.lng]);
                trailLine.addLatLng([p.lat, p.lng]);
                progressFill.style.width = (idx / totalPoints * 100) + '%';
                statusBox.innerHTML =
                    '<b>时间</b>：' + p.time + '<br>' +
                    '<b>状态</b>：' + (p.status === 1 ? '🔴 载客' : '🟢 空载') + '<br>' +
                    '<b>速度</b>：<span style="color:' + speedColor(p.speed) +
                    ';font-weight:bold;">' + p.speed + ' km/h</span>';
                idx++;
                var delay = idx < totalPoints
                    ? Math.min(Math.max(points[idx].time_diff * timeScale * 1000, 30), 800) : 0;
                timerHandle = setTimeout(step, delay);
            }}
            playBtn.onclick = function() {{
                if (timerHandle) return;
                this.innerText = '⏸ 播放中...';
                step();
            }};
            document.getElementById('resetBtn').onclick = function() {{
                clearTimeout(timerHandle);
                timerHandle = null; idx = 0;
                carMarker.setLatLng([points[0].lat, points[0].lng]);
                trailLine.setLatLngs([]);
                statusBox.innerText = '点击播放后显示状态';
                progressFill.style.width = '0%';
                playBtn.innerText = '▶ 播放动画';
            }};
        }}
        waitForMap();
    }})();
    </script>
    '''
    return m.get_root().render() + animation_script


# =====================================================================
# 核心改动2：多车地图 —— 图层面板与单车统一，每辆车颜色区分
# =====================================================================

def build_multi_vehicle_animation_html(vehicle_dfs, color_map, start_time, road_match=False):
    vehicle_dfs = {vid: df for vid, df in vehicle_dfs.items() if len(df) > 0}
    if not vehicle_dfs:
        return None

    all_lats = pd.concat([df['lati'] for df in vehicle_dfs.values()])
    all_lons = pd.concat([df['long'] for df in vehicle_dfs.values()])
    m = folium.Map(location=[all_lats.mean(), all_lons.mean()], zoom_start=12,
                   tiles='OpenStreetMap')

    vehicles_payload = []
    max_elapsed = 0.0

    for vid, df in vehicle_dfs.items():
        color = color_map.get(vid, '#333333')

        # 每辆车都用 add_standard_layers 添加统一结构的6个图层（带车辆ID前缀）
        add_standard_layers(m, df, vid, color=color)

        df = df.copy().reset_index(drop=True)
        points = []
        for _, row in df.iterrows():
            elapsed = (row['time'] - start_time).total_seconds()
            points.append({
                'lat': row['lati'], 'lng': row['long'],
                'status': int(row['status']), 'speed': float(row['speed']),
                'time': row['time'].strftime('%H:%M:%S'),
                'elapsed': float(elapsed),
            })
        if points:
            max_elapsed = max(max_elapsed, points[-1]['elapsed'])
        vehicles_payload.append({'id': vid, 'color': color, 'points': points})

    boundary = load_shenzhen_boundary()
    fg_boundary = folium.FeatureGroup(name='深圳市边界', show=True)
    add_boundary_to_map(fg_boundary, boundary)
    fg_boundary.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    add_road_match_banner(m, road_match)

    target_play_seconds = 25
    time_scale_real_per_play = (max_elapsed / target_play_seconds) if max_elapsed > 0 else 1.0

    vehicles_json = json.dumps(vehicles_payload)
    map_var_name = m.get_name()

    legend_rows = ''.join(
        f'<div><span style="display:inline-block;width:14px;height:14px;border-radius:50%;'
        f'background:{v["color"]};vertical-align:middle;margin-right:6px;"></span>车辆 {v["id"]}</div>'
        for v in vehicles_payload
    )
    status_rows = ''.join(
        f'<div id="status-{v["id"]}" style="margin-bottom:2px;">车辆{v["id"]}：等待播放</div>'
        for v in vehicles_payload
    )

    animation_script = f'''
    <div id="multi-anim-controls" style="position:fixed; bottom:30px; right:20px; z-index:9999;
         background:rgba(255,255,255,0.96); padding:12px 16px;
         border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.2);
         font-family:'Microsoft YaHei',sans-serif; min-width:220px; max-width:260px;">
        <div style="margin-bottom:10px;">
            <button id="multiPlayBtn" style="padding:6px 14px; font-size:14px; cursor:pointer;
                background:#2196F3; color:white; border:none; border-radius:6px;">▶ 播放对比动画</button>
            <button id="multiResetBtn" style="padding:6px 14px; font-size:14px; cursor:pointer;
                background:#6c757d; color:white; border:none; border-radius:6px; margin-left:6px;">⟲ 重置</button>
        </div>
        <div style="width:100%; height:5px; background:#e9ecef; border-radius:3px; margin-bottom:8px;">
            <div id="multiProgressFill" style="height:100%; width:0%; background:#2196F3;
                 border-radius:3px; transition:width 0.1s;"></div>
        </div>
        <div style="border-top:1px solid #e9ecef; margin:6px 0 8px 0;"></div>
        <div style="font-size:12px; color:#333; line-height:1.7; max-height:160px; overflow-y:auto;">
            {status_rows}
        </div>
        <div style="border-top:1px solid #e9ecef; margin:8px 0 6px 0;"></div>
        <div style="font-size:12px; color:#333; line-height:1.8;">{legend_rows}</div>
    </div>
    <script>
    (function() {{
        var vehicles = {vehicles_json};
        var timeScale = {time_scale_real_per_play};
        var maxElapsed = {max_elapsed};
        var tickMs = 50;
        var playElapsedMs = 0;
        var tickHandle = null;
        function waitForMap() {{
            if (typeof {map_var_name} === 'undefined') {{ setTimeout(waitForMap, 100); return; }}
            var targetMap = {map_var_name};
            vehicles.forEach(function(v) {{
                var iconHtml = '<div style="width:22px;height:22px;border-radius:50%;border:3px solid ' +
                    v.color + ';background:white;display:flex;align-items:center;justify-content:center;' +
                    'font-size:12px;">🚕</div>';
                var icon = L.divIcon({{ html: iconHtml, iconSize: [22, 22], className: '' }});
                v.marker = L.marker([v.points[0].lat, v.points[0].lng], {{icon: icon}}).addTo(targetMap);
                v.trail = L.polyline([], {{color: v.color, weight: 5, opacity: 0.92}}).addTo(targetMap);
                v.idx = 0;
                v.statusEl = document.getElementById('status-' + v.id);
            }});
            var progressFill = document.getElementById('multiProgressFill');
            var playBtn = document.getElementById('multiPlayBtn');
            function allDone() {{
                return vehicles.every(function(v) {{ return v.idx >= v.points.length; }});
            }}
            function tick() {{
                playElapsedMs += tickMs;
                var realElapsed = (playElapsedMs / 1000) * timeScale;
                vehicles.forEach(function(v) {{
                    while (v.idx < v.points.length && v.points[v.idx].elapsed <= realElapsed) {{
                        var p = v.points[v.idx];
                        v.marker.setLatLng([p.lat, p.lng]);
                        v.trail.addLatLng([p.lat, p.lng]);
                        if (v.statusEl) {{
                            v.statusEl.innerHTML = '<b>车辆' + v.id + '</b>：' + p.time +
                                ' · ' + (p.status === 1 ? '🔴载客' : '🟢空载') +
                                ' · ' + p.speed + 'km/h';
                        }}
                        v.idx++;
                    }}
                }});
                progressFill.style.width = Math.min(100, (realElapsed / maxElapsed * 100)) + '%';
                if (realElapsed >= maxElapsed || allDone()) {{
                    clearInterval(tickHandle);
                    tickHandle = null;
                    playBtn.innerText = '✅ 播放完成';
                    progressFill.style.width = '100%';
                }}
            }}
            playBtn.onclick = function() {{
                if (tickHandle) return;
                this.innerText = '⏸ 播放中...';
                tickHandle = setInterval(tick, tickMs);
            }};
            document.getElementById('multiResetBtn').onclick = function() {{
                clearInterval(tickHandle); tickHandle = null; playElapsedMs = 0;
                vehicles.forEach(function(v) {{
                    v.idx = 0;
                    v.marker.setLatLng([v.points[0].lat, v.points[0].lng]);
                    v.trail.setLatLngs([]);
                    if (v.statusEl) {{ v.statusEl.innerHTML = '车辆' + v.id + '：等待播放'; }}
                }});
                progressFill.style.width = '0%';
                playBtn.innerText = '▶ 播放对比动画';
            }};
        }}
        waitForMap();
    }})();
    </script>
    '''
    return m.get_root().render() + animation_script


# =====================================================================
# 核心改动3：全车位置快照 —— 只保留原点图，移除热力图和聚合图
# =====================================================================

def build_fleet_snapshot_map(df, highlight_row=None, dots_sample_size=3000,
                              bbox=None, enable_draw=True):
    """
    全车位置快照地图：仅原点图模式。
    - 载客：红色圆点；空载：绿色圆点
    - 支持高亮指定车辆（红色星标）
    - 支持框选范围显示
    - 支持Draw工具画矩形框选
    """
    if len(df) == 0:
        return None

    if highlight_row is not None:
        map_center = [highlight_row['lati'], highlight_row['long']]
        zoom = 14
    else:
        map_center = SHENZHEN_CENTER
        zoom = 11

    m = folium.Map(location=map_center, zoom_start=zoom,
                   tiles='OpenStreetMap', prefer_canvas=True)

    # ===== 原点图：唯一显示模式 =====
    df_sample = (df.sample(n=min(dots_sample_size, len(df)), random_state=42)
                 if len(df) > dots_sample_size else df)

    fg_busy = folium.FeatureGroup(name='载客轨迹', show=True)   # 沿用图示名称
    fg_idle = folium.FeatureGroup(name='空载轨迹', show=True)

    for _, row in df_sample.iterrows():
        is_busy = row['status'] == 1
        color = '#E63946' if is_busy else '#06D6A0'
        popup_text = (f"车辆id: {int(row['id'])}<br>"
                      f"状态: {'载客' if is_busy else '空载'}<br>"
                      f"速度: {row['speed']} km/h")
        marker = folium.CircleMarker(
            location=[row['lati'], row['long']],
            radius=4, color=color, fill=True,
            fill_color=color, fill_opacity=0.75, weight=1,
            popup=folium.Popup(popup_text, max_width=160),
        )
        if is_busy:
            marker.add_to(fg_busy)
        else:
            marker.add_to(fg_idle)

    fg_busy.add_to(m)
    fg_idle.add_to(m)

    # ===== 高亮搜索车辆 =====
    if highlight_row is not None:
        fg_highlight = folium.FeatureGroup(name='起点/终点', show=True)   # 复用图示图层名
        folium.Marker(
            location=[highlight_row['lati'], highlight_row['long']],
            popup=folium.Popup(
                f"<b>车辆 {int(highlight_row['id'])}</b><br>"
                f"状态：{'载客' if highlight_row['status'] == 1 else '空载'}<br>"
                f"速度：{highlight_row['speed']} km/h",
                max_width=200,
            ),
            icon=folium.Icon(color='red', icon='star'),
            tooltip=f"车辆 {int(highlight_row['id'])}（已定位）",
        ).add_to(fg_highlight)
        fg_highlight.add_to(m)

    # ===== 框选范围静态展示 =====
    if bbox is not None:
        south, north, west, east = bbox
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color='#9b5de5', weight=2.5, fill=True,
            fill_color='#9b5de5', fill_opacity=0.06,
            dash_array='6, 4', tooltip='当前框选范围',
        ).add_to(m)

    # ===== 深圳边界 =====
    boundary = load_shenzhen_boundary()
    fg_boundary = folium.FeatureGroup(name='深圳市边界', show=True)
    add_boundary_to_map(fg_boundary, boundary)
    fg_boundary.add_to(m)

    # ===== 框选工具 =====
    if enable_draw:
        Draw(
            export=False,
            draw_options={
                'polyline': False, 'polygon': False, 'circle': False,
                'marker': False, 'circlemarker': False,
                'rectangle': {'shapeOptions': {'color': '#9b5de5'}},
            },
            edit_options={'edit': False, 'remove': True},
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_picker_map():
    m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap')
    boundary = load_shenzhen_boundary()
    m = add_boundary_to_map(m, boundary)
    return m


def main():
    st.set_page_config(page_title='出租车GPS查询系统', layout='wide')

    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {height: 0px; min-height: 0px;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    div[data-testid="stTabs"] > div:first-child {
        background-color: transparent; padding: 0; margin-left: 0; margin-right: 0;
        margin-bottom: 0.5rem; border-bottom: 1px solid #e9ecef;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-size: 14px !important; font-weight: 400 !important; color: #666 !important;
        padding: 10px 20px !important; border: none !important;
        border-bottom: 3px solid transparent !important;
        background: transparent !important; border-radius: 0 !important; margin: 0 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: #333 !important; background: #f5f5f5 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #E63946 !important; border-bottom: 3px solid #E63946 !important;
        font-weight: 500 !important; background: transparent !important;
    }
    div[data-testid="stTabs"] > div:first-child > div { border-bottom: none !important; gap: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; align-items:baseline; gap:12px;
                padding: 0.5rem 0 0.8rem 0;
                border-bottom: 1px solid #e9ecef; margin-bottom:0;">
        <span style="font-size:1.4rem; font-weight:600;">🚕 出租车GPS数据查询</span>
        <span style="font-size:12px; color:#888;">深圳 · 2013-10-22</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        '🚗  单车轨迹查询与动画', '📍  全车位置查询', '🗺  地图选点（路网校正/ETA）', '🚦  多车轨迹动画对比'
    ])

    # ========== Tab 1：单车轨迹 ==========
    with tab1:
        vehicle_ids = list_vehicle_ids()
        col_control, col_display = st.columns([1, 3])

        with col_control:
            st.subheader('查询条件')
            default_index = 0
            if 'selected_vehicle' in st.session_state and st.session_state['selected_vehicle'] in vehicle_ids:
                default_index = vehicle_ids.index(st.session_state['selected_vehicle'])
            selected_id = st.selectbox('选择车辆id', vehicle_ids, index=default_index)

            df_full = load_vehicle_data(selected_id)
            if len(df_full) > 0:
                min_time = df_full['time'].min()
                max_time = df_full['time'].max()
                time_range = st.slider(
                    '选择时间范围',
                    min_value=min_time.to_pydatetime(),
                    max_value=max_time.to_pydatetime(),
                    value=(min_time.to_pydatetime(),
                           min(min_time + pd.Timedelta(minutes=30), max_time).to_pydatetime()),
                    format='HH:mm', step=pd.Timedelta(minutes=1),
                )
                start_time, end_time = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
                st.write(f'起始：{start_time.strftime("%H:%M:%S")}')
                st.write(f'结束：{end_time.strftime("%H:%M:%S")}')
                df_filtered = df_full[(df_full['time'] >= start_time) & (df_full['time'] <= end_time)]
                st.metric('轨迹点数', len(df_filtered))
                st.metric('载客点数', int((df_filtered['status'] == 1).sum()))
                st.metric('空载点数', int((df_filtered['status'] == 0).sum()))
                st.caption('动画播放建议选择30分钟以内范围')
                st.divider()
                road_match_1 = st.checkbox('启用路网校正', value=False, key='road_match_tab1',
                                            help='功能开发中，当前为占位开关')
            else:
                df_filtered = df_full
                road_match_1 = False
                st.warning('该车辆没有数据')

        with col_display:
            combined_html = build_combined_map_html(df_filtered, selected_id, road_match=road_match_1)
            if combined_html is not None:
                components.html(combined_html, height=780, scrolling=False)
            else:
                st.info('当前时间范围内没有轨迹点，请调整滑块')

    # ========== Tab 2：全车位置（仅原点图）==========
    with tab2:
        col_control2, col_map2 = st.columns([1, 4])

        with col_control2:
            st.subheader('选择时间点')
            query_time = st.time_input('查询时刻', value=pd.Timestamp('08:00:00').time())
            df_snapshot = load_minute_snapshot(query_time.hour, query_time.minute)

            st.divider()
            st.write('🖊️ 框选范围筛选')
            st.caption('在右侧地图用矩形工具画一个区域，点击下方按钮即可只看区域内车辆')
            current_bbox = st.session_state.get('fleet_bbox')
            col_bbox_apply, col_bbox_clear = st.columns(2)
            with col_bbox_apply:
                apply_bbox_clicked = st.button('✅ 应用框选范围', use_container_width=True)
            with col_bbox_clear:
                if st.button('🗑 清除范围', use_container_width=True):
                    st.session_state['fleet_bbox'] = None
                    st.rerun()

            if current_bbox is not None:
                st.caption(f'当前范围：纬度[{current_bbox[0]:.4f}, {current_bbox[1]:.4f}]，'
                           f'经度[{current_bbox[2]:.4f}, {current_bbox[3]:.4f}]')

            df_display = filter_by_bbox(df_snapshot, current_bbox) if current_bbox else df_snapshot

            st.divider()
            st.metric('当前显示车辆总数', len(df_display))
            if len(df_display) > 0:
                st.metric('载客车辆数', int((df_display['status'] == 1).sum()))
                st.metric('空载车辆数', int((df_display['status'] == 0).sum()))
            st.caption('原点图最多抽样3000辆展示，保证渲染性能')
            st.divider()

            st.write('🔍 查找指定车辆')
            search_id = st.text_input('输入车辆id', placeholder='如：22223')
            search_result = None
            if search_id.strip():
                matched = (df_snapshot[df_snapshot['id'] == int(search_id.strip())]
                           if search_id.strip().isdigit() else pd.DataFrame())
                if len(matched) > 0:
                    search_result = matched.iloc[0]
                    status_str = '载客' if search_result['status'] == 1 else '空载'
                    st.success(f"找到车辆 {search_id}")
                    st.write(f"**状态**：{status_str}")
                    st.write(f"**速度**：{search_result['speed']} km/h")
                    st.write(f"**位置**：{search_result['lati']:.5f}, {search_result['long']:.5f}")
                else:
                    st.warning(f"该时刻未找到车辆 {search_id}，可能不在线")

            st.divider()
            st.write('💡 点击地图上车辆位置可自动选中最近车辆')

        with col_map2:
            m2 = build_fleet_snapshot_map(
                df_display,
                highlight_row=search_result,
                bbox=current_bbox,
            )

            if m2 is not None:
                map_data2 = st_folium(m2, width=None, height=650, key='fleet_map')

                if apply_bbox_clicked and map_data2:
                    drawing = map_data2.get('last_active_drawing')
                    new_bbox = extract_bbox_from_drawing(drawing)
                    if new_bbox is not None:
                        st.session_state['fleet_bbox'] = new_bbox
                        st.rerun()
                    else:
                        st.warning('未检测到已绘制的矩形，请先在地图上用左侧工具栏画一个矩形区域')

                if map_data2 and map_data2.get('last_clicked'):
                    click_lat = map_data2['last_clicked']['lat']
                    click_lng = map_data2['last_clicked']['lng']
                    nearest = find_nearest_vehicle(df_display, click_lat, click_lng)
                    if nearest is not None:
                        nearest_id = str(int(nearest['id']))
                        if st.session_state.get('clicked_vehicle') != nearest_id:
                            st.session_state['clicked_vehicle'] = nearest_id
                            st.session_state['selected_vehicle'] = nearest_id
                            st.rerun()
                        st.success(f"已选中车辆：{nearest_id}（状态："
                                   f"{'载客' if nearest['status'] == 1 else '空载'}，"
                                   f"速度：{nearest['speed']}km/h）")
            else:
                st.info('该时刻没有数据')

        # 需求5：点击车辆后弹出动画回放面板
        clicked_id = st.session_state.get('clicked_vehicle')
        if clicked_id:
            st.divider()
            with st.expander(f'🎬 车辆 {clicked_id} 动画轨迹回放设置', expanded=True):
                df_clicked_full = load_vehicle_data(clicked_id)
                if len(df_clicked_full) == 0:
                    st.warning('该车辆没有可用的轨迹数据')
                else:
                    min_t = df_clicked_full['time'].min()
                    max_t = df_clicked_full['time'].max()
                    default_start = pd.Timestamp.combine(min_t.date(), query_time)
                    default_start = max(min(default_start, max_t), min_t)
                    default_end = min(default_start + pd.Timedelta(minutes=20), max_t)

                    col_t1, col_t2, col_t3 = st.columns([2, 1, 1])
                    with col_t1:
                        anim_range = st.slider(
                            '选择该车的动画播放时间范围（建议60分钟以内）',
                            min_value=min_t.to_pydatetime(),
                            max_value=max_t.to_pydatetime(),
                            value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
                            format='HH:mm', step=pd.Timedelta(minutes=1),
                            key='clicked_vehicle_time_range',
                        )
                    with col_t2:
                        road_match_2 = st.checkbox('启用路网校正', value=False, key='road_match_tab2',
                                                    help='功能开发中，当前为占位开关')
                    with col_t3:
                        play_clicked = st.button('▶ 开始播放', key='play_clicked_vehicle',
                                                  use_container_width=True)

                    if play_clicked:
                        anim_start = pd.Timestamp(anim_range[0])
                        anim_end = pd.Timestamp(anim_range[1])
                        df_anim = df_clicked_full[(df_clicked_full['time'] >= anim_start) &
                                                   (df_clicked_full['time'] <= anim_end)]
                        if len(df_anim) == 0:
                            st.info('所选时间范围内该车辆没有轨迹点，请调整范围')
                        else:
                            anim_html = build_combined_map_html(df_anim, clicked_id,
                                                                 road_match=road_match_2)
                            components.html(anim_html, height=700, scrolling=False)

    # ========== Tab 3：地图选点 ==========
    with tab3:
        st.subheader('点击地图选取坐标点')
        st.caption('此功能为后续路网校正（06阶段）与ETA预测（07阶段）预留接口')

        col_picker_control, col_picker_map = st.columns([1, 4])
        with col_picker_map:
            m3 = build_picker_map()
            map_data = st_folium(m3, width=None, height=800, key='picker_map')
        with col_picker_control:
            st.write('已选取的坐标：')
            if map_data and map_data.get('last_clicked'):
                clicked_lat = map_data['last_clicked']['lat']
                clicked_lng = map_data['last_clicked']['lng']
                st.success(f'纬度：{clicked_lat:.6f}')
                st.success(f'经度：{clicked_lng:.6f}')
                st.code(f'[{clicked_lat:.6f}, {clicked_lng:.6f}]', language='python')
                st.caption('该坐标可直接作为后续路网匹配/ETA计算的起点或终点输入')
            else:
                st.info('请在地图上点击选取一个位置')

    # ========== Tab 4：多车轨迹动画对比 ==========
    with tab4:
        st.subheader('多车轨迹动画对比')
        st.caption(f'各车以不同颜色同步播放，最多 {MAX_VEHICLES_FOR_ANIMATION} 辆')

        col_control4, col_map4 = st.columns([1, 3])
        vehicle_dfs_full = None
        range_start = range_end = None
        color_map = {}
        play_multi = False

        with col_control4:
            vehicle_ids_all = list_vehicle_ids()
            selected_ids = st.multiselect(
                '选择车辆id（可多选）',
                vehicle_ids_all,
                default=vehicle_ids_all[:2] if len(vehicle_ids_all) >= 2 else vehicle_ids_all,
            )
            if len(selected_ids) > MAX_VEHICLES_FOR_ANIMATION:
                st.warning(f'最多支持 {MAX_VEHICLES_FOR_ANIMATION} 辆，已自动截取')
                selected_ids = selected_ids[:MAX_VEHICLES_FOR_ANIMATION]

            road_match_4 = st.checkbox('启用路网校正', value=False, key='road_match_tab4',
                                        help='功能开发中，当前为占位开关')

            if not selected_ids:
                st.info('请至少选择一辆车')
            else:
                vehicle_dfs_full = {vid: load_vehicle_data(vid) for vid in selected_ids}
                vehicle_dfs_full = {vid: df for vid, df in vehicle_dfs_full.items() if len(df) > 0}

                if not vehicle_dfs_full:
                    st.warning('所选车辆均没有可用数据')
                else:
                    union_min = min(df['time'].min() for df in vehicle_dfs_full.values())
                    union_max = max(df['time'].max() for df in vehicle_dfs_full.values())
                    default_end = min(union_min + pd.Timedelta(minutes=20), union_max)

                    shared_range = st.slider(
                        '选择共同的时间范围',
                        min_value=union_min.to_pydatetime(),
                        max_value=union_max.to_pydatetime(),
                        value=(union_min.to_pydatetime(), default_end.to_pydatetime()),
                        format='HH:mm', step=pd.Timedelta(minutes=1),
                    )
                    st.caption('车辆越多，建议范围越短')

                    range_start = pd.Timestamp(shared_range[0])
                    range_end = pd.Timestamp(shared_range[1])
                    color_map = {vid: MULTI_VEHICLE_COLORS[i % len(MULTI_VEHICLE_COLORS)]
                                 for i, vid in enumerate(vehicle_dfs_full.keys())}

                    st.divider()
                    for vid, df in vehicle_dfs_full.items():
                        df_in_range = df[(df['time'] >= range_start) & (df['time'] <= range_end)]
                        st.markdown(
                            f'<span style="display:inline-block;width:12px;height:12px;'
                            f'border-radius:50%;background:{color_map[vid]};vertical-align:middle;'
                            f'margin-right:6px;"></span>车辆 {vid}：{len(df_in_range)} 个轨迹点',
                            unsafe_allow_html=True,
                        )

                    play_multi = st.button('▶ 生成并播放多车对比动画', use_container_width=True)

        with col_map4:
            if play_multi and vehicle_dfs_full:
                vehicle_dfs_ranged = {
                    vid: df[(df['time'] >= range_start) & (df['time'] <= range_end)]
                    for vid, df in vehicle_dfs_full.items()
                }
                vehicle_dfs_ranged = {vid: df for vid, df in vehicle_dfs_ranged.items() if len(df) > 0}
                if not vehicle_dfs_ranged:
                    st.info('所选时间范围内没有任何车辆的轨迹点')
                else:
                    multi_html = build_multi_vehicle_animation_html(
                        vehicle_dfs_ranged, color_map,
                        start_time=range_start, road_match=road_match_4,
                    )
                    if multi_html is not None:
                        components.html(multi_html, height=780, scrolling=False)
            else:
                st.info('设置好车辆与时间范围后，点击左侧按钮生成动画')


if __name__ == '__main__':
    main()
