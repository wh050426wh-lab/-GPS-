import os
import sys

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import json
import numpy as np
import streamlit.components.v1 as components
from folium.plugins import FastMarkerCluster, HeatMap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import VEHICLE_CACHE_DIR, MINUTE_CACHE_DIR, SHENZHEN_BOUNDARY_FILE  # noqa: E402

SHENZHEN_CENTER = [22.52847, 114.05454]
DATA_DATE = '2013-10-22'


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
    """读取某一分钟所有车辆的位置（分钟缓存文件名格式：HH-MM.csv）"""
    minute_str = f'{hour:02d}-{minute:02d}'
    file_path = os.path.join(MINUTE_CACHE_DIR, f'{minute_str}.csv')
    if not os.path.exists(file_path):
        return pd.DataFrame()
    return pd.read_csv(file_path, parse_dates=['time'])


@st.cache_data
def load_shenzhen_boundary():
    """加载深圳市边界GeoJSON数据"""
    with open(SHENZHEN_BOUNDARY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_boundary_to_map(target, boundary_geojson):
    """给地图对象或图层组叠加深圳市行政区边界线，颜色统一为蓝色。
    只保留一份定义（之前重复定义了两次，第二份会覆盖第一份，现已合并）。"""
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


def add_legend(m):
    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; z-index: 9999;
                background-color: white; padding: 10px; border: 2px solid grey;
                border-radius: 5px; font-size: 14px;">
        <b>图例</b><br>
        <span style="color:#E63946;">●</span> 载客轨迹<br>
        <span style="color:#06D6A0;">●</span> 空载轨迹<br>
        <span style="color:#3388ff;">- - -</span> 深圳市边界
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def build_trajectory_map(df):
    """构建单车轨迹静态地图：载客/空载分色图层、上下车点、起终点、深圳边界、图层控制面板"""
    if len(df) == 0:
        return None
    center_lat, center_lon = df['lati'].mean(), df['long'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

    df = df.copy()
    df['status_grp'] = (df['status'] != df['status'].shift()).cumsum()

    fg_busy = folium.FeatureGroup(name='载客轨迹', show=True)
    fg_idle = folium.FeatureGroup(name='空载轨迹', show=True)

    for _, seg in df.groupby('status_grp'):
        if len(seg) < 2:
            continue
        points = seg[['lati', 'long']].values.tolist()
        if seg['status'].iloc[0] == 1:
            folium.PolyLine(points, color='#E63946', weight=4, opacity=0.8, tooltip='载客').add_to(fg_busy)
        else:
            folium.PolyLine(points, color='#06D6A0', weight=4, opacity=0.8, tooltip='空载').add_to(fg_idle)

    fg_busy.add_to(m)
    fg_idle.add_to(m)

    fg_pickup = folium.FeatureGroup(name='上车点', show=True)
    fg_dropoff = folium.FeatureGroup(name='下车点', show=True)

    df['status_prev'] = df['status'].shift(1)
    for _, row in df[(df['status'] == 1) & (df['status_prev'] == 0)].iterrows():
        folium.Marker([row['lati'], row['long']], popup=f"上车 {row['time']}",
                      icon=folium.Icon(color='green', icon='play')).add_to(fg_pickup)
    for _, row in df[(df['status'] == 0) & (df['status_prev'] == 1)].iterrows():
        folium.Marker([row['lati'], row['long']], popup=f"下车 {row['time']}",
                      icon=folium.Icon(color='orange', icon='stop')).add_to(fg_dropoff)

    fg_pickup.add_to(m)
    fg_dropoff.add_to(m)

    fg_endpoints = folium.FeatureGroup(name='起点/终点', show=True)
    folium.Marker([df.iloc[0]['lati'], df.iloc[0]['long']], popup=f"起点 {df.iloc[0]['time']}",
                  icon=folium.Icon(color='blue', icon='home')).add_to(fg_endpoints)
    folium.Marker([df.iloc[-1]['lati'], df.iloc[-1]['long']], popup=f"终点 {df.iloc[-1]['time']}",
                  icon=folium.Icon(color='black', icon='flag')).add_to(fg_endpoints)
    fg_endpoints.add_to(m)

    boundary = load_shenzhen_boundary()
    fg_boundary = folium.FeatureGroup(name='深圳市边界', show=True)
    add_boundary_to_map(fg_boundary, boundary)
    fg_boundary.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def build_picker_map():
    """提供一个可点击选点的地图，返回地图对象（预留给06路网校正/07ETA阶段）"""
    m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap')
    boundary = load_shenzhen_boundary()
    m = add_boundary_to_map(m, boundary)
    return m


def build_combined_map_html(df, vehicle_id):
    """生成包含静态轨迹（背景）+ 动态播放（叠加）的单一地图HTML。
    动画按真实时间差控制播放节奏，并实时显示当前速度。"""
    if len(df) == 0:
        return None

    m = build_trajectory_map(df)

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

    # 修正：用 json.dumps 生成合法JSON，而不是手动字符串替换
    # （手动replace('True','true')在边界情况下有误替换风险，json.dumps从根本上避免这个问题）
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
            <div id="progressFill" style="height:100%; width:0%; background:#2196F3; border-radius:3px; transition:width 0.1s;"></div>
        </div>

        <div id="statusBox" style="font-size:12px; color:#555; margin-bottom:10px; min-height:40px;">
            点击播放后显示状态
        </div>

        <div style="border-top:1px solid #e9ecef; margin-bottom:8px;"></div>

        <div style="font-size:12px; color:#333; line-height:2;">
            <div><span style="display:inline-block;width:24px;height:4px;background:#E63946;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>载客轨迹</div>
            <div><span style="display:inline-block;width:24px;height:4px;background:#06D6A0;border-radius:2px;vertical-align:middle;margin-right:6px;"></span>空载轨迹</div>
            <div>🟢 上车点 &nbsp; 🟠 下车点</div>
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
            if (typeof {map_var_name} === 'undefined') {{
                setTimeout(waitForMap, 100);
                return;
            }}
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
                    '<b>速度</b>：<span style="color:' + speedColor(p.speed) + ';font-weight:bold;">' + p.speed + ' km/h</span>';
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

    combined_html = m.get_root().render() + animation_script
    return combined_html


def find_nearest_vehicle(df, click_lat, click_lng):
    """在df中找出距离点击位置最近的车辆"""
    if len(df) == 0:
        return None
    distances = np.sqrt((df['lati'] - click_lat) ** 2 + (df['long'] - click_lng) ** 2)
    nearest_idx = distances.idxmin()
    return df.loc[nearest_idx]


def build_fleet_snapshot_map_with_highlight(df, show_mode='cluster', highlight_row=None, dots_sample_size=2000):
    """构建全车位置快照地图：聚合点位/原点图/热力图三种模式，支持高亮指定车辆并自动定位。
    原点图模式会对车辆做抽样（默认2000辆），避免上万个CircleMarker导致页面卡顿或渲染失败。"""
    if len(df) == 0:
        return None

    # 修正：地图中心点在创建Map对象时就决定好，而不是创建后再修改 m.location/m.zoom_start
    # （folium的Map对象创建后修改这两个属性不会触发重新渲染，之前的写法不会生效）
    if highlight_row is not None:
        map_center = [highlight_row['lati'], highlight_row['long']]
        zoom = 14
    else:
        map_center = SHENZHEN_CENTER
        zoom = 11

    m = folium.Map(location=map_center, zoom_start=zoom, tiles='OpenStreetMap', prefer_canvas=True)

    # ===== 聚合点位图层 =====
    fg_cluster = folium.FeatureGroup(name='聚合点位', show=(show_mode == 'cluster'))
    callback = """
    function (row) {
        var icon = L.divIcon({
            className: 'car-dot',
            html: '<div style="background-color:' + (row[2] === 1 ? '#E63946' : '#06D6A0') +
                  '; width:10px; height:10px; border-radius:50%; border:1px solid white;"></div>',
            iconSize: [10, 10]
        });
        var marker = L.marker(new L.LatLng(row[0], row[1]), {icon: icon});
        marker.bindPopup('车辆id: ' + row[3] + '<br>状态: ' + (row[2] === 1 ? '载客' : '空载') +
                          '<br>速度: ' + row[4] + ' km/h');
        return marker;
    };
    """
    marker_data = df[['lati', 'long', 'status', 'id', 'speed']].values.tolist()
    FastMarkerCluster(marker_data, callback=callback).add_to(fg_cluster)
    fg_cluster.add_to(m)

    # ===== 原点图图层：抽样后用CircleMarker逐个画，数量可控不会卡顿 =====
    fg_dots = folium.FeatureGroup(name=f'原点图(抽样{min(dots_sample_size, len(df))}辆)', show=(show_mode == 'dots'))
    if show_mode == 'dots':
        df_sample = df.sample(n=min(dots_sample_size, len(df)), random_state=42) if len(df) > dots_sample_size else df
        for _, row in df_sample.iterrows():
            color = '#E63946' if row['status'] == 1 else '#06D6A0'
            folium.CircleMarker(
                location=[row['lati'], row['long']],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=1,
                popup=f"车辆id: {int(row['id'])}<br>状态: {'载客' if row['status']==1 else '空载'}<br>速度: {row['speed']} km/h",
            ).add_to(fg_dots)
    fg_dots.add_to(m)

    # ===== 热力图图层 =====
    fg_heat = folium.FeatureGroup(name='热力图', show=(show_mode == 'heatmap'))
    heat_data = df[['lati', 'long']].values.tolist()
    HeatMap(heat_data, radius=10, blur=8, min_opacity=0.4).add_to(fg_heat)
    fg_heat.add_to(m)

    # ===== 载客车辆热力图（单独图层）=====
    fg_heat_busy = folium.FeatureGroup(name='载客车辆热力图', show=False)
    busy_data = df[df['status'] == 1][['lati', 'long']].values.tolist()
    if busy_data:
        HeatMap(busy_data, radius=10, blur=8, min_opacity=0.4,
                gradient={0.4: '#06D6A0', 0.7: '#FFD166', 1.0: '#E63946'}).add_to(fg_heat_busy)
    fg_heat_busy.add_to(m)

    # ===== 高亮搜索车辆 =====
    if highlight_row is not None:
        folium.Marker(
            location=[highlight_row['lati'], highlight_row['long']],
            popup=folium.Popup(
                f"<b>车辆 {int(highlight_row['id'])}</b><br>"
                f"状态：{'载客' if highlight_row['status'] == 1 else '空载'}<br>"
                f"速度：{highlight_row['speed']} km/h",
                max_width=200
            ),
            icon=folium.Icon(color='red', icon='star'),
            tooltip=f"车辆 {int(highlight_row['id'])}（已定位）"
        ).add_to(m)

    # ===== 深圳边界 =====
    boundary = load_shenzhen_boundary()
    fg_boundary = folium.FeatureGroup(name='深圳市边界', show=True)
    add_boundary_to_map(fg_boundary, boundary)
    fg_boundary.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    st.set_page_config(page_title='出租车GPS查询系统', layout='wide')

    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header[data-testid="stHeader"] {height: 0px; min-height: 0px;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    div[data-testid="stTabs"] > div:first-child {
        background-color: transparent;
        padding: 0;
        margin-left: 0;
        margin-right: 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #e9ecef;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #666 !important;
        padding: 10px 20px !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        background: transparent !important;
        border-radius: 0 !important;
        margin: 0 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: #333 !important;
        background: #f5f5f5 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #E63946 !important;
        border-bottom: 3px solid #E63946 !important;
        font-weight: 500 !important;
        background: transparent !important;
    }
    div[data-testid="stTabs"] > div:first-child > div {
        border-bottom: none !important;
        gap: 0 !important;
    }
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

    tab1, tab2, tab3 = st.tabs(['🚗  单车轨迹查询与动画', '📍  全车位置查询', '🗺  地图选点（路网校正/ETA）'])

    # ========== Tab 1：单车轨迹（静态图 + 动画叠加在同一张地图）==========
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
                    value=(
                        min_time.to_pydatetime(),
                        min(min_time + pd.Timedelta(minutes=30), max_time).to_pydatetime()
                    ),
                    format='HH:mm',
                    step=pd.Timedelta(minutes=1),
                )

                start_time, end_time = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
                st.write(f'起始：{start_time.strftime("%H:%M:%S")}')
                st.write(f'结束：{end_time.strftime("%H:%M:%S")}')

                df_filtered = df_full[(df_full['time'] >= start_time) & (df_full['time'] <= end_time)]

                st.metric('轨迹点数', len(df_filtered))
                st.metric('载客点数', int((df_filtered['status'] == 1).sum()))
                st.metric('空载点数', int((df_filtered['status'] == 0).sum()))
                st.caption('动画播放建议选择30分钟以内范围，避免过长不流畅')
            else:
                df_filtered = df_full
                st.warning('该车辆没有数据')

        with col_display:
            combined_html = build_combined_map_html(df_filtered, selected_id)
            if combined_html is not None:
                components.html(combined_html, height=780, scrolling=False)
            else:
                st.info('当前时间范围内没有轨迹点，请调整滑块')

    # ========== Tab 2：全车位置 ==========
    with tab2:
        col_control2, col_map2 = st.columns([1, 4])

        with col_control2:
            st.subheader('选择时间点')
            query_time = st.time_input('查询时刻', value=pd.Timestamp('08:00:00').time())
            df_snapshot = load_minute_snapshot(query_time.hour, query_time.minute)

            st.metric('该时刻车辆总数', len(df_snapshot))
            if len(df_snapshot) > 0:
                st.metric('载客车辆数', int((df_snapshot['status'] == 1).sum()))
                st.metric('空载车辆数', int((df_snapshot['status'] == 0).sum()))
            show_mode = st.radio('显示模式', ['聚合点位', '原点图', '热力图'], horizontal=True)
            show_mode_map = {'聚合点位': 'cluster', '原点图': 'dots', '热力图': 'heatmap'}
            show_mode_key = show_mode_map[show_mode]
            if show_mode_key == 'dots':
                st.caption('原点图为节省性能，已自动抽样最多2000辆车展示')
            st.divider()

            st.write('🔍 查找指定车辆')
            search_id = st.text_input('输入车辆id', placeholder='如：22223')
            search_result = None
            if search_id.strip():
                matched = df_snapshot[df_snapshot['id'] == int(search_id.strip())] \
                    if search_id.strip().isdigit() else pd.DataFrame()
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
            m2 = build_fleet_snapshot_map_with_highlight(
                df_snapshot,
                show_mode=show_mode_key,
                highlight_row=search_result
            )

            if m2 is not None:
                map_data2 = st_folium(m2, width=None, height=650, key='fleet_map')

                if map_data2 and map_data2.get('last_clicked'):
                    click_lat = map_data2['last_clicked']['lat']
                    click_lng = map_data2['last_clicked']['lng']
                    nearest = find_nearest_vehicle(df_snapshot, click_lat, click_lng)
                    if nearest is not None:
                        nearest_id = str(int(nearest['id']))
                        if st.session_state.get('selected_vehicle') != nearest_id:
                            st.session_state['selected_vehicle'] = nearest_id
                            st.rerun()

                        st.success(f"已自动选中最近车辆：{nearest_id}（状态："
                                   f"{'载客' if nearest['status'] == 1 else '空载'}，"
                                   f"速度：{nearest['speed']}km/h）。请切换到其他标签页查看详情")
            else:
                st.info('该时刻没有数据')

    # ========== Tab 3：地图选点 ==========
    with tab3:
        st.subheader('点击地图选取坐标点')
        st.caption('此功能为后续路网校正（06阶段）与ETA预测（07阶段）预留接口：点击地图任意位置，获取经纬度坐标')

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


if __name__ == '__main__':
    main()