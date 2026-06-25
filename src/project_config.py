import os

# 项目根目录（自动定位，基于本文件位置往上推一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 数据路径 =====
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'sample')
DATA_CLEANED_DIR = os.path.join(BASE_DIR, 'data', 'cleaned')
DATA_OD_DIR = os.path.join(BASE_DIR, 'data', 'od')
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'cache')

# 具体文件路径（先用样本数据跑通逻辑，后续切换为 raw）
RAW_FILE = os.path.join(DATA_RAW_DIR, 'TaxiData.csv')
SHENZHEN_BOUNDARY_FILE = os.path.join(BASE_DIR, 'data', 'raw', '深圳市.json')
SAMPLE_FILE = os.path.join(DATA_SAMPLE_DIR, 'taxi_sample.csv')
CLEANED_FILE = os.path.join(DATA_CLEANED_DIR, 'taxi_cleaned.csv')
OD_FILE = os.path.join(DATA_OD_DIR, 'od_data.csv')
MAP_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'maps')
OD_MARKERS_FILE = os.path.join(DATA_OD_DIR, 'od_markers.csv')   # OD启动阶段的标记结果（上车点/下车点）
# OD订单表
OD_ORDERS_FILE = os.path.join(DATA_OD_DIR, 'od_orders.csv')

# 车辆缓存目录（按车辆id分文件存轨迹）
VEHICLE_CACHE_DIR = os.path.join(DATA_CACHE_DIR, 'vehicle_cache')

# 分钟缓存目录（每分钟一个文件，存当时刻每辆车的位置）
MINUTE_CACHE_DIR = os.path.join(DATA_CACHE_DIR, 'minute_cache')

# OD缓存（给热力图/统计分析用，这里先和OD订单表保持一致，后续可扩展聚合统计文件）
OD_CACHE_DIR = os.path.join(DATA_CACHE_DIR, 'od_cache')

# ===== 字段定义 =====
# 原始数据无 header，需手动指定列名
RAW_COLUMNS = ['id', 'time', 'long', 'lati', 'status', 'speed']

# ===== 其他配置 =====
CHUNK_SIZE = 5_000_000      # 大文件分块读取参考值
DUP_TIME_GAP_THRESHOLD = 60  # 异常状态判断的时间差阈值（秒）

if __name__ == '__main__':
    # 自检：确保目录都存在
    for d in [DATA_RAW_DIR, DATA_SAMPLE_DIR, DATA_CLEANED_DIR, DATA_OD_DIR, DATA_CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
    print('BASE_DIR =', BASE_DIR)
    print('目录检查完成')