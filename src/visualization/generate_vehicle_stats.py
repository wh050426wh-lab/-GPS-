"""
生成所有车辆的全天载客率、总里程、载客里程、空载里程统计
输出：outputs/vehicle_stats.csv
运行方式：python generate_vehicle_stats.py
"""

import os
import sys
import pandas as pd
from geopy.distance import geodesic

# ============================================================
# 导入项目配置
# ============================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import VEHICLE_CACHE_DIR, BASE_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vehicle_stats.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 深圳经纬度范围（过滤漂移点用）
# ============================================================
LNG_MIN, LNG_MAX = 113.7, 114.7
LAT_MIN, LAT_MAX = 22.3, 22.9
MAX_SINGLE_DIST_KM = 5.0  # 单段超过5km视为漂移，跳过


# ============================================================
# 计算单辆车的统计数据
# ============================================================
def calc_vehicle_stats(vid):
    path = os.path.join(VEHICLE_CACHE_DIR, f"{vid}.csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if len(df) < 2:
        return None

    total_km = 0.0
    loaded_km = 0.0
    unloaded_km = 0.0

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # 过滤深圳范围外的漂移点
        if not (LNG_MIN <= prev["long"] <= LNG_MAX and LAT_MIN <= prev["lati"] <= LAT_MAX):
            continue
        if not (LNG_MIN <= curr["long"] <= LNG_MAX and LAT_MIN <= curr["lati"] <= LAT_MAX):
            continue

        # 计算两点距离
        try:
            dist = geodesic(
                (prev["lati"], prev["long"]),
                (curr["lati"], curr["long"])
            ).km
        except Exception:
            continue

        # 过滤单段异常跳跃
        if dist > MAX_SINGLE_DIST_KM:
            continue

        total_km += dist
        if prev["status"] == 1:
            loaded_km += dist
        else:
            unloaded_km += dist

    if total_km == 0:
        return None

    return {
        "车辆ID": vid,
        "总里程_km": round(total_km, 3),
        "载客里程_km": round(loaded_km, 3),
        "空载里程_km": round(unloaded_km, 3),
        "载客率": round(loaded_km / total_km, 4)
    }


# ============================================================
# 主流程：遍历所有车辆
# ============================================================
if __name__ == "__main__":
    vehicle_files = [
        f.replace(".csv", "")
        for f in os.listdir(VEHICLE_CACHE_DIR)
        if f.endswith(".csv")
    ]

    total = len(vehicle_files)
    print(f"共发现 {total} 辆车，开始统计...")

    results = []
    for i, vid in enumerate(vehicle_files):
        r = calc_vehicle_stats(vid)
        if r:
            results.append(r)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  进度：{i+1}/{total}，有效：{len(results)}")

    if not results:
        print("❌ 没有有效数据")
    else:
        df = pd.DataFrame(results)
        df = df.sort_values("车辆ID").reset_index(drop=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n✅ 完成！共统计 {len(df)} 辆车")
        print(f"   导出路径：{OUTPUT_FILE}")
        print(f"\n汇总统计：")
        print(f"   平均载客率：{df['载客率'].mean()*100:.1f}%")
        print(f"   平均总里程：{df['总里程_km'].mean():.1f} km")
        print(f"   平均载客里程：{df['载客里程_km'].mean():.1f} km")
        print(f"   平均空载里程：{df['空载里程_km'].mean():.1f} km")