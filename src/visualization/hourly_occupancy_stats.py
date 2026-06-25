import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import MINUTE_CACHE_DIR, OD_CACHE_DIR  # noqa: E402


def main():
    print('开始统计按小时载客车辆数与载客率...')

    results = []

    for hour in range(24):
        hourly_busy_counts = []
        hourly_total_counts = []

        # 这个小时内有60个分钟文件，比如08点对应 08-00.csv 到 08-59.csv
        for minute in range(60):
            file_name = f'{hour:02d}-{minute:02d}.csv'
            file_path = os.path.join(MINUTE_CACHE_DIR, file_name)

            if not os.path.exists(file_path):
                continue

            df_minute = pd.read_csv(file_path, usecols=['status'])
            total = len(df_minute)
            busy = (df_minute['status'] == 1).sum()

            hourly_total_counts.append(total)
            hourly_busy_counts.append(busy)

        if len(hourly_total_counts) == 0:
            continue

        avg_total = sum(hourly_total_counts) / len(hourly_total_counts)
        avg_busy = sum(hourly_busy_counts) / len(hourly_busy_counts)
        occupancy_rate = avg_busy / avg_total if avg_total > 0 else 0

        results.append({
            '小时': hour,
            '平均在线车辆数': round(avg_total, 1),
            '平均载客车辆数': round(avg_busy, 1),
            '载客率': round(occupancy_rate, 4),
        })

        print(f'第 {hour:02d} 时：在线 {avg_total:.0f} 辆，载客 {avg_busy:.0f} 辆，载客率 {occupancy_rate*100:.1f}%')

    df_result = pd.DataFrame(results)
    output_path = os.path.join(OD_CACHE_DIR, 'hourly_occupancy_rate.csv')
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print('=' * 50)
    print(f'统计完成，结果已保存：{output_path}')
    print(df_result)


if __name__ == '__main__':
    main()