import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import RAW_FILE, CLEANED_FILE, OD_MARKERS_FILE, RAW_COLUMNS, DUP_TIME_GAP_THRESHOLD  # noqa: E402

DATA_DATE = '2013-10-22'
CHUNK_SIZE = 5_000_000  # 每块行数，内存紧张可以调小

# 深圳市大致经纬度范围（用于坐标越界过滤）
LON_MIN, LON_MAX = 113.75, 114.65
LAT_MIN, LAT_MAX = 22.45, 22.85
SPEED_MAX = 120  # km/h，速度上限


def clean_one_block(df):
    """对一个完整的数据块（保证车辆id不跨块）执行完整清洗 + OD启动，
    返回：清洗后的数据、OD标记数据（上车点/下车点）、本块统计信息
    """
    # 补日期 + 转换时间类型
    df['time'] = DATA_DATE + ' ' + df['time'].astype(str)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    df = df.sort_values(by=['id', 'time']).reset_index(drop=True)

    # ---- 1. 去重 ----
    df_dup = df[df.duplicated(subset=['id', 'time'], keep=False)].reset_index()
    if len(df_dup) > 0:
        dup_grp = (
            df_dup.groupby(['id', 'time'])
            .agg(stat_cnt=('status', 'count'), stat_sum=('status', 'sum'))
            .reset_index()
        )
        dup_mrg = pd.merge(df_dup, dup_grp, on=['id', 'time'], how='left')

        def dup_check(x):
            cnt, s = x.stat_cnt.max(), x.stat_sum.max()
            if cnt == 2 and s == 0:
                return x['index'].values[0]
            elif cnt == 2 and s == 1:
                return x.loc[x.status == 0, 'index'].values[0]
            elif cnt == 2 and s == 2:
                return x['index'].values[0]
            elif cnt == 3 and s == 0:
                return x['index'].values[0]
            elif cnt == 3 and s == 1:
                return x.loc[x.status == 0, 'index'].values[0]
            elif cnt == 3 and s == 2:
                return x.loc[x.status == 1, 'index'].values[0]
            elif cnt == 3 and s == 3:
                return x['index'].values[0]
            return x['index'].values[0]

        kp_index = dup_mrg.groupby(['id', 'time']).apply(dup_check)
        drp_index = dup_mrg.loc[~dup_mrg['index'].isin(kp_index.values), 'index']
        df = df.loc[~df.index.isin(drp_index.values)].reset_index(drop=True)
    n_dup = len(df_dup)

    # ---- 2. 短时间状态跳变异常 ----
    df['status_up'] = df['status'].shift(1)
    df['status_down'] = df['status'].shift(-1)
    df['id_up'] = df['id'].shift(1)
    df['id_down'] = df['id'].shift(-1)
    df['time_up'] = df['time'].shift(1)
    df['time_down'] = df['time'].shift(-1)

    cond_1 = df['status'] != df['status_down']
    cond_2 = df['status'] != df['status_up']
    cond_3 = df['id'] == df['id_up']
    cond_4 = df['id'] == df['id_down']
    cond_5 = (df['time_down'] - df['time_up']).dt.seconds < DUP_TIME_GAP_THRESHOLD

    df_abn = df[cond_1 & cond_2 & cond_3 & cond_4 & cond_5]
    n_abn_jump = len(df_abn)
    df = df.loc[~df.index.isin(df_abn.index)].reset_index(drop=True)

    # ---- 3. 坐标越界 + 速度越界（逐行判断） ----
    cond_geo = (df['long'] < LON_MIN) | (df['long'] > LON_MAX) | \
               (df['lati'] < LAT_MIN) | (df['lati'] > LAT_MAX)
    cond_speed_max = df['speed'] > SPEED_MAX

    n_geo = int(cond_geo.sum())
    n_speed_max = int((cond_speed_max & ~cond_geo).sum())  # 避免重复计数

    df = df[~(cond_geo | cond_speed_max)].reset_index(drop=True)

    # ---- 4. 整车全天异常（速度恒为0 / status恒定为0或1） ----
    def is_abnormal_vehicle(g):
        all_zero_speed = (g['speed'] == 0).all()
        all_status_0 = (g['status'] == 0).all()
        all_status_1 = (g['status'] == 1).all()
        return all_zero_speed or all_status_0 or all_status_1

    abnormal_ids = df.groupby('id').filter(is_abnormal_vehicle)['id'].unique()
    n_abn_vehicle_rows = int(df['id'].isin(abnormal_ids).sum())
    n_abn_vehicle_count = len(abnormal_ids)

    df = df[~df['id'].isin(abnormal_ids)].reset_index(drop=True)

    # ---- OD 启动：计算 status_chg，并落盘标记结果 ----
    df['status_up'] = df['status'].shift(1)
    df['id_up'] = df['id'].shift(1)
    df['status_chg'] = df['status'] - df['status_up']
    df['id_chg'] = df['id'] - df['id_up']

    df_pickup = df.loc[(df['status_chg'] == 1) & (df['id_chg'] == 0)].copy()
    df_dropoff = df.loc[(df['status_chg'] == -1) & (df['id_chg'] == 0)].copy()

    df_pickup['point_type'] = 'pickup'
    df_dropoff['point_type'] = 'dropoff'

    df_markers = pd.concat([df_pickup, df_dropoff], ignore_index=True)
    df_markers = df_markers[['id', 'time', 'long', 'lati', 'status', 'speed', 'point_type']]
    df_markers = df_markers.sort_values(by=['id', 'time']).reset_index(drop=True)

    n_pickup = len(df_pickup)
    n_dropoff = len(df_dropoff)

    df_out = df[['id', 'time', 'long', 'lati', 'status', 'speed']]

    stats = {
        'n_dup': n_dup,
        'n_abn_jump': n_abn_jump,
        'n_geo': n_geo,
        'n_speed_max': n_speed_max,
        'n_abn_vehicle_rows': n_abn_vehicle_rows,
        'n_abn_vehicle_count': n_abn_vehicle_count,
        'n_pickup': n_pickup,
        'n_dropoff': n_dropoff,
    }
    return df_out, df_markers, stats


def main():
    print(f'开始分块处理：{RAW_FILE}')
    print(f'每块大小：{CHUNK_SIZE} 行')

    first_write = True
    total_raw = 0
    total_cleaned = 0
    totals = {
        'n_dup': 0, 'n_abn_jump': 0, 'n_geo': 0, 'n_speed_max': 0,
        'n_abn_vehicle_rows': 0, 'n_abn_vehicle_count': 0,
        'n_pickup': 0, 'n_dropoff': 0,
    }

    leftover = pd.DataFrame(columns=RAW_COLUMNS)

    reader = pd.read_csv(
        RAW_FILE, header=None, names=RAW_COLUMNS,
        chunksize=CHUNK_SIZE, dtype={'id': 'int64', 'status': 'int64', 'speed': 'int64'}
    )

    for i, chunk in enumerate(reader):
        total_raw += len(chunk)

        chunk = pd.concat([leftover, chunk], ignore_index=True)

        last_id = chunk['id'].iloc[-1]
        leftover = chunk[chunk['id'] == last_id].copy()
        chunk_complete = chunk[chunk['id'] != last_id].copy()

        if len(chunk_complete) == 0:
            print(f'第 {i+1} 块：暂无完整车辆数据，继续累积')
            continue

        df_out, df_markers, stats = clean_one_block(chunk_complete)

        total_cleaned += len(df_out)
        for k in totals:
            totals[k] += stats[k]

        df_out.to_csv(CLEANED_FILE, mode='a', header=first_write, index=False)
        df_markers.to_csv(OD_MARKERS_FILE, mode='a', header=first_write, index=False)
        first_write = False

        print(f'第 {i+1} 块完成：原始 {len(chunk_complete)} -> 清洗后 {len(df_out)} | '
              f'重复{stats["n_dup"]} 跳变{stats["n_abn_jump"]} 坐标越界{stats["n_geo"]} '
              f'超速{stats["n_speed_max"]} 全天异常车辆{stats["n_abn_vehicle_count"]}辆/'
              f'{stats["n_abn_vehicle_rows"]}行 | 上车{stats["n_pickup"]} 下车{stats["n_dropoff"]}')

    if len(leftover) > 0:
        df_out, df_markers, stats = clean_one_block(leftover)
        total_cleaned += len(df_out)
        for k in totals:
            totals[k] += stats[k]
        df_out.to_csv(CLEANED_FILE, mode='a', header=first_write, index=False)
        df_markers.to_csv(OD_MARKERS_FILE, mode='a', header=first_write, index=False)
        print(f'末尾遗留车辆处理完成：{len(df_out)} 行')

    print('=' * 60)
    print('全部完成！')
    print(f'原始总行数：{total_raw}')
    print(f'清洗后总行数：{total_cleaned}')
    print(f'累计删除行数：{total_raw - total_cleaned} （占比 {(total_raw-total_cleaned)/total_raw*100:.2f}%）')
    print(f'  - 重复记录：{totals["n_dup"]}')
    print(f'  - 短时间跳变异常：{totals["n_abn_jump"]}')
    print(f'  - 坐标越界：{totals["n_geo"]}')
    print(f'  - 超速(>120km/h)：{totals["n_speed_max"]}')
    print(f'  - 全天异常车辆：{totals["n_abn_vehicle_count"]} 辆，共 {totals["n_abn_vehicle_rows"]} 行')
    print(f'累计上车点：{totals["n_pickup"]}  累计下车点：{totals["n_dropoff"]}')
    print(f'结果已保存到：{CLEANED_FILE}')
    print(f'OD标记数据已保存到：{OD_MARKERS_FILE}')


if __name__ == '__main__':
    main()