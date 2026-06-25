import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import SAMPLE_FILE, CLEANED_FILE, RAW_COLUMNS, DUP_TIME_GAP_THRESHOLD  # noqa: E402

# 数据对应的日期（原始 time 字段只有时分秒，需要补日期才能转换为完整时间戳）
DATA_DATE = '2013-10-22'


def load_data(file_path, has_header=True):
    """读取数据并统一字段名。
    has_header=True 用于读取已经处理过的样本/中间文件（自带表头）；
    has_header=False 用于读取最原始的 TaxiData.csv（无表头）。
    """
    if has_header:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, header=None, names=RAW_COLUMNS)
    print(f'读取完成，共 {len(df)} 行')
    return df


def sort_and_convert_time(df):
    """补日期 -> 转换为 datetime -> 按 id,time 排序"""
    # 补日期：将 '21:09:38' 拼接为 '2013-10-22 21:09:38'
    df['time'] = DATA_DATE + ' ' + df['time'].astype(str)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

    df = df.sort_values(by=['id', 'time']).reset_index(drop=True)
    print('排序与时间转换完成')
    return df


def remove_duplicates(df):
    """处理 id,time 重复的记录（重复数量为2或3的多种 status 组合）"""
    df_dup = df[df.duplicated(subset=['id', 'time'], keep=False)].reset_index()
    print(f'发现重复记录（含全部重复行）：{len(df_dup)} 条')

    if len(df_dup) == 0:
        return df

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
        # 超过3个重复的极端情况，保底返回第一个，避免 apply 报错
        return x['index'].values[0]

    kp_index = dup_mrg.groupby(['id', 'time']).apply(dup_check)
    drp_index = dup_mrg.loc[~dup_mrg['index'].isin(kp_index.values), 'index']

    before = len(df)
    df = df.loc[~df.index.isin(drp_index.values)].reset_index(drop=True)
    print(f'去重完成：{before} -> {len(df)} 行（删除 {before - len(df)} 行）')
    return df


def remove_outliers(df):
    """剔除短时间内状态异常跳变的记录（0-1-0 或 1-0-1，时间差小于阈值）"""
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
    print(f'发现异常跳变记录：{len(df_abn)} 条')

    before = len(df)
    df = df.loc[~df.index.isin(df_abn.index)].reset_index(drop=True)
    print(f'异常值剔除完成：{before} -> {len(df)} 行（删除 {before - len(df)} 行）')
    return df


def start_od_extraction(df):
    """OD 启动：计算 status_chg，筛选上车点(0->1)和下车点(1->0)"""
    # 重新计算 shift（因为前面剔除异常值后索引已重置，shift 需要基于最新数据）
    df['status_up'] = df['status'].shift(1)
    df['id_up'] = df['id'].shift(1)

    df['status_chg'] = df['status'] - df['status_up']
    df['id_chg'] = df['id'] - df['id_up']

    df_pickup = df.loc[(df['status_chg'] == 1) & (df['id_chg'] == 0)]
    df_dropoff = df.loc[(df['status_chg'] == -1) & (df['id_chg'] == 0)]

    print(f'初步识别上车点：{len(df_pickup)} 个')
    print(f'初步识别下车点：{len(df_dropoff)} 个')
    return df_pickup, df_dropoff


def main():
    df = load_data(SAMPLE_FILE)
    print(df.dtypes)
    print('-' * 50)

    df = sort_and_convert_time(df)
    print('-' * 50)

    df = remove_duplicates(df)
    print('-' * 50)

    df = remove_outliers(df)
    print('-' * 50)

    df_pickup, df_dropoff = start_od_extraction(df)
    print('-' * 50)

    # 保存清洗后的数据，供后续 OD 提取阶段直接使用
    df.to_csv(CLEANED_FILE, index=False)
    print(f'清洗后数据已保存到 {CLEANED_FILE}')


if __name__ == '__main__':
    main()