import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_config import RAW_FILE, SAMPLE_FILE, RAW_COLUMNS

# 只读取前 N 行作为样本，不会加载全部 1.88GB
N_SAMPLE = 50000

df_sample = pd.read_csv(RAW_FILE, header=None, names=RAW_COLUMNS, nrows=N_SAMPLE)
df_sample.to_csv(SAMPLE_FILE, index=False)

print(f'样本已生成，共 {len(df_sample)} 行，保存到 {SAMPLE_FILE}')
print(df_sample.head())