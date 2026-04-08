"""
从 grampa_s_aureus_7_25_with_GRAMPA.csv 提取 Activity 列
添加到 train.csv 和 test.csv 中
"""

import pandas as pd
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 文件路径
grampa_file = os.path.join(script_dir, 'grampa_s_aureus_7_25_with_GRAMPA.csv')
train_file = os.path.join(script_dir, 'train.csv')
test_file = os.path.join(script_dir, 'test.csv')

# 读取 grampa 文件，建立 ID -> Activity 映射
print(f"读取: {grampa_file}")
grampa_df = pd.read_csv(grampa_file)
print(f"  共 {len(grampa_df)} 条记录")
print(f"  列: {list(grampa_df.columns)}")

# 创建 ID -> Activity 的字典
id_to_activity = dict(zip(grampa_df['ID'], grampa_df['Activity']))
print(f"  Activity 范围: {grampa_df['Activity'].min():.4f} ~ {grampa_df['Activity'].max():.4f}")

# 处理 train.csv
print(f"\n处理: {train_file}")
train_df = pd.read_csv(train_file)
print(f"  原始列: {list(train_df.columns)}")
print(f"  共 {len(train_df)} 条记录")

# 添加 Activity 列
train_df['Activity'] = train_df['ID'].map(id_to_activity)
matched = train_df['Activity'].notna().sum()
print(f"  匹配成功: {matched}/{len(train_df)}")

if train_df['Activity'].isna().any():
    missing = train_df[train_df['Activity'].isna()]['ID'].tolist()
    print(f"  警告: {len(missing)} 个ID未找到Activity: {missing[:5]}...")

# 保存
train_df.to_csv(train_file, index=False)
print(f"  已保存: {train_file}")
print(f"  新列: {list(train_df.columns)}")

# 处理 test.csv
print(f"\n处理: {test_file}")
test_df = pd.read_csv(test_file)
print(f"  原始列: {list(test_df.columns)}")
print(f"  共 {len(test_df)} 条记录")

# 添加 Activity 列
test_df['Activity'] = test_df['ID'].map(id_to_activity)
matched = test_df['Activity'].notna().sum()
print(f"  匹配成功: {matched}/{len(test_df)}")

if test_df['Activity'].isna().any():
    missing = test_df[test_df['Activity'].isna()]['ID'].tolist()
    print(f"  警告: {len(missing)} 个ID未找到Activity: {missing[:5]}...")

# 保存
test_df.to_csv(test_file, index=False)
print(f"  已保存: {test_file}")
print(f"  新列: {list(test_df.columns)}")

print("\n完成!")
