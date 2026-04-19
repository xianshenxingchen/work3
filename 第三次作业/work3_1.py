import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，防止图表乱码
# Windows系统通常使用 'SimHei'，Mac/Linux可能需要 'Arial Unicode MS' 或其他
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 数据读取与预处理 (承接上一任务)
# ==========================================
print("--- 正在进行数据预处理 ---")
col_names = [
    '交易类型', '交易时间', '交易卡号', '刷卡类型', '线路号',
    '车辆编号', '上车站点', '下车站点', '驾驶员编号', '运营公司编号'
]

# 注意：sep=',' 是因为你提供的样本数据是逗号分隔的
df = pd.read_csv('ICData.csv', sep=',', names=col_names, header=0)

# 时间解析
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour

# 计算搭乘站点数并清洗
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()
df = df[df['ride_stops'] != 0]

# 筛选上车记录 (刷卡类型 == 0)
# 为了后续统计准确，我们只保留上车数据进行分析
df_boarding = df[df['刷卡类型'] == 0].copy()

print(f"预处理完成。上车刷卡总记录数: {len(df_boarding)}")

# ==========================================
# (a) 早晚时段刷卡量统计 (必须使用 numpy)
# ==========================================
print("\n--- (a) 早晚时段刷卡量统计 ---")

# 将数据转换为 numpy 数组以便使用 numpy 操作
# 这里我们主要对 'hour' 列进行操作
hours_array = df_boarding['hour'].values

# 1. 早峰前时段 (hour < 7)
# 使用 numpy 布尔索引
mask_morning = hours_array < 7
count_morning = np.sum(mask_morning)  # 统计 True 的数量

# 2. 深夜时段 (hour >= 22)
# 使用 numpy.where 示例 (虽然布尔索引也可以，这里演示 where 的用法)
# np.where 返回满足条件的索引或值，这里我们要统计数量
indices_night = np.where(hours_array >= 22)
count_night = len(indices_night[0])

# 3. 计算百分比
total_count = len(hours_array)
percent_morning = (count_morning / total_count) * 100
percent_night = (count_night / total_count) * 100

print(f"早峰前时段 (00:00-06:59) 刷卡量: {count_morning} 次, 占比: {percent_morning:.2f}%")
print(f"深夜时段 (22:00-23:59) 刷卡量: {count_night} 次, 占比: {percent_night:.2f}%")
print(f"全天总上车刷卡量: {total_count} 次")

# ==========================================
# (b) 24小时刷卡量分布可视化
# ==========================================
print("\n--- (b) 正在绘制 24小时分布图 ---")

# 1. 统计每个小时的刷卡量
# value_counts 统计频次，sort_index 确保按 0-23 顺序排列
hourly_counts = df_boarding['hour'].value_counts().sort_index()

# 确保 0-23 小时都存在（即使某些小时数据为0），防止绘图错位
all_hours = np.arange(24)
# reindex 用于对齐索引，fill_value=0 填充缺失的小时
hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)

# 2. 绘图设置
plt.figure(figsize=(12, 6))

# 定义颜色策略：
# 默认灰色，早峰前(<7)红色，深夜(>=22)橙色
colors = []
for h in all_hours:
    if h < 7:
        colors.append('#ff9999')  # 浅红 (早峰前)
    elif h >= 22:
        colors.append('#ffcc99')  # 浅橙 (深夜)
    else:
        colors.append('#66b3ff')  # 蓝色 (正常时段)

# 3. 绘制柱状图
bars = plt.bar(all_hours, hourly_counts, color=colors, edgecolor='black', alpha=0.8)

# 4. 图表装饰
plt.title('24小时公交上车刷卡量分布图', fontsize=16)
plt.xlabel('小时 (0-23)', fontsize=12)
plt.ylabel('刷卡量 (次)', fontsize=12)

# 设置 x 轴刻度：0-23，步长为 2
plt.xticks(np.arange(0, 24, 2))

# 添加网格线 (仅 y 轴方向)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 创建图例句柄 (手动创建颜色块用于图例)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ff9999', edgecolor='black', label='早峰前 (< 07:00)'),
    Patch(facecolor='#66b3ff', edgecolor='black', label='日间时段'),
    Patch(facecolor='#ffcc99', edgecolor='black', label='深夜 (>= 22:00)')
]
plt.legend(handles=legend_elements, loc='upper right')

# 5. 保存并显示
plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
print("图表已保存为 hour_distribution.png")
plt.show()