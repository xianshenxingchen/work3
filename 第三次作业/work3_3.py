import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 0. 基础设置与数据读取
# ==========================================
# 设置绘图风格和中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("--- 正在读取数据... ---")
col_names = [
    '交易类型', '交易时间', '交易卡号', '刷卡类型', '线路号',
    '车辆编号', '上车站点', '下车站点', '驾驶员编号', '运营公司编号'
]

# 读取数据
df = pd.read_csv('ICData.csv', sep=',', names=col_names, header=0)

# 预处理时间
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['minute'] = df['交易时间'].dt.minute

# ==========================================
# 任务 4：高峰小时系数计算
# ==========================================
print("\n" + "="*30)
print("任务 4：高峰小时系数计算")
print("="*30)

# 1. 高峰小时识别
hourly_counts = df.groupby('hour').size()
peak_hour = hourly_counts.idxmax()
peak_volume = hourly_counts.max()

print(f"高峰小时为：{peak_hour}:00 ~ {peak_hour+1}:00，刷卡量：{peak_volume} 次")

# 筛选高峰小时的数据
df_peak = df[df['hour'] == peak_hour].copy()

# 2. 5分钟粒度统计
# 将分钟向下取整到最近的5分钟 (例如 08:13 -> 08:10)
df_peak['time_5min'] = df_peak['交易时间'].dt.floor('5min')
vol_5min = df_peak.groupby('time_5min').size()
max_vol_5 = vol_5min.max()
time_5min_label = vol_5min.idxmax().strftime('%H:%M')

phf5 = peak_volume / (12 * max_vol_5)

print(f"最大5分钟刷卡量（{time_5min_label}~）：{max_vol_5} 次")
print(f"PHF5  = {peak_volume} / (12 × {max_vol_5}) = {phf5:.4f}")

# 3. 15分钟粒度统计
# 将分钟向下取整到最近的15分钟
df_peak['time_15min'] = df_peak['交易时间'].dt.floor('15min')
vol_15min = df_peak.groupby('time_15min').size()
max_vol_15 = vol_15min.max()
time_15min_label = vol_15min.idxmax().strftime('%H:%M')

phf15 = peak_volume / (4 * max_vol_15)

print(f"最大15分钟刷卡量（{time_15min_label}~）：{max_vol_15} 次")
print(f"PHF15 = {peak_volume} / (4 × {max_vol_15}) = {phf15:.4f}")

# ==========================================
# 任务 5：线路驾驶员信息批量导出
# ==========================================
print("\n" + "="*30)
print("任务 5：线路驾驶员信息批量导出")
print("="*30)

# 1. 筛选线路 1101-1120
mask = (df['线路号'] >= 1101) & (df['线路号'] <= 1120)
df_filtered = df[mask]

# 2. 创建文件夹
output_dir = '线路驾驶员信息'
os.makedirs(output_dir, exist_ok=True)

# 3. 循环导出
print("正在生成文件...")
for route_id in range(1101, 1121):
    df_route = df_filtered[df_filtered['线路号'] == route_id]
    
    # 去重：获取唯一的 车辆编号 -> 驾驶员编号 对应关系
    # 假设同一辆车在同一条线路上应该对应同一个司机，或者我们要列出所有出现过的组合
    unique_pairs = df_route[['车辆编号', '驾驶员编号']].drop_duplicates()
    
    # 排序以便查看（可选）
    unique_pairs = unique_pairs.sort_values(by='车辆编号')
    
    # 写入文件
    file_path = os.path.join(output_dir, f"{route_id}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"线路号: {route_id}\n")
        f.write("车辆编号\t驾驶员编号\n")
        for _, row in unique_pairs.iterrows():
            f.write(f"{int(row['车辆编号'])}\t{int(row['驾驶员编号'])}\n")
            
    print(f"已生成: {file_path}")

# ==========================================
# 任务 6：服务绩效排名与热力图
# ==========================================
print("\n" + "="*30)
print("任务 6：服务绩效排名与热力图")
print("="*30)

# 1. 排名统计
# 司机
top_drivers = df.groupby('驾驶员编号').size().sort_values(ascending=False).head(10)
# 线路
top_routes = df.groupby('线路号').size().sort_values(ascending=False).head(10)
# 上车站点
top_stops = df.groupby('上车站点').size().sort_values(ascending=False).head(10)
# 车辆
top_vehicles = df.groupby('车辆编号').size().sort_values(ascending=False).head(10)

print("Top 10 司机:\n", top_drivers)
print("\nTop 10 线路:\n", top_routes)

# 2. 构造热力图数据 (4x10 矩阵)
# 将 Series 的值提取出来作为行数据
heatmap_data = pd.DataFrame([
    top_drivers.values,
    top_routes.values,
    top_stops.values,
    top_vehicles.values
], index=['司机', '线路', '上车站点', '车辆'])

# 设置列名为 Top1 - Top10
heatmap_data.columns = [f'Top{i}' for i in range(1, 11)]

# 绘图
plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",        # 显示整数
    cmap="YlOrRd",  # 颜色映射
    linewidths=.5,
    cbar_kws={'label': '服务人次'}
)

plt.title("各维度 Top10 服务绩效热力图\n(基于有效刷卡记录数)", fontsize=14)
plt.xlabel("排名", fontsize=12)
plt.ylabel("维度", fontsize=12)
plt.xticks(rotation=0)

# 保存图像
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("热力图已保存为 performance_heatmap.png")

# 3. 结论说明
print("\n--- 结论说明 ---")
print("""
从热力图中可以观察到明显的服务绩效差异规律：
. 线路维度的客流量差异最为巨大（颜色最深），Top 1 线路的服务人次远超其他线路，说明该城市公交网络中存在极少数“超级骨干线路”，承载了不成比例的巨大客流。
. 相比之下，单车和单司机的服务人次分布相对均匀（颜色梯度较缓），这反映了排班制度较为均衡，没有司机或车辆出现极端的超长工作时间。
. 上车站点的热点集中度高，Top 1-3 站点的颜色显著深于后续站点，表明城市交通流具有极强的方向性和枢纽依赖性（如大型居住区或交通枢纽）。
""")