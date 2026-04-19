import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 读取数据与预处理 ---
col_names = [
    '交易类型', '交易时间', '交易卡号', '刷卡类型', '线路号',
    '车辆编号', '上车站点', '下车站点', '驾驶员编号', '运营公司编号'
]

print("正在读取数据...")
# 读取CSV，header=0表示用文件第一行作为列名（会被names参数覆盖），
# 或者如果文件本身没有列名行，应设 header=None。
# 根据你的写法，这里假设文件有表头，但我们要强制用col_names。
df = pd.read_csv('ICData.csv', sep=',', names=col_names, header=0)

# 计算搭乘站点数：下车站点 - 上车站点
# 假设站点编号是数字且递增的
df['ride_stops'] = df['下车站点'] - df['上车站点']

# 过滤掉可能的脏数据（如上下车顺序反了，或者站点编号非数字导致的计算错误）
df = df[df['ride_stops'] > 0]


# --- 2. 定义函数 ---
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 分组聚合
    result = df.groupby(route_col)[stops_col].agg(
        mean_stops='mean',
        std_stops='std'
    ).reset_index()

    # 降序排列
    result = result.sort_values(by='mean_stops', ascending=False).reset_index(drop=True)
    return result


# --- 3. 调用函数并打印前10行 ---
result_df = analyze_route_stops(df)
print("计算结果（前10行）：")
print(result_df.head(10))


# --- 4. 可视化（修正后的代码） ---
# 取均值最高的前15条线路
top_15 = result_df.head(15)

plt.figure(figsize=(10, 8))

# 核心修正：显式指定 x, y, hue
# 注意：Seaborn 0.14+ 版本必须指定 hue 才能使用 palette
ax = sns.barplot(
    data=top_15,
    x='mean_stops',      # 数值轴（X轴）
    y='线路号',          # 分类轴（Y轴）
    hue='线路号',        # 用于映射颜色的分类变量
    xerr='std_stops',    # 误差棒
    palette='Blues_d',   # 颜色主题
    capsize=0.3,         # 误差棒横线宽度
    legend=False         # 关闭图例，因为颜色已经对应了Y轴标签，图例会显得多余
)

# 设置标题和标签
plt.title('Top 15 线路平均搭乘站点数', fontsize=16)
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)

# X轴范围从0开始
plt.xlim(0, top_15['mean_stops'].max() * 1.1)

# 保存图像
plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
print("图像已保存为 route_stops.png")

# 显示图像
plt.show()