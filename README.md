# work3
第三次作业
# 程博洋-25348056-第三次人工智能编程作业

## 1. 任务拆解与 AI 协作策略
依据任务的分块逐步处理，然后再合并处理

## 2. 核心 Prompt 迭代记录
初代 Prompt：import pandas as pd

# 设置 pandas 显示选项，防止输出时被省略
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ==========================================
# 1. 读取数据
# ==========================================
print("--- 步骤 1: 读取数据 ---")
try:
    # 读取制表符分隔的文件
    df = pd.read_csv('ICData.csv', sep='\t')

    # 打印前5行
    print("数据集前5行：")
    print(df.head())

    # 打印基本信息
    print("\n数据集基本信息：")
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("\n各列数据类型：")
    print(df.dtypes)

except FileNotFoundError:
    print("错误：未找到 'ICData.csv' 文件，请确保文件在当前目录下。")
    # 为了演示后续代码，这里退出或创建一个模拟DataFrame，实际运行请确保文件存在
    exit()

# ==========================================
# 2. 时间解析
# ==========================================
print("\n--- 步骤 2: 时间解析 ---")
# 将交易时间转换为 datetime 类型
df['交易时间'] = pd.to_datetime(df['交易时间'])

# 提取小时字段
df['hour'] = df['交易时间'].dt.hour

print("已提取 'hour' 列，前5行预览：")
print(df[['交易时间', 'hour']].head())

# ==========================================
# 3. 构造衍生字段 & 清洗异常
# ==========================================
print("\n--- 步骤 3: 构造衍生字段 & 清洗 ---")
# 计算搭乘站点数：|下车站点 - 上车站点|
# 确保参与计算的列是数值类型，以防读取时出错
df['上车站点'] = pd.to_numeric(df['上车站点'])
df['下车站点'] = pd.to_numeric(df['下车站点'])

df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# 找出 ride_stops 为 0 的异常记录
abnormal_rows = df[df['ride_stops'] == 0]
num_abnormal = len(abnormal_rows)

# 删除异常记录
df_clean = df[df['ride_stops'] != 0].copy()

print(f"发现 ride_stops 为 0 的异常记录共: {num_abnormal} 行")
print(f"已删除异常记录，当前数据集剩余行数: {len(df_clean)}")

# ==========================================
# 4. 缺失值检查与处理
# ==========================================
print("\n--- 步骤 4: 缺失值检查与处理 ---")
missing_values = df_clean.isnull().sum()
print("各列缺失值数量：")
print(missing_values)

# 处理策略：删除包含缺失值的行
# 理由：对于公交刷卡交易数据，关键字段（如卡号、时间、站点）的缺失会导致该条记录无法用于分析，且难以准确填充，故采取删除策略。
initial_count = len(df_clean)
df_final = df_clean.dropna()
final_count = len(df_final)

if initial_count == final_count:
    print("\n检查完毕：数据集中无缺失值，无需处理。")
else:
    print(f"\n处理策略：删除包含缺失值的行。")
    print(f"删除了 {initial_count - final_count} 行缺失数据。")

print("\n最终数据集信息：")
print(f"最终行数: {df_final.shape[0]}")
print(f"最终列数: {df_final.shape[1]}")
AI 生成的问题：无法正确的读取数据
优化后的 Prompt：df = pd.read_csv('ICData.csv', sep='\t')

# 【关键修复】去掉所有列名前后的空格
df.columns = df.columns.str.strip()

# 然后再执行你的时间解析代码
df['交易时间'] = pd.to_datetime(df['交易时间'])

## 3. Debug 记录
报错现象：<img width="1448" height="969" alt="image" src="https://github.com/user-attachments/assets/6280db95-80d7-4014-86f5-e7f7bc865d1f" />

解决过程：分析问题，发现根本原因：使用的 Seaborn 版本较新（v0.13.0及以上），而教程或代码是基于旧版本编写的。
机制变化：在旧版本中，Seaborn 会自动将 x 和 y 参数传递给 Matplotlib 来绘制误差线（errorbar）。但在新版本中，Seaborn 接管了误差线的绘制逻辑，不再直接透传 xerr 或 yerr 数组给 Matplotlib，导致 Matplotlib 收到的数据形状不匹配（它期望标量，却收到了数组），从而引发崩溃。
解决思路：显式地指定 x 和 y 参数，而不是依赖 Seaborn 的自动推断。当参数显式指定时，新版本的 Seaborn 能够正确处理内部逻辑。

## 4. 人工代码审查（逐行中文注释）
（贴出任务4 PHF 计算的核心代码，并加上你自己的逐行中文注释）
```python
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
# 强制用col_names。
df = pd.read_csv('ICData.csv', sep=',', names=col_names, header=0)

# 计算搭乘站点数：下车站点 - 上车站点
# 假设站点编号是数字且递增的
df['ride_stops'] = df['下车站点'] - df['上车站点']

# 过滤掉可能的错误数据
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
