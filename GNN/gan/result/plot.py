import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# 读取Excel文件中的两个表格
# sheet_name=0 表示第一个表格，sheet_name=1 表示第二个表格
# 如果表格有表头，可添加 header=0 参数（默认）；如果没有表头，用 header=None
print("当前后端：", matplotlib.get_backend())
df1 = pd.read_excel('color1_datasets.xlsx', sheet_name=0, header=None)  # 第一个表格
df2 = pd.read_excel('color1_datasets.xlsx', sheet_name=1, header=None)  # 第二个表格
df3 = pd.read_excel('color1_datasets.xlsx', sheet_name=2, header=None)
df4 = pd.read_excel('color1_datasets.xlsx', sheet_name=3, header=None)
df5 = pd.read_excel('color1_datasets.xlsx', sheet_name=4, header=None)
df6 = pd.read_excel('color1_datasets.xlsx', sheet_name=5, header=None)
df7 = pd.read_excel('color1_datasets.xlsx', sheet_name=6, header=None)
df8 = pd.read_excel('color1_datasets.xlsx', sheet_name=7, header=None)
# 提取数据
x1 = df1.iloc[1:, 0]  # 第一个表格的第一列作为x轴
y1 = pd.to_numeric(df1.iloc[1:, 1], errors='coerce')  # 第一个表格的第二列作为第一条线
x2 = df2.iloc[1:, 0]
y2 = pd.to_numeric(df2.iloc[1:, 1], errors='coerce')   # 第二个表格的第二列作为第二条线
x3 = df3.iloc[1:, 0]
y3 = pd.to_numeric(df3.iloc[1:, 1], errors='coerce')
x4 = df4.iloc[1:, 0]
y4 = pd.to_numeric(df4.iloc[1:, 1], errors='coerce')
x5 = df5.iloc[1:, 0]
y5 = pd.to_numeric(df5.iloc[1:, 1], errors='coerce')
x6 = df6.iloc[1:, 0]
y6 = pd.to_numeric(df6.iloc[1:, 1], errors='coerce')
x7 = df7.iloc[1:, 0]
y7 = pd.to_numeric(df7.iloc[1:, 1], errors='coerce')
x8 = df8.iloc[1:, 0]
y8 = pd.to_numeric(df8.iloc[1:, 1], errors='coerce')
# 创建画布并绘图
plt.figure(figsize=(10, 6))  # 设置图表大小

# 绘制两条折线
plt.plot(x1, y1, label='bayes', marker='o', linestyle='-', color='b')
plt.plot(x2, y2, label='40k', marker='s', linestyle='--', color='r')
plt.plot(x3, y3, label='10k', marker='*', linestyle='-', color='g')
plt.plot(x4, y4, label='2k', marker='+', linestyle='-', color='y')
plt.plot(x5, y5, label='1k', marker='D', linestyle='-.', color='purple')  # 菱形marker，点划线
plt.plot(x6, y6, label='500', marker='x', linestyle=':', color='orange')  # x形marker，虚线
plt.plot(x7, y7, label='200', marker='^', linestyle='-', color='c')
plt.plot(x8, y8, label='100', marker='v', linestyle='--', color='m')
# 添加图表元素
plt.xlabel('TV', fontsize=14)
plt.ylabel('val_acc', fontsize=14)
plt.title('color1_datasets', fontsize=16)
plt.legend(fontsize=18)  # 显示图例
plt.grid(alpha=0.3)  # 添加网格线
plt.tight_layout()  # 自动调整布局

# 显示图表
plt.show()