import matplotlib
matplotlib.use('QtAgg')  # 强制指定QtAgg
import matplotlib.pyplot as plt
import numpy as np

# 极简绘图：仅画一条直线
x = np.linspace(0, 10, 100)
y = x + 1
plt.plot(x, y)
plt.title('QtAgg Test')
plt.show()

# 打印关键信息，辅助排查
print("QtAgg后端是否可用：", matplotlib.rcParams['backend'] == 'QtAgg')
print("PyQt5/PySide2是否安装：")
try:
    import PyQt5
    print("✅ PyQt5已安装")
except ImportError:
    try:
        import PySide2
        print("✅ PySide2已安装")
    except ImportError:
        print("❌ 缺少Qt绑定库（PyQt5/PySide2）")
# 极简绘图：仅画一条直线
x = np.linspace(0, 10, 100)
y = x + 1
plt.plot(x, y)
plt.title('QtAgg Test')
plt.show()