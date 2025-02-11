import numpy as np
import matplotlib.pyplot as plt

# 横坐标范围
x = np.arange(24)  # 0到23

# 生成一个形状为 (24, 10) 的矩阵，表示10条曲线的纵坐标
y = np.random.randn(24, 10)  # 随机生成10条曲线的数据

# 绘制图形
plt.figure(figsize=(10, 6))  # 设置画布大小
plt.plot(x, y, label=[f'Curve {i+1}' for i in range(10)])  # 绘制10条曲线

# 添加标题和标签
plt.title('Ten Curves from 0 to 23')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)  # 添加网格
plt.show()