# import numpy as np
# import matplotlib.pyplot as plt

# # 横坐标范围
# x = np.arange(24)  # 0到23

# # 生成一个形状为 (24, 10) 的矩阵，表示10条曲线的纵坐标
# y = np.random.randn(24, 10)  # 随机生成10条曲线的数据

# # 绘制图形
# plt.figure(figsize=(10, 6))  # 设置画布大小
# plt.plot(x, y, label=[f'Curve {i+1}' for i in range(10)])  # 绘制10条曲线

# # 添加标题和标签
# plt.title('Ten Curves from 0 to 23')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# # 显示图例
# plt.legend()

# # 显示图形
# plt.grid(True)  # 添加网格
# plt.show()

import torch
x = torch.ones(7,1)  # keep original initialization

# Create a lower triangular matrix of ones
x = torch.tril(x, diagonal=-1)


# Create an upper triangular matrix of ones with zero diagonal
# y = torch.triu(x, diagonal=-1)

# Create mask for upper and lower triangles
# mask = torch.ones_like(x)
# n = x.size(1)
# for i in range(x.size(0)):
#     for j in range(n):
#         if j < i or j >= n-i:
#             mask[i,j] = 0

# # Apply mask to x
# x = x * mask
print(x)

y = torch.zeros(7, 4)
diags = torch.arange(6)# [:, None]  # Values for each diagonal [0,1,2,3,4,5]
offsets = torch.arange(1, 7)  # Diagonal offsets [1,2,3,4,5,6]
for i, (d, offset) in enumerate(zip(diags, offsets)):
    print(d, offset)
    y.diagonal(-offset).fill_(d)
print(y)

diagonal_sums = torch.diagonal(y, offset=-torch.arange(1, 7), dim1=0, dim2=1).sum(dim=1)
print("Diagonal sums:", diagonal_sums.tolist())
# # Zero out upper triangle
# for i in range(0, 2):
#     y[:, :i+1] = 0
# # Zero out lower triangle
# for i in range(0, 2):
#     y[:, -(i+1):] = 0

# print(y)