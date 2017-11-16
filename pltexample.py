#coding=utf-8


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x)

z = np.cos(x**2)

plt.figure(figsize=(8, 4)) #创建图表

plt.plot(x, y, label='$sin(x)$', color='red', linewidth=2)
plt.plot(x, z, 'g--', label='$co s(x^2)$', lw=3)

#x轴y轴名称以及标题栏
plt.xlabel('Time(s)')
plt.ylabel('volt')
plt.title('First python firgure')

#X轴y轴的的范围
plt.xlim(0,2)
plt.ylim(-1.2, 2)

#显示线段代表的名称
plt.legend(loc = 'right')

# #画子图，其中数值的意思为将整个图分为2行，2列，在第一个图进行操作
# plt.subplot(221)

# #画子图加颜色
# for idx, color in enumerate('rgbyck'):
#     plt.subplot(321 + idx, axisbg=color)


plt.show()