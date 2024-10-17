from math import factorial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random
import numpy as np

p1 = (0,0)
p4 = (1,1)
index_x = random.uniform(0, 1)
p2 = (index_x, random.uniform(index_x, 1))
index_x = random.uniform(0, 1)
p3 = (index_x, random.uniform(index_x, 1))

pp2 = (0.5,0.7)
pp3 = (0.4,0.8)
points = [p1,pp2,pp3,p4]# 在此处修改坐标
N = len(points)
n = N - 1
px = []
py = []
fig = plt.figure(figsize=(5,4))
for T in range(1001):
    t = T*0.001
    x,y = 0,0
    for i in range(N):
        B = factorial(n)*t**i*(1-t)**(n-i)/(factorial(i)*factorial(n-i))
        x += points[i][0]*B
        y += points[i][1]*B
    px.append(x)
    py.append(y)
f = interp1d(px, py, kind='linear', axis=-1)
matrix = np.random.random(size=(196,256))

# x = matrix[0][0]
# y = matrix[0][1]
# out = f(matrix)
# out_x = f(x)
# out_y = f(y)
plt.plot(px,py,color='r')
plt.plot([i[0] for i in points],[i[1] for i in points],'r.')
# points = [(-0.01,0.01),(0,0.5),(0.5,1),(0.985,1.01)]
# px = []
# py = []
# for T in range(1001):
#     t = T*0.001
#     x,y = 0,0
#     for i in range(N):
#         B = factorial(n)*t**i*(1-t)**(n-i)/(factorial(i)*factorial(n-i))
#         x += points[i][0]*B
#         y += points[i][1]*B
#     px.append(x)
#     py.append(y)
# f = interp1d(px, py, kind='linear', axis=-1)
# matrix = np.random.random(size=(196,256))
# plt.plot(px,py,color='b')
# plt.plot([i[0] for i in points],[i[1] for i in points],'b.')
# ax = fig.add_subplot(111)
# ax.patch.set_facecolor((252/255,224/255,224/255))
plt.show()
print(1)
