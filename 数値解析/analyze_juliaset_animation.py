#モジュールインポート
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from mpl_toolkits.mplot3d import Axes3D

#定数等準備
N,M=200,500
a, b = -0.122, 0.745
x_list, y_list, t_list = [], [], []
X, Y = -0.0967, 0.1558
ims = []
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.set_zlabel("time")
plt.title("first number:Re, Im = -0.0967, 0.1558")

#イテレーション
for i in range(N):
    x_list.append(X)
    y_list.append(Y)
    t_list.append(i/10)
    X, Y = X**2 - Y**2 + a, 2 * X * Y + b 
    #print("Re=",X, ": Im=",Y)
    im = ax.plot(x_list, y_list, t_list, marker="o",linestyle='None', color = "blue")
    ims.append(im)
ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()