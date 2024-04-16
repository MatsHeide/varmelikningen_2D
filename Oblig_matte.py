print("Hello world")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

print("2D varmelikning")


alpha = 2
delta_x = 1

plate_length = 75
max_iter_time = 90


delta_t = (delta_x ** 2) / (4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)


u = np.empty((max_iter_time, plate_length, plate_length))

# Initialbetingelser på innsiden
u_initial = 5

x_values = np.arange(0, plate_length, 1)
initial_condition = x_values**2
for i in [u]:
    u[0, 20:30,:] = initial_condition

# Definer grenser
u_top = 0.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0

# Initialbetingelser i top, bunn, venstre og høyre 
u[:, 0, :] = u_top
u[:, -1, :] = u_bottom
u[:, :, 0] = u_left
u[:, :, -1] = u_right


def calculate(u):
    for k in range(0, max_iter_time - 1):
        for i in range(1, plate_length - 1):
            for j in range(1, plate_length - 1):
                u[k + 1, i, j] = gamma * (u[k][i + 1][j] + u[k][i - 1][j] + u[k][i][j + 1] + u[k][i][j - 1] - 4 * u[k][i][j]) + u[k][i][j]
    return u

u =calculate(u)

x = np.arange(0, plate_length, delta_x)
y = np.arange(0, plate_length, delta_x)
X, Y = np.meshgrid(x, y)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def animate(k):
    ax.clear()
    ax.plot_surface(X, Y, u[k], cmap=plt.cm.jet, rstride=1, cstride=1, edgecolor='none')
    ax.set_xlabel('X akse')
    ax.set_ylabel('Y akse')
    ax.set_zlabel('Temperatur')
    ax.set_title(f"Temperatur fordeling over tid {k * delta_t:.2f}")


anim = FuncAnimation(fig, animate, frames=max_iter_time, interval=50)


plt.show()
