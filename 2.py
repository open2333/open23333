import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter



# Constants
g = 9.81  # Acceleration due to gravity
m1 = 1.0  # Mass of the first pendulum
m2 = 1.0  # Mass of the second pendulum
l1 = 1.0  # Length of the first pendulum
l2 = 1.0  # Length of the second pendulum

# Time settings
t_max = 20  # maximum time for the simulation
dt = 0.05  # time step
n_steps = int(t_max / dt)  # number of time steps
time = np.linspace(0, t_max, n_steps)

# Initial conditions [theta1, omega1, theta2, omega2]
initial_conditions = [np.pi / 4, 0, np.pi / 4, 0]

# Initialize arrays to store the data
theta1 = np.zeros(n_steps)
omega1 = np.zeros(n_steps)
theta2 = np.zeros(n_steps)
omega2 = np.zeros(n_steps)
theta1[0], omega1[0], theta2[0], omega2[0] = initial_conditions


# Function to compute derivatives
def derivatives(state, t):
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    dydx[1] = ((m2 * l2 * state[3] ** 2 * np.sin(delta) * np.cos(delta) +
                m2 * g * np.sin(state[2]) * np.cos(delta) +
                m2 * l2 * state[3] ** 2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(state[0])) / denom1)

    dydx[2] = state[3]

    denom2 = (l2 / l1) * denom1
    dydx[3] = ((-l1 / l2 * state[1] ** 2 * np.sin(delta) * np.cos(delta) +
                (m1 + m2) * g * np.sin(state[0]) * np.cos(delta) -
                m2 * g * np.sin(state[2]) -
                (m1 + m2) * l1 * state[1] ** 2 * np.sin(delta)) / denom2)

    return dydx


# Perform the numerical integration (4th-order Runge-Kutta method)
for i in range(1, n_steps):
    # Compute derivatives for the current state
    k1 = derivatives([theta1[i - 1], omega1[i - 1], theta2[i - 1], omega2[i - 1]], i * dt)
    k2 = derivatives([theta1[i - 1] + 0.5 * dt * k1[0], omega1[i - 1] + 0.5 * dt * k1[1],
                      theta2[i - 1] + 0.5 * dt * k1[2], omega2[i - 1] + 0.5 * dt * k1[3]], (i + 0.5) * dt)
    k3 = derivatives([theta1[i - 1] + 0.5 * dt * k2[0], omega1[i - 1] + 0.5 * dt * k2[1],
                      theta2[i - 1] + 0.5 * dt * k2[2], omega2[i - 1] + 0.5 * dt * k2[3]], (i + 0.5) * dt)
    k4 = derivatives([theta1[i - 1] + dt * k3[0], omega1[i - 1] + dt * k3[1],
                      theta2[i - 1] + dt * k3[2], omega2[i - 1] + dt * k3[3]], (i + 1) * dt)

    # Update the state
    theta1[i] = theta1[i - 1] + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    omega1[i] = omega1[i - 1] + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    theta2[i] = theta2[i - 1] + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    omega2[i] = omega2[i - 1] + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
line, = ax.plot([], [], 'o-', lw=2)


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    x1 = l1 * np.sin(theta1[i])
    y1 = -l1 * np.cos(theta1[i])
    x2 = x1 + l2 * np.sin(theta2[i])
    y2 = y1 - l2 * np.cos(theta2[i])
    line.set_data([0, x1, x2], [0, y1, y2])
    return (line,)


ani = FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=dt * 1000, blit=True)

# Create a PillowWriter object
writer = PillowWriter(fps=20)

# Save the animation

plt.show()
