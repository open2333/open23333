# Importing required libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
g = 9.81  # gravity
l1, l2, l3 = 1.0, 1.0, 1.0  # lengths
m1, m2, m3 = 1.0, 1.0, 1.0  # masses

# Time settings
t_max = 10
dt = 0.05
t = np.linspace(0, t_max, int(t_max / dt))

# Initial conditions [theta1, omega1, theta2, omega2, theta3, omega3]
initial_conditions = [np.pi / 3.9, 0, np.pi / 3.9, 0, np.pi / 3.9, 0]


# Function to return derivatives
def derivatives(y, t):
    theta1, omega1, theta2, omega2, theta3, omega3 = y

    dydt = np.zeros_like(y)

    # For theta1 and omega1
    dydt[0] = omega1
    dydt[1] = -((g * (2 * m1 + m2 + m3) * np.sin(theta1) + m2 * g * np.sin(theta1 - 2 * theta2) + m3 * g * np.sin(
        theta1 - 2 * theta3) + 2 * np.sin(theta1 - theta2) * m2 * (
                             omega2 ** 2 * l2 + omega1 ** 2 * l1 * np.cos(theta1 - theta2)) + 2 * np.sin(
        theta1 - theta3) * m3 * (omega3 ** 2 * l3 + omega1 ** 2 * l1 * np.cos(theta1 - theta3))) / (l1 * (
                2 * m1 + m2 + m3 - m2 * np.cos(2 * theta1 - 2 * theta2) - m3 * np.cos(2 * theta1 - 2 * theta3))))

    # For theta2 and omega2
    dydt[2] = omega2
    dydt[3] = (((m1 + m2) * l1 * omega1 ** 2 * np.sin(theta1 - theta2) + m2 * l2 * omega2 ** 2 * np.sin(
        theta2 - theta1) * np.cos(theta1 - theta2) + (m1 + m2 + m3) * g * np.sin(theta1) * np.cos(theta1 - theta2) - (
                            m2 + m3) * g * np.sin(theta2)) / (l2 * (
                m1 + m2 + m3 - m2 * np.cos(2 * theta1 - 2 * theta2) - m3 * np.cos(2 * theta1 - 2 * theta3))))

    # For theta3 and omega3
    dydt[4] = omega3
    dydt[5] = (((m1 + m2 + m3) * l1 * omega1 ** 2 * np.sin(theta1 - theta3) + m3 * l3 * omega3 ** 2 * np.sin(
        theta3 - theta1) * np.cos(theta1 - theta3) + (m1 + m2 + m3) * g * np.sin(theta1) * np.cos(
        theta1 - theta3) - m3 * g * np.sin(theta3)) / (l3 * (
                m1 + m2 + m3 - m2 * np.cos(2 * theta1 - 2 * theta2) - m3 * np.cos(2 * theta1 - 2 * theta3))))

    return dydt


# Solving the ODE using odeint
solution = odeint(derivatives, initial_conditions, t)

# Extract the solutions for plotting
theta1, omega1, theta2, omega2, theta3, omega3 = solution.T

# Create the animation
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
line, = ax.plot([], [], 'o-', lw=2)


# Initialize the plot
def init():
    line.set_data([], [])
    return (line,)


# Animation function
def animate(i):
    x1 = l1 * np.sin(theta1[i])
    y1 = -l1 * np.cos(theta1[i])
    x2 = x1 + l2 * np.sin(theta2[i])
    y2 = y1 - l2 * np.cos(theta2[i])
    x3 = x2 + l3 * np.sin(theta3[i])
    y3 = y2 - l3 * np.cos(theta3[i])
    line.set_data([0, x1, x2, x3], [0, y1, y2, y3])
    return (line,)


ani = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=dt * 1000, blit=True)



plt.show()

x3_vals = np.zeros(len(t))
y3_vals = np.zeros(len(t))

# Extract positions for animation and trajectory plotting
for i in range(len(t)):
    x1 = l1 * np.sin(theta1[i])
    y1 = -l1 * np.cos(theta1[i])
    x2 = x1 + l2 * np.sin(theta2[i])
    y2 = y1 - l2 * np.cos(theta2[i])
    x3 = x2 + l3 * np.sin(theta3[i])
    y3 = y2 - l3 * np.cos(theta3[i])
    x3_vals[i] = x3
    y3_vals[i] = y3

# Plotting the trajectory
plt.figure()
plt.plot(x3_vals, y3_vals, label='End of Third Pendulum', color='r')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory of the Third Pendulum')









