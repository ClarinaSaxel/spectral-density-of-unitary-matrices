import matplotlib.pyplot as plt
import numpy as np

# map the interval [-1, 1]
z_points = np.linspace(-1, 1, 1000)
phi_z = (1j - z_points) / (1j + z_points)

# map som random points outside of the interval [-1, 1]
x_points = [-100, -2, 5, 30]
phi_x = [(1j - x) / (1j + x) for x in x_points]

# Unit circle for reference
theta = np.linspace(0, 2 * np.pi, 500)
unit_circle = np.exp(1j * theta)

plt.figure(figsize=(6, 6))
plt.plot(unit_circle.real, unit_circle.imag, 'k--', label='Unit circle')
plt.plot(phi_z.real, phi_z.imag, label=r"\varphi(z) = $\frac{i - z}{i + z}$")

plt.title("Cayley Transform on [-1, 1]")
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.show()