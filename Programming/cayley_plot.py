import numpy as np
import matplotlib.pyplot as plt

# Generate real numbers from -5 to 5 (dense for smooth curve)
z_real = np.linspace(-5, 5, 1000)
z_complex = z_real + 0j  # Make them complex numbers

# Cayley transform: φ(z) = (i - z) / (i + z)
phi_z = (1j - z_complex) / (1j + z_complex)

# Split into those from [-1, 1] and the rest
mask = np.abs(z_real) <= 1
phi_inner = phi_z[mask]
phi_outer = phi_z[~mask]

# Unit circle for reference
theta = np.linspace(0, 2 * np.pi, 500)
unit_circle = np.exp(1j * theta)

# Plot
plt.figure(figsize=(6, 6))
plt.plot(unit_circle.real, unit_circle.imag, 'k--', label='Unit circle')
plt.plot(phi_inner.real, phi_inner.imag, 'blue', label=r'$\varphi(z), z \in [-1, 1]$')
plt.plot(phi_outer.real, phi_outer.imag, 'orange', label=r'$\varphi(z), z \notin [-1, 1]$')

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.gca().set_aspect('equal')
plt.title("Cayley Transform of Real Line")
plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid(True)
plt.show()