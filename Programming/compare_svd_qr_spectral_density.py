import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX style for text in plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

def inverse_cayley(z):
    """Inverse Cayley transform: maps S^1 to R."""
    return 1j * (1 - z) / (1 + z)

def gaussian(x, sigma):
    """Gaussian regularization function."""
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))


# Parameters
n = 50  # Matrix size
sigmas = [0.15, 0.45, 0.75]  # Regularization widths
num_points = 500  # Points on the unit circle

# Generate a single random matrix A
A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
# SVD-based unitary
U, _, _ = np.linalg.svd(A, full_matrices=False)
eigvals_svd = np.linalg.eigvals(U)
# QR-based unitary
Q, _ = np.linalg.qr(A)
eigvals_qr = np.linalg.eigvals(Q)

# Points on the unit circle
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
z = np.exp(1j * theta)
x = inverse_cayley(z)
x_eigs_svd = inverse_cayley(eigvals_svd)
x_eigs_qr = inverse_cayley(eigvals_qr)

# Plot: 2 rows (SVD, QR) x 3 columns (sigmas)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for row, (x_eigs, label) in enumerate(zip([x_eigs_svd, x_eigs_qr], ["SVD-based", "QR-based"])):
    for col, sigma in enumerate(sigmas):
        phi_sigma = np.zeros_like(x, dtype=float)
        for x_i in x_eigs:
            phi_sigma += gaussian(x.real - x_i.real, sigma)
        jacobian = (x.real**2 + 1) / 2
        phi_hat = phi_sigma * jacobian
        phi_hat /= np.trapezoid(phi_hat, theta)
        ax = axes[row, col]
        ax.plot(theta, phi_hat, color='C0', lw=2)
        ax.set_xlim(np.pi/2, 3 * np.pi/2)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xticks([np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
        if col == 0:
            ax.set_ylabel(f'{label}')
            if row == 1:
                ax.set_xlabel(r'$\theta$ (angle on unit circle)')
        if row == 0:
            ax.set_title(fr'$\sigma={sigma}$')
        ax.legend(frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('compare_svd_qr_spectral_density.png', dpi=300)
plt.show()
