import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

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

n = 20  # Matrix size (adjust as needed)
num_trials = 90  # Number of random samples

eigvals_svd = []
eigvals_qr = []
for _ in range(num_trials):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # SVD-based unitary (like MATLAB's orth)
    U1, _, _ = np.linalg.svd(A, full_matrices=False)
    eigvals_svd.extend(np.linalg.eigvals(U1))
    # QR-based unitary
    Q, _ = qr(A)
    eigvals_qr.extend(np.linalg.eigvals(Q))

circle = np.exp(1j * np.linspace(0, 2 * np.pi, 500))

fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# SVD/orth plot
axes[0].scatter(np.real(eigvals_svd), np.imag(eigvals_svd), s=36, alpha=0.8, color='C0', marker='*', label='Eigenvalues')
axes[0].plot(np.real(circle), np.imag(circle), 'k--', linewidth=1, label='Unit circle')
axes[0].set_title(r'Eigenvalues: SVD-based unitary')
axes[0].set_xlabel(r'Re$(\lambda)$')
axes[0].set_ylabel(r'Im$(\lambda)$')
axes[0].set_aspect('equal')
axes[0].set_xlim(-1.2, 1.2)
axes[0].set_ylim(-1.2, 1.2)
axes[0].legend(loc='upper right', fontsize=9, frameon=False)

# QR plot
axes[1].scatter(np.real(eigvals_qr), np.imag(eigvals_qr), s=36, alpha=0.8, color='C1', marker='*', label='Eigenvalues')
axes[1].plot(np.real(circle), np.imag(circle), 'k--', linewidth=1, label='Unit circle')
axes[1].set_title(r'Eigenvalues: QR-based unitary')
axes[1].set_xlabel(r'Re$(\lambda)$')
axes[1].set_ylabel(r'Im$(\lambda)$')
axes[1].set_aspect('equal')
axes[1].set_xlim(-1.2, 1.2)
axes[1].set_ylim(-1.2, 1.2)
axes[1].legend(loc='upper right', fontsize=9, frameon=False)

for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.5)

# Save figures for thesis use
fig.savefig('eigenvalue_comparison.png', dpi=300)
fig.savefig('eigenvalue_comparison.pdf')
plt.show()