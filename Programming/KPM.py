import numpy as np
from scipy.sparse.linalg import gmres

def kpm_unitary(U, M, n_vec, t_points):
    """
    Kernel Polynomial Method for a unitary matrix U using the implicit Cayley transform.
    Parameters:
        U: np.ndarray, shape (n, n)
            Unitary matrix (UU^* = I)
        M: int
            Number of Chebyshev polynomials
        n_vec: int
            Number of random vectors for trace estimation
        t_points: np.ndarray
            Points at which to evaluate the spectral density (in [-1, 1])
    Returns:
        phi_tilde: np.ndarray
            Approximated spectral density at t_points
    """
    n = U.shape[0]
    zeta = np.zeros(M + 1, dtype=np.complex128)
    for l in range(n_vec):
        v0 = np.random.randn(n) + 1j * np.random.randn(n)
        v0 = v0 / np.linalg.norm(v0)
        v_km1 = None
        v_k = v0.copy()
        for k in range(M + 1):
            zeta[k] += np.vdot(v0, v_k)
            # Solve (I + U) w = v_k
            w_k, _info = gmres(np.eye(n) + U, v_k, atol=1e-8)
            if _info != 0:
                raise RuntimeError("GMRES did not converge")
            if k == 0:
                v_kp1 = 1j * (2 * w_k - v0)
            else:
                v_kp1 = 2j * (2 * w_k - v_k) - v_km1
            v_km1 = v_k
            v_k = v_kp1
    zeta /= n_vec
    mu = np.zeros(M + 1, dtype=np.complex128)
    for k in range(M + 1):
        mu[k] = (2 - (k == 0)) / (n * np.pi) * zeta[k]
    # Evaluate Chebyshev polynomials at t_points
    T = np.polynomial.chebyshev.chebvander(t_points, M)
    phi_tilde = (T @ mu).real / np.sqrt(1 - t_points**2)
    return phi_tilde


# Example usage: SVD vs QR comparison
n = 50
M = 30
n_vec = 10
num_points = 200
epsilon = 1e-3
t_points = np.linspace(-1 + epsilon, 1 - epsilon, num_points)
A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
# SVD-based unitary
U, _, _ = np.linalg.svd(A, full_matrices=False)
phi_svd = kpm_unitary(U, M, n_vec, t_points)
# QR-based unitary
Q, _ = np.linalg.qr(A)
phi_qr = kpm_unitary(Q, M, n_vec, t_points)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axes[0].plot(t_points, phi_svd, color='C0', label='SVD-based KPM')
axes[0].set_title('SVD-based unitary')
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel(r'$\tilde{\phi}_M(x)$')
axes[0].grid(True, linestyle=':', alpha=0.5)
axes[0].legend()

axes[1].plot(t_points, phi_qr, color='C1', label='QR-based KPM')
axes[1].set_title('QR-based unitary')
axes[1].set_xlabel(r'$x$')
axes[1].grid(True, linestyle=':', alpha=0.5)
axes[1].legend()

fig.suptitle('KPM Spectral Density Estimate: SVD vs QR Unitary')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('compare_svd_qr_kpm.png', dpi=300)
plt.show()