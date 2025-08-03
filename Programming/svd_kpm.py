import numpy as np
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

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
    T = np.polynomial.chebyshev.chebvander(t_points, M)
    phi_tilde = (T @ mu).real / np.sqrt(1 - t_points**2)
    return phi_tilde

# Parameters
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

plt.figure(figsize=(7, 5))
plt.plot(t_points, phi_svd, color='C0', label='SVD-based KPM')
plt.xlabel(r'$x$')
plt.ylabel(r'$\tilde{\phi}_M(x)$')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('svd_kpm.png', dpi=300)
plt.show()