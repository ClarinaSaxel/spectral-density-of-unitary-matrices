import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

n = 10  # Du kan sette n til ønsket verdi

plt.ion()  # Interaktiv modus for kontinuerlig plotting
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

for _ in range(100):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    
    # U1: orthogonalization via SVD (similar to MATLAB's orth)
    U1, _, _ = np.linalg.svd(A, full_matrices=False)
    ax1.plot(np.real(np.linalg.eigvals(U1)), np.imag(np.linalg.eigvals(U1)), '+')
    
    # U2: QR decomposition
    Q, _ = qr(A)
    ax2.plot(np.real(np.linalg.eigvals(Q)), np.imag(np.linalg.eigvals(Q)), '+')

# Gi figurene titler og aksenavn hvis ønsket
ax1.set_title('Eigenvalues of U1 (from orth)')
ax2.set_title('Eigenvalues of U2 (from QR)')
plt.show()
input("Press Enter to exit...")