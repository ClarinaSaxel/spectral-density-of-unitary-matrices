import streamlit as st
import numpy as np
import pandas as pd

# Title of the app
st.title("The Cayley Transform")

st.latex(r"\varphi(z) = (i - z)(i + z)^{-1}")

st.subheader("Application for Unitary Matrices")

# Generate a random complex matrix 
def generate_random_complex_matrix(matrix_size):
    return np.random.randn(matrix_size, matrix_size) + 1j * np.random.randn(matrix_size, matrix_size)

# Generate a unitary matrix using QR decomposition
def make_unitary(matrix):
    # QR decomposition
    Q, R = np.linalg.qr(matrix)
    return Q

# Convert matrix to LaTeX format
def matrix_to_latex(matrix, decimals=3):
    latex_str = "\\begin{bmatrix}\n"
    for row in matrix:
        row_str = " & ".join([str(np.round(x, decimals)) for x in row])
        latex_str += row_str + " \\\\\n"
    latex_str += "\\end{bmatrix}"
    return latex_str

# Matrix size input
matrix_size = st.slider('Matrix Size', min_value=2, max_value=20, value=5, step=1)

# Use session state to store the generated matrix
if 'A' not in st.session_state:
    st.session_state['A'] = None
    st.session_state['U'] = None
    st.session_state['U_eigenvalues'] = None
    st.session_state['U_norms'] = None
    st.session_state['H'] = None
    st.session_state['H_eigenvalues'] = None

if st.button("Generate"):
    st.session_state['A'] = np.round(generate_random_complex_matrix(matrix_size), 3)
    # Reset when new random matrix is generated
    st.session_state['U'] = None  
    st.session_state['U_eigenvalues'] = None
    st.session_state['U_norms'] = None
    st.session_state['H'] = None
    st.session_state['H_eigenvalues'] = None

if st.session_state['A'] is not None:
    with st.expander("Random complex matrix"):
        st.latex(matrix_to_latex(st.session_state['A']))
        if st.button("Make unitary"):
            st.session_state['U'] = make_unitary(st.session_state['A'])

if st.session_state['U'] is not None:
    with st.expander("Unitary matrix"):
        st.latex(matrix_to_latex(st.session_state['U']))
        if st.button("Calculate eigenvalues of U"):
            st.session_state["U_eigenvalues"] = np.linalg.eigvals(st.session_state['U'])
            st.session_state["U_norms"] = np.abs(st.session_state["U_eigenvalues"])
        if st.session_state["U_eigenvalues"] is not None and st.session_state["U_norms"] is not None:
            with st.expander("Eigenvalues of U and their norms"):
                df = pd.DataFrame({
                    "Eigenvalue": [str(np.round(ev, 3)) for ev in st.session_state["U_eigenvalues"]],
                    "Norm": np.round(st.session_state["U_norms"], 3)
                })
                st.table(df)
        if st.button("Transform to Hermitian (Cayley)"):
            # Cayley transform: H = i (I - U)(I + U)^{-1}
            U = st.session_state['U']
            I = np.eye(U.shape[0])
            cayley_H = 1j * (I - U) @ np.linalg.inv(I + U)
            st.session_state['H'] = cayley_H

if 'H' in st.session_state and st.session_state['H'] is not None:
    with st.expander("Hermitian matrix (via Cayley transform)"):
        st.latex(matrix_to_latex(st.session_state['H']))
        if st.button("Calculate eigenvalues of H"):
            st.session_state["H_eigenvalues"] = np.linalg.eigvals(st.session_state['H'])
            with st.expander("Eigenvalues of H"):
                df = pd.DataFrame({
                    "Eigenvalue": [str(np.round(ev, 3)) for ev in st.session_state["H_eigenvalues"]]
                })
                st.table(df)