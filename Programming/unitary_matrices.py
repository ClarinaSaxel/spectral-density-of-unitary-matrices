import streamlit as st
import numpy as np
import cmath

# Title of the app
st.title("Unitary Matrices")

st.write("A random matrix")

# Generate random complex matrix 
def generate_random_matrix(matrix_size):
    return np.random.randn(matrix_size, matrix_size)

# Matrix size input
matrix_size = st.slider('Matrix Size', min_value=2, max_value=10, value=5, step=1)

if st.button("Generate"):
    st.write(generate_random_matrix(matrix_size))


if __name__ == '__main__':

    print(generate_random_matrix(4))