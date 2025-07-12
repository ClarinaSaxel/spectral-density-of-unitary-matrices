import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cmath

# Title of the app
st.title("The Cayley Transform")

st.latex(r"\varphi(z) = (i - z)(i + z)^{-1}")

# Calculate phi of [-1, 1]
def varphi(z):
    return (complex(0, 1) - z) / (complex(0, 1) + z)

# Plot the Cayley transform
xpoints = np.array([-1, 1])
ypoints = np.array([varphi(x) for x in xpoints])

if st.button("Calculate Cayley Transform"):
    plt.figure()
    plt.plot(xpoints, ypoints)
    st.pyplot(plt)