# Tanh Activation Function Visualiser (Streamlit app)
# Run with:  streamlit run tanh_app.py
#
# What this app does:
#   1. Shows the formula of the Tanh activation function
#   2. Plots the function and its derivative (gradient)
#   3. Lets you test one neuron:  z = w.x + b  ->  output = f(z)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------
# 1. MATHS: the activation function and its derivative
# ---------------------------------------------------------------
def activation(z):
    """Tanh: squashes any input into a value between -1 and 1."""
    return np.tanh(z)


def derivative(z):
    """Slope of tanh: 1 - f(z)^2."""
    return 1 - np.tanh(z) ** 2


# ---------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------
def neuron_output(inputs, weights, bias):
    """One neuron: weighted sum plus bias (z), then the activation function."""
    z = float(np.dot(inputs, weights) + bias)
    return z, float(activation(z))


def plot_function(z_min, z_max, z_value, show_derivative=True):
    """Draw the activation function (and optionally its derivative)."""
    z = np.linspace(z_min, z_max, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, activation(z), linewidth=2, label="Tanh  f(z)")
    if show_derivative:
        ax.plot(z, derivative(z), "--", label="Derivative  f'(z)")
    ax.scatter([z_value], [activation(z_value)], color="red", zorder=5,
               label="Selected input")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z  (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.set_title("Tanh Activation Function")
    ax.grid(True)
    ax.legend()
    return fig


# ---------------------------------------------------------------
# 3. PAGE LAYOUT
# ---------------------------------------------------------------
st.set_page_config(page_title="Tanh Activation Function", layout="centered")
st.title("Hyperbolic Tangent (Tanh) Activation Function")
st.write("Tanh is similar to the sigmoid function, but it maps any input value to an output between -1 and 1.")
st.latex(r"f(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}")

# Sidebar controls
st.sidebar.header("Controls")
z_min, z_max = st.sidebar.slider("Range of z (x-axis)", -10.0, 10.0,
                                 (-5.0, 5.0), step=0.5)
z_value = st.sidebar.slider("Pick an input value z", -10.0, 10.0,
                            0.8, step=0.01)
show_derivative = st.sidebar.checkbox("Show derivative (gradient)", value=True)

if z_min >= z_max:
    st.error("The minimum of the range must be smaller than the maximum.")
    st.stop()

tab_graph, tab_neuron, tab_table, tab_notes = st.tabs(
    ["Graph", "Neuron calculator", "Table of values", "Key points"])

# ---- Tab 1: graph ----
with tab_graph:
    fig = plot_function(z_min, z_max, z_value, show_derivative)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    col1.metric("Output f(z)", f"{float(activation(z_value)):.4f}")
    col2.metric("Derivative f'(z)", f"{float(derivative(z_value)):.4f}")

# ---- Tab 2: one neuron, step by step ----
with tab_neuron:
    st.write("Change the inputs (x), weights (w) and bias (b). "
             "The neuron computes z = w1*x1 + w2*x2 + ... + b and then applies "
             "the Tanh function.")
    default_inputs = [1.0, 0.0, 1.0]
    default_weights = [0.5, -0.2, 0.3]
    inputs, weights = [], []
    cols = st.columns(len(default_inputs))
    for i, col in enumerate(cols):
        inputs.append(col.number_input(f"x{i + 1}", value=default_inputs[i],
                                       step=0.1, key=f"x{i}"))
        weights.append(col.number_input(f"w{i + 1}", value=default_weights[i],
                                        step=0.1, key=f"w{i}"))
    bias = st.number_input("Bias b", value=0.0, step=0.1)

    z_neuron, out_neuron = neuron_output(inputs, weights, bias)
    c1, c2 = st.columns(2)
    c1.metric("Weighted sum z", f"{z_neuron:.4f}")
    c2.metric("Neuron output f(z)", f"{out_neuron:.4f}")
    st.caption("Using the same inputs and weights as the lecture's H1 neuron: z = 0.8, so tanh(0.8) is about 0.664. A negative z gives a negative output, e.g. tanh(-0.1) is about -0.0997.")

# ---- Tab 3: table of values ----
with tab_table:
    samples = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    st.table({
        "z": samples,
        "f(z)": [round(float(activation(s)), 4) for s in samples],
        "f'(z)": [round(float(derivative(s)), 4) for s in samples],
    })

# ---- Tab 4: key points ----
with tab_notes:
    st.markdown(
        "- **Range:** -1 to 1.\n"
        "- **f(0) = 0**, so the output is centred around zero (sigmoid is centred at 0.5).\n"
        "- **Non-linear**, smooth S-shaped curve.\n"
        "- **Negative inputs give negative outputs**, positive inputs give positive outputs.\n"
        "- **Weakness:** like sigmoid, the curve flattens for large |z| and gradients can vanish."
    )
