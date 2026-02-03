import streamlit as st
from src.design_checker import check_design

st.title("Directional Coupler Design Checker")

kappa = st.slider("Coupling coefficient κ", 0.01, 0.08, 0.05)
L = st.slider("Interaction length L (µm)", 5.0, 50.0, 20.0)
noise = st.slider("Fabrication variation (%)", 0.0, 20.0, 10.0) / 100

if st.button("Check Design"):
    result = check_design(kappa, L, noise=noise)

    st.write(f"Nominal split ratio: {result['nominal_split']:.3f}")
    st.write(f"Mean error: {result['mean_error']:.3f}")
    st.write(f"Variation (std): {result['std_error']:.3f}")

    if result["passes"]:
        st.success("✅ Design is ROBUST and manufacturable")
    else:
        st.error("❌ Design is NOT robust")
        for reason in result["reasons"]:
            st.write("•", reason)
