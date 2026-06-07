import streamlit as st

def apply_style():
    st.markdown("""
    <style>
    /* Glassmorphism & Dark Theme */
    .stApp { background: #0b0e14; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Space Mono', sans-serif; }
    .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)