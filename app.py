import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Popularity Classifier 🌸", layout="wide")

# =========================
# CUSTOM PINK BLOSSOM HEADER
# =========================
st.markdown("""
    <div style="
        background-color:#ffe6f0;
        padding:20px;
        border-radius:12px;
        margin-bottom:20px;
        border: 2px solid #ffb3d2;
    ">
        <h1 style="color:#d14a7c; text-align:center; font-size:38px;">
            🌸 Roblox Game Popularity Classifier 🌸
        </h1>
        <p style="color:#5a2a41; text-align:center;">
            Prediksi tingkat popularitas game Roblox menggunakan model Machine Learning (SVM dan KNN) dengan tema Pink Blossom.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_all():
    svm = joblib.load("svm_model.pkl")
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    evaluation = joblib.load("evaluation.pkl")
    return svm, knn, scaler, evaluation

svm_model, knn_model, scaler, evaluation = load_all()

# =========================
# SIDEBAR PINK STYLE
# =========================
st.sidebar.markdown("""
    <div style="
        background-color:#ffe6f0;
        padding:15px;
        border-radius:10px;
        border:2px solid #ffb3d2;
    ">
        <h2 style="color:#d14a7c; text-align:center;">🌺 Input Data</h2>
        <p style="color:#5a2a41;">Masukkan data aktivitas pengguna game Roblox.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# INPUT FIELDS
visits = st.sidebar.number_input("Jumlah Visits", min_value=0)
likes = st.sidebar.number_input("Jumlah Likes", min_value=0)
dislikes = st.sidebar.number_input("Jumlah Dislikes", min_value=0)
favourites = st.sidebar.number_input("Jumlah Favourites", min_value=0)

# =========================
# PREDICTION
# =========================
if st.sidebar.button("🌸 Prediksi"):
    x = np.array([[visits, likes, dislikes, favourites]])
    x_scaled = scaler.transform(x)

    svm_pred = svm_model.predict(x_scaled)[0]
    knn_pred = knn_model.predict(x_scaled)[0]

    label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

    st.markdown("""
        <h2 style="color:#d14a7c; margin-top:20px;">🔮 Hasil Prediksi</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div style="background-color:#ffe6f0; padding:20px; border-radius:10px; border:2px solid #ffb3d2;">
                <h3 style="color:#d14a7c;">SVM</h3>
                <p style="color:#5a2a41; font-size:20px;"><b>{label_map[svm_pred]}</b></p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color:#fff0f6; padding:20px; border-radius:10px; border:2px solid #d14a7c;">
                <h3 style="color:#d14a7c;">KNN</h3>
                <p style="color:#5a2a41; font-size:20px;"><b>{label_map[knn_pred]}</b></p>
            </div>
        """, unsafe_allow_html=True)

# =========================
# VISUALISASI EVALUASI
# =========================
st.markdown("""
    <h2 style="color:#d14a7c; margin-top:40px;">📊 Visualisasi Evaluasi Model</h2>
""", unsafe_allow_html=True)

svm_matrix = evaluation["svm_matrix"]
knn_matrix = evaluation["knn_matrix"]

def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="pink")
    ax.set_title(title, color="#d14a7c")
    ax.set_xlabel("Predicted", color="#5a2a41")
    ax.set_ylabel("Actual", color="#5a2a41")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="#5a2a41")

    return fig

colA, colB = st.columns(2)

with colA:
    st.pyplot(plot_matrix(svm_matrix, "Confusion Matrix - SVM"))

with colB:
    st.pyplot(plot_matrix(knn_matrix, "Confusion Matrix - KNN"))

# =========================
# METRIK EVALUASI
# =========================
st.markdown("""
    <h2 style="color:#d14a7c; margin-top:40px;">📈 Perbandingan Metrik Evaluasi</h2>
""", unsafe_allow_html=True)

svm_report = evaluation["svm_report"]
knn_report = evaluation["knn_report"]

st.subheader("🌺 SVM Classification Report")
st.code(svm_report)

st.subheader("🌺 KNN Classification Report")
st.code(knn_report)

st.write("---")
st.caption("🌸 © 2025 — Roblox Popularity ML Deployment | Pink Blossom Theme 🌸")
