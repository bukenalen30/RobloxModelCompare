import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Popularity Classifier 🌸", layout="wide")

# ==============================================
st.markdown("""
<div style="background-color:#ffe6f0; padding:20px; border-radius:12px; border:2px solid #ffb3d2; margin-bottom:20px;">
    <h1 style="color:#d14a7c; text-align:center;">🌸 Roblox Game Popularity Classifier 🌸</h1>
    <p style="color:#5a2a41; text-align:center;">Prediksi tingkat popularitas game Roblox menggunakan model SVM & KNN.</p>
</div>
""", unsafe_allow_html=True)

# ==============================================
@st.cache_resource
def load_all():
    svm = joblib.load("svm_model.pkl")
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    evaluation = joblib.load("evaluation.pkl")
    return svm, knn, scaler, evaluation

svm_model, knn_model, scaler, evaluation = load_all()

# Pastikan model bukan numpy array
if isinstance(svm_model, np.ndarray):
    st.error("❌ ERROR: svm_model is numpy.ndarray — periksa file svm_model.pkl di Colab.")
if isinstance(knn_model, np.ndarray):
    st.error("❌ ERROR: knn_model is numpy.ndarray — periksa file knn_model.pkl di Colab.")()

# ==============================================
# 7 fitur (Wajib karena scaler expect 7 fitur)
# ==============================================
feature_cols = [
    "Active", "Visits", "Favourites", "Likes", "Dislikes",
    "like_ratio", "fav_per_visit"
]

st.sidebar.write("Scaler expects:", getattr(scaler, "n_features_in_", "?"))
st.sidebar.write("Feature names:", getattr(scaler, "feature_names_in_", "no names"))

# ==============================================
st.sidebar.markdown("""
<div style="background-color:#ffe6f0; padding:15px; border-radius:10px; border:2px solid #ffb3d2;">
    <h3 style="color:#d14a7c; text-align:center;">🌺 Input Data</h3>
</div>
""", unsafe_allow_html=True)

active = st.sidebar.number_input("Active", min_value=0)
visits = st.sidebar.number_input("Visits", min_value=0)
favourites = st.sidebar.number_input("Favourites", min_value=0)
likes = st.sidebar.number_input("Likes", min_value=0)
dislikes = st.sidebar.number_input("Dislikes", min_value=0)

# fitur turunan
like_ratio = likes / (likes + dislikes) if (likes + dislikes) > 0 else 0
fav_per_visit = favourites / visits if visits > 0 else 0

# ==============================================
if st.sidebar.button("🌸 Prediksi"):

    x_df = pd.DataFrame([[
        active, visits, favourites, likes, dislikes,
        like_ratio, fav_per_visit
    ]], columns=feature_cols)

    st.write("### 🔍 Input DataFrame:")
    st.dataframe(x_df)

    x_scaled = scaler.transform(x_df)

    svm_pred = svm_model.predict(x_scaled)[0]
    knn_pred = knn_model.predict(x_scaled)[0]

    label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

    st.subheader("🔮 Hasil Prediksi")
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**SVM:** {label_map[svm_pred]}")

    with col2:
        st.info(f"**KNN:** {label_map[knn_pred]}")

# ==============================================
st.header("📊 Visualisasi Evaluasi Model")

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
    st.pyplot(fig)

colA, colB = st.columns(2)
with colA:
    plot_matrix(svm_matrix, "Confusion Matrix - SVM")
with colB:
    plot_matrix(knn_matrix, "Confusion Matrix - KNN")

# ==============================================
st.header("📈 Perbandingan Metrik Evaluasi")

st.subheader("SVM Classification Report")
st.code(evaluation["svm_report"])

st.subheader("KNN Classification Report")
st.code(evaluation["knn_report"])

st.write("---")
st.caption("🌸 © 2025 — Roblox Popularity ML Deployment | Pink Blossom Theme 🌸")
