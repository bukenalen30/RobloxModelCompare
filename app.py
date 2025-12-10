import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Roblox Popularity Classifier", layout="wide")

# =========================
# LOAD MODEL & EVALUATION
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
# UI
# =========================
st.title("🎮 Roblox Game Popularity Classification Web App")
st.write("Aplikasi ini memprediksi tingkat popularitas game Roblox menggunakan model SVM dan KNN.")

st.sidebar.title("Input Data Aktivitas")
visits = st.sidebar.number_input("Visits", min_value=0)
likes = st.sidebar.number_input("Likes", min_value=0)
dislikes = st.sidebar.number_input("Dislikes", min_value=0)
favourites = st.sidebar.number_input("Favourites", min_value=0)

# =========================
# PREDICTION
# =========================
if st.sidebar.button("Prediksi"):
    x = np.array([[visits, likes, dislikes, favourites]])
    x_scaled = scaler.transform(x)

    svm_pred = svm_model.predict(x_scaled)[0]
    knn_pred = knn_model.predict(x_scaled)[0]

    label_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}

    st.subheader("🔮 Hasil Prediksi")
    col1, col2 = st.columns(2)

    with col1:
        st.success(f"**SVM:** {label_map[svm_pred]}")

    with col2:
        st.info(f"**KNN:** {label_map[knn_pred]}")

# =========================
# VISUALISASI EVALUASI  
# =========================
st.header("📊 Visualisasi Evaluasi Model")

svm_matrix = evaluation["svm_matrix"]
knn_matrix = evaluation["knn_matrix"]

# Plot confusion matrix
def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    st.pyplot(fig)

colA, colB = st.columns(2)

with colA:
    plot_matrix(svm_matrix, "Confusion Matrix - SVM")

with colB:
    plot_matrix(knn_matrix, "Confusion Matrix - KNN")

# =========================
# METRIK EVALUASI
# =========================
st.header("📈 Perbandingan Metrik Evaluasi")

svm_report = evaluation["svm_report"]
knn_report = evaluation["knn_report"]

st.subheader("SVM Classification Report")
st.code(svm_report)

st.subheader("KNN Classification Report")
st.code(knn_report)

st.write("---")
st.caption("© 2025 — Roblox Popularity ML Deployment")
