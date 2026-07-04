import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("WineQT.csv")

# =====================================================
# HEADER
# =====================================================

st.set_page_config(
    page_title="Wine Quality Analysis",
    layout="wide"
)

st.title("🍷 Wine Quality Analysis")
st.subheader("Proyek Akhir Machine Learning")

st.write("""
Aplikasi ini digunakan untuk melakukan analisis karakteristik wine
berdasarkan dataset WineQT dan simulasi hasil Q-Learning.
""")

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Dashboard",
        "Dataset",
        "Visualisasi",
        "Prediksi State"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.header("Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data",
        len(df)
    )

    col2.metric(
        "Jumlah Kolom",
        len(df.columns)
    )

    col3.metric(
        "Missing Value",
        int(df.isnull().sum().sum())
    )

    st.write(df.head())

# =====================================================
# DATASET
# =====================================================

elif menu == "Dataset":

    st.header("Dataset WineQT")

    st.dataframe(df)

# =====================================================
# VISUALISASI
# =====================================================

elif menu == "Visualisasi":

    st.header("Visualisasi Data")

    fitur = st.selectbox(
        "Pilih Fitur",
        df.columns
    )

    fig, ax = plt.subplots()

    ax.hist(df[fitur], bins=20)

    ax.set_title(f"Distribusi {fitur}")

    st.pyplot(fig)

# =====================================================
# PREDIKSI STATE
# =====================================================

elif menu == "Prediksi State":

    st.header("Simulasi Karakteristik Wine")

    alcohol = st.slider(
        "Alcohol",
        8.0,
        15.0,
        10.0
    )

    ph = st.slider(
        "pH",
        2.5,
        4.5,
        3.2
    )

    sulphates = st.slider(
        "Sulphates",
        0.3,
        2.5,
        0.6
    )

    input_data = pd.DataFrame(
        {
            "alcohol":[alcohol],
            "pH":[ph],
            "sulphates":[sulphates]
        }
    )

    scaler = MinMaxScaler()

    scaler.fit(
        df[
            [
                "alcohol",
                "pH",
                "sulphates"
            ]
        ]
    )

    scaled = scaler.transform(input_data)

    alcohol_scaled = scaled[0][0]

    state = int(alcohol_scaled * 5)

    if state > 4:
        state = 4

    st.success(
        f"State Wine = {state}"
    )

    rekomendasi = {
        0:"Decrease",
        1:"Maintain",
        2:"Maintain",
        3:"Increase",
        4:"Increase"
    }

    st.info(
        f"Rekomendasi Action : {rekomendasi[state]}"
    )