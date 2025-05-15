import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pickle 
import sklearn

wage = pd.read_csv("wage.csv")

st.title("Analisis Promedio Universitario")

tab1, tab2, tab3 = st.tabs(["Análisis Univariado", "Análisis Bivariado", "Predicción de Promedio"])

with tab1:
    st.header("Analisis Univariado")
    fig, ax =plt.subplots(1, 4, figsize = (15,6))

    ax[0].set_title("Promedio Universidad", fontsize = 20)
    ax[1].set_title("Puntaje pruebas de Estado", fontsize = 20)
    ax[2].set_title("Género", fontsize = 20)
    ax[3].set_title("¿Vive en el campus?", fontsize = 20)

    ax[0].hist(wage["promedio"])
    ax[1].hist(wage["icfes"])
    conteo = wage["genero"].value_counts()
    conteo2 = wage["vivienda"].value_counts()
    ax[2].bar(conteo.index, conteo.values)
    ax[3].bar(conteo2.index, conteo2.values)
    fig.tight_layout()
    st.pyplot(fig)

with tab2:
    st.header("Analisis Bivariado")
    fig, ax =plt.subplots(1, 3, figsize = (15,7))


    sns.scatterplot(data=wage, x="icfes", y="promedio", ax=ax[0])
    xlabel = ax[0].set_xlabel("Puntaje pruebas de Estado")
    ylabel = ax[0].set_ylabel("Promedio Universidad")
    sns.violinplot(data=wage, x="genero", y="promedio", ax=ax[1])
    xlabel = ax[1].set_xlabel("Género")
    ylabel = ax[1].set_ylabel("Promedio Universidad")
    sns.violinplot(data=wage, x="vivienda", y="promedio", ax=ax[2])
    xlabel = ax[2].set_xlabel("¿Vive en el campus?")
    ylabel = ax[2].set_ylabel("Promedio Universidad")

    fig.tight_layout()
    st.pyplot(fig)

with open("model.pickle", "rb") as f:
    modelo = pickle.load(f)

with tab3:
    icfes = st.slider("Puntaje pruebas de Estado", 1, 36)
    genero = st.selectbox("Seleccione su género:", ["Hombre", "Mujer"])
    if genero == "Hombre":
        genero = 1
    else:
        genero = 0
    vivienda = st.selectbox("¿Vive en el campus?", ["Si", "No"])
    if vivienda == "Si":
        vivienda = 1
    else:
        vivienda = 0
    if st.button("Predecir"):
        pred = modelo.predict(np.array([[vivienda, genero, icfes]]))
        st.write(f"Su promedio sería {round(pred[0], 1)}")

