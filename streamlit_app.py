import pandas as pd
import streamlit as st
from joblib import load
from sklearn.datasets import load_iris


@st.cache_resource
def load_scaler():
    return load("scaler.joblib")


@st.cache_resource
def load_model():
    return load("model.joblib")


@st.cache_resource
def load_dataset():
    iris = load_iris(as_frame=True)
    return iris.data, iris.target, iris.target_names


iris_df, target, target_names = load_dataset()
complete_dataframe = pd.concat((iris_df, target.map(lambda i: target_names[i])), axis=1)

with st.sidebar:
    st.header("Iris App")
    st.write("Aplicación para estimar la especie de una flor de Iris")

    st.markdown("[Estadísticas](#estad-sticas-del-dataset)")
    st.markdown("[Visualización](#visualizaci-n)")
    st.markdown("[Estimación](#estimaci-n-de-especie-de-iris)")

    st.write("Código en [GitHub](https://github.com/pablonoya/iris-app/)")

st.subheader("Estadísticas del dataset")
st.dataframe(iris_df.describe())

st.subheader("Visualización")

selection = st.multiselect(
    "Elige 2 columnas del dataset",
    iris_df.columns,
    default=list(iris_df.columns[:2]),
    max_selections=2,
)


if len(selection) == 2:
    st.scatter_chart(
        complete_dataframe,
        x=selection[0],
        y=selection[1],
        color="target",
    )

st.header("Estimación de especie de Iris")

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.subheader("Parámetros de la flor")

    sepal_length = st.slider("Longitud del sépalo (cm.)", 4.0, 8.0, 5.1, step=0.1)
    sepal_width = st.slider("Ancho del sépalo (cm.)", 2.0, 5.0, 3.5, step=0.1)
    petal_length = st.slider("Longitud del pétalo (cm.)", 1.0, 7.0, 1.4, step=0.1)
    petal_width = st.slider("Ancho del pétalo (cm.)", 0.1, 3.0, 0.2, step=0.1)

with col2:
    st.subheader("Especie estimada")

    model = load_model()
    scaler = load_scaler()

    class_names = ["Setosa", "Versicolor", "Virginica"]
    images = ["setosa.jpg", "versicolor.jpg", "virginica.jpg"]

    inputs = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(inputs)[0]

    st.image("img/" + images[prediction], caption=class_names[prediction])
