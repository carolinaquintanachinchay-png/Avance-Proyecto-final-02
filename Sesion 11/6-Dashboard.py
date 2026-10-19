from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


CARPETA = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA / "slump_limpio.csv"
VARIABLE_OBJETIVO = "SLUMP (cm)"
PREDICTORES = [
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "SP",
    "Coarse Aggr.",
    "Fine Aggr.",
]
NUMERO_PLIEGUES = 5
SEMILLA = 42


st.set_page_config(page_title="Dashboard de asentamiento", page_icon="📊", layout="wide")


@st.cache_data
def cargar_datos():
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_DATOS}. Ejecuta primero 1-Limpieza Datos.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    requeridas = PREDICTORES + [VARIABLE_OBJETIVO]
    faltantes = [columna for columna in requeridas if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias: {faltantes}")

    for columna in requeridas:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos = datos.dropna(subset=requeridas).copy()
    if datos.empty:
        raise ValueError("El CSV no contiene filas numéricas completas para analizar.")
    return datos


@st.cache_data
def calcular_validacion(datos):
    x = datos[PREDICTORES]
    y = datos[VARIABLE_OBJETIVO]
    if len(datos) < NUMERO_PLIEGUES:
        raise ValueError(
            f"Se necesitan al menos {NUMERO_PLIEGUES} filas; se encontraron {len(datos)}."
        )

    modelos = {
        "Regresión lineal": Pipeline(
            [
                ("escalado", StandardScaler()),
                ("regresion", LinearRegression()),
            ]
        ),
        "Regresión polinomial (grado 2)": Pipeline(
            [
                ("polinomio", PolynomialFeatures(degree=2, include_bias=False)),
                ("escalado", StandardScaler()),
                ("regresion", LinearRegression()),
            ]
        ),
    }
    cv = KFold(n_splits=NUMERO_PLIEGUES, shuffle=True, random_state=SEMILLA)
    predicciones = pd.DataFrame({"observado": y.reset_index(drop=True)})
    metricas = []

    for nombre, modelo in modelos.items():
        valores = cross_val_predict(modelo, x, y, cv=cv)
        predicciones[nombre] = valores
        metricas.append(
            {
                "Modelo": nombre,
                "R²": r2_score(y, valores),
                "MAE (cm)": mean_absolute_error(y, valores),
                "RMSE (cm)": np.sqrt(mean_squared_error(y, valores)),
            }
        )
        modelo.fit(x, y)

    return pd.DataFrame(metricas).sort_values("RMSE (cm)"), predicciones, modelos


def dibujar_histograma(datos):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        datos[VARIABLE_OBJETIVO],
        bins="auto",
        color="#2563eb",
        edgecolor="white",
        alpha=0.9,
    )
    ax.axvline(
        datos[VARIABLE_OBJETIVO].mean(),
        color="#dc2626",
        linestyle="--",
        label=f"Media: {datos[VARIABLE_OBJETIVO].mean():.2f} cm",
    )
    ax.axvline(
        datos[VARIABLE_OBJETIVO].median(),
        color="#16a34a",
        linestyle=":",
        label=f"Mediana: {datos[VARIABLE_OBJETIVO].median():.2f} cm",
    )
    ax.set(title="Distribución del asentamiento", xlabel="Slump (cm)", ylabel="Frecuencia")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def dibujar_correlacion(datos):
    columnas = PREDICTORES + [VARIABLE_OBJETIVO]
    matriz = datos[columnas].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    imagen = ax.imshow(matriz, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(imagen, ax=ax, label="Correlación de Pearson")
    ax.set_xticks(range(len(columnas)), labels=columnas, rotation=45, ha="right")
    ax.set_yticks(range(len(columnas)), labels=columnas)
    ax.set_title("Correlación entre ingredientes y asentamiento")
    for fila in range(len(columnas)):
        for columna in range(len(columnas)):
            ax.text(
                columna,
                fila,
                f"{matriz.iloc[fila, columna]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.tight_layout()
    return fig, matriz


def main():
    st.title("Dashboard de análisis del asentamiento (Slump)")
    st.caption(
        "Exploración de la mezcla, estadística descriptiva, relaciones y predicción "
        "experimental del asentamiento."
    )

    try:
        datos = cargar_datos()
        metricas, predicciones, modelos = calcular_validacion(datos)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    promedio = datos[VARIABLE_OBJETIVO].mean()
    mediana = datos[VARIABLE_OBJETIVO].median()
    desviacion = datos[VARIABLE_OBJETIVO].std()
    correlacion_agua = datos["Water"].corr(datos[VARIABLE_OBJETIVO])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Muestras", f"{len(datos)}")
    kpi2.metric("Slump promedio", f"{promedio:.2f} cm")
    kpi3.metric("Mediana de slump", f"{mediana:.2f} cm")
    kpi4.metric("Correlación agua / slump", f"{correlacion_agua:.3f}")

    st.divider()
    tab_resumen, tab_graficos, tab_modelos, tab_prediccion, tab_datos = st.tabs(
        ["Resumen", "Gráficos", "Modelos", "Predecir", "Datos"]
    )

    with tab_resumen:
        st.subheader("Estadística del asentamiento")
        estadisticas = datos[VARIABLE_OBJETIVO].describe()
        resumen = pd.DataFrame(
            {
                "Estadístico": [
                    "Cantidad",
                    "Media",
                    "Desviación estándar",
                    "Mínimo",
                    "Primer cuartil (Q1)",
                    "Mediana",
                    "Tercer cuartil (Q3)",
                    "Máximo",
                ],
                "Valor": [
                    int(estadisticas["count"]),
                    estadisticas["mean"],
                    estadisticas["std"],
                    estadisticas["min"],
                    estadisticas["25%"],
                    estadisticas["50%"],
                    estadisticas["75%"],
                    estadisticas["max"],
                ],
            }
        )
        st.dataframe(resumen, hide_index=True, width="stretch")
        st.write(f"Desviación estándar: **{desviacion:.2f} cm**.")
        st.info(
            "El archivo no tiene fechas. El número de muestra representa el orden de "
            "los registros, no una serie temporal de calendario."
        )

    with tab_graficos:
        st.subheader("Distribución y relaciones")
        columna_hist, columna_agua = st.columns(2)
        with columna_hist:
            st.pyplot(dibujar_histograma(datos), clear_figure=True)
        with columna_agua:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(
                datos["Water"],
                datos[VARIABLE_OBJETIVO],
                color="#2563eb",
                alpha=0.75,
                edgecolor="white",
            )
            pendiente, intercepto = np.polyfit(datos["Water"], datos[VARIABLE_OBJETIVO], 1)
            x_linea = np.linspace(datos["Water"].min(), datos["Water"].max(), 100)
            ax.plot(x_linea, pendiente * x_linea + intercepto, color="#dc2626")
            ax.set(
                title=f"Agua y slump (r = {correlacion_agua:.3f})",
                xlabel="Water",
                ylabel="Slump (cm)",
            )
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        fig, matriz = dibujar_correlacion(datos)
        st.pyplot(fig, clear_figure=True)
        correlaciones = (
            matriz[VARIABLE_OBJETIVO]
            .drop(VARIABLE_OBJETIVO)
            .sort_values(key=lambda valores: valores.abs(), ascending=False)
            .rename("Correlación con slump")
            .to_frame()
        )
        st.caption("Correlación describe asociación; no demuestra causa y efecto.")
        st.dataframe(correlaciones.round(3), width="stretch")

        fig3d = plt.figure(figsize=(9, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        puntos = ax3d.scatter(
            datos["Water"],
            datos["Slag"],
            datos[VARIABLE_OBJETIVO],
            c=datos["SP"],
            cmap="viridis",
            s=45,
            alpha=0.85,
        )
        fig3d.colorbar(puntos, ax=ax3d, pad=0.12, label="SP")
        ax3d.set(
            title="Slump según agua y escoria (color = SP)",
            xlabel="Water",
            ylabel="Slag",
            zlabel="Slump (cm)",
        )
        fig3d.tight_layout()
        st.pyplot(fig3d, clear_figure=True)

        if "FLOW (cm)" in datos.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(datos["No."] if "No." in datos else np.arange(1, len(datos) + 1),
                    datos[VARIABLE_OBJETIVO], marker="o", markersize=3, label="Slump")
            ax.plot(datos["No."] if "No." in datos else np.arange(1, len(datos) + 1),
                    datos["FLOW (cm)"], marker="o", markersize=3, label="Flow")
            ax.set(
                title="Slump y Flow por número de muestra (no es tiempo calendario)",
                xlabel="Número de muestra",
                ylabel="Medición (cm)",
            )
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    with tab_modelos:
        st.subheader("Comparación de modelos")
        st.caption(
            f"Validación cruzada de {NUMERO_PLIEGUES} partes; solo ingredientes como "
            "predictores. MAE/RMSE más bajos y R² más alto son preferibles."
        )
        st.dataframe(metricas.round(3), hide_index=True, width="stretch")
        mejor = metricas.iloc[0]
        st.write(f"Menor RMSE: **{mejor['Modelo']}**, {mejor['RMSE (cm)']:.2f} cm.")
        st.warning(
            "Los resultados se basan en 103 muestras y son exploratorios. La validación "
            "con datos nuevos es necesaria antes de usar el modelo para decisiones reales."
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        limite_min = min(predicciones["observado"].min(), predicciones.iloc[:, 1:].min().min())
        limite_max = max(predicciones["observado"].max(), predicciones.iloc[:, 1:].max().max())
        for nombre in modelos:
            ax.scatter(
                predicciones["observado"],
                predicciones[nombre],
                alpha=0.7,
                label=nombre,
            )
        ax.plot([limite_min, limite_max], [limite_min, limite_max], "k--", label="Ideal")
        ax.set(
            title="Predicción frente a valor observado",
            xlabel="Slump observado (cm)",
            ylabel="Slump predicho en validación (cm)",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with tab_prediccion:
        st.subheader("Estimar slump para una mezcla")
        st.caption("Ingresa ingredientes dentro de los rangos observados en el dataset.")
        nombre_modelo = st.selectbox("Modelo", list(modelos))
        with st.form("formulario_prediccion"):
            valores = {}
            columnas = st.columns(2)
            for indice, ingrediente in enumerate(PREDICTORES):
                minimo = float(datos[ingrediente].min())
                maximo = float(datos[ingrediente].max())
                paso = max((maximo - minimo) / 100, 0.01)
                valores[ingrediente] = columnas[indice % 2].number_input(
                    ingrediente,
                    min_value=minimo,
                    max_value=maximo,
                    value=float(datos[ingrediente].median()),
                    step=paso,
                    format="%.2f",
                )
            enviado = st.form_submit_button("Calcular estimación")

        if enviado:
            entrada = pd.DataFrame([valores], columns=PREDICTORES)
            estimacion = modelos[nombre_modelo].predict(entrada)[0]
            st.success(f"Asentamiento estimado: **{estimacion:.2f} cm**")
            st.caption(
                "Estimación del modelo dentro del rango de datos observado; no es "
                "garantía del resultado de una mezcla física."
            )

    with tab_datos:
        st.subheader("Dataset limpio usado en el análisis")
        st.write(f"Filas: {len(datos)} | Columnas: {len(datos.columns)}")
        st.dataframe(datos, width="stretch")
        st.download_button(
            "Descargar datos limpios (CSV)",
            data=datos.to_csv(index=False).encode("utf-8-sig"),
            file_name="slump_limpio.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()