from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CARPETA = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA / "slump_limpio.csv"
ARCHIVO_GRAFICO = CARPETA / "grafico_slump_vs_agua.png"
ARCHIVO_CORRELACION = CARPETA / "grafico_correlacion_slump.png"
ARCHIVO_HISTOGRAMA = CARPETA / "histograma_slump.png"
ARCHIVO_3D = CARPETA / "grafico_3d_slump.png"

VARIABLE_X = "Water"
VARIABLE_Y = "SLUMP (cm)"
INGREDIENTES = [
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "SP",
    "Coarse Aggr.",
    "Fine Aggr.",
]


def main():
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_DATOS}. Ejecuta primero 1-Limpieza Datos.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    columnas_graficos = INGREDIENTES + [VARIABLE_Y]
    faltantes = [columna for columna in columnas_graficos if columna not in datos]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias para el gráfico: {faltantes}")

    for columna in columnas_graficos:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos = datos.dropna(subset=columnas_graficos)
    if len(datos) < 2:
        raise ValueError("Se necesitan al menos dos filas válidas para hacer el gráfico.")

    x = datos[VARIABLE_X].to_numpy()
    y = datos[VARIABLE_Y].to_numpy()
    correlacion = float(np.corrcoef(x, y)[0, 1])
    pendiente, intercepto = np.polyfit(x, y, 1)
    x_tendencia = np.linspace(x.min(), x.max(), 100)
    y_tendencia = pendiente * x_tendencia + intercepto

    figura, eje = plt.subplots(figsize=(9, 6))
    eje.scatter(x, y, alpha=0.75, edgecolor="white", linewidth=0.5, label="Muestras")
    eje.plot(
        x_tendencia,
        y_tendencia,
        color="tab:red",
        linewidth=2,
        label="Tendencia lineal",
    )
    eje.axhline(
        y.mean(),
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label=f"Slump promedio: {y.mean():.2f} cm",
    )
    eje.set_title("Relación entre la cantidad de agua y el asentamiento")
    eje.set_xlabel("Water")
    eje.set_ylabel("Slump (cm)")
    eje.text(
        0.03,
        0.97,
        f"Correlación de Pearson: {correlacion:.3f}",
        transform=eje.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(ARCHIVO_GRAFICO, dpi=150, bbox_inches="tight")

    print(f"Gráfico guardado en: {ARCHIVO_GRAFICO}")
    print(f"Correlación entre Water y {VARIABLE_Y}: {correlacion:.3f}")
    print("La línea muestra una tendencia lineal descriptiva; no demuestra causalidad.")

    matriz = datos[INGREDIENTES + [VARIABLE_Y]].corr()
    figura_correlacion, eje_correlacion = plt.subplots(figsize=(10, 8))
    imagen = eje_correlacion.imshow(matriz, cmap="coolwarm", vmin=-1, vmax=1)
    figura_correlacion.colorbar(imagen, ax=eje_correlacion, label="Correlación de Pearson")
    etiquetas = INGREDIENTES + [VARIABLE_Y]
    eje_correlacion.set_xticks(range(len(etiquetas)), labels=etiquetas, rotation=45, ha="right")
    eje_correlacion.set_yticks(range(len(etiquetas)), labels=etiquetas)
    eje_correlacion.set_title("Correlación entre ingredientes y asentamiento")
    for fila in range(len(etiquetas)):
        for columna in range(len(etiquetas)):
            eje_correlacion.text(
                columna,
                fila,
                f"{matriz.iloc[fila, columna]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    figura_correlacion.tight_layout()
    figura_correlacion.savefig(ARCHIVO_CORRELACION, dpi=150, bbox_inches="tight")

    figura_histograma, eje_histograma = plt.subplots(figsize=(9, 6))
    eje_histograma.hist(
        datos[VARIABLE_Y],
        bins="auto",
        color="steelblue",
        edgecolor="white",
        alpha=0.85,
    )
    eje_histograma.axvline(
        datos[VARIABLE_Y].mean(),
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"Media: {datos[VARIABLE_Y].mean():.2f} cm",
    )
    eje_histograma.axvline(
        datos[VARIABLE_Y].median(),
        color="tab:green",
        linestyle=":",
        linewidth=2,
        label=f"Mediana: {datos[VARIABLE_Y].median():.2f} cm",
    )
    eje_histograma.set_title("Distribución del asentamiento")
    eje_histograma.set_xlabel("Slump (cm)")
    eje_histograma.set_ylabel("Frecuencia")
    eje_histograma.grid(axis="y", alpha=0.3)
    eje_histograma.legend()
    figura_histograma.tight_layout()
    figura_histograma.savefig(ARCHIVO_HISTOGRAMA, dpi=150, bbox_inches="tight")

    figura_3d = plt.figure(figsize=(10, 7))
    eje_3d = figura_3d.add_subplot(111, projection="3d")
    puntos_3d = eje_3d.scatter(
        datos["Water"],
        datos["Slag"],
        datos[VARIABLE_Y],
        c=datos["SP"],
        cmap="viridis",
        s=55,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    figura_3d.colorbar(puntos_3d, ax=eje_3d, pad=0.12, label="SP")
    eje_3d.set_title("Asentamiento según agua y escoria")
    eje_3d.set_xlabel("Water")
    eje_3d.set_ylabel("Slag")
    eje_3d.set_zlabel("Slump (cm)")
    figura_3d.tight_layout()
    figura_3d.savefig(ARCHIVO_3D, dpi=150, bbox_inches="tight")

    print(f"Matriz de correlación guardada en: {ARCHIVO_CORRELACION}")
    print(f"Histograma guardado en: {ARCHIVO_HISTOGRAMA}")
    print(f"Gráfico 3D guardado en: {ARCHIVO_3D}")
    print(
        "Para futuras predicciones, usa los ingredientes como variables de entrada; "
        "estos gráficos son exploratorios y no sustituyen la validación de un modelo."
    )
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()