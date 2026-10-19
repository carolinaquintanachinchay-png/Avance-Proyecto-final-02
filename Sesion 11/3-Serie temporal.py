from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CARPETA = Path(__file__).resolve().parent
ARCHIVO_DATOS = CARPETA / "slump_limpio.csv"
ARCHIVO_GRAFICO = CARPETA / "serie_slump_flow.png"
ARCHIVO_SERIE = CARPETA / "serie_slump_flow.csv"

VARIABLE_ORDEN = "No."
VARIABLE_SLUMP = "SLUMP (cm)"
VARIABLE_FLOW = "FLOW (cm)"
VENTANA_MOVIL = 10


def main():
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_DATOS}. Ejecuta primero 1-Limpieza Datos.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    columnas_requeridas = [VARIABLE_ORDEN, VARIABLE_SLUMP, VARIABLE_FLOW]
    faltantes = [columna for columna in columnas_requeridas if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias en los datos: {faltantes}")

    serie = datos[columnas_requeridas].copy()
    for columna in columnas_requeridas:
        serie[columna] = pd.to_numeric(serie[columna], errors="coerce")

    serie = serie.dropna(subset=columnas_requeridas).sort_values(VARIABLE_ORDEN)
    if serie.empty:
        raise ValueError("No hay filas válidas para construir las series.")

    serie["SLUMP_media_movil"] = serie[VARIABLE_SLUMP].rolling(
        window=VENTANA_MOVIL, min_periods=1
    ).mean()
    serie["FLOW_media_movil"] = serie[VARIABLE_FLOW].rolling(
        window=VENTANA_MOVIL, min_periods=1
    ).mean()

    correlacion = serie[VARIABLE_SLUMP].corr(serie[VARIABLE_FLOW])
    print("SERIES POR NÚMERO DE MUESTRA")
    print(
        "El archivo no tiene fechas ni horas. El eje horizontal representa el número "
        "de muestra, no tiempo calendario."
    )
    print(f"Muestras analizadas: {len(serie)}")
    print(
        f"Correlación de Pearson entre {VARIABLE_SLUMP} y {VARIABLE_FLOW}: "
        f"{correlacion:.3f}"
    )
    print(
        "La correlación mide asociación entre las mediciones; no prueba causalidad "
        "ni una tendencia temporal."
    )
    print(
        f"Promedios móviles calculados con una ventana de {VENTANA_MOVIL} muestras "
        "(para visualizar variación local)."
    )

    figura, ejes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ejes[0].plot(
        serie[VARIABLE_ORDEN],
        serie[VARIABLE_SLUMP],
        marker="o",
        markersize=3,
        linewidth=1,
        alpha=0.6,
        label="Medición",
    )
    ejes[0].plot(
        serie[VARIABLE_ORDEN],
        serie["SLUMP_media_movil"],
        linewidth=2,
        label=f"Media móvil ({VENTANA_MOVIL} muestras)",
    )
    ejes[0].set_ylabel("Slump (cm)")
    ejes[0].set_title("Slump por número de muestra")
    ejes[0].legend()
    ejes[0].grid(True, alpha=0.3)

    ejes[1].plot(
        serie[VARIABLE_ORDEN],
        serie[VARIABLE_FLOW],
        marker="o",
        markersize=3,
        linewidth=1,
        alpha=0.6,
        color="tab:orange",
        label="Medición",
    )
    ejes[1].plot(
        serie[VARIABLE_ORDEN],
        serie["FLOW_media_movil"],
        linewidth=2,
        color="tab:red",
        label=f"Media móvil ({VENTANA_MOVIL} muestras)",
    )
    ejes[1].set_xlabel("Número de muestra (orden del archivo)")
    ejes[1].set_ylabel("Flow (cm)")
    ejes[1].set_title("Flow por número de muestra")
    ejes[1].legend()
    ejes[1].grid(True, alpha=0.3)

    figura.suptitle(
        f"Slump y Flow por muestra (correlación de Pearson: {correlacion:.3f})"
    )
    figura.tight_layout()
    figura.savefig(ARCHIVO_GRAFICO, dpi=150, bbox_inches="tight")

    serie.to_csv(ARCHIVO_SERIE, index=False, encoding="utf-8-sig")
    print(f"\nGráfico guardado en: {ARCHIVO_GRAFICO}")
    print(f"Series y medias móviles guardadas en: {ARCHIVO_SERIE}")
    plt.show()
    plt.close(figura)


if __name__ == "__main__":
    main()