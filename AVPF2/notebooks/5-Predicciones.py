from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = PROYECTO / "data"
CARPETA_IMAGENES = PROYECTO / "imagenes"
CARPETA_MODELOS = PROYECTO / "modelos"
ARCHIVO_DATOS = CARPETA_DATOS / "dataset.csv"
ARCHIVO_METRICAS = CARPETA_DATOS / "metricas_prediccion_slump.csv"
ARCHIVO_PREDICCIONES = CARPETA_DATOS / "predicciones_validacion_slump.csv"
ARCHIVO_GRAFICO = CARPETA_IMAGENES / "comparacion_predicciones_slump.png"
ARCHIVO_MODELO_LINEAL = CARPETA_MODELOS / "modelo_regresion_lineal_slump.joblib"
ARCHIVO_MODELO_POLINOMIAL = CARPETA_MODELOS / "modelo_regresion_polinomial_slump.joblib"

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


def crear_modelos():
    modelo_lineal = Pipeline(
        [
            ("escalado", StandardScaler()),
            ("regresion", LinearRegression()),
        ]
    )
    modelo_polinomial = Pipeline(
        [
            ("polinomio", PolynomialFeatures(degree=2, include_bias=False)),
            ("escalado", StandardScaler()),
            ("regresion", LinearRegression()),
        ]
    )
    return {
        "Regresión lineal": modelo_lineal,
        "Regresión polinomial (grado 2)": modelo_polinomial,
    }


def main():
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_DATOS}. Ejecuta primero notebooks/1-Limpieza Datos.py."
        )

    datos = pd.read_csv(ARCHIVO_DATOS)
    requeridas = PREDICTORES + [VARIABLE_OBJETIVO]
    faltantes = [columna for columna in requeridas if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias para entrenar: {faltantes}")

    datos = datos[requeridas].apply(pd.to_numeric, errors="coerce").dropna()
    if len(datos) < NUMERO_PLIEGUES:
        raise ValueError(
            f"Se requieren al menos {NUMERO_PLIEGUES} filas completas; se encontraron {len(datos)}."
        )

    x = datos[PREDICTORES]
    y = datos[VARIABLE_OBJETIVO]
    validacion = KFold(n_splits=NUMERO_PLIEGUES, shuffle=True, random_state=SEMILLA)
    modelos = crear_modelos()
    predicciones = pd.DataFrame({"observado": y.reset_index(drop=True)})
    filas_metricas = []

    predicciones["Referencia: media"] = np.full(len(y), y.mean())
    for nombre, modelo in modelos.items():
        estimaciones = cross_val_predict(modelo, x, y, cv=validacion)
        predicciones[nombre] = estimaciones
        filas_metricas.append(
            {
                "modelo": nombre,
                "R2_validacion": r2_score(y, estimaciones),
                "MAE": mean_absolute_error(y, estimaciones),
                "RMSE": np.sqrt(mean_squared_error(y, estimaciones)),
            }
        )

    predicciones["Referencia: media"] = y.mean()
    filas_metricas.append(
        {
            "modelo": "Referencia: media",
            "R2_validacion": r2_score(y, predicciones["Referencia: media"]),
            "MAE": mean_absolute_error(y, predicciones["Referencia: media"]),
            "RMSE": np.sqrt(mean_squared_error(y, predicciones["Referencia: media"])),
        }
    )
    metricas = pd.DataFrame(filas_metricas).sort_values("RMSE").reset_index(drop=True)

    print(f"Filas utilizadas: {len(datos)}")
    print(f"Validación: {NUMERO_PLIEGUES}-fold cruzada, aleatoria (semilla {SEMILLA})")
    print(f"Variable a predecir: {VARIABLE_OBJETIVO}")
    print(f"Predictores: {', '.join(PREDICTORES)}")
    print("\nMÉTRICAS DE VALIDACIÓN (menor MAE/RMSE es mejor; mayor R² es mejor)")
    print(metricas.round(3).to_string(index=False))

    mejor_nombre = metricas.loc[0, "modelo"]
    if mejor_nombre == "Referencia: media":
        print(
            "\nAdvertencia: los modelos probados no superaron la predicción constante "
            "de la media en RMSE; no se recomienda usarlos para predicciones confiables."
        )
    else:
        print(f"\nMenor RMSE en esta validación: {mejor_nombre}")
        print(
            "La validación cruzada es una estimación interna; se necesita más información "
            "o datos nuevos para confirmar el desempeño fuera de esta muestra."
        )

    predicciones.insert(0, "muestra", np.arange(1, len(predicciones) + 1))
    metricas.to_csv(ARCHIVO_METRICAS, index=False, encoding="utf-8-sig")
    predicciones.to_csv(ARCHIVO_PREDICCIONES, index=False, encoding="utf-8-sig")
    print("\nPREDICCIONES DE VALIDACIÓN CRUZADA POR MUESTRA (cm)")
    print(predicciones.round(2).to_string(index=False))

    figura, eje = plt.subplots(figsize=(8, 7))
    colores = {
        "Regresión lineal": "tab:blue",
        "Regresión polinomial (grado 2)": "tab:orange",
    }
    limite_min = min(y.min(), predicciones.iloc[:, 2:].min().min())
    limite_max = max(y.max(), predicciones.iloc[:, 2:].max().max())
    for nombre in modelos:
        eje.scatter(
            y,
            predicciones[nombre],
            alpha=0.75,
            label=nombre,
            color=colores[nombre],
        )
    eje.plot(
        [limite_min, limite_max],
        [limite_min, limite_max],
        "k--",
        label="Predicción perfecta",
    )
    eje.set_title("Predicciones de slump frente a valores observados")
    eje.set_xlabel("Slump observado (cm)")
    eje.set_ylabel("Slump predicho por validación cruzada (cm)")
    eje.grid(True, alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(ARCHIVO_GRAFICO, dpi=150, bbox_inches="tight")

    for nombre, modelo in modelos.items():
        modelo.fit(x, y)
    joblib.dump(modelos["Regresión lineal"], ARCHIVO_MODELO_LINEAL)
    joblib.dump(modelos["Regresión polinomial (grado 2)"], ARCHIVO_MODELO_POLINOMIAL)

    print(f"\nMétricas guardadas en: {ARCHIVO_METRICAS}")
    print(f"Predicciones de validación guardadas en: {ARCHIVO_PREDICCIONES}")
    print(f"Gráfico de comparación guardado en: {ARCHIVO_GRAFICO}")
    print(f"Modelo lineal entrenado guardado en: {ARCHIVO_MODELO_LINEAL}")
    print(f"Modelo polinomial entrenado guardado en: {ARCHIVO_MODELO_POLINOMIAL}")
    print(
        "\nLos archivos de modelo se ajustaron con todos los datos disponibles y pueden "
        "usarse para predicción exploratoria; sus métricas deben evaluarse con datos nuevos."
    )
    print("\nMostrando el gráfico comparativo de valores observados y predichos.")
    plt.show()
    plt.close(figura)


if __name__ == "__main__":
    main()