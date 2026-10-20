from pathlib import Path

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = PROYECTO / "data"
ARCHIVO_DATOS = CARPETA_DATOS / "dataset.csv"
ARCHIVO_RESUMEN = CARPETA_DATOS / "resumen_estadistico.csv"
ARCHIVO_CORRELACIONES = CARPETA_DATOS / "correlaciones_slump.csv"

VARIABLE_ID = "No."
VARIABLE_OBJETIVO = "SLUMP (cm)"
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
    columnas_requeridas = [VARIABLE_ID, VARIABLE_OBJETIVO, *INGREDIENTES]
    faltantes = [columna for columna in columnas_requeridas if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias en el CSV limpio: {faltantes}")

    columnas_numericas = datos.select_dtypes(include="number").columns.tolist()
    variables = [columna for columna in columnas_numericas if columna != VARIABLE_ID]
    if not variables:
        raise ValueError("No hay columnas numéricas disponibles para calcular estadísticas.")

    resumen = datos[variables].describe(percentiles=[0.25, 0.5, 0.75]).T
    resumen = resumen.rename(
        columns={
            "count": "cantidad",
            "mean": "media",
            "std": "desviacion_estandar",
            "min": "minimo",
            "25%": "Q1",
            "50%": "mediana",
            "75%": "Q3",
            "max": "maximo",
        }
    )
    resumen["moda"] = datos[variables].mode().iloc[0]
    resumen["rango"] = resumen["maximo"] - resumen["minimo"]
    resumen["IQR"] = resumen["Q3"] - resumen["Q1"]
    resumen["varianza_muestral"] = datos[variables].var()
    resumen["asimetria"] = datos[variables].skew()
    resumen["curtosis"] = datos[variables].kurt()
    resumen["coeficiente_variacion_pct"] = (
        resumen["desviacion_estandar"] / resumen["media"].abs() * 100
    )
    resumen.index.name = "variable"
    resumen = resumen.round(3)

    print("ESTADÍSTICAS DESCRIPTIVAS")
    print(resumen.to_string())

    estadisticas_slump = resumen.loc[VARIABLE_OBJETIVO]
    print(f"\nRESUMEN DE {VARIABLE_OBJETIVO}")
    print(f"Cantidad de mediciones: {int(estadisticas_slump['cantidad'])}")
    print(f"Media: {estadisticas_slump['media']:.2f} cm")
    print(f"Mediana: {estadisticas_slump['mediana']:.2f} cm")
    print(f"Moda: {estadisticas_slump['moda']:.2f} cm")
    print(f"Desviación estándar muestral: {estadisticas_slump['desviacion_estandar']:.2f} cm")
    print(f"Varianza muestral: {estadisticas_slump['varianza_muestral']:.2f} cm²")
    print(
        f"Rango: {estadisticas_slump['minimo']:.2f}–"
        f"{estadisticas_slump['maximo']:.2f} cm "
        f"(amplitud {estadisticas_slump['rango']:.2f} cm)"
    )
    print(
        f"Cuartiles: Q1={estadisticas_slump['Q1']:.2f} cm, "
        f"mediana={estadisticas_slump['mediana']:.2f} cm, "
        f"Q3={estadisticas_slump['Q3']:.2f} cm"
    )

    correlaciones = (
        datos[INGREDIENTES + [VARIABLE_OBJETIVO]]
        .corr()[VARIABLE_OBJETIVO]
        .drop(VARIABLE_OBJETIVO)
    )
    correlaciones = correlaciones.reindex(
        correlaciones.abs().sort_values(ascending=False).index
    ).rename("correlacion_con_slump").to_frame()
    correlaciones.index.name = "ingrediente"

    print(f"\nCORRELACIÓN DE INGREDIENTES CON {VARIABLE_OBJETIVO}")
    print(correlaciones.round(3).to_string())

    resumen.to_csv(ARCHIVO_RESUMEN, encoding="utf-8-sig")
    correlaciones.round(3).to_csv(ARCHIVO_CORRELACIONES, encoding="utf-8-sig")
    print(f"\nResumen completo guardado en: {ARCHIVO_RESUMEN}")
    print(f"Correlaciones guardadas en: {ARCHIVO_CORRELACIONES}")


if __name__ == "__main__":
    main()