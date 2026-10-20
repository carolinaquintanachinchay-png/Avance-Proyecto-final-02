from pathlib import Path

import pandas as pd


PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = PROYECTO / "data"
ARCHIVO = CARPETA_DATOS / "dataset.xlsx"
ARCHIVO_LIMPIO = CARPETA_DATOS / "dataset.csv"

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
OTRAS_MEDICIONES = ["FLOW (cm)", "Compressive Strength (MPa)"]
COLUMNAS_REQUERIDAS = [
    VARIABLE_ID,
    *INGREDIENTES,
    VARIABLE_OBJETIVO,
    *OTRAS_MEDICIONES,
]


def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ARCHIVO}")

    datos = pd.read_excel(ARCHIVO)
    datos.columns = datos.columns.str.strip()

    faltantes = [columna for columna in COLUMNAS_REQUERIDAS if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el Excel: {faltantes}")

    datos = datos[COLUMNAS_REQUERIDAS].copy()
    filas_originales = len(datos)
    for columna in COLUMNAS_REQUERIDAS:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

    faltantes_iniciales = datos.isna().sum()
    invalidos = int(datos.isna().any(axis=1).sum())
    duplicados = int(datos.duplicated().sum())
    datos = datos.drop_duplicates().dropna(subset=COLUMNAS_REQUERIDAS).copy()

    if datos.empty:
        raise ValueError("No quedaron filas válidas después de revisar los datos.")

    print(f"Filas originales: {filas_originales}")
    print("\nValores vacíos/no numéricos por columna antes de limpiar:")
    print(faltantes_iniciales.to_string())
    print(f"Filas eliminadas por valores vacíos/no numéricos: {invalidos}")
    print(f"Filas duplicadas eliminadas: {duplicados}")
    print(f"Filas para análisis: {len(datos)}")
    print("\nValores faltantes por columna:")
    print(datos.isna().sum().to_string())

    print(f"\nResumen de {VARIABLE_OBJETIVO}:")
    print(datos[VARIABLE_OBJETIVO].describe().round(2).to_string())

    print("\nPosibles valores atípicos por regla IQR (se reportan, no se eliminan):")
    for columna in INGREDIENTES + OTRAS_MEDICIONES + [VARIABLE_OBJETIVO]:
        q1 = datos[columna].quantile(0.25)
        q3 = datos[columna].quantile(0.75)
        rango_iqr = q3 - q1
        limite_inferior = q1 - 1.5 * rango_iqr
        limite_superior = q3 + 1.5 * rango_iqr
        cantidad = int(
            ((datos[columna] < limite_inferior) | (datos[columna] > limite_superior)).sum()
        )
        print(f"{columna}: {cantidad}")

    correlaciones_ingredientes = datos[INGREDIENTES + [VARIABLE_OBJETIVO]].corr()[
        VARIABLE_OBJETIVO
    ].drop(VARIABLE_OBJETIVO)
    correlaciones_ingredientes = correlaciones_ingredientes.reindex(
        correlaciones_ingredientes.abs().sort_values(ascending=False).index
    )
    print(f"\nCorrelación de ingredientes con {VARIABLE_OBJETIVO}:")
    print(correlaciones_ingredientes.round(3).to_string())

    print(f"\nCorrelación de otras mediciones con {VARIABLE_OBJETIVO}:")
    print(datos[OTRAS_MEDICIONES + [VARIABLE_OBJETIVO]].corr()[VARIABLE_OBJETIVO]
          .drop(VARIABLE_OBJETIVO).round(3).to_string())

    datos.to_csv(ARCHIVO_LIMPIO, index=False, encoding="utf-8-sig")
    print(f"\nDatos listos para análisis guardados en: {ARCHIVO_LIMPIO}")
    print(f"Variable objetivo: {VARIABLE_OBJETIVO}")
    print(f"Predictores basados en ingredientes (sin {VARIABLE_ID}): {', '.join(INGREDIENTES)}")
    print("FLOW y resistencia se reportan aparte porque son otras mediciones, no ingredientes.")


if __name__ == "__main__":
    main()