# Análisis del asentamiento (Slump) de mezclas de concreto

## Título del Proyecto

**Análisis exploratorio y predicción del asentamiento del concreto (Slump).**

## Objetivo

Limpiar y analizar datos de mezclas de concreto para explorar cómo se relacionan sus ingredientes con el asentamiento (`SLUMP`, en centímetros). El proyecto calcula estadísticas y correlaciones, genera visualizaciones y compara modelos de regresión lineal y polinomial para estimar el asentamiento.

El análisis es exploratorio: las correlaciones no demuestran causalidad y las predicciones deben validarse con datos nuevos antes de usarse para tomar decisiones reales.

## Dataset

- **Archivo de entrada:** `excel_slump.csv (1).xlsx`, ubicado en la carpeta raíz.
- **Registros:** el dashboard está preparado para las 103 muestras del conjunto de datos.
- **Variables:** número de muestra (`No.`), cemento (`Cement`), escoria (`Slag`), ceniza volante (`Fly ash`), agua (`Water`), superplastificante (`SP`), agregado grueso (`Coarse Aggr.`), agregado fino (`Fine Aggr.`), asentamiento (`SLUMP (cm)`), flujo (`FLOW (cm)`) y resistencia a compresión (`Compressive Strength (MPa)`).
- **Variable objetivo:** `SLUMP (cm)`.
- **Preparación:** el primer script valida las columnas, convierte los valores a numéricos y elimina filas duplicadas o incompletas. El archivo limpio se guarda como `AVPF2/slump_limpio.csv`.

El dataset no incluye fechas; por ello, los gráficos por número de muestra representan el orden de los registros, no una evolución en tiempo calendario.

## Tecnologías Utilizadas

- **Python** para el procesamiento y análisis.
- **Pandas y NumPy** para manipulación de datos y cálculos.
- **Matplotlib** para gráficos.
- **Scikit-learn** para escalado, regresión y validación cruzada.
- **Streamlit** para el dashboard interactivo.
- **Openpyxl** para leer el archivo Excel.
- **HTML y CSS** para la página de presentación `index.html`.
- **Streamlit** para creación del Dashboard interactivo.

## KPIs

El dashboard muestra estos indicadores principales:

| Indicador | Qué representa |
|---|---|
| Muestras | Número de registros completos incluidos en el análisis. |
| Slump promedio | Media del asentamiento observado, en centímetros. |
| Mediana de slump | Valor central del asentamiento, en centímetros. |
| Correlación agua / slump | Correlación de Pearson entre el agua y el asentamiento; indica asociación, no causalidad. |

En la pestaña **Modelos** se muestran métricas adicionales para comparar las predicciones:

- **MAE (cm):** error absoluto medio; menor es mejor.
- **RMSE (cm):** raíz del error cuadrático medio; penaliza más los errores grandes y menor es mejor.
- **R²:** medida del ajuste predictivo; un valor mayor suele ser preferible, aunque debe interpretarse junto con las otras métricas.

Las métricas de regresión se calculan con validación cruzada de 5 partes. Son estimaciones sobre los datos disponibles, no garantías de desempeño con muestras nuevas.

## Dashboard

El dashboard se ejecuta con Streamlit y contiene cinco pestañas:

- **Resumen:** estadísticas descriptivas del asentamiento.
- **Gráficos:** distribución del slump, relaciones con los ingredientes, correlaciones y gráficos por número de muestra.
- **Modelos:** comparación de regresión lineal y polinomial usando validación cruzada.
- **Predecir:** estimación del asentamiento a partir de las cantidades de ingredientes ingresadas.
- **Datos:** exploración de los datos utilizados.

## Instalación y ejecución

Se requiere Python 3.10 o posterior y `pip`. Desde PowerShell, en la carpeta raíz del proyecto, crea y activa un entorno virtual e instala las dependencias:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Primero genera los datos limpios:

```powershell
python ".\AVPF2\1-Limpieza Datos.py"
```

Después puedes ejecutar los demás análisis:

```powershell
python ".\AVPF2\2-Estadítica.py"
python ".\AVPF2\3-Serie temporal.py"
python ".\AVPF2\4-Graficos.py"
python ".\AVPF2\5-Predicciones.py"
```

Para iniciar el dashboard:

```powershell
streamlit run ".\AVPF2\6-Dashboard.py"
```

Streamlit mostrará una dirección local para abrir el dashboard en el navegador. También puedes abrir `index.html` para ver la página de presentación.

## Archivos principales

- `index.html`: página de presentación.
- `requirements.txt`: dependencias de Python.
- `AVPF2/1-Limpieza Datos.py`: limpieza y validación del archivo de entrada.
- `AVPF2/2-Estadítica.py`: estadísticas descriptivas y correlaciones.
- `AVPF2/3-Serie temporal.py`: gráficos de slump y flow por número de muestra.
- `AVPF2/4-Graficos.py`: gráficos exploratorios.
- `AVPF2/5-Predicciones.py`: comparación de modelos y exportación de métricas y predicciones.
- `AVPF2/6-Dashboard.py`: aplicación interactiva de Streamlit.

 
## Integrantes
 
- Integrante 1: Quintana Chinchay Carolina Anais
- Integrante 2: Nuñez Chunga Elizabeth Asteria
- Integrante 3: Cunyarache Correa Gleidy Melissa
