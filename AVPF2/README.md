# Análisis del asentamiento (Slump) de mezclas de concreto

## Estructura del proyecto

```text
AVPF2/
├── data/
│   ├── dataset.xlsx
│   ├── dataset.csv
│   ├── correlaciones_slump.csv
│   ├── metricas_prediccion_slump.csv
│   ├── predicciones_validacion_slump.csv
│   ├── resumen_estadistico.csv
│   └── serie_slump_flow.csv
├── dashboard/
│   └── Dashboard_AVPF2.html
├── notebooks/
│   ├── 1-Limpieza Datos.py
│   ├── 2-Estadítica.py
│   ├── 3-Serie temporal.py
│   ├── 4-Graficos.py
│   ├── 5-Predicciones.py
│   ├── 6-Dashboard.py
│   └── requirements.txt
├── reportes/
│   └── Informe.pdf
├── imagenes/
├── modelos/
├── README.md
└── requirements.txt
```

- `data/dataset.xlsx` es el archivo Excel original. `data/dataset.csv` contiene los datos limpios que consumen los análisis y el dashboard.
- `dashboard/Dashboard_AVPF2.html` es la página HTML de presentación del proyecto.
- `notebooks/` contiene los scripts Python de limpieza, análisis, gráficos, predicción y la aplicación de Streamlit.
- `notebooks/requirements.txt` declara las dependencias en la misma carpeta que el script principal para que Streamlit Community Cloud las instale al desplegarlo. `requirements.txt` en la raíz de `AVPF2/` sirve para la instalación local.
- `reportes/Informe.pdf` contiene el informe del proyecto.
- `imagenes/` almacena las gráficas generadas y `modelos/` los modelos entrenados.
- Los CSV de estadísticas, correlaciones, métricas, predicciones y series también se guardan en `data/`.

## Objetivo

Limpiar y analizar datos de mezclas de concreto para explorar cómo se relacionan sus ingredientes con el asentamiento (`SLUMP`, en centímetros). El proyecto calcula estadísticas y correlaciones, genera visualizaciones y compara modelos de regresión lineal y polinomial para estimar el asentamiento.

El análisis es exploratorio: las correlaciones no demuestran causalidad y las predicciones deben validarse con datos nuevos antes de usarse para tomar decisiones reales.

## Dataset

El conjunto de datos **Concrete Slump Test** contiene 103 muestras. Incluye siete variables de entrada (cemento, escoria, ceniza volante, agua, superplastificante, agregado grueso y agregado fino) y tres mediciones de salida: `SLUMP (cm)`, `FLOW (cm)` y resistencia a compresión (`Compressive Strength (MPa)`). Las dosificaciones se expresan en kg/m³; slump y flow, en cm; y la resistencia, en MPa.

El archivo no contiene fechas; por ello, los gráficos por número de muestra representan el orden de los registros, no una evolución en tiempo calendario.

## Tecnologías

- Python, Pandas y NumPy para procesamiento y análisis de datos.
- Matplotlib para visualizaciones.
- Scikit-learn para regresión y validación cruzada.
- Streamlit para el dashboard interactivo.
- Openpyxl para leer el Excel.
- HTML y CSS para la página de presentación.

## KPIs y métricas del modelo

El dashboard muestra el número de muestras, el slump promedio, la mediana de slump y la correlación de Pearson entre agua y slump. En la pestaña de modelos también presenta MAE, RMSE y R² para comparar el desempeño predictivo. Una correlación indica asociación, no causalidad; los resultados predictivos son estimaciones sobre los datos disponibles.

## Instalación y ejecución

Abre PowerShell en esta carpeta (`AVPF2/`), crea el entorno virtual e instala las dependencias:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Primero limpia el Excel original para generar `data/dataset.csv`:

```powershell
python ".\notebooks\1-Limpieza Datos.py"
```

Después puedes ejecutar los análisis:

```powershell
python ".\notebooks\2-Estadítica.py"
python ".\notebooks\3-Serie temporal.py"
python ".\notebooks\4-Graficos.py"
python ".\notebooks\5-Predicciones.py"
```

Para iniciar el dashboard interactivo de Streamlit:

```powershell
streamlit run ".\notebooks\6-Dashboard.py"
```

El dashboard interactivo también está disponible como aplicación Streamlit. El archivo `dashboard/Dashboard_AVPF2.html` es una página de presentación estática; se puede abrir directamente en un navegador.

## Fuente de datos

UCI Machine Learning Repository. (s. f.). *Concrete Slump Test*. https://doi.org/10.24432/C5FG7D

## Integrantes

- Integrante 1: Quintana Chinchay Carolina Anais
- Integrante 2: Nuñez Chunga Elizabeth Asteria
- Integrante 3: Cunyarache Correa Gleidy Melissa
