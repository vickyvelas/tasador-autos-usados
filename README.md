# ¿Cuánto vale un auto usado? · Tasador de autos usados

Proyecto integrador de Ciencia de Datos · UTN FRM · 2026 · comisión 5K09, grupo 03
(García Nicolás, Ramos Ignacio, Velasco Victoria, Via Tomás, Villanueva Sergio).

**App publicada:** _https://tasador-autos-usados.streamlit.app_

> Con la marca, el modelo, el año y la ficha técnica, el tasador estima el precio de un auto usado con un
> error típico de **6,6 %** (USD 1.428 de error promedio en test), **4,3 veces menos** que tasar todo al
> precio mediano (USD 6.201).

## La cadena completa

```
CarOne (API GraphQL) ─┐
                      ├─► pipeline/ (Airflow) ─► bronce (crudo) ─► plata (CSV, 18 columnas) ─► calidad
DeRuedas (HTML)    ───┘                                                     │
                                                                            ▼
                     notebooks/entrega3_modelo.ipynb   análisis y modelo (Entregas 2 y 3)
                                                                            │
                     build_app.py   misma limpieza y mismo modelo que el notebook, semilla 42
                                                                            ▼
                     app_data/  ─►  app.py (Streamlit)  ─►  URL pública
```

| Carpeta / archivo | Qué es |
|---|---|
| `pipeline/` | Pipeline de Airflow que cosecha los avisos y arma el dataset (Entrega 1). Ver `pipeline/README.MD` |
| `notebooks/entrega3_modelo.ipynb` | Análisis y modelo (Entregas 2 y 3), el mismo Colab que se presentó |
| `tasador.py` | Limpieza y `Pipeline` de scikit-learn, idénticos al notebook. Lo usan `build_app.py` y la app |
| `build_app.py` | Toma el CSV del pipeline, entrena el modelo con semilla 42 y genera `app_data/` |
| `geocodificar.py` | Pide a la API de Nominatim (OpenStreetMap) las coordenadas de las ubicaciones nuevas del CSV. `build_app.py` lo llama solo |
| `data/ubicaciones_coordenadas.csv`, `data/provincias.geojson` | Coordenadas ya obtenidas (caché de la API) y contorno de las provincias para el mapa |
| `app_data/` | Lo que lee la app: `modelo.joblib`, `autos.csv` (cada auto con su tasación), `ubicaciones.csv` (datos del mapa) y `metricas.json` |
| `app.py` | La aplicación de Streamlit |
| `data/dataset.csv` | Cosecha del 9/9/2026 (capa plata): 9.039 filas, 8.909 después de limpiar |
| `requirements.txt` | Versiones fijas de la app |

## Qué muestra la app

Una "copa de martini":

- **Portada:** la idea central y tres números (error en test, error del piso, autos analizados).
- **Los datos:** distribución de precios. En *Cómo lo sabemos*: el pipeline (fuentes, bronce, plata, calidad),
  qué es una fila, los filtros paso por paso con las filas que quedan y las columnas descartadas por fuga.
- **Lo que encontramos:** un auto pierde cerca del 8 % de su valor por año. En *Cómo lo sabemos*: una hipótesis que
  se sostuvo (el año es lo que más pesa) y una que no (el kilometraje casi no agrega aparte del año), con números.
- **El modelo:** comparación del piso y los 4 modelos (media ± desvío en validación cruzada agrupada). En
  *Cómo lo sabemos*: métrica, piso, validación, grilla, test, error por tramo, modelos raros, importancia y fugas.
- **Explorá:** todos los autos coloreados según si están por encima o por debajo de la tasación del modelo, con
  filtros de marca, año, precio y caja, y la ficha de cada auto al hacer clic.
- **Probá el tasador:** cargás un auto y el modelo estima el precio. Sólo deja elegir marcas y modelos que el
  modelo vio y avisa cuando un valor está fuera de lo habitual.

## Cómo correr todo

### 1. El pipeline (opcional: el CSV ya está en `data/`)

Requiere Docker Desktop y el Astro CLI.

```bash
cd pipeline
astro dev start
```

En http://localhost:8080, correr el DAG `tp1_5K09_03_autos_mensual` (con `forzar_descarga` para cosechar).
Deja el CSV de plata en `pipeline/include/output/silver/final_AAAA-MM-DD.csv`. Una cosecha completa tarda un
par de horas.

### 2. El análisis y el modelo

Abrir `notebooks/entrega3_modelo.ipynb` en Google Colab o Jupyter y ejecutar todas las celdas.

### 3. La app en tu compu

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 4. Regenerar la app si cambian los datos

```bash
python build_app.py --datos pipeline/include/output/silver/final_AAAA-MM-DD.csv
```

Sin `--datos` usa `data/dataset.csv`. Repite la misma limpieza, la misma partición (semilla 42), la comparación de
modelos, la grilla de 32 combinaciones, el test y la importancia, y reescribe `app_data/`. Tarda unos minutos.

**Mapa.** `build_app.py` llama a `geocodificar.py`, que toma las `ubicacion_corregida` del CSV, busca cuáles no están en
`data/ubicaciones_coordenadas.csv` y le pide la latitud y longitud de cada una a Nominatim (1 consulta por segundo). Acepta el
resultado sólo si cae dentro de su provincia; si no responde la API o el punto cae afuera, usa el centro de la provincia y la
ubicación queda como *aproximada* (el cartel del mapa lo avisa). Las respuestas de la API quedan guardadas en esa tabla, así
cada ciudad se pide una sola vez. Con `--sin-api` no se consulta nada (por ejemplo, sin internet).
Después se suben los cambios a GitHub y Streamlit Cloud vuelve a publicar la app solo.

Las versiones de `requirements.txt` están fijas a propósito: con otras versiones de scikit-learn el boosting da
números un poco distintos a los del notebook.

## Publicarla en Streamlit Community Cloud

https://share.streamlit.io → **Create app** → este repositorio, rama `main`, archivo `app.py`.
En *Advanced settings* elegir Python 3.12.

## Resultados (cosecha del 9/9/2026, semilla 42)

| Modelo | Error promedio (validación cruzada agrupada) |
|---|---|
| Piso: tasar todo a la mediana | USD 6.201 ± 190 |
| Regresión lineal (Ridge) | USD 2.151 ± 128 |
| Random forest | USD 1.518 ± 100 |
| Boosting sin ajustar | USD 1.478 ± 102 |
| **Boosting ajustado (elegido)** | **USD 1.408 ± 87** |

Test (20 % de los vehículos, medido una sola vez): **USD 1.428**, error relativo mediano **6,6 %**.

Los nombres repetidos (Leapmotor/Leapmotors, Eco sport/EcoSport, Up/Up!, Stepway/Sandero Stepway) se unifican sólo para mostrarlos en la app; el modelo usa la misma normalización del notebook, por eso los números coinciden.
