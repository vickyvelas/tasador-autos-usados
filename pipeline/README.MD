# Tasador de autos usados · pipeline de datos (grupo 5K09-03)

Pipeline en Airflow que junta publicaciones de autos usados de **CarOne** (API GraphQL)
y **DeRuedas** (scraping del HTML), las guarda crudas (bronce), las unifica en un CSV con
un esquema único (plata), controla la calidad y arma un ZIP entregable.

## Cómo correrlo

Requisitos: Docker Desktop abierto y el [Astro CLI](https://www.astronomer.io/docs/astro/cli/install-cli) (`brew install astro` en Mac).

```bash
cd pipeline              # desde la raíz de este repositorio
astro dev start          # la primera vez tarda unos minutos
chmod -R 777 include     # para que Airflow pueda escribir en include/output
```

Abrir http://localhost:8080 (usuario y clave `admin` / `admin` si los pide).

- El DAG se llama `tp1_5K09_03_autos_mensual`.
- **Para cosechar a propósito:** Trigger → *Trigger con config* → tildar `forzar_descarga`.
  Una cosecha completa tarda un par de horas (pausa de 1,55 s entre avisos para no saturar los sitios).
- La primera corrida cosecha aunque no se tilde nada, porque todavía no hay cosecha previa.
- Para apagarlo: `astro dev stop`.
- Tests: `astro dev pytest` (normalización y estructura del DAG).

## Qué hace el DAG

| # | Tarea | Tipo | Qué hace |
|---|---|---|---|
| 1 | `crear_carpetas_trabajo` | task | Crea las carpetas de bronce, plata y respaldo |
| 2 | `validar_disponibilidad_fuentes` | sensor (reschedule, cada 60 s, 30 min, soft_fail) | Espera a que las dos fuentes devuelvan marcas |
| 3 | `obtener_marcas_actuales` | task | Lista de marcas de DeRuedas (vacía si no responde) |
| 4 | `elegir_ruta_datos` | branch (ALL_DONE) | Con marcas → cosecha; sin marcas → respaldo congelado |
| 5 | `chequear_refresh_mensual` | short circuit | Sigue sólo si se forzó o pasaron 30 días desde la última cosecha válida |
| 6 | `cosecha_bronce_carone` | task | Todos los usados en stock de CarOne → `bronze/carone/fecha=AAAA-MM-DD/carone_raw.json` |
| 7 | `cosecha_bronce_deruedas` | task mapeada (`expand`, una por marca) | Cada aviso → `bronze/deruedas/marca=<marca>/id_<cod>.html.gz`. Incremental: no vuelve a bajar lo que ya está. Ante un 429 reintenta en 10 min |
| 8 | `consolidar_capa_plata` | task | Parsea bronce, aplica el esquema de 18 columnas, saca duplicados → `silver/final_AAAA-MM-DD.csv` |
| 9 | `usar_respaldo_congelado` | task | Devuelve `include/frozen/autos_semilla.csv` (la cosecha del 7/9/2026, 9.145 filas) |
| 10 | `validar_calidad_dataset` | task (NONE_FAILED_MIN_ONE_SUCCESS) | Junta las dos ramas. Falla si hay menos de `meta_volumen` filas, faltan columnas, hay ids repetidos o más de 5 % de precios vacíos. Escribe `output/calidad.json` y, si fue cosecha, actualiza la Variable `autos_fecha_ultima_ingesta` |
| 11 | `generar_entregable_zip` | task | `output/tp1_5K09_03.zip` con `dataset.csv`, `codigo_dag.py`, `bronce.txt`, `calidad.json` y los logs de la corrida |

Configuración (`airflow_settings.yaml`): Variables `grupo` (5K09-03) y `meta_volumen` (9000), y las
Connections `carone_api` y `deruedas_web`, que documentan las dos fuentes.

## Estructura

```
dags/dag_tasador_vehiculos.py      orquestación (qué corre, cuándo y en qué orden)
include/autos/collectors/carone.py  API GraphQL de CarOne
include/autos/collectors/deruedas.py scraping de DeRuedas (buscador + parseo de cada aviso)
include/autos/normalize.py         limpieza: números argentinos, motor en litros, moneda
include/autos/schema.py            columnas del CSV de plata
include/frozen/autos_semilla.csv   respaldo si las fuentes no responden
tests/                             tests de normalización y del DAG
```

## Esquema de plata

`fuente, id_publicacion, marca, modelo, version, fabricado_en, kilometraje, precio, moneda,
motor_lt, potencia_hp, transmision, traccion, combustible, consumo_lt_100km, ubicacion, url,
fecha_ingesta` + `grupo`. Una fila = una publicación de un auto usado.

- Un dato que no se pudo leer queda **vacío**, no en 0.
- `precio` queda en su moneda original (`moneda` = ARS o USD); la conversión a dólares se hace en el análisis.
- En CarOne, `ubicacion` es el id del concesionario (la API no da la ciudad).
