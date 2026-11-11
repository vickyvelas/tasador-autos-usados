"""
grupo 5K09-03 · Proyecto integrador: tasador de autos usados

Pipeline mensual que cosecha avisos de autos usados de CarOne (API GraphQL) y
DeRuedas (scraping), los guarda crudos en bronce, los unifica en plata (CSV con
esquema único), controla la calidad y arma un ZIP entregable.

Se agenda @daily, pero sólo cosecha si pasaron DIAS_ENTRE_COSECHAS desde la
última cosecha exitosa (o si se dispara con forzar_descarga=True). Si las fuentes
no responden, usa el respaldo congelado include/frozen/autos_semilla.csv.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import pendulum
import requests
from airflow.sdk import Param, Variable, dag, task, PokeReturnValue as ResultadoValidacion
from airflow.task.trigger_rule import TriggerRule

### logica de negocio
from autos import schema
from autos.collectors import carone, deruedas
from autos.normalize import _slug

log = logging.getLogger(__name__)

TZ = "America/Argentina/Buenos_Aires"
DIR_SALIDA = Path("/usr/local/airflow/include/output")
DIR_BRONCE = DIR_SALIDA / "bronze"
DIR_PLATA = DIR_SALIDA / "silver"
DIR_FROZEN = Path("/usr/local/airflow/include/frozen")

VAR_ULTIMA_COSECHA = "autos_fecha_ultima_ingesta"
DIAS_ENTRE_COSECHAS = 30
PAUSA_ENTRE_AVISOS_S = 1.55
MAX_PAGINAS_POR_MARCA = 300


def _hoy() -> str:
    return pendulum.now(TZ).to_date_string()


@dag(
    dag_id="tp1_5K09_03_autos_mensual",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 8, 1, tz=TZ),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=2,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    tags=["proyecto-integrador", "vehiculos"],
    doc_md=__doc__,
    params={
        "forzar_descarga": Param(False, type="boolean", title="Forzar recolección ahora"),
    },
)
def pipeline_vehiculos():

    @task
    def crear_carpetas_trabajo():
        for carpeta in [DIR_BRONCE / "carone", DIR_BRONCE / "deruedas", DIR_PLATA, DIR_FROZEN]:
            carpeta.mkdir(parents=True, exist_ok=True)
        return True

    @task.sensor(poke_interval=60, timeout=1800, mode="reschedule", soft_fail=True)
    def validar_disponibilidad_fuentes():
        m_carone = carone.get_available_brands()
        m_deruedas = deruedas.get_available_brands()
        log.info(f"marcas CarOne={len(m_carone)} DeRuedas={len(m_deruedas)}")
        return ResultadoValidacion(is_done=bool(m_carone and m_deruedas))

    @task
    def obtener_marcas_actuales() -> list[str]:
        # Devuelve [] (no falla) si no hay marcas: así el branch puede elegir el respaldo
        marcas = deruedas.get_available_brands()
        log.info(f"{len(marcas)} marcas en DeRuedas: {marcas}")
        return marcas

    @task.branch(trigger_rule=TriggerRule.ALL_DONE)
    def elegir_ruta_datos(marcas_ok) -> str:
        # ALL_DONE: corre aunque el sensor haya quedado salteado o las marcas hayan fallado
        if marcas_ok:
            return "chequear_refresh_mensual"
        log.warning("Sin marcas disponibles: se usa el respaldo congelado")
        return "usar_respaldo_congelado"

    @task.short_circuit
    def chequear_refresh_mensual(**context) -> bool:
        if context["params"]["forzar_descarga"]:
            log.info("forzar_descarga=True: se cosecha")
            return True
        ultima = Variable.get(VAR_ULTIMA_COSECHA, default=None)
        if not ultima:
            log.info("No hay cosecha previa: se cosecha")
            return True
        dias = (pendulum.now(TZ) - pendulum.parse(ultima, tz=TZ)).days
        log.info(f"Última cosecha {ultima}, hace {dias} días")
        return dias >= DIAS_ENTRE_COSECHAS

    @task(retries=3, retry_delay=pendulum.duration(minutes=10))
    def cosecha_bronce_carone() -> str:
        datos_api = carone.search()
        if not datos_api:
            raise ValueError("CarOne no devolvió autos")
        carpeta = DIR_BRONCE / "carone" / f"fecha={_hoy()}"
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_archivo = carpeta / "carone_raw.json"
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            json.dump(datos_api, f, ensure_ascii=False, indent=2)
        log.info(f"CarOne: {len(datos_api)} autos -> {ruta_archivo}")
        return str(ruta_archivo)

    @task(
        map_index_template="{{ task.op_kwargs['marca'] }}",
        retries=3,
        retry_delay=pendulum.duration(minutes=10),
    )
    def cosecha_bronce_deruedas(marca: str) -> list[str]:
        marca_dir = DIR_BRONCE / "deruedas" / f"marca={_slug(marca)}"
        marca_dir.mkdir(parents=True, exist_ok=True)

        saved_paths, vistos = [], set()
        nuevos = fallidos = 0
        for page in range(1, MAX_PAGINAS_POR_MARCA + 1):
            links = deruedas.fetch_search_page_links(marca, page)
            nuevos_links = [u for u in links if u not in vistos]
            # Corta si la página viene vacía o repite la anterior (evita un bucle infinito)
            if not nuevos_links:
                break
            vistos.update(nuevos_links)

            for url in nuevos_links:
                car_id = url.split("cod=")[-1]
                file_path = marca_dir / f"id_{car_id}.html.gz"
                ## descargar solo si el aviso no está ya en bronce (incremental)
                if not file_path.exists():
                    try:
                        html = deruedas.download_listing(url)
                    except requests.exceptions.HTTPError as e:
                        if e.response is not None and e.response.status_code == 429:
                            raise  # Airflow reintenta en 10 min; lo ya bajado no se repite
                        fallidos += 1
                        log.warning(f"{url}: {e}")
                        continue
                    except requests.exceptions.RequestException as e:
                        fallidos += 1
                        log.warning(f"{url}: {e}")
                        continue
                    with gzip.open(file_path, "wt", encoding="utf-8") as f:
                        f.write(html)
                    nuevos += 1
                    time.sleep(PAUSA_ENTRE_AVISOS_S)
                saved_paths.append(str(file_path))
            time.sleep(PAUSA_ENTRE_AVISOS_S)

        log.info(f"{marca}: {len(saved_paths)} avisos ({nuevos} nuevos, {fallidos} con error)")
        return saved_paths

    @task
    def consolidar_capa_plata(json_carone: str, paths_dr_nested) -> str:
        DIR_PLATA.mkdir(parents=True, exist_ok=True)
        registros = []

        ### CarOne: el JSON de bronce
        with open(json_carone, "r", encoding="utf-8") as f:
            registros.extend(json.load(f))
        n_carone = len(registros)

        ### DeRuedas: parsear cada HTML.gz de bronce
        no_parseados = 0
        for path_list in paths_dr_nested:
            for file_path in path_list or []:
                try:
                    with gzip.open(file_path, "rt", encoding="utf-8") as f:
                        html = f.read()
                except OSError as e:
                    no_parseados += 1
                    log.warning(f"No se pudo leer {file_path}: {e}")
                    continue
                car_id = Path(file_path).name.replace("id_", "").replace(".html.gz", "")
                url_original = f"https://www.deruedas.com.ar/resul.asp?cod={car_id}"
                data = deruedas.parse_html_to_dict(html, url_original)
                if data:
                    registros.append(data)
                else:
                    no_parseados += 1

        df = pd.DataFrame(registros).reindex(columns=schema.COLUMNAS)
        for col in schema.NUMERICAS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["grupo"] = Variable.get("grupo", default="5K09-03")
        antes = len(df)
        df = df.drop_duplicates(subset=["fuente", "id_publicacion"]).reset_index(drop=True)
        log.info(
            f"Plata: CarOne={n_carone}, DeRuedas={antes - n_carone} "
            f"(sin parsear {no_parseados}), duplicados quitados={antes - len(df)}, filas={len(df)}"
        )

        fecha = _hoy()
        ruta_final = DIR_PLATA / f"final_{fecha}.csv"
        df.to_csv(ruta_final, index=False, encoding="utf-8")
        return str(ruta_final)

    @task
    def usar_respaldo_congelado() -> str:
        ruta_semilla = DIR_FROZEN / "autos_semilla.csv"
        if ruta_semilla.exists():
            log.warning(f"Usando respaldo congelado: {ruta_semilla}")
            return str(ruta_semilla)
        raise FileNotFoundError(f"sin respaldo disponible en {ruta_semilla}")

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS, retries=0)
    def validar_calidad_dataset(ruta_plata, ruta_congelado) -> dict:
        # Junta las dos ramas: recibe la ruta de la que haya corrido (la otra llega como None)
        ruta_csv = ruta_plata or ruta_congelado
        df = pd.read_csv(ruta_csv)
        meta_v = int(Variable.get("meta_volumen", default=9000))

        faltan = [c for c in schema.COLUMNAS if c not in df.columns]
        reporte = {
            "archivo": Path(ruta_csv).name,
            "origen": "cosecha" if ruta_plata else "respaldo_congelado",
            "filas": len(df),
            "meta_volumen": meta_v,
            "filas_por_fuente": df["fuente"].value_counts().to_dict() if "fuente" in df else {},
            "ids_duplicados": int(df.duplicated(subset=["fuente", "id_publicacion"]).sum()) if not faltan else None,
            "columnas_faltantes": faltan,
            "pct_vacios": {c: round(float(df[c].isna().mean()) * 100, 1) for c in schema.COLUMNAS if c in df},
        }
        log.info("Reporte de calidad:\n" + json.dumps(reporte, ensure_ascii=False, indent=2))

        errores = []
        if len(df) < meta_v:
            errores.append(f"volumen insuficiente: {len(df)}/{meta_v}")
        if faltan:
            errores.append(f"faltan columnas del esquema: {faltan}")
        if reporte["ids_duplicados"]:
            errores.append(f"{reporte['ids_duplicados']} publicaciones duplicadas")
        if reporte["pct_vacios"].get("precio", 0) > 5:
            errores.append(f"precio vacío en {reporte['pct_vacios']['precio']} % de las filas")
        if errores:
            raise ValueError("; ".join(errores))

        if ruta_plata:
            # Sólo una cosecha válida reinicia el reloj de 30 días
            Variable.set(VAR_ULTIMA_COSECHA, _hoy())

        reporte["ruta"] = ruta_csv
        ruta_reporte = DIR_SALIDA / "calidad.json"
        ruta_reporte.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
        return reporte

    @task
    def generar_entregable_zip(reporte: dict, **context) -> str:
        dag_run = context["dag_run"]
        path_zip = DIR_SALIDA / "tp1_5K09_03.zip"
        path_bronce_txt = DIR_SALIDA / "bronce.txt"
        with open(path_bronce_txt, "w", encoding="utf-8") as f:
            for file in sorted(DIR_BRONCE.rglob("*.*")):
                f.write(f"{file}\n")
        with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(reporte["ruta"], arcname="dataset.csv")
            zipf.write(__file__, arcname="codigo_dag.py")
            zipf.write(path_bronce_txt, arcname="bronce.txt")
            zipf.write(DIR_SALIDA / "calidad.json", arcname="calidad.json")
            ### logs de esta corrida
            log_dir = f"/usr/local/airflow/logs/dag_id={dag_run.dag_id}/run_id={dag_run.run_id}"
            if os.path.exists(log_dir):
                for root, _, archivos in os.walk(log_dir):
                    for a in archivos:
                        fp = os.path.join(root, a)
                        zipf.write(fp, arcname=os.path.join("logs", os.path.relpath(fp, log_dir)))
        log.info(f"Entregable: {path_zip}")
        return str(path_zip)

    ### grafo
    setup = crear_carpetas_trabajo()
    vivas = validar_disponibilidad_fuentes()
    marcas = obtener_marcas_actuales()
    setup >> vivas >> marcas

    camino = elegir_ruta_datos(marcas)
    refresh = chequear_refresh_mensual()
    congelado = usar_respaldo_congelado()
    camino >> [refresh, congelado]

    c_raw = cosecha_bronce_carone()
    d_raw = cosecha_bronce_deruedas.expand(marca=marcas)
    refresh >> [c_raw, d_raw]

    plata = consolidar_capa_plata(c_raw, d_raw)
    reporte = validar_calidad_dataset(plata, congelado)
    generar_entregable_zip(reporte)


dag_obj = pipeline_vehiculos()
