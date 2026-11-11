"""Coordenadas de las ubicaciones de los avisos, para el mapa de la app.

build_app.py llama a `obtener(ubicaciones)` con las ubicaciones corregidas del CSV ("Mendoza, Godoy Cruz",
"Tortuguitas", ...). Para cada una:

  1. Si ya está en la caché (data/ubicaciones_coordenadas.csv), se usa esa.
  2. Si no, se consulta la API oficial de georreferenciación de Argentina (Georef, apis.datos.gob.ar).
     DeRuedas casi siempre publica el DEPARTAMENTO ("Mendoza, Godoy Cruz", "Córdoba, San Alberto"), así que
     primero se busca el departamento y, adentro, su localidad cabecera (el punto queda sobre la ciudad; si no
     hay, el centro del departamento). Si no es un departamento, se busca como localidad. "Capital" es la
     capital de la provincia, que se pide a Wikidata con sus coordenadas. Nada está escrito a mano.
     No se usa Nominatim/OpenStreetMap: ubica los departamentos en la capital de la provincia
     (ej. San Alberto en Córdoba capital) y ese error pasa el control por provincia.
  3. Toda coordenada se valida: tiene que caer dentro de la caja de su provincia (data/provincias.geojson).
  4. Si la API no responde o no encuentra el lugar, se usa el centro de la provincia y se marca aproximada.

Las coordenadas nuevas que vienen de la API se guardan en la caché, así la próxima vez no se vuelve a consultar.
Se puede correr solo:  python coordenadas.py data/dataset.csv
"""
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

AQUI = Path(__file__).parent
CACHE = AQUI / "data" / "ubicaciones_coordenadas.csv"
GEOJSON = AQUI / "data" / "provincias.geojson"
API = "https://apis.datos.gob.ar/georef/api"
COLUMNAS = ["ubicacion_corregida", "provincia", "latitud", "longitud", "aproximada", "fuente"]

WIKIDATA = "https://query.wikidata.org/sparql"
# Capitales de las provincias argentinas (Q44753 = "provincia de Argentina", P36 = capital, P625 = coordenadas).
CONSULTA_CAPITALES = """SELECT ?provLabel ?capLabel ?coord WHERE {
  ?prov wdt:P31 wd:Q44753; wdt:P36 ?cap. ?cap wdt:P625 ?coord.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "es". } }"""


def _clave(texto):
    """Nombre comparable: sin tildes, en minúsculas, sin "Provincia de" ni lo que va después de una coma.
    'Provincia de Tierra del Fuego, Antártida e Islas…' y 'Tierra Del Fuego' -> 'tierra del fuego'."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower().strip()
    t = t.removeprefix("provincia de ").split(",")[0].strip()
    return " ".join(t.split())


def _provincias():
    """Caja (lon/lat mín y máx) y un punto central de cada provincia, sacados de sus límites."""
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    out = {}
    for f in geo["features"]:
        polys = f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]
        pts = np.array([p for poly in polys for p in poly[0]])
        # punto central: el centroide del polígono más grande (Tierra del Fuego tiene islas; así no cae en el mar)
        grande = max(polys, key=lambda poly: len(poly[0]))
        x, y = np.array(grande[0])[:, 0], np.array(grande[0])[:, 1]
        a = x[:-1] * y[1:] - x[1:] * y[:-1]
        cx = ((x[:-1] + x[1:]) * a).sum() / (3 * a.sum())
        cy = ((y[:-1] + y[1:]) * a).sum() / (3 * a.sum())
        out[_clave(f["properties"]["nombre"])] = {"nombre": f["properties"]["nombre"],
                                                  "caja": (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()),
                                                  "centro": (float(cy), float(cx))}
    return out


def _km(a, b):
    """Distancia aproximada en km entre dos puntos (lat, lon)."""
    dlat, dlon = np.radians(b[0] - a[0]), np.radians(b[1] - a[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(a[0])) * np.cos(np.radians(b[0])) * np.sin(dlon / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def _dentro(prov_info, lat, lon, margen=0.05):
    x0, x1, y0, y1 = prov_info["caja"]
    return x0 - margen <= lon <= x1 + margen and y0 - margen <= lat <= y1 + margen


def _georef(sesion, recurso, nombre, provincia, **extra):
    """Primer resultado de Georef (localidades o departamentos) o None si no hay o no hay red."""
    try:
        r = sesion.get(f"{API}/{recurso}", timeout=15,
                       params={"nombre": nombre, "provincia": provincia, "max": 1, "campos": "nombre,centroide", **extra})
        r.raise_for_status()
        items = r.json().get(recurso, [])
    except (requests.RequestException, ValueError):
        return None
    if not items:
        return None
    c = items[0]["centroide"]
    return float(c["lat"]), float(c["lon"]), items[0]["nombre"]


def _buscar(sesion, lugar, prov, pausa):
    """Busca un lugar en Georef. DeRuedas casi siempre publica el DEPARTAMENTO ("Mendoza, Godoy Cruz",
    "Córdoba, Punilla"), así que primero se busca el departamento y, adentro, su localidad del mismo nombre
    (la cabecera: así el punto queda sobre la ciudad). Devuelve (lat, lon, aproximada, fuente) o None."""
    depto = _georef(sesion, "departamentos", lugar, prov)
    time.sleep(pausa)
    if depto:
        ciudad = _georef(sesion, "localidades", lugar, prov, departamento=depto[2])
        time.sleep(pausa)
        # la localidad sólo se acepta si está cerca del departamento (a veces hay homónimos en otra punta de la provincia)
        if ciudad and _km(ciudad[:2], depto[:2]) <= 100:
            return ciudad[0], ciudad[1], False, "georef: localidad cabecera"
        return depto[0], depto[1], True, "georef: centro del departamento"
    ciudad = _georef(sesion, "localidades", lugar, prov)   # no es un departamento: se busca como localidad
    time.sleep(pausa)
    if ciudad:
        return ciudad[0], ciudad[1], False, "georef: localidad"
    return None


def _separar(ubicacion, provincia_de):
    """'Córdoba, San Alberto' -> ('Córdoba', 'San Alberto'); 'Tortuguitas' -> ('Buenos Aires', 'Tortuguitas')."""
    lugar = ubicacion.split(",", 1)[1].strip() if "," in ubicacion else ubicacion.strip()
    return provincia_de(ubicacion), lugar


def _nombre_oficial(sesion, provincia, memo):
    """Nombre oficial de la provincia según Georef ('Capital Federal' -> 'Ciudad Autónoma de Buenos Aires').
    Sin red, se usa el nombre tal como vino."""
    if provincia not in memo:
        try:
            r = sesion.get(f"{API}/provincias", params={"nombre": provincia, "max": 1, "campos": "nombre"}, timeout=15)
            items = r.json().get("provincias", [])
            memo[provincia] = items[0]["nombre"] if items else provincia
        except (requests.RequestException, ValueError):
            memo[provincia] = provincia
    return memo[provincia]


def _capitales(sesion):
    """{clave de provincia: (lat, lon, nombre de la capital)} desde Wikidata. Vacío si no hay red."""
    try:
        r = sesion.get(WIKIDATA, params={"query": CONSULTA_CAPITALES},
                       headers={"Accept": "application/sparql-results+json"}, timeout=30)
        filas = r.json()["results"]["bindings"]
    except (requests.RequestException, ValueError, KeyError):
        return {}
    out = {}
    for f in filas:
        lon, lat = f["coord"]["value"].removeprefix("Point(").removesuffix(")").split()
        out[_clave(f["provLabel"]["value"])] = (float(lat), float(lon), f["capLabel"]["value"])
    return out


def obtener(ubicaciones, provincia_de, usar_api=True, pausa=0.2):
    """DataFrame con COLUMNAS para cada ubicación distinta. Actualiza la caché con lo que trae la API."""
    provs = _provincias()
    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=COLUMNAS)
    if "fuente" not in cache.columns:
        cache["fuente"] = "caché"
    conocidas = set(cache["ubicacion_corregida"])
    nuevas, filas = [], []
    sesion = requests.Session()
    sesion.headers["User-Agent"] = "tasador-autos-usados (proyecto academico)"
    memo, capitales = {}, None
    for u in sorted(set(pd.Series(ubicaciones).dropna())):
        if u in conocidas:
            continue
        prov, lugar = _separar(u, provincia_de)
        prov = _nombre_oficial(sesion, prov, memo) if usar_api else prov
        info = provs.get(_clave(prov))
        fila = None
        if usar_api and info:
            res = None
            if lugar.lower() == "capital":   # "Mendoza, Capital": la capital de la provincia, según Wikidata
                if capitales is None:
                    capitales = _capitales(sesion)
                c = capitales.get(_clave(prov))
                res = (c[0], c[1], False, f"wikidata: capital provincial ({c[2]})") if c else None
            res = res or _buscar(sesion, lugar, prov, pausa)
            if res and _dentro(info, res[0], res[1]):
                fila = {"ubicacion_corregida": u, "provincia": info["nombre"], "latitud": round(res[0], 6),
                        "longitud": round(res[1], 6), "aproximada": res[2], "fuente": res[3]}
                nuevas.append(fila)
        if fila is None and info:
            # sin red o sin resultado válido: el centro de la provincia (no se guarda en la caché: se reintenta)
            fila = {"ubicacion_corregida": u, "provincia": info["nombre"], "latitud": round(info["centro"][0], 6),
                    "longitud": round(info["centro"][1], 6), "aproximada": True, "fuente": "centro de la provincia"}
        if fila is None:
            fila = {"ubicacion_corregida": u, "provincia": prov, "latitud": np.nan, "longitud": np.nan,
                    "aproximada": True, "fuente": "sin datos"}
        filas.append(fila)
    if nuevas:
        cache = pd.DataFrame(nuevas) if cache.empty else pd.concat([cache, pd.DataFrame(nuevas)], ignore_index=True)
        cache[COLUMNAS].to_csv(CACHE, index=False, encoding="utf-8")
    partes = [d for d in (cache[COLUMNAS], pd.DataFrame(filas, columns=COLUMNAS)) if not d.empty]
    todo = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLUMNAS)
    todo = todo.drop_duplicates("ubicacion_corregida", keep="first")
    pedidas = set(pd.Series(ubicaciones).dropna())
    return todo[todo["ubicacion_corregida"].isin(pedidas)][COLUMNAS].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    import tasador as T

    df, _ = T.limpiar(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else AQUI / "data" / "dataset.csv"))
    ub = df["ubicacion"].map(T.arreglar_codificacion).str.replace(r"\s+", " ", regex=True).str.strip()
    res = obtener(ub, T.provincia_de)
    print(res["fuente"].value_counts().to_string())
    print("sin coordenadas:", int(res["latitud"].isna().sum()), "de", len(res))
