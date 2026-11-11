"""Coordenadas de las ubicaciones del dataset, pedidas a una API de geocodificación (Nominatim / OpenStreetMap).

build_app.py llama a `completar_coordenadas`: toma las `ubicacion_corregida` del CSV, busca cuáles todavía no están en la
tabla guardada (data/ubicaciones_coordenadas.csv), le pide la latitud y longitud de cada una a la API y las agrega a la tabla.
La tabla funciona de caché: cada ciudad se pide una sola vez, y las siguientes corridas no consultan nada.

Controles para no dibujar una ciudad en cualquier lado:
  * el resultado se acepta sólo si cae dentro de la caja de su provincia (sacada de data/provincias.geojson);
  * si la API no responde, no hay red o el resultado cae afuera, se usa el centro de la provincia;
  * toda ubicación que no sea una localidad (un departamento, por ejemplo) o que haya caído en el centro queda con
    aproximada=True, y el mapa lo avisa en el cartel.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://nominatim.openstreetmap.org/search"
# Nominatim pide un User-Agent que identifique a la aplicación y como máximo 1 consulta por segundo.
USER_AGENT = "tasador-autos-usados-proyecto-universitario/1.0"
PAUSA = 1.1
LOCALIDADES = {"city", "town", "village", "hamlet", "suburb", "neighbourhood", "city_district", "quarter", "locality", "municipality"}
ALIAS_PROVINCIA = {"Capital Federal": "Ciudad Autónoma de Buenos Aires", "Tierra Del Fuego": "Tierra del Fuego"}
COLUMNAS = ["ubicacion_corregida", "provincia", "latitud", "longitud", "aproximada"]


def cajas_provincias(ruta_geojson):
    """{provincia: (lat_min, lat_max, lon_min, lon_max)} del geojson."""
    geo = json.loads(Path(ruta_geojson).read_text(encoding="utf-8"))
    cajas = {}
    for f in geo["features"]:
        pts = [p for poligono in f["geometry"]["coordinates"] for anillo in poligono for p in anillo]
        lons, lats = [p[0] for p in pts], [p[1] for p in pts]
        cajas[f["properties"]["nombre"]] = (min(lats), max(lats), min(lons), max(lons))
    return cajas


def caja_de(cajas, provincia):
    return cajas.get(ALIAS_PROVINCIA.get(provincia, provincia))


def pedir_a_la_api(ciudad, provincia, timeout=15):
    """Una consulta a Nominatim. Devuelve (lat, lon, tipo) o None si no hay resultado."""
    # "Mendoza, Godoy Cruz" -> se busca "Godoy Cruz, Mendoza, Argentina"
    lugar = ciudad.split(",", 1)[1].strip() if "," in ciudad else ciudad
    q = f"{lugar}, {provincia}, Argentina"
    url = URL + "?" + urllib.parse.urlencode({"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "ar"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        datos = json.loads(r.read().decode("utf-8"))
    if not datos:
        return None
    d = datos[0]
    return float(d["lat"]), float(d["lon"]), d.get("addresstype") or d.get("type") or ""


def buscar(ciudad, provincia, cajas, consulta=pedir_a_la_api):
    """(lat, lon, aproximada, vino_de_la_api) de una ciudad: de la API si cae dentro de su provincia, si no el centro de la provincia."""
    caja = caja_de(cajas, provincia)
    try:
        res = consulta(ciudad, provincia)
    except Exception as e:   # sin red, API caída, límite de consultas...
        print(f"  [geocodificar] {ciudad}: la API no respondió ({type(e).__name__})")
        res = None
    if res and caja and caja[0] <= res[0] <= caja[1] and caja[2] <= res[1] <= caja[3]:
        lat, lon, tipo = res
        return lat, lon, tipo not in LOCALIDADES, True
    if caja:   # centro de la caja de la provincia (no se guarda en la tabla: la próxima corrida lo vuelve a intentar)
        return (caja[0] + caja[1]) / 2, (caja[2] + caja[3]) / 2, True, False
    return None, None, True, False


def completar_coordenadas(ciudades, ruta_tabla, ruta_geojson, consultar_api=True, consulta=pedir_a_la_api, pausa=PAUSA):
    """Devuelve la tabla (COLUMNAS) con todas las `ciudades` ({ubicacion_corregida: provincia}).

    Las que ya estaban en `ruta_tabla` se reutilizan; las nuevas se piden a la API (si consultar_api) y se guardan en la tabla."""
    ruta_tabla = Path(ruta_tabla)
    tabla = pd.read_csv(ruta_tabla) if ruta_tabla.exists() else pd.DataFrame(columns=COLUMNAS)
    nuevas = [c for c in ciudades if c not in set(tabla["ubicacion_corregida"])]
    if nuevas:
        cajas = cajas_provincias(ruta_geojson)
        print(f"[geocodificar] {len(nuevas)} ubicaciones nuevas" + ("" if consultar_api else " (sin consultar la API: centro de la provincia)"))
        filas = []
        for ciudad in nuevas:
            prov = ciudades[ciudad]
            if consultar_api:
                lat, lon, aprox, de_api = buscar(ciudad, prov, cajas, consulta)
                time.sleep(pausa)
            else:
                lat, lon, aprox, de_api = buscar(ciudad, prov, cajas, consulta=lambda *_: None)
            filas.append({"ubicacion_corregida": ciudad, "provincia": prov, "latitud": lat, "longitud": lon,
                          "aproximada": aprox, "_api": de_api})
        nuevas_df = pd.DataFrame(filas)
        # A la tabla guardada sólo van las que respondió la API (y pasaron el control); las que cayeron al centro de la
        # provincia se usan en esta corrida pero se vuelven a intentar en la próxima.
        guardar = pd.concat([tabla, nuevas_df[nuevas_df["_api"]][COLUMNAS]], ignore_index=True)
        if len(guardar) > len(tabla):
            guardar.to_csv(ruta_tabla, index=False, encoding="utf-8")
        tabla = pd.concat([tabla, nuevas_df[COLUMNAS]], ignore_index=True)
    return tabla
