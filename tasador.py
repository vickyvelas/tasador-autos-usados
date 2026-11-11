"""Limpieza y pipeline del tasador, idénticos al Colab de la Entrega 3.

build_app.py y app.py importan este módulo, así el modelo guardado con joblib
se puede volver a cargar en la app (la clase IngenieriaAutos tiene que existir
con el mismo nombre de módulo).

Mapa del archivo (de arriba hacia abajo):
  constantes            semilla, tipo de cambio, columnas que usa el modelo y reglas de texto
  norm_text / normalizar_categorica   cómo se unifican "AUTOMÁTICO" y "automatic", etc.
  clave_marca / clave_modelo          cómo se unifican "CITROEN" y "Citroën", "Eco sport" y "EcoSport", "Leapmotors" y "Leapmotor"
  IngenieriaAutos       primer paso del Pipeline: limpia las categorías adentro del modelo (sin fuga)
  limpiar()             la limpieza del notebook: precio en dólares, filtros, clave de vehículo y duplicados
  preprocesador()       ColumnTransformer: qué se le hace a cada tipo de columna
  pipeline()            las 3 etapas juntas: ingeniería -> preprocesador -> modelo con objetivo logarítmico
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# --- constantes -------------------------------------------------------------
SEMILLA = 42            # misma semilla en toda la cadena: partición, modelos, TargetEncoder
TIPO_CAMBIO = 1565.0    # ARS por USD usado para pasar los precios en pesos a dólares (decisión del grupo, Entrega 3)
URL_DATOS = "https://sourceforge.net/p/g-utn-frm08/codedataset01-cs-datos2026-test/ci/master/tree/dataset.csv?format=raw"

# Columnas que entran al modelo, según cómo se tratan:
NUM_COLS = ["fabricado_en", "kilometraje", "motor_lt", "potencia_hp", "consumo_lt_100km"]  # números: mediana + escala
CAT_COLS = ["transmision", "traccion", "combustible", "marca_n"]                            # pocas categorías: one-hot
TE_COLS = ["modelo_n"]  # muchísimas categorías (cientos de modelos): TargetEncoder en vez de one-hot

# Reglas para unificar el texto libre de los portales. Se prueban en orden y gana la primera que coincide.
# (patrón, etiqueta final). Ej: "AUTOMÁTICO SECUENCIAL" -> "automatico"; "4x4", "AWD" o "Integral" -> "4x4".
REGLAS_CAJA = [("manual", "manual"), ("auto|cvt", "automatico")]
REGLAS_TRAC = [("delan", "delantera"), ("tras", "trasera"), ("4x4|awd|integral", "4x4")]
REGLAS_COMB = [("nafta", "nafta"), ("diesel", "diesel"), ("ctric", "electrico"), ("brid|hev", "hibrido"), ("gas", "gnc")]


def norm_text(v):
    """Texto en minúsculas, sin tildes ni espacios de más. 'CITROËN  C4' -> 'citroen c4'.
    Es lo que hace que CarOne (todo en mayúsculas) y DeRuedas (Capitalizado) hablen de la misma marca."""
    if pd.isna(v):
        return ""
    v = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", v.lower()).strip()


def normalizar_categorica(v, reglas):
    """Devuelve la etiqueta de la primera regla que coincide; si ninguna, NaN (el modelo lo imputa después)."""
    v = norm_text(v)
    for patron, etiqueta in reglas:
        if re.search(patron, v):
            return etiqueta
    return np.nan


# --- marcas y modelos escritos distinto en los portales ---------------------------------------------
# CarOne escribe CITROEN / CRUZE y DeRuedas Citroen / Cruze; además hay variantes como "Eco sport" y "EcoSport",
# "Up" y "Up!", "Leapmotor" y "Leapmotors". Para el modelo y para la clave del vehículo, marca y modelo se reducen
# a una clave: minúsculas, sin tildes y sin espacios ni signos. Dos escrituras de la misma cosa dan la misma clave.
# Lo que la clave sola no resuelve (un plural, una abreviatura, un nombre incompleto) va en estas tablas.
MARCA_ALIAS = {"leapmotors": "leapmotor", "vw": "volkswagen", "mercedes": "mercedesbenz", "chevy": "chevrolet"}
MODELO_ALIAS = {("renault", "stepway"): "sanderostepway"}  # (marca, modelo) -> modelo; "Stepway" es el Sandero Stepway


def _sin_signos(v):
    return re.sub(r"[^a-z0-9]", "", norm_text(v))


def clave_marca(v):
    """'CITROËN', 'Citroen' -> 'citroen' · 'Mercedes-Benz' -> 'mercedesbenz' · 'Leapmotors' -> 'leapmotor'."""
    k = _sin_signos(v)
    return MARCA_ALIAS.get(k, k)


def clave_modelo(marca, modelo):
    """'Eco sport', 'EcoSport' -> 'ecosport' · 'Up!', 'up' -> 'up' (la marca sólo se usa para los alias)."""
    k = _sin_signos(modelo)
    return MODELO_ALIAS.get((clave_marca(marca), k), k)


# --- ubicación (NO entra al modelo; sólo se usa para la prueba de la hipótesis y el mapa) -------------------------
def arreglar_codificacion(v):
    """'CÃ³rdoba' -> 'Córdoba': la cosecha de DeRuedas guardó texto UTF-8 leído como Latin-1."""
    if pd.isna(v):
        return v
    v = str(v).strip()
    try:
        return v.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return v


def provincia_de(ubicacion):
    """'Mendoza, Godoy Cruz' -> 'Mendoza'. Las ubicaciones de CarOne sin provincia ('San Martin') son de Buenos Aires."""
    if pd.isna(ubicacion):
        return "Sin especificar"
    v = re.sub(r"\s+", " ", str(ubicacion)).strip()
    if v in ("SanMartin", "San Martin", "San Martín", "Tortuguitas"):
        return "Buenos Aires"
    prov = v.split(",")[0].strip().title()
    if "Capital Federal" in prov or prov in ("Caba", "Cap. Federal", "Ciudad De Buenos Aires"):
        return "Capital Federal"
    if "Bs" in prov or "Buenos Aires" in prov:
        return "Buenos Aires"
    return prov


def eta_cuadrado(y, grupos):
    """Qué parte de la variación del precio explican los grupos (0 = nada, 1 = todo). ANOVA de un factor."""
    d = pd.DataFrame({"y": y, "g": grupos}).dropna()
    entre = d.groupby("g")["y"].agg(["mean", "size"])
    ss_entre = (entre["size"] * (entre["mean"] - d["y"].mean()) ** 2).sum()
    return float(ss_entre / ((d["y"] - d["y"].mean()) ** 2).sum())


def claves_marca_modelo(df):
    """Dos Series alineadas con df: clave de marca y clave de modelo (lo que usan el Pipeline y la clave del vehículo)."""
    marcas = df["marca"].map(clave_marca)
    modelos = pd.Series([clave_modelo(a, b) for a, b in zip(df["marca"], df["modelo"])], index=df.index)
    return marcas, modelos


class IngenieriaAutos(BaseEstimator, TransformerMixin):
    """Primer paso del Pipeline. Está adentro del Pipeline (y no como limpieza previa) para que se
    aplique igual al entrenar, al validar y cuando alguien carga un auto en la app.

    No aprende nada de los datos (fit no hace nada), así que no puede filtrar información del test."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        # caja, tracción y combustible: de texto libre a pocas etiquetas
        X_new["transmision"] = X_new["transmision"].apply(lambda v: normalizar_categorica(v, REGLAS_CAJA))
        X_new["traccion"] = X_new["traccion"].apply(lambda v: normalizar_categorica(v, REGLAS_TRAC))
        X_new["combustible"] = X_new["combustible"].apply(lambda v: normalizar_categorica(v, REGLAS_COMB))
        # marca_n y modelo_n: la marca y el modelo normalizados. modelo_n lleva la marca adelante para que
        # un "Cruze" de Chevrolet no se mezcle con otro "Cruze". Es la columna que va al TargetEncoder.
        marca_n, modelo_n = claves_marca_modelo(X_new)
        X_new["marca_n"] = marca_n
        X_new["modelo_n"] = marca_n + " " + modelo_n
        # estas columnas ya cumplieron su función (o no deben entrar al modelo): "moneda" y "clave" darían fuga
        return X_new.drop(columns=["marca", "modelo", "moneda", "clave"])


def limpiar(df_raw: pd.DataFrame):
    """Devuelve (df limpio, pasos) con la misma limpieza del Colab.
    `pasos` = lista de (paso, filas que quedan, por qué): es la tabla que muestra la app en "Cómo lo sabemos"."""
    pasos = [("Filas en la capa plata del pipeline", len(df_raw), "CSV de plata: CarOne + DeRuedas")]
    df_raw = df_raw.copy()
    # objetivo: precio en dólares. Si está en pesos se divide por el tipo de cambio fijo.
    df_raw["precio_usd"] = np.where(df_raw["moneda"] == "ARS", df_raw["precio"] / TIPO_CAMBIO, df_raw["precio"])
    # sin precio (0 o vacío) no hay qué predecir
    df = df_raw[df_raw["precio_usd"] > 0].copy()
    pasos.append(("Con precio publicado", len(df), "sin precio no hay objetivo"))
    # un motor de más de 40 litros no existe: es un error de carga (ej. 1000 en lugar de 1.0)
    df = df[~(df["motor_lt"] > 40)].copy()
    pasos.append(("Motor con valor posible", len(df), "motor_lt > 40 es un error de carga"))
    # consumo 0 significa "no informado", no "consume 0 litros": se pasa a vacío para que se impute
    df["consumo_lt_100km"] = df["consumo_lt_100km"].replace(0.0, np.nan)
    # "clave" identifica al vehículo físico: marca|modelo|versión|año|km (todo normalizado).
    # Sirve para dos cosas: sacar avisos repetidos y, más adelante, agrupar la validación
    # (el mismo auto nunca queda a la vez en train y en test).
    marca_k, modelo_k = claves_marca_modelo(df)  # "Eco sport" y "EcoSport" son el mismo auto
    df["clave"] = (marca_k + "|" + modelo_k + "|" +
                   df["version"].map(norm_text) + "|" + df["fabricado_en"].astype(str) + "|" +
                   df["kilometraje"].astype(str))
    # mismo vehículo y mismo precio = el mismo aviso publicado dos veces
    df = df.drop_duplicates(subset=["clave", "precio"]).copy()
    pasos.append(("Sin avisos repetidos", len(df), "mismo auto (marca, modelo, versión, año, km) y mismo precio"))
    return df, pasos


# Columnas de la capa plata que NO entran al modelo (datos de la publicación, no del auto, o el objetivo "precio").
# Quedan guardadas aparte en la app para mostrarlas (versión, url, fuente...).
COLUMNAS_SIN_USO = ["fuente", "id_publicacion", "ubicacion", "version", "fecha_ingesta", "grupo", "url", "precio"]


def preprocesador():
    """Qué se le hace a cada grupo de columnas antes del modelo."""
    return ColumnTransformer([
        # números: los vacíos se completan con la mediana y se llevan a la misma escala
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_COLS),
        # categorías chicas: los vacíos se completan con el valor más frecuente y se pasan a columnas 0/1;
        # handle_unknown="ignore" = si en la app llega una categoría nueva, no se rompe
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
        # modelo (cientos de valores): cada modelo se reemplaza por el precio típico de sus avisos de entrenamiento.
        # sklearn lo calcula con validación cruzada interna, así el auto no "ve" su propio precio.
        ("te", TargetEncoder(target_type="continuous", random_state=SEMILLA), TE_COLS),
    ])


def pipeline(estimador):
    """Las tres etapas: ingeniería -> preprocesador -> modelo. El modelo aprende sobre log(1 + precio) y
    devuelve el precio en dólares otra vez (np.expm1): los precios tienen cola larga y así el error se reparte mejor."""
    return Pipeline([
        ("ingenieria", IngenieriaAutos()),
        ("prep", preprocesador()),
        ("modelo", TransformedTargetRegressor(regressor=estimador, func=np.log1p, inverse_func=np.expm1)),
    ])


def pipeline_hgb(**params):
    """El pipeline con HistGradientBoosting (el modelo elegido); params son los hiperparámetros de la grilla."""
    return pipeline(HistGradientBoostingRegressor(random_state=SEMILLA, **params))
