"""Tasador de autos usados · Entrega 4 · grupo 5K09-03

La app sólo lee lo que generó build_app.py (app_data/). Para regenerarla con datos
nuevos del pipeline: python build_app.py --datos <final_AAAA-MM-DD.csv>
"""
import json
import unicodedata
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import tasador as T  # noqa: F401  (necesario para cargar el modelo con joblib)

AQUI = Path(__file__).parent
DATA = AQUI / "app_data"

st.set_page_config(page_title="Tasador de autos usados · 5K09-03", page_icon="🚗", layout="wide")
alt.data_transformers.disable_max_rows()

# --- colores: significan lo mismo en toda la app ---------------------------
# Una paleta para toda la app: cada color significa siempre lo mismo (tallo y copa).
AZUL = "#2a78d6"      # el modelo / la tasación
GRIS = "#9a9a9a"      # referencia (piso, otros autos, datos sin modelo)
TINTA = "#3d3d3a"     # precios reales
VERDE = "#1a9e77"     # más barato que la tasación
NARANJA = "#d95926"   # más caro que la tasación
VEREDICTOS = ["Más barato que la tasación", "En precio (±10 %)", "Más caro que la tasación"]
ESCALA_VEREDICTO = alt.Scale(domain=VEREDICTOS, range=[VERDE, GRIS, NARANJA])
ESCALA_PRECIO = alt.Scale(type="log", domain=[1500, 150000], clamp=True)
MARCAS_PRECIO = [2000, 5000, 10000, 20000, 50000, 100000]
PUNTOS_MIL = "replace(format(datum.value, ',.0f'), ',', '.')"  # 10.000 en vez de 10,000


COL_ES = {"fabricado_en": "Año", "potencia_hp": "Potencia (HP)", "marca": "Marca", "modelo": "Modelo",
          "kilometraje": "Kilometraje", "transmision": "Caja", "motor_lt": "Motor (litros)", "traccion": "Tracción",
          "consumo_lt_100km": "Consumo", "combustible": "Combustible"}
FUENTES = {"deruedas": "DeRuedas", "carone": "CarOne"}


# Eje con separador de miles argentino (10.000 y no 10,000). Se usa en casi todos los gráficos.
def eje(titulo, valores=None, **kw):
    if valores is not None:
        kw["values"] = valores
    return alt.Axis(title=titulo, labelExpr=PUNTOS_MIL, **kw)


# Formatos al estilo argentino: usd(1234) -> "USD 1.234"; num(1234.5, 1) -> "1.234,5".
def ancla(nombre):
    """Punto de destino de un link del índice (barra lateral). Deja un margen arriba para que el título no quede pegado."""
    st.html(f'<div id="{nombre}" style="position:relative; top:-70px; height:0"></div>')


def alfabetico(valores):
    """Orden alfabético sin distinguir mayúsculas ni tildes (Citroën con las C)."""
    return sorted(valores, key=lambda v: unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower())


def usd(x):
    return f"USD {x:,.0f}".replace(",", ".")


def num(x, dec=0):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# --- diseño: estilos de las tarjetas, el índice lateral y el encabezado ---------------------------------------
# Todo el CSS está acá. Los colores de las tarjetas siguen la paleta de arriba (azul = modelo, gris = piso,
# verde = autos/datos, naranja = hallazgo/alerta), así que un color significa lo mismo en tarjetas y gráficos.
st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg, #e8f1fc 0%, #f4f8fd 100%); }
.ficha { background:#1d4f91; color:#fff; border-radius:14px; padding:.8rem 1rem; font-size:.82rem; margin-bottom:1rem; line-height:1.5; }
.ficha ul { margin:.4rem 0 0 0; padding-left:1.1rem; }
.idx-titulo { font-weight:800; font-size:1.15rem; color:#1d4f91; margin:0 0 .6rem 0; }
.idx-grupo { font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:#6b7a90; margin:1rem 0 .35rem 2px; font-weight:700; }
a.idx { display:flex; align-items:center; gap:.65rem; background:#fff; border:1px solid #d6e4f7; border-left:5px solid var(--c);
        border-radius:12px; padding:.55rem .8rem; margin:.3rem 0; text-decoration:none !important; color:#25324a !important;
        font-weight:600; box-shadow:0 1px 3px rgba(30,70,130,.08); transition:all .15s ease; }
a.idx:hover { background:var(--c); color:#fff !important; transform:translateX(4px); box-shadow:0 4px 10px rgba(30,70,130,.22); }
a.idx .ic { font-size:1.2rem; }
a.idx small { display:block; font-weight:400; opacity:.75; font-size:.72rem; }
.hero { background:linear-gradient(120deg,#1d4f91 0%,#2a78d6 55%,#3fa7e0 100%); border-radius:18px; padding:1.6rem 1.8rem; color:#fff; margin-bottom:1rem; }
.hero h1 { color:#fff; margin:0; padding:0; font-size:2.4rem; }
.hero p { margin:.5rem 0 0 0; font-size:1.1rem; opacity:.95; }
.kpi { border-radius:16px; padding:1rem 1.2rem; color:#fff; box-shadow:0 3px 10px rgba(0,0,0,.12); height:100%; }
.kpi .v { font-size:2rem; font-weight:800; line-height:1.1; }
.kpi .l { font-size:.9rem; opacity:.95; margin-top:.25rem; }
.kpi .s { font-size:.75rem; opacity:.8; margin-top:.35rem; }
.dato { border-radius:14px; padding:.8rem 1rem; background:var(--bg); border:1px solid var(--bd); border-left:6px solid var(--bd); margin:.25rem 0 .75rem 0; }
.dato .v { font-size:1.5rem; font-weight:800; color:var(--tx); line-height:1.15; }
.dato .l { font-size:.85rem; color:#3d3d3a; }
.kicker { font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#1d4f91; margin:.4rem 0 -.6rem 0; }
.kicker.copa { color:#d95926; }
.concl { border-left:6px solid #1d4f91; background:#f3f7fc; border-radius:0 12px 12px 0; padding:.9rem 1.2rem; font-size:1.12rem; margin:.4rem 0 1rem 0; }
.panel { background:linear-gradient(160deg,#1d4f91 0%,#2f6fa8 100%); color:#fff; border-radius:18px; padding:1.2rem 1.4rem; box-shadow:0 6px 18px rgba(20,60,120,.3); }
.panel .t { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; opacity:.8; }
.panel .v { font-size:2.4rem; font-weight:800; line-height:1.1; }
.panel .r { font-size:1.02rem; font-weight:700; margin-top:.5rem; }
.panel .s { font-size:.82rem; opacity:.85; margin-top:.2rem; }
.panel table { width:100%; font-size:.85rem; margin-top:.4rem; border-collapse:collapse; }
.panel td { padding:.3rem .2rem; border-top:1px solid rgba(255,255,255,.25); }
</style>
""", unsafe_allow_html=True)

# Colores de las tarjetas: (fondo/borde, texto)
TARJ = {"azul": ("#2a78d6", "#2a78d6"), "verde": ("#1a9e77", "#1a9e77"), "naranja": ("#d95926", "#d95926"),
        "gris": ("#6b7280", "#6b7280"), "violeta": ("#7a4fd0", "#7a4fd0")}
TARJ_SUAVE = {"azul": ("#eaf2fd", "#2a78d6"), "verde": ("#e6f6f0", "#1a9e77"), "naranja": ("#fcefe8", "#d95926"),
              "gris": ("#f1f2f4", "#6b7280"), "violeta": ("#f1ecfb", "#7a4fd0")}


def kpi(valor, etiqueta, color="azul", pie=""):
    """Tarjeta grande de color sólido: el dato clave de la portada."""
    return (f'<div class="kpi" style="background:{TARJ[color][0]}"><div class="v">{valor}</div>'
            f'<div class="l">{etiqueta}</div><div class="s">{pie}</div></div>')


def dato(valor, etiqueta, color="azul"):
    """Cuadro suave que destaca un número dentro de una sección."""
    bg, c = TARJ_SUAVE[color]
    return (f'<div class="dato" style="--bg:{bg};--bd:{c};--tx:{c}"><div class="v">{valor}</div>'
            f'<div class="l">{etiqueta}</div></div>')


# Carga de lo que generó build_app.py. La app NO calcula nada del modelo: sólo lee app_data/.
@st.cache_data
def cargar():
    m = json.loads((DATA / "metricas.json").read_text(encoding="utf-8"))
    a, me, d = m["fecha_datos"].split("-")
    m["fecha_datos"] = f"{int(d)}/{int(me)}/{a}"
    autos = pd.read_csv(DATA / "autos.csv")
    autos["dif"] = autos["precio_usd"] / autos["tasacion_usd"] - 1
    autos["veredicto"] = np.select(
        [autos["dif"] < -0.10, autos["dif"] > 0.10], [VEREDICTOS[0], VEREDICTOS[2]], VEREDICTOS[1])
    autos["dif_txt"] = (autos["dif"] * 100).round(0).astype(int).astype(str) + " %"
    autos["auto"] = autos["marca"] + " " + autos["modelo"]
    autos["fuente"] = autos["fuente"].map(FUENTES).fillna(autos["fuente"])
    autos["trac"] = autos["traccion"].map(lambda v: T.normalizar_categorica(v, T.REGLAS_TRAC))
    return m, autos


@st.cache_resource
def cargar_modelo():
    return joblib.load(DATA / "modelo.joblib")


M, AUTOS = cargar()
MODELO = cargar_modelo()
# Atajos a los números que usan los títulos: test, comparación de modelos, piso, elegido y columnas probadas.
TEST = M["test"]
COMP = {c["modelo"]: c for c in M["comparacion"]}
PISO = COMP["Piso (mediana)"]
AJ = COMP["HistGradientBoosting ajustado"]
CP = {c["variante"]: c for c in M["columnas_probadas"]}
N_AUTOS = M["pasos"][-1]["filas"]

# =========================================================================
# PORTADA
# =========================================================================
# PORTADA: la idea central en una oración + tres números que la sostienen (error en test, piso, cantidad de autos).
# ÍNDICE: menú lateral para navegar. Cada link apunta a un ancla (`ancla("id")`) puesta justo antes de la sección.
def idx(destino, icono, titulo, sub, color):
    return (f'<a class="idx" href="#{destino}" target="_self" style="--c:{color}"><span class="ic">{icono}</span>'
            f'<span>{titulo}<small>{sub}</small></span></a>')


st.sidebar.markdown(
    '<div class="ficha"><b>Ciencia de Datos</b> · UTN FRM · 2026<br>Proyecto Integrador<br>'
    'Comisión <b>5K09</b> · Grupo <b>03</b>'
    '<ul><li>García, Nicolás</li><li>Ramos, Ignacio</li><li>Velasco, Victoria</li><li>Via, Tomás</li>'
    '<li>Villanueva, Sergio</li></ul></div>'
    '<div class="idx-titulo">🧭 Índice</div>'
    + idx("portada", "🏁", "Portada", "La idea en una línea", "#1d4f91")
    + '<div class="idx-grupo">El tallo · la historia</div>'
    + idx("datos", "🗂️", "1 · Los datos", "De dónde salen y cómo se limpian", "#2a78d6")
    + idx("hallazgos", "🔍", "2 · Lo que encontramos", "Año, km y ubicación", "#2a78d6")
    + idx("modelo", "🤖", "3 · El modelo", "Qué tan bien tasa", "#2a78d6")
    + '<div class="idx-grupo">La copa · ahora vos</div>'
    + idx("explora", "🔎", "Explorá los autos", "Filtrá y mirá cada uno", "#d95926")
    + idx("proba", "🧮", "Probá el tasador", "Cargá tu auto", "#d95926")
    + '<div class="idx-grupo">Más</div>'
    + idx("pie", "📦", "Repositorio", "Cómo regenerar todo", "#6b7280"),
    unsafe_allow_html=True)
# La copa de martini del modelo "tallo y copa": el tallo se lee en orden, la copa la maneja quien visita.
st.sidebar.markdown(
    '<div style="text-align:center;margin-top:1.2rem"><svg width="120" height="150" viewBox="0 0 120 150">'
    '<polygon points="10,10 110,10 60,80" fill="#cfe2f7" stroke="#2a78d6" stroke-width="3"/>'
    '<polygon points="22,22 98,22 60,70" fill="#2a78d6" opacity=".55"/>'
    '<line x1="60" y1="80" x2="60" y2="135" stroke="#d95926" stroke-width="4"/>'
    '<line x1="35" y1="138" x2="85" y2="138" stroke="#d95926" stroke-width="4" stroke-linecap="round"/>'
    '<circle cx="82" cy="30" r="5" fill="#d95926"/></svg></div>', unsafe_allow_html=True)
st.sidebar.caption("Dentro de cada sección, el desplegable *Cómo lo sabemos* tiene la evidencia técnica.")
st.caption("Proyecto integrador · Ciencia de Datos · UTN FRM · 2026 · grupo 5K09-03")
ancla("portada")
st.markdown(
    f'''<div class="hero"><div style="opacity:.85;font-size:.85rem;letter-spacing:.06em;margin-bottom:.3rem">CIENCIA DE DATOS · UTN FRM · 2026 · COMISIÓN 5K09 · GRUPO 03</div><h1>🚗 ¿Cuánto vale un auto usado?</h1>
    <p>Con la marca, el modelo, el año y la ficha técnica, el tasador estima el precio con un error típico de
    <b>{num(TEST['rel_mediano'] * 100, 1)} %</b>: <b>{num(PISO['mae'] / TEST['mae'], 1)} veces menos error</b>
    que tasar todo al precio mediano.</p></div>''', unsafe_allow_html=True)
st.markdown(
    f'<div class="concl">El precio de un usado se puede estimar con un error de <b>{usd(TEST["mae"])}</b> usando año, '
    f'kilometraje, motor, potencia, marca y modelo. El <b>año</b> es lo que más pesa y los autos de alta gama son los que '
    f'más se le escapan.</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(kpi(usd(TEST["mae"]), "Error promedio del tasador (test)", "azul",
                "En el 20 % de autos que el modelo nunca vio, medido una sola vez"), unsafe_allow_html=True)
c2.markdown(kpi(usd(PISO["mae"]), "Error si tasáramos todo a la mediana (piso)", "gris",
                "El número a vencer: sin modelo, mismo método de validación"), unsafe_allow_html=True)
c3.markdown(kpi(num(N_AUTOS), "Autos usados analizados", "verde",
                f"CarOne y DeRuedas, cosecha del {M['fecha_datos']}"), unsafe_allow_html=True)
st.caption("Integrantes: García Nicolás · Ramos Ignacio · Velasco Victoria · Via Tomás · Villanueva Sergio")
st.divider()

# =========================================================================
# TALLO 1 · LOS DATOS
# =========================================================================
# TALLO 1 · LOS DATOS. El título afirma algo con números calculados (mediana y percentil 90 del precio).
p10, p90 = M["precio_p10_p90"]
ancla("datos")
st.markdown('<div class="kicker">Los datos</div>', unsafe_allow_html=True)
st.subheader(f"Los datos · La mitad de los autos cuesta menos de {usd(M['precio_mediano'])}, "
             f"pero el 10 % más caro pasa los {usd(p90)}")
d1, d2, d3, d4 = st.columns(4)
d1.markdown(dato(num(N_AUTOS), "autos usados, ya limpios", "verde"), unsafe_allow_html=True)
d2.markdown(dato(usd(M["precio_mediano"]), "precio mediano", "azul"), unsafe_allow_html=True)
d3.markdown(dato(usd(p90), "piso del 10 % más caro", "naranja"), unsafe_allow_html=True)
d4.markdown(dato(num(M["ubicacion"]["n_ciudades"]), f"ciudades en {M['ubicacion']['n_provincias']} provincias", "violeta"),
            unsafe_allow_html=True)
# Histograma de precios: marca de barras, x = precio en tramos de USD 2.000, y = cantidad de autos (count).
# Se cortó en USD 80.000 para que se lea; la línea azul punteada es la mediana.
hist = alt.Chart(AUTOS[AUTOS["precio_usd"] <= 80000]).mark_bar(color=TINTA, opacity=0.85).encode(
    x=alt.X("precio_usd:Q", bin=alt.Bin(step=2000), axis=eje("Precio publicado (USD), tramos de 2.000")),
    y=alt.Y("count():Q", axis=eje("Cantidad de autos")),
    tooltip=[alt.Tooltip("precio_usd:Q", bin=alt.Bin(step=2000), title="Desde (USD)", format=",.0f"),
             alt.Tooltip("count():Q", title="Autos")],
)
mediana = alt.Chart(pd.DataFrame({"m": [M["precio_mediano"]]})).mark_rule(
    color=AZUL, strokeDash=[4, 4], size=2).encode(x="m:Q")
etiqueta = alt.Chart(pd.DataFrame({"m": [M["precio_mediano"]], "t": [f"mediana {usd(M['precio_mediano'])}"]})).mark_text(
    align="left", dx=6, dy=-120, color=AZUL, fontWeight="bold").encode(x="m:Q", text="t:N")
st.altair_chart((hist + mediana + etiqueta).properties(height=320), width="stretch")
st.markdown(
    f"**Qué mirar:** casi todos los autos están entre {usd(p10)} y {usd(p90)}, con una cola larga hacia "
    f"los caros (se cortó en USD 80.000 para que se lea). Por esa asimetría el modelo aprende sobre el "
    f"**logaritmo** del precio, y los errores se miden en porcentaje además de en dólares.")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**De dónde salen.** Un pipeline en **Airflow** (DAG `tp1_5K09_03_autos_mensual`, repositorio del grupo) junta los
avisos de dos sitios argentinos de autos usados y los ordena en capas, como un *medallion*:
"""
)
    st.markdown("""
- **1 · Fuentes.** Un sensor espera a que CarOne (API GraphQL) y DeRuedas (HTML) respondan. Si no responden, se usa un respaldo congelado de la última cosecha buena.
- **2 · Bronce (crudo).** Se guarda lo que devuelve cada sitio sin tocar: el JSON de CarOne por fecha y el HTML comprimido de cada aviso de DeRuedas, una tarea por marca. Si algo sale mal después, se reprocesa sin volver a bajar.
- **3 · Plata (limpio).** Se parsea el bronce a un CSV con un esquema único de 18 columnas: números argentinos a número, motor en litros, precio en su moneda, datos ilegibles vacíos (no en 0) y sin ids repetidos.
- **4 · Calidad.** La corrida falla si hay menos de 9.000 filas, faltan columnas, hay ids repetidos o más del 5 % de precios vacíos. Deja un calidad.json y un ZIP con el CSV y los logs.
- **5 · Esta app.** build_app.py toma el CSV de plata, aplica la limpieza de abajo, entrena el modelo con semilla 42 y genera los datos de la app.
""")
    st.markdown(f"""
Corre una vez por mes (o cuando se fuerza) y respeta una pausa entre avisos para no saturar los sitios. Esta app usa
la cosecha del **{M['fecha_datos']}**.

**Qué es una fila.** Una publicación de un auto usado: marca, modelo, versión, año, km, precio y moneda, motor,
potencia, caja, tracción, combustible y consumo. El precio en pesos se pasa a dólares a
{num(M['tipo_cambio'])} ARS/USD (`precio_usd`, el objetivo).

**Qué se filtró, paso por paso** (ya en el análisis, sobre la capa plata):
""")
    pasos = pd.DataFrame(M["pasos"]).rename(columns={"paso": "Paso", "filas": "Filas que quedan", "por_que": "Por qué"})
    pasos["Se van"] = (-pasos["Filas que quedan"].diff()).fillna(0).astype(int)
    st.dataframe(pasos[["Paso", "Filas que quedan", "Se van", "Por qué"]], hide_index=True, width="stretch")
    fuentes = ", ".join(f"{FUENTES.get(k, k)}: {num(v)} autos" for k, v in M["por_fuente"].items())
    st.markdown(f"""
Por fuente, después de limpiar: {fuentes}.

**Columnas que no entran al modelo.** `fuente`, `id_publicacion`, `url`, `fecha_ingesta` y `grupo` son datos de
la publicación, no del auto. `version` es texto libre y queda resumida en motor y potencia. `ubicacion` se
descartó en la Entrega 2 (casi no explica el precio). `precio` es el objetivo en su moneda original: dejarlo
sería una **fuga**, porque es la respuesta. Ninguna columna que entra se conoce *después* de publicar.
""")

st.divider()

# =========================================================================
# TALLO 2 · LO QUE ENCONTRAMOS
# =========================================================================
# TALLO 2 · LO QUE ENCONTRAMOS. El título usa la depreciación anual que calculó build_app.py (paso 11).
dep = M["depreciacion_anual"]
ancla("hallazgos")
st.markdown('<div class="kicker">Lo que encontramos</div>', unsafe_allow_html=True)
st.subheader(f"Lo que encontramos · Un auto pierde cerca del {num(dep * 100)} % de su valor por cada año de antigüedad")
h1, h2, h3 = st.columns(3)
h1.markdown(dato(f"−{num(dep * 100)} % por año", "lo que pierde un auto de su valor", "naranja"), unsafe_allow_html=True)
h2.markdown(dato(num(M["fuga"]["spearman_anio"], 2), "correlación año ↔ precio (Spearman)", "azul"), unsafe_allow_html=True)
h3.markdown(dato(f"+{usd(CP['sin fabricado_en']['delta'])}", "de error si el modelo no conoce el año", "verde"),
            unsafe_allow_html=True)
por_anio = pd.DataFrame(M["mediana_por_anio"]).rename(columns={"fabricado_en": "anio", "median": "mediana", "size": "autos"})
# Nube de puntos: un punto por auto (con un poquito de desplazamiento horizontal para que no se tapen).
# La línea es la mediana por año. Eje de precio logarítmico: una recta ahí significa pérdida porcentual constante.
puntos = alt.Chart(AUTOS[AUTOS["fabricado_en"].between(2000, 2026)]).transform_calculate(
    anio_j="datum.fabricado_en + (random() - 0.5) * 0.6").mark_circle(
    size=12, opacity=0.25, color=GRIS).encode(
    x=alt.X("anio_j:Q", title="Año de fabricación", scale=alt.Scale(domain=[1999.5, 2026.5]),
            axis=alt.Axis(format="d", tickMinStep=1)),
    y=alt.Y("precio_usd:Q", scale=ESCALA_PRECIO, axis=eje("Precio publicado (USD, escala log)", MARCAS_PRECIO)),
)
linea = alt.Chart(por_anio[por_anio["anio"].between(2000, 2026)]).mark_line(
    color=TINTA, strokeWidth=3, point=alt.OverlayMarkDef(color=TINTA, size=40)).encode(
    x="anio:Q", y=alt.Y("mediana:Q", scale=ESCALA_PRECIO),
    tooltip=[alt.Tooltip("anio:Q", title="Año"), alt.Tooltip("mediana:Q", title="Precio mediano", format=",.0f"),
             alt.Tooltip("autos:Q", title="Autos")],
)
st.altair_chart((puntos + linea).properties(height=360), width="stretch")
st.markdown(
    "**Qué mirar:** la línea es el precio mediano de cada año. En escala logarítmica es casi recta: el auto no "
    "pierde una cantidad fija de dólares por año sino un **porcentaje** parecido. La nube gris muestra que para "
    "un mismo año los precios varían mucho: lo que explica esa diferencia es la marca, el modelo y la ficha.")

with st.expander("Cómo lo sabemos"):
    km = CP["sin kilometraje"]
    ref = CP["referencia (todas)"]
    anio = CP["sin fabricado_en"]
    st.markdown(f"""
**Hipótesis que se sostuvo: el año es lo que más pesa en el precio.** Correlación de Spearman entre año y precio:
**{num(M['fuga']['spearman_anio'], 2)}**. Y cuando se saca el año del modelo, el error sube
**{usd(anio['delta'])}** ({num(anio['delta'] / ref['desvio'], 1)} desvíos), más que cualquier otra columna.

**Hipótesis que no se sostuvo: el kilometraje agrega mucho aparte del año.** Sacarlo sólo sube el error
**{usd(km['delta'])}** ({num(km['delta'] / ref['desvio'], 1)} desvíos, menos que el ruido de la validación): como los
km crecen con la antigüedad, lo que dicen ya lo dice el año.

**Cómo se construyó el gráfico.** Puntos: cada auto con año entre 2000 y 2026. Línea: mediana del precio en
dólares por año (no promedio, para que los autos de lujo no la arrastren). El eje vertical es logarítmico. El
{num(dep * 100)} % sale de ajustar una recta al logaritmo de esas medianas (años con 30 autos o más):
pendiente *b* (negativa), pérdida anual = 1 − e<sup>b</sup>.

**De dónde salen los números de las hipótesis.** Se entrenó el mismo boosting sacando una columna (o un grupo) por
vez, con la misma validación cruzada agrupada. Si el error sube menos que el desvío de la validación
({usd(ref['desvio'])}), esa columna no aporta algo que el resto no diga ya.
""", unsafe_allow_html=True)
    cp = pd.DataFrame(M["columnas_probadas"])
    cp = cp[cp["variante"] != "referencia (todas)"].sort_values("delta", ascending=False)
    cp["Variante"] = cp["variante"].str.replace("fabricado_en", "año").str.replace("potencia_hp", "potencia") \
        .str.replace("motor_lt", "motor").str.replace("consumo_lt_100km", "consumo") \
        .str.replace("transmision", "caja").str.replace("traccion", "tracción")
    cp["Error en validación"] = cp["mae"].map(usd)
    cp["Cuánto sube"] = cp["delta"].map(lambda v: "+" + usd(v))
    cp["¿Más que el ruido?"] = np.where(cp["delta"] > ref["desvio"], "Sí", "No")
    st.dataframe(cp[["Variante", "Error en validación", "Cuánto sube", "¿Más que el ruido?"]], hide_index=True,
                 width="stretch")


# ---- HIPÓTESIS DESCARTADA · LA UBICACIÓN --------------------------------------------------------------------
# Los números salen de build_app.py (paso 8: el boosting con provincia y con ciudad, y el η² de cada una) y el mapa
# de app_data/ubicaciones.csv (cada ciudad con sus coordenadas, cuántos autos tiene y su precio mediano).
UB = M["ubicacion"]
CPV = {c["variante"]: c for c in M["columnas_probadas"]}
c_prov, c_ciu = CPV["con provincia"], CPV["con provincia y ciudad"]
st.markdown("### 📍 Hipótesis que descartamos: «el precio depende de dónde se vende el auto»")
st.markdown(
    f"Se la dimos al modelo y **no movió la aguja**: con la provincia el error pasó de {usd(CPV['referencia (todas)']['mae'])} a "
    f"{usd(c_prov['mae'])}, y con provincia y ciudad a {usd(c_ciu['mae'])}. Son diferencias de dólares, mucho menores que el "
    f"ruido de la validación (± {usd(CPV['referencia (todas)']['desvio'])}). Por eso la ubicación **no entra al modelo**.")
u1, u2, u3, u4 = st.columns(4)
u1.markdown(dato(f"{num(UB['eta2_provincia'] * 100, 1)} %", "del precio lo explica la provincia", "violeta"), unsafe_allow_html=True)
u2.markdown(dato(f"{num(UB['eta2_marca'] * 100, 0)} %", "del precio lo explica la marca", "azul"), unsafe_allow_html=True)
u3.markdown(dato(f"{num(UB['pct_top_provincias'] * 100, 0)} %", "de los autos, en Mendoza, Córdoba y Buenos Aires", "naranja"),
            unsafe_allow_html=True)
u4.markdown(dato(f"{'+' if c_prov['delta'] >= 0 else '−'}{usd(abs(c_prov['delta']))}", "de error al sumar la provincia", "gris"),
            unsafe_allow_html=True)


@st.cache_data
def cargar_mapa():
    ub = pd.read_csv(DATA / "ubicaciones.csv")
    geo = json.loads((DATA / "provincias.geojson").read_text(encoding="utf-8"))
    return ub, geo


UBIC, GEO = cargar_mapa()
UBIC_OK = UBIC.dropna(subset=["latitud", "longitud"])
provs = alfabetico(UBIC_OK["provincia"].unique())
sel_prov = st.multiselect("Provincias en el mapa", provs, placeholder="Todas las provincias")
mapa_df = UBIC_OK[UBIC_OK["provincia"].isin(sel_prov)] if sel_prov else UBIC_OK
contorno = alt.Chart(alt.Data(values=GEO["features"])).mark_geoshape(fill="#eef1f5", stroke="#b8c0cc", strokeWidth=0.6)
puntos_mapa = alt.Chart(mapa_df).mark_circle(opacity=0.8, stroke="white", strokeWidth=0.7).encode(
    longitude="longitud:Q", latitude="latitud:Q",
    size=alt.Size("autos:Q", scale=alt.Scale(range=[40, 2200]), legend=alt.Legend(title="Autos")),
    color=alt.Color("mediana_usd:Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Precio mediano (USD)")),
    tooltip=[alt.Tooltip("ubicacion_corregida:N", title="Ciudad"), alt.Tooltip("autos:Q", title="Autos"),
             alt.Tooltip("mediana_usd:Q", title="Precio mediano (USD)", format=",.0f")])
col_mapa, col_barras = st.columns([3, 2])
col_mapa.altair_chart((contorno + puntos_mapa).project("mercator").properties(height=560), width="stretch")
por_prov = (UBIC.groupby("provincia").agg(autos=("autos", "sum")).reset_index().sort_values("autos", ascending=False).head(8))
barras = alt.Chart(por_prov).mark_bar(color=AZUL).encode(
    y=alt.Y("provincia:N", sort="-x", title=None), x=alt.X("autos:Q", title="Autos publicados"),
    tooltip=[alt.Tooltip("provincia:N", title="Provincia"), alt.Tooltip("autos:Q", title="Autos")])
col_barras.altair_chart(barras.properties(height=320, title="Dónde están los autos (8 provincias con más)"), width="stretch")
col_barras.caption(f"Cada círculo es una ciudad: tamaño = cantidad de autos, color = precio mediano. "
                   f"Faltan {UB['sin_coordenadas']} ciudades sin coordenadas (con pocos autos cada una).")
st.markdown(
    "**Qué mirar:** casi todo el mercado está en Mendoza y Córdoba, y los precios medianos de las ciudades grandes son "
    "parecidos. Lo que cambia el precio es *qué auto es* (marca, modelo, año), no *dónde* se publica.")

st.divider()

# =========================================================================
# TALLO 3 · EL MODELO
# =========================================================================
# TALLO 3 · EL MODELO. "veces" = cuántas veces menor es el error del modelo que el del piso.
veces = PISO["mae"] / TEST["mae"]
ancla("modelo")
st.markdown('<div class="kicker">El modelo</div>', unsafe_allow_html=True)
st.subheader(f"El modelo · El tasador se equivoca en promedio {usd(TEST['mae'])} por auto, "
             f"{num(veces, 1)} veces menos que tasar todo a la mediana")
m1, m2, m3, m4 = st.columns(4)
m1.markdown(dato(usd(TEST["mae"]), "error promedio en test", "azul"), unsafe_allow_html=True)
m2.markdown(dato(f"{num(TEST['rel_mediano'] * 100, 1)} %", "error relativo mediano", "verde"), unsafe_allow_html=True)
m3.markdown(dato(f"{num(veces, 1)}×", "mejor que el piso (mediana)", "naranja"), unsafe_allow_html=True)
m4.markdown(dato(usd(AJ["mae"]), f"validación cruzada (± {usd(AJ['desvio'])})", "gris"), unsafe_allow_html=True)
# Barras con el error de cada modelo en validación cruzada (con barra de ± 1 desvío). Gris = piso, azul = elegido.
comp = pd.DataFrame(M["comparacion"])
NOMBRES = {"Piso (mediana)": "Piso: tasar todo a la mediana", "Lineal (Ridge)": "Regresión lineal (Ridge)",
           "Random forest": "Random forest", "HistGradientBoosting": "Boosting sin ajustar",
           "HistGradientBoosting ajustado": "Boosting ajustado (elegido)"}
comp["tipo"] = np.where(comp["modelo"] == "HistGradientBoosting ajustado", "Elegido",
                        np.where(comp["modelo"] == "Piso (mediana)", "Piso", "Probado"))
comp["lo"], comp["hi"] = comp["mae"] - comp["desvio"], comp["mae"] + comp["desvio"]
comp["txt"] = comp["mae"].map(usd)
comp["nombre"] = comp["modelo"].map(NOMBRES)
orden = list(comp.sort_values("mae", ascending=False)["nombre"])
escala_tipo = alt.Scale(domain=["Piso", "Probado", "Elegido"], range=[GRIS, "#a9c8ee", AZUL])
base = alt.Chart(comp).encode(y=alt.Y("nombre:N", sort=orden, title=None, axis=alt.Axis(labelLimit=260)))
barras = base.mark_bar().encode(
    x=alt.X("mae:Q", axis=eje("Error promedio en validación cruzada por vehículo (USD, ± 1 desvío)")),
    color=alt.Color("tipo:N", scale=escala_tipo, legend=None),
    tooltip=[alt.Tooltip("nombre:N", title="Modelo"), alt.Tooltip("mae:Q", title="MAE", format=",.0f"),
             alt.Tooltip("desvio:Q", title="Desvío", format=",.0f")])
bigotes = base.mark_rule(color=TINTA).encode(x="lo:Q", x2="hi:Q")
textos = base.mark_text(align="left", dx=8, color=TINTA).encode(x="hi:Q", text="txt:N")
st.altair_chart((barras + bigotes + textos).properties(height=230), width="stretch")
st.markdown(
    f"**Qué mirar:** la barra gris es lo que se logra sin aprender nada. Los árboles (boosting) le ganan al modelo "
    f"lineal por varios desvíos. El elegido, en azul, es el boosting con la grilla ajustada: "
    f"{usd(AJ['mae'])} ± {num(AJ['desvio'])} en validación y **{usd(TEST['mae'])} en test**, que es el número honesto.")

with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
**Métrica: MAE en dólares.** Se lee directo ("me equivoco X dólares por auto") y el costo de equivocarse en una
tasación es lineal; el RMSE castigaría de más a los pocos autos de lujo. Acompaña el **error relativo mediano**:
**{num(TEST['rel_mediano'] * 100, 1)} %** en test.

**Piso:** tasar todo con la mediana da {usd(PISO['mae'])} ± {num(PISO['desvio'])}.

**Validación.** 80/20 con `GroupShuffleSplit` y `GroupKFold` de 5 pedazos, agrupando por vehículo
(marca + modelo + versión + año + km): el mismo auto publicado dos veces nunca queda de los dos lados.
{num(M['n_train'])} filas de entrenamiento y {num(M['n_test'])} de test. Semilla {M['semilla']}. Todo (limpieza
de categorías, imputación, escalado, `TargetEncoder` y logaritmo del objetivo) va adentro de un `Pipeline`.
""")
    st.markdown("**Comparación con media y desvío (validación cruzada agrupada)**")
    st.dataframe(pd.DataFrame({"Modelo": comp["nombre"], "Error promedio (MAE)": comp["mae"].map(usd),
                               "Desvío entre los 5 pedazos": comp["desvio"].map(lambda v: "± " + num(v))}),
                 hide_index=True, width="stretch")
    mp = M["mejores_parametros"]
    gr = pd.DataFrame(M["grilla"])
    st.markdown(f"**Ajuste del elegido.** `GridSearchCV` sobre {len(gr)} combinaciones, con la misma validación agrupada. "
                f"Ganó: {mp['max_iter']} árboles, {mp['max_leaf_nodes']} hojas por árbol, tasa de aprendizaje "
                f"{num(mp['learning_rate'], 2)} y al menos {mp['min_samples_leaf']} autos por hoja. Bajó el error de "
                f"{usd(COMP['HistGradientBoosting']['mae'])} (sin ajustar) a {usd(AJ['mae'])}. Las 5 mejores combinaciones:")
    top = gr.head(5).copy()
    top = pd.DataFrame({
        "Árboles": top["max_iter"], "Hojas": top["max_leaf_nodes"], "Tasa": top["learning_rate"].map(lambda v: num(v, 2)),
        "Autos por hoja": top["min_samples_leaf"], "Error validación": top["mae_val"].map(usd),
        "Desvío": top["desvio_val"].map(lambda v: num(v)), "Error en train": top["mae_train"].map(usd)})
    st.dataframe(top, hide_index=True, width="stretch")
    st.caption("Las cinco quedan dentro de un desvío entre sí: el ajuste mejora poco y el modelo no depende de una "
               "combinación exacta. El error en train es menor que en validación (hay algo de sobreajuste, esperable en boosting).")

    st.markdown(f"**Test, una sola vez: {usd(TEST['mae'])}.** Dónde se equivoca: el error en dólares crece con el precio, "
                "pero en porcentaje se mantiene parecido.")
    # Dónde se equivoca, como en el Colab: por cada tramo de precio del test, DOS barras con su propio eje:
    #   azul   = error promedio en dólares (MAE)            -> eje de la IZQUIERDA
    #   naranja = error relativo mediano, en %              -> eje de la DERECHA
    # xOffset las pone una al lado de la otra; resolve_scale(y="independent") da un eje vertical a cada una.
    tramo = pd.DataFrame(TEST["por_tramo"])
    orden_tramo = list(tramo["tramo"])
    tramo["rel_pct"] = tramo["rel"] * 100
    tramo["mae_txt"] = tramo["mae"].map(lambda v: num(v))
    tramo["rel_txt"] = tramo["rel"].map(lambda v: num(v * 100, 1) + "%")
    NAR_REL = "#C8602F"  # el naranja del gráfico del Colab
    series = ["MAE (USD)", "Error relativo mediano (%)"]
    desfase = alt.XOffset("serie:N", scale=alt.Scale(domain=series))
    x_tramo = alt.X("tramo:N", sort=orden_tramo, title="Tramo de mercado (USD)", axis=alt.Axis(labelAngle=0))
    base_mae = alt.Chart(tramo.assign(serie=series[0])).encode(x=x_tramo, xOffset=desfase)
    base_rel = alt.Chart(tramo.assign(serie=series[1])).encode(x=x_tramo, xOffset=desfase)
    y_mae = alt.Y("mae:Q", axis=alt.Axis(title="MAE absoluto (USD)", titleColor=AZUL, labelColor=AZUL,
                                         labelExpr=PUNTOS_MIL, orient="left"))
    y_rel = alt.Y("rel_pct:Q", axis=alt.Axis(title="Error relativo mediano (%)", titleColor=NAR_REL,
                                             labelColor=NAR_REL, orient="right"))
    g_mae = base_mae.mark_bar(color=AZUL).encode(y=y_mae, tooltip=[
        alt.Tooltip("tramo:N", title="Tramo"), alt.Tooltip("mae:Q", title="MAE (USD)", format=",.0f")])
    t_mae = base_mae.mark_text(dy=-6, color=TINTA).encode(y="mae:Q", text="mae_txt:N")
    g_rel = base_rel.mark_bar(color=NAR_REL).encode(y=y_rel, tooltip=[
        alt.Tooltip("tramo:N", title="Tramo"), alt.Tooltip("rel_pct:Q", title="Error relativo mediano (%)", format=".1f")])
    t_rel = base_rel.mark_text(dy=-6, color=TINTA).encode(y="rel_pct:Q", text="rel_txt:N")
    st.altair_chart(alt.layer(g_mae + t_mae, g_rel + t_rel).resolve_scale(y="independent").properties(height=300),
                    width="stretch")
    st.caption("Cada tramo tiene ~20 % de los autos de test. Azul (eje izquierdo): cuántos dólares se equivoca en promedio. "
               "Naranja (eje derecho): cuánto se equivoca en porcentaje del precio.")

    te = pd.DataFrame(M["target_encoder"])
    sm = TEST["sesgo_moneda"]
    st.markdown(f"""
**Dos confusiones concretas.**
- Los autos **publicados en dólares** quedan tasados en promedio {usd(abs(sm.get('USD', 0)))} por debajo de su precio;
  los publicados en pesos casi no tienen sesgo ({usd(abs(sm.get('ARS', 0)))}). El dólar implícito del mercado de autos
  no es el tipo de cambio que usamos para convertir.
- Los **modelos raros** se tasan peor: el `TargetEncoder` aprende de autos del mismo modelo y, si hay pocos, aprende poco.
""")
    st.dataframe(pd.DataFrame({"Avisos de ese modelo en train": te["avisos_en_train"], "Autos en test": te["autos_test"],
                               "Error promedio": te["mae"].map(usd),
                               "Error relativo mediano": te["rel"].map(lambda v: num(v * 100, 1) + " %")}),
                 hide_index=True, width="stretch")
    # En qué se apoya: importancia por permutación (sólo columnas que importan, nombres en castellano).
    imp = pd.DataFrame(M["importancia"])
    imp = imp[imp["sube_mae"] > 0].assign(columna=lambda d: d["columna"].map(COL_ES).fillna(d["columna"]))
    st.markdown("**En qué se apoya.** Importancia por permutación en test: cuánto sube el error al desordenar cada columna.")
    g_imp = alt.Chart(imp).mark_bar(color=AZUL).encode(
        x=alt.X("sube_mae:Q", axis=eje("Cuánto sube el error (USD) al desordenar la columna")),
        y=alt.Y("columna:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        tooltip=[alt.Tooltip("columna:N", title="Columna"), alt.Tooltip("sube_mae:Q", title="Sube (USD)", format=",.0f")])
    st.altair_chart(g_imp.properties(height=26 * len(imp)), width="stretch")
    st.caption(f"Si se desordena el año, el error en test sube {usd(imp['sube_mae'].iloc[0])}: es lo que más usa el modelo, "
               "igual que lo que mostró la sección anterior.")
    fu = M["fuga"]
    st.markdown(f"""
**Fugas descartadas.** El año pesa mucho pero se conoce antes de publicar (Spearman {num(fu['spearman_anio'], 2)}, lejos
de 1). Validar al azar en lugar de por vehículo da {usd(fu['mae_azar'])} contra {usd(fu['mae_agrupado'])}: unos
{usd(fu['mae_agrupado'] - fu['mae_azar'])} más optimista, chico porque sólo {num(fu['repetidos'])} de
{num(fu['vehiculos_train'])} vehículos se repiten. El `TargetEncoder` tampoco copia precios: si lo hiciera, los
modelos raros saldrían *mejor*, y salen peor (tabla de arriba).
""")

st.divider()

# =========================================================================
# COPA · EXPLORÁ Y PROBÁ
# =========================================================================
# COPA · Explorá (panorama + filtros + detalle) y Probá el tasador (caso nuevo).
ancla("explora")
ancla("proba")
st.markdown('<div class="kicker copa">La copa · tu turno</div>', unsafe_allow_html=True)
st.header("Ahora vos")
tab_exp, tab_probar = st.tabs(["🔎 Explorá los autos", "🧮 Probá el tasador"])

with tab_exp:
    st.markdown("Cada punto es un auto publicado. El color compara su precio con lo que dijo el tasador "
                "(una tasación hecha **sin haber visto ese auto**). Pasá el mouse para ver el detalle o hacé clic en un punto.")
    # Controles que achican el universo. Cada uno filtra el DataFrame `d` que alimenta tanto las 4 métricas
    # de arriba como el gráfico: marca (multiselect), año y precio (sliders) y caja (radio).
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4])
    marcas = alfabetico(AUTOS["marca"].unique())
    sel_marcas = f1.multiselect("Marca", marcas, placeholder="Todas las marcas")
    a_min, a_max = int(AUTOS["fabricado_en"].min()), int(AUTOS["fabricado_en"].max())
    # el slider abarca TODOS los años de los datos (por defecto, el rango completo)
    rango_anio = f2.slider("Año", a_min, a_max, (a_min, a_max))
    pmax = int(np.ceil(AUTOS["precio_usd"].quantile(0.99) / 1000) * 1000)
    rango_precio = f3.slider("Precio publicado (USD)", 0, pmax, (0, pmax), step=1000)
    CAJAS = {"Todas": None, "Manual": "manual", "Automática": "automatico"}
    caja = f4.radio("Caja", list(CAJAS), horizontal=False)

    d = AUTOS[AUTOS["fabricado_en"].between(*rango_anio) & AUTOS["precio_usd"].between(*rango_precio)]
    if sel_marcas:
        d = d[d["marca"].isin(sel_marcas)]
    if CAJAS[caja]:
        d = d[d["caja"] == CAJAS[caja]]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Autos en la vista", num(len(d)))
    k2.metric("Precio mediano", usd(d["precio_usd"].median()) if len(d) else "–")
    k3.metric("Error mediano del tasador en estos autos",
              f"{num((d['precio_usd'] / d['tasacion_usd'] - 1).abs().median() * 100, 1)} %" if len(d) else "–",
              help="Diferencia típica entre el precio publicado y la tasación. Más alta = el tasador se fía menos en ese grupo.")
    k4.metric("Más baratos que la tasación", f"{num((d['veredicto'] == VEREDICTOS[0]).mean() * 100)} %" if len(d) else "–",
              help="Autos publicados más de 10 % por debajo de lo que estima el modelo: posibles oportunidades (o autos con algún problema).")

    # Detalle a pedido: al hacer clic en un punto se guarda su fila y abajo aparece la ficha con la tasación del modelo.
    clic = alt.selection_point(name="auto", fields=["idx"], on="click", empty=False)
    dd = d.reset_index(names="idx")
    jitter = alt.Chart(dd).transform_calculate(anio_j="datum.fabricado_en + (random() - 0.5) * 0.7")
    disp = jitter.mark_circle(size=28, opacity=0.65).encode(
        x=alt.X("anio_j:Q", title="Año de fabricación", axis=alt.Axis(format="d"),
                scale=alt.Scale(domain=[rango_anio[0] - 0.5, rango_anio[1] + 0.5])),
        y=alt.Y("precio_usd:Q", scale=ESCALA_PRECIO, axis=eje("Precio publicado (USD, escala log)", MARCAS_PRECIO)),
        color=alt.Color("veredicto:N", scale=ESCALA_VEREDICTO, title=None,
                        legend=alt.Legend(orient="top")),
        size=alt.condition(clic, alt.value(160), alt.value(28)),
        tooltip=[alt.Tooltip("auto:N", title="Auto"), alt.Tooltip("version:N", title="Versión"),
                 alt.Tooltip("fabricado_en:Q", title="Año"),
                 alt.Tooltip("kilometraje:Q", title="Km", format=",.0f"),
                 alt.Tooltip("precio_usd:Q", title="Precio publicado (USD)", format=",.0f"),
                 alt.Tooltip("tasacion_usd:Q", title="Tasación del modelo (USD)", format=",.0f"),
                 alt.Tooltip("dif_txt:N", title="Diferencia vs tasación"),
                 alt.Tooltip("fuente:N", title="Fuente")],
    ).add_params(clic).properties(height=460)
    evento = st.altair_chart(disp, width="stretch", on_select="rerun", key="explora")

    elegido = None
    try:
        sel = evento.selection.get("auto") if evento else None
        if sel:
            elegido = AUTOS.loc[sel[0]["idx"]]
    except Exception:
        elegido = None
    if elegido is not None:
        st.markdown(f"#### {elegido['marca']} {elegido['modelo']} · {elegido['version']}")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precio publicado", usd(elegido["precio_usd"]),
                  help=f"Original: {num(elegido['precio'])} {elegido['moneda']}")
        e2.metric("Tasación del modelo", usd(elegido["tasacion_usd"]),
                  delta=f"{num(elegido['dif'] * 100)} % vs tasación", delta_color="off")
        e3.metric("Año · km", f"{int(elegido['fabricado_en'])} · {num(elegido['kilometraje'])} km")
        e4.metric("Avisos de este modelo en entrenamiento", num(elegido["avisos_modelo"]))
        if elegido["avisos_modelo"] < 10:
            st.warning("El tasador vio menos de 10 autos de este modelo: su tasación es poco confiable.")
        if isinstance(elegido["url"], str):
            st.markdown(f"[Ver la publicación original]({elegido['url']}) · fuente: {elegido['fuente']}")
    else:
        st.caption("Tip: hacé clic en un punto para ver la ficha completa de ese auto.")

with tab_probar:
    st.markdown("Cargá un auto y el tasador estima su precio. Sólo se pueden elegir marcas y modelos que el "
                "modelo vio; si un número se sale de lo habitual, la app te avisa.")
    # Rangos que vio el modelo (calculados en build_app.py, paso 12) para avisar si el usuario carga algo raro.
    R = M["rangos"]
    REAL = {c: AUTOS.loc[AUTOS[c] > 0, c].quantile([0.01, 0.99]).tolist() for c in
            ["fabricado_en", "kilometraje", "motor_lt", "potencia_hp"]}
    cuenta = AUTOS.groupby(["marca", "modelo"]).size().rename("n").reset_index()
    c_a, c_b = st.columns(2)
    marcas_p = alfabetico(AUTOS["marca"].unique())
    marca = c_a.selectbox("Marca", marcas_p, index=marcas_p.index("Volkswagen") if "Volkswagen" in marcas_p else 0)
    opciones = cuenta[cuenta["marca"] == marca].copy()
    opciones = opciones.set_index("modelo").loc[alfabetico(opciones["modelo"])].reset_index()  # A-Z
    modelo = c_b.selectbox("Modelo", opciones["modelo"].tolist(),
                           format_func=lambda m: f"{m}  ({int(opciones.set_index('modelo').loc[m, 'n'])} avisos)")
    ref_m = AUTOS[(AUTOS["marca"] == marca) & (AUTOS["modelo"] == modelo)]

    def tipico(col, por_defecto):
        v = ref_m[col].median()
        return float(v) if pd.notna(v) else por_defecto

    c1, c2, c3 = st.columns(3)
    anio = c1.slider("Año de fabricación", int(R["fabricado_en"][2]), int(R["fabricado_en"][3]),
                     int(np.clip(round(tipico("fabricado_en", 2018)), R["fabricado_en"][2], R["fabricado_en"][3])))
    km_v = c2.number_input("Kilometraje", min_value=0, max_value=500000,
                           value=int(round(tipico("kilometraje", 80000), -3)), step=5000)
    motor = c3.number_input("Motor (litros)", min_value=0.6, max_value=float(R["motor_lt"][3]),
                            value=float(np.clip(round(tipico("motor_lt", 1.6), 1), 0.6, R["motor_lt"][3])), step=0.1, format="%.1f")
    c4, c5, c6 = st.columns(3)
    hp = c4.number_input("Potencia (HP)", min_value=40.0, max_value=float(R["potencia_hp"][3]),
                         value=float(np.clip(round(tipico("potencia_hp", 110)), 40, R["potencia_hp"][3])), step=5.0, format="%.0f")
    caja_p = c5.selectbox("Caja", ["Automática", "Manual"],
                          index=0 if (ref_m["caja"] == "automatico").mean() > 0.5 else 1)
    comb_opts = {"Diésel": "diesel", "Eléctrico": "electrico", "GNC": "gnc", "Híbrido": "hibrido", "Nafta": "nafta"}  # A-Z
    comb_moda = ref_m["comb"].mode()
    comb_def = list(comb_opts.values()).index(comb_moda.iloc[0]) if len(comb_moda) and comb_moda.iloc[0] in comb_opts.values() else 0
    comb = c6.selectbox("Combustible", list(comb_opts.keys()), index=comb_def)
    TRACS = {"Delantera": "delantera", "Trasera": "trasera", "4x4": "4x4"}
    trac_moda = ref_m["trac"].mode()
    trac_def = list(TRACS.values()).index(trac_moda.iloc[0]) if len(trac_moda) and trac_moda.iloc[0] in TRACS.values() else 0
    trac = st.radio("Tracción", list(TRACS), index=trac_def, horizontal=True)

    # El caso nuevo, con las MISMAS columnas que recibió el modelo al entrenar (sin precio). IngenieriaAutos lo limpia
    # igual que a los datos de entrenamiento. Consumo en blanco = el modelo usa la mediana. "moneda" y "clave" son relleno.
    caso = pd.DataFrame([{
        "marca": marca, "modelo": modelo, "fabricado_en": anio, "kilometraje": km_v, "moneda": "USD",
        "motor_lt": motor, "potencia_hp": hp, "transmision": caja_p, "traccion": trac,
        "combustible": comb_opts[comb], "consumo_lt_100km": np.nan, "clave": "",
    }])
    # Predicción del modelo guardado (el mismo que dio el test).
    precio = float(MODELO.predict(caso)[0])
    n_train_mod = int(ref_m["avisos_modelo"].iloc[0]) if len(ref_m) else 0
    TE = {r["avisos_en_train"]: r["rel"] for r in M["target_encoder"]}
    tramo_te = "1–9" if n_train_mod < 10 else "10–49" if n_train_mod < 50 else "50+"
    # Rango probable: error relativo mediano del test para modelos con tantos avisos como éste
    # (los modelos raros tienen más error, ver la tabla "Avisos de ese modelo en train").
    margen = TE.get(tramo_te, TEST["rel_mediano"])

    # Avisos: modelo con pocos autos, o valores fuera del rango (percentil 1 a 99) de lo que vio el modelo.
    avisos = []
    n_mod = len(ref_m)
    if n_mod < 10:
        avisos.append(f"Hay sólo {n_mod} avisos de {marca} {modelo}: con tan pocos autos el error en test fue de "
                      "varios miles de dólares. Tomalo como orientación.")
    for col, v, nombre in [("fabricado_en", anio, "El año"), ("kilometraje", km_v, "El kilometraje"),
                           ("motor_lt", motor, "El motor"), ("potencia_hp", hp, "La potencia")]:
        p01, p99 = REAL[col]
        if v < p01 or v > p99:
            avisos.append(f"{nombre} ({num(v, 1 if col == 'motor_lt' else 0)}) está fuera del rango donde está el 98 % "
                          f"de los autos que vio el modelo ({num(p01, 1 if col == 'motor_lt' else 0)} a "
                          f"{num(p99, 1 if col == 'motor_lt' else 0)}).")
    if anio >= 2025 and km_v > 60000:
        avisos.append("Un auto tan nuevo con tantos kilómetros es raro en los datos.")

    r1, r2 = st.columns([1, 1.4])
    with r1:
        # Percentil: de todos los autos publicados, qué porcentaje cuesta menos que esta tasación.
        pct = float((AUTOS["precio_usd"] < precio).mean() * 100)
        # Comparables: los 4 avisos del mismo modelo más parecidos en año y km (con su precio real y la tasación del modelo).
        comp_html = ""
        if len(ref_m):
            cerca = ref_m.assign(dist=(ref_m["fabricado_en"] - anio).abs() * 10000 + (ref_m["kilometraje"] - km_v).abs() / 10).nsmallest(4, "dist")
            filas = "".join(
                f"<tr><td>{r.marca} {r.modelo} · {int(r.fabricado_en)} · {num(r.kilometraje)} km</td>"
                f"<td style='text-align:right'>{usd(r.precio_usd)} real<br>{usd(r.tasacion_usd)} tasado</td></tr>"
                for r in cerca.itertuples())
            comp_html = f"<div class='r' style='margin-top:.9rem'>Comparables en los datos</div><table>{filas}</table>"
        st.markdown(
            f"<div class='panel'><div class='t'>Tasación estimada</div><div class='v'>{usd(precio)}</div>"
            f"<div class='s'>≈ ARS {num(precio * M['tipo_cambio'])} (cotización de referencia {num(M['tipo_cambio'])} ARS/USD)</div>"
            f"<div class='r'>Rango probable: {usd(precio * (1 - margen))} a {usd(precio * (1 + margen))} (± {num(margen * 100, 1)} %)</div>"
            f"<div class='s'>Error mediano en test para modelos con {tramo_te} avisos; {marca} {modelo} tiene {n_train_mod} en entrenamiento.</div>"
            f"<div class='r'>Percentil {num(pct)} del mercado ({num(len(AUTOS))} autos)</div>{comp_html}</div>",
            unsafe_allow_html=True)
        for a_ in avisos:
            st.warning(a_)
    with r2:
        if len(ref_m):
            otros = alt.Chart(ref_m).mark_circle(color=GRIS, opacity=0.6, size=40).encode(
                x=alt.X("fabricado_en:Q", title="Año", axis=alt.Axis(format="d"),
                        scale=alt.Scale(zero=False, padding=12)),
                y=alt.Y("precio_usd:Q", scale=alt.Scale(zero=False), axis=eje("Precio (USD)")),
                tooltip=[alt.Tooltip("version:N", title="Versión"), alt.Tooltip("fabricado_en:Q", title="Año"),
                         alt.Tooltip("kilometraje:Q", title="Km", format=",.0f"),
                         alt.Tooltip("precio_usd:Q", title="Precio publicado", format=",.0f")])
            tuyo = alt.Chart(pd.DataFrame({"a": [anio], "p": [precio]})).mark_point(
                shape="diamond", size=320, filled=True, color=AZUL).encode(
                x="a:Q", y="p:Q", tooltip=[alt.Tooltip("p:Q", title="Tasación de tu auto", format=",.0f")])
            st.altair_chart((otros + tuyo).properties(
                height=300, title=f"Tu auto (rombo azul) entre los {n_mod} {marca} {modelo} publicados"),
                width="stretch")

ancla("pie")
st.divider()
st.caption(f"Datos: cosecha del {M['fecha_datos']} del pipeline de Airflow del grupo. Modelo: HistGradientBoosting "
           f"con objetivo logarítmico, semilla {M['semilla']}, regenerado con build_app.py.")
