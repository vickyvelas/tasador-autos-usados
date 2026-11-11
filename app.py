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

st.set_page_config(page_title="Tasador de autos usados · 5K09 grupo 03", layout="wide")
alt.data_transformers.disable_max_rows()

# --- colores: significan lo mismo en toda la app ---------------------------
# Una paleta para toda la app: cada color significa siempre lo mismo (tallo y copa).
AZUL = "#2c5f8f"      # el modelo / la tasación
GRIS = "#9a9a9a"      # referencia (piso, otros autos, datos sin modelo)
TINTA = "#3d3d3a"     # precios reales
VERDE = "#3f7d6b"     # más barato que la tasación
NARANJA = "#b5603a"   # más caro que la tasación
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
[data-testid="stSidebar"] { background:#f4f6f9; }
.ficha { background:#22364d; color:#fff; border-radius:10px; padding:.55rem .8rem; font-size:.74rem; margin-bottom:.6rem; line-height:1.4; }
.ficha ul { margin:.3rem 0 0 0; padding-left:1rem; font-size:.72rem; }
.ficha li { margin:0; }
.idx-titulo { font-weight:700; font-size:.85rem; letter-spacing:.06em; text-transform:uppercase; color:#2c3e50; margin:0 0 .3rem 0; }
.idx-grupo { font-size:.66rem; letter-spacing:.08em; text-transform:uppercase; color:#6b7a90; margin:.6rem 0 .2rem 2px; font-weight:700; }
a.idx { display:flex; align-items:center; gap:.5rem; font-size:.86rem; background:#fff; border:1px solid #e1e6ec; border-left:3px solid var(--c);
        border-radius:8px; padding:.3rem .65rem; margin:.15rem 0; text-decoration:none !important; color:#25324a !important;
        font-weight:600; box-shadow:none; transition:all .15s ease; }
a.idx.act { background:var(--c); color:#fff !important; font-weight:700; }
a.idx.act small { opacity:.9; }
a.idx:hover { background:#eef2f6; transform:translateX(3px); }
a.idx .ic { font-size:1.2rem; }
a.idx small { display:none; }
.hero { background:#22364d; border-radius:10px; padding:1.6rem 1.8rem; color:#fff; margin-bottom:1rem; }
.hero h1 { color:#fff; margin:0; padding:0; font-size:2.2rem; font-weight:700; }
.hero p { margin:.5rem 0 0 0; font-size:1.1rem; opacity:.95; }
.kpi { border-radius:10px; padding:1rem 1.2rem; color:#fff; box-shadow:none; height:100%; }
.kpi .v { font-size:1.9rem; font-weight:700; line-height:1.1; }
.kpi .l { font-size:.9rem; opacity:.95; margin-top:.25rem; }
.kpi .s { font-size:.75rem; opacity:.8; margin-top:.35rem; }
.dato { border-radius:8px; padding:.7rem 1rem; background:var(--bg); border:1px solid #e1e6ec; border-left:4px solid var(--bd); margin:.25rem 0 .75rem 0; }
.dato .v { font-size:1.4rem; font-weight:700; color:var(--tx); line-height:1.15; }
.dato .l { font-size:.85rem; color:#3d3d3a; }
@media (max-width: 820px) {
  .hero { padding:1rem 1.1rem; } .hero h1 { font-size:1.5rem; } .hero p { font-size:.95rem; }
  .kpi .v { font-size:1.5rem; } .kpi { margin-bottom:.5rem; } .dato .v { font-size:1.15rem; }
  .concl { font-size:1rem; padding:.7rem .9rem; } .panel .v { font-size:1.8rem; }
}
.kicker { font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; color:#2c5f8f; margin:.4rem 0 -.6rem 0; }
.kicker.copa { color:#b5603a; }
.concl { border-left:4px solid #2c5f8f; background:#f6f8fa; border-radius:0 8px 8px 0; padding:.9rem 1.2rem; font-size:1.12rem; margin:.4rem 0 1rem 0; }
.explica { background:#f4f7fb; border-left:5px solid #2c5f8f; border-radius:8px; padding:.75rem 1rem; font-size:.92rem; line-height:1.45; }
.explica .t { display:block; font-weight:700; color:#2c5f8f; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; margin-top:.55rem; }
.explica .t:first-child { margin-top:0; }
.explica p { margin:.15rem 0 0 0; }
.alerta {background:#fdecea; border:2px solid #c0392b; color:#8a1c12; border-radius:10px; padding:.6rem .8rem;
  margin-bottom:.6rem; font-size:.9rem; font-weight:500; animation:parp 1.4s ease-in-out 3;}
@keyframes parp {50% {box-shadow:0 0 0 6px rgba(192,57,43,.35);}}
.panel { background:#22364d; color:#fff; border-radius:10px; padding:.9rem 1.1rem; }
.panel .t { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; opacity:.8; }
.panel .v { font-size:2rem; font-weight:700; line-height:1.1; }
.panel .r { font-size:.95rem; font-weight:700; margin-top:.35rem; }
.panel .s { font-size:.82rem; opacity:.85; margin-top:.2rem; }
.panel .nota { background:rgba(255,255,255,.12); border-radius:8px; padding:.35rem .6rem; font-size:.72rem; margin:.4rem 0; }
.panel .barra { position:relative; height:8px; background:rgba(255,255,255,.28); border-radius:6px; margin:.8rem 0 .3rem 0; }
.panel .punto { position:absolute; top:-5px; width:18px; height:18px; border-radius:50%; background:#fff; border:3px solid #b5603a; transform:translateX(-50%); }
.panel .ext { display:flex; justify-content:space-between; font-size:.72rem; opacity:.8; margin-bottom:.5rem; }
.panel table { width:100%; font-size:.78rem; margin-top:.2rem; border-collapse:collapse; }
.panel td { padding:.15rem .2rem; border-top:1px solid rgba(255,255,255,.25); }
</style>
""", unsafe_allow_html=True)

# Colores de las tarjetas: (fondo/borde, texto)
TARJ = {"azul": ("#2c5f8f", "#2c5f8f"), "verde": ("#3f7d6b", "#3f7d6b"), "naranja": ("#b5603a", "#b5603a"),
        "gris": ("#5f6b7a", "#5f6b7a"), "violeta": ("#6a5a8c", "#6a5a8c")}
TARJ_SUAVE = {"azul": ("#f3f6fa", "#2c5f8f"), "verde": ("#f2f7f5", "#3f7d6b"), "naranja": ("#faf4f0", "#b5603a"),
              "gris": ("#f4f5f7", "#5f6b7a"), "violeta": ("#f5f3f8", "#6a5a8c")}


def kpi(valor, etiqueta, color="azul", pie=""):
    """Tarjeta grande de color sólido: el dato clave de la portada."""
    return (f'<div class="kpi" style="background:{TARJ[color][0]}"><div class="v">{valor}</div>'
            f'<div class="l">{etiqueta}</div><div class="s">{pie}</div></div>')


def explica(que, por_que, mirar, conclusion):
    """Cuadro que le explica al usuario, paso a paso, qué muestra el gráfico y por qué lo elegimos."""
    return ('<div class="explica">'
            f'<span class="t">Qué estás viendo</span><p>{que}</p>'
            f'<span class="t">Por qué elegimos este gráfico</span><p>{por_que}</p>'
            f'<span class="t">Qué mirar</span><p>{mirar}</p>'
            f'<span class="t">Qué nos dice</span><p><b>{conclusion}</b></p></div>')


def dato(valor, etiqueta, color="azul"):
    """Cuadro suave que destaca un número dentro de una sección."""
    bg, c = TARJ_SUAVE[color]
    return (f'<div class="dato" style="--bg:{bg};--bd:{c};--tx:{c}"><div class="v">{valor}</div>'
            f'<div class="l">{etiqueta}</div></div>')


def sticker(texto, x, y, color=None, align="left", size=13, valor=None):
    """Etiqueta pegada DENTRO de un gráfico. x e y son encodings de Altair (alt.X, alt.value(px)...). Si x es un dato, `valor` es
    su número. Se dibuja dos veces: una con halo blanco (para leerse sobre barras y puntos) y otra encima con el color."""
    base = alt.Chart(pd.DataFrame({"t": [texto], "m": [valor if valor is not None else 0]})).encode(x=x, y=y, text="t:N")
    kw = dict(align=align, baseline="middle", fontSize=size, fontWeight="bold")
    return base.mark_text(stroke="white", strokeWidth=5, color="white", **kw) + base.mark_text(color=color or TINTA, **kw)


def px(v):
    return alt.value(v)


# Carga de lo que generó build_app.py. La app NO calcula nada del modelo: sólo lee app_data/.
@st.cache_data
def cargar(version):   # version: fecha de los archivos, para no servir una copia vieja
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


M, AUTOS = cargar((DATA / "metricas.json").stat().st_mtime_ns + (DATA / "autos.csv").stat().st_mtime_ns)
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
    return (f'<a class="idx" href="#{destino}" target="_self" style="--c:{color}">'
            f'<span>{titulo}<small>{sub}</small></span></a>')


st.sidebar.markdown(
    '<div class="ficha"><b>Ciencia de Datos</b> · UTN FRM · 2026<br>Proyecto Integrador<br>'
    'Comisión <b>5K09</b> · Grupo <b>03</b>'
    '<ul><li>García, Nicolás</li><li>Ramos, Ignacio</li><li>Velasco, Victoria</li><li>Via, Tomás</li>'
    '<li>Villanueva, Sergio</li></ul></div>'
    '<div class="idx-titulo">Índice</div>'
    + idx("portada", "", "Portada", "La idea en una línea", "#22364d")
    + '<div class="idx-grupo">El tallo · la historia</div>'
    + idx("datos", "", "1 · Los datos", "De dónde salen y cómo se limpian", "#2c5f8f")
    + idx("hallazgos", "", "2 · Lo que encontramos", "Año, km y ubicación", "#2c5f8f")
    + idx("modelo", "", "3 · El modelo", "Qué tan bien tasa", "#2c5f8f")
    + '<div class="idx-grupo">La copa · ahora vos</div>'
    + idx("explora", "", "Explorá y probá", "Filtrá los autos y tasá el tuyo", "#b5603a")
    + '<div class="idx-grupo">Más</div>'
    + idx("pie", "", "Repositorio", "Cómo regenerar todo", "#6b7280"),
    unsafe_allow_html=True)
# La copa de martini del modelo "tallo y copa": el tallo se lee en orden, la copa la maneja quien visita.
st.sidebar.markdown(
    '<div class="copa-menu" style="text-align:center;margin-top:.6rem"><svg width="70" height="88" viewBox="0 0 120 150">'
    '<polygon points="10,10 110,10 60,80" fill="#dde6ef" stroke="#2c5f8f" stroke-width="3"/>'
    '<polygon points="22,22 98,22 60,70" fill="#2c5f8f" opacity=".5"/>'
    '<line x1="60" y1="80" x2="60" y2="135" stroke="#b5603a" stroke-width="4"/>'
    '<line x1="35" y1="138" x2="85" y2="138" stroke="#b5603a" stroke-width="4" stroke-linecap="round"/>'
    '<circle cx="82" cy="30" r="5" fill="#b5603a"/></svg></div>', unsafe_allow_html=True)
st.sidebar.caption("Dentro de cada sección, el desplegable *Cómo lo sabemos* tiene la evidencia técnica.")
st.caption("Integrantes: García Nicolás · Ramos Ignacio · Velasco Victoria · Via Tomás · Villanueva Sergio")
ancla("portada")
st.markdown(
    f'''<div class="hero"><div style="opacity:.85;font-size:.85rem;letter-spacing:.06em;margin-bottom:.3rem">CIENCIA DE DATOS · UTN FRM · 2026 · COMISIÓN 5K09 · GRUPO 03</div><h1>¿Cuánto vale un auto usado?</h1>
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
# Fuentes de los datos, con sus logos (app_data/logos/). Los autos por fuente salen de metricas.json.
import base64
carone_b64 = base64.b64encode((DATA / "logos" / "carone.svg").read_bytes()).decode()   # el logo de CarOne es blanco: va sobre un chip oscuro
lg0, lg1, lg2, lg3 = st.columns([1, 1.4, 1, 1.4], vertical_alignment="center")
lg0.markdown(f'<div style="background:#22364d;border-radius:10px;padding:8px 12px;display:inline-block"><img src="data:image/svg+xml;base64,{carone_b64}" width="56"></div>', unsafe_allow_html=True)
lg1.markdown(f"**CarOne** · {num(M['por_fuente'].get('carone', 0))} autos<br><span style='font-size:.8rem;color:#6b7280'>API GraphQL</span>", unsafe_allow_html=True)
lg2.image(str(DATA / "logos" / "deruedas.png"), width=130)
lg3.markdown(f"**DeRuedas** · {num(M['por_fuente'].get('deruedas', 0))} autos<br><span style='font-size:.8rem;color:#6b7280'>páginas HTML</span>", unsafe_allow_html=True)
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
g_h, e_h = st.columns([2.2, 1], gap="large")
p90_regla = alt.Chart(pd.DataFrame({"m": [p90]})).mark_rule(color="#C8602F", strokeDash=[4, 4], size=2).encode(x="m:Q")
st_med = sticker(f"mitad de los autos: menos de {usd(M['precio_mediano'])}", alt.X("m:Q"), px(14), AZUL, valor=M["precio_mediano"])
st_p90 = sticker(f"10 % más caro: más de {usd(p90)}", alt.X("m:Q"), px(40), "#C8602F", valor=p90)
g_h.altair_chart((hist + mediana + p90_regla + st_med + st_p90).properties(
    height=340, title=alt.TitleParams(f"{num(N_AUTOS)} autos usados, ya limpios (cortado en USD 80.000)", anchor="start", fontSize=13, color="#555")), width="stretch")
e_h.markdown(explica(
    "Cada barra cuenta cuántos autos cuestan entre dos precios (en tramos de USD 2.000): cuanto más alta, más autos. "
    "La línea punteada marca el precio del medio, la mediana.",
    "Un histograma es la forma más simple de ver cómo se reparten los precios: dónde hay muchos autos y dónde pocos.",
    f"Casi todos los autos están entre {usd(p10)} y {usd(p90)}. A la derecha queda una cola larga de autos caros "
    "(el gráfico se corta en USD 80.000 para que se lea).",
    "Hay muchos autos baratos y pocos muy caros: ahí es donde después le va a costar más al modelo."), unsafe_allow_html=True)

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
g_a, e_a = st.columns([2.2, 1], gap="large")
st_dep = sticker(f"−{num(dep * 100)} % por año: un año más viejo vale ~{num(dep * 100)} % menos", alt.value(14), px(16), "#C8602F")
g_a.altair_chart((puntos + linea + st_dep).properties(height=380), width="stretch")
e_a.markdown(explica(
    "Cada punto gris es un auto: cuanto más a la derecha, más nuevo; cuanto más arriba, más caro. "
    "La línea negra une el precio del medio de cada año.",
    "Un gráfico de puntos sirve para ver si dos cosas van juntas. Probamos con el año y el precio. "
    "El precio va en escala logarítmica para que una pérdida de porcentaje parecido se vea como una recta.",
    "La línea baja hacia los autos más viejos y es casi recta: el auto no pierde dólares fijos por año, "
    f"pierde cerca de {num(dep * 100)} % de su valor. Para un mismo año los precios varían mucho: eso lo explican la marca, el modelo y la ficha.",
    f"El año es lo que más pesa: la correlación con el precio es {num(M['fuga']['spearman_anio'], 2)} y, si el modelo no lo conoce, "
    f"el error sube {usd(CP['sin fabricado_en']['delta'])}."), unsafe_allow_html=True)

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
st.markdown("### Hipótesis que descartamos: «el precio depende de dónde se vende el auto»")
st.markdown(
    f"Se la dimos al modelo y **no movió la aguja**: con la provincia el error pasó de {usd(CPV['referencia (todas)']['mae'])} a "
    f"{usd(c_prov['mae'])}, y con provincia y ciudad a {usd(c_ciu['mae'])}. Son diferencias de dólares, mucho menores que el "
    f"ruido de la validación (± {usd(CPV['referencia (todas)']['desvio'])}). Por eso la ubicación **no entra al modelo**.")
@st.cache_data
def cargar_mapa(version):   # version = fecha de los archivos: si cambian, se vuelve a leer y no queda un caché viejo
    ub = pd.read_csv(DATA / "ubicaciones.csv")
    ub["mas_publicados"] = ub["mas_publicados"].astype(str).str.replace("<br>", "\n", regex=False)
    geo = json.loads((DATA / "provincias.geojson").read_text(encoding="utf-8"))
    return ub, geo


UBIC, GEO = cargar_mapa((DATA / "ubicaciones.csv").stat().st_mtime_ns + len(list(pd.read_csv(DATA / "ubicaciones.csv", nrows=0).columns)))
UBIC_OK = UBIC.dropna(subset=["latitud", "longitud"])
st.markdown(explica(
    "Un círculo por ciudad: cuanto más grande, más autos se publican ahí; cuanto más oscuro, más caro es el precio del medio. "
    "Podés elegir una provincia para acercarte, y pasar el mouse por un círculo para ver sus autos.",
    "Para saber si el precio cambia según dónde se vende, lo más directo es mirarlo sobre un mapa.",
    f"Casi todos los autos están en Mendoza y Córdoba, y los colores de las ciudades grandes son parecidos: no hay zonas mucho más caras. "
    f"La provincia explica sólo {num(UB['eta2_provincia'] * 100, 1)} % del precio; la marca, {num(UB['eta2_marca'] * 100, 0)} %.",
    f"Se lo dimos al modelo y no mejoró (+{usd(abs(c_prov['delta']))}): descartamos la ubicación."), unsafe_allow_html=True)
provs = alfabetico(UBIC_OK["provincia"].unique())
sel_prov = st.multiselect("Provincias en el mapa", provs, placeholder="Todas las provincias")
mapa_df = UBIC_OK[UBIC_OK["provincia"].isin(sel_prov)] if sel_prov else UBIC_OK
# Mapa con zoom y desplazamiento (pydeck, con mapa base real). Un círculo por ciudad: el radio sale de la raíz de la
# cantidad de autos y el color (de claro a oscuro) del precio mediano. Contorno de provincias de app_data/provincias.geojson.
import pydeck as pdk
t = (mapa_df["mediana_usd"] - UBIC_OK["mediana_usd"].min()) / (UBIC_OK["mediana_usd"].max() - UBIC_OK["mediana_usd"].min())
mapa_df = mapa_df.assign(
    r=(mapa_df["autos"] ** 0.5) * 2500 + 3000,
    rojo=(254 + (181 - 254) * t).round().astype(int), verde=(215 + (96 - 215) * t).round().astype(int),
    azul=(170 + (58 - 170) * t).round().astype(int), mediana_txt=mapa_df["mediana_usd"].map(usd),
    nota_ubic=mapa_df["aproximada"].map(lambda v: "(ubicación aproximada)" if v else ""))
capa_prov = pdk.Layer("GeoJsonLayer", GEO, stroked=True, filled=True, get_fill_color=[238, 241, 245],
                     get_line_color=[120, 130, 145], line_width_min_pixels=1)   # sólo Argentina, sin mapa base del mundo
capa_ciu = pdk.Layer("ScatterplotLayer", mapa_df, get_position=["longitud", "latitud"], get_radius="r",
                     get_fill_color=["rojo", "verde", "azul", 210], get_line_color=[255, 255, 255], line_width_min_pixels=1,
                     stroked=True, pickable=True, radius_min_pixels=3, radius_max_pixels=45)
# Vista inicial: todo el país o, si se eligió una o más provincias, un zoom que las encuadra (caja de sus límites).
ALIAS_GEO = {"Capital Federal": "Ciudad Autónoma de Buenos Aires", "Tierra Del Fuego": "Tierra del Fuego"}


def puntos_de(geom):
    """Todos los puntos [lon, lat] de un MultiPolygon del geojson."""
    return [p for poligono in geom["coordinates"] for anillo in poligono for p in anillo]


if sel_prov:
    nombres = {ALIAS_GEO.get(p, p) for p in sel_prov}
    pts = [p for f in GEO["features"] if f["properties"]["nombre"] in nombres for p in puntos_de(f["geometry"])]
    if not pts:   # si el nombre no coincide, se usan las ciudades elegidas
        pts = mapa_df[["longitud", "latitud"]].values.tolist()
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    # centro = medio de la caja; zoom = el que deja entrar la caja completa en el mapa (≈ 550 × 520 px)
    lat0, lon0 = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    ancho, alto = max(max(lons) - min(lons), 0.3), max(max(lats) - min(lats), 0.3)
    z_ancho = np.log2(386 * 0.7 / ancho)
    z_alto = np.log2(365 * 0.7 * np.cos(np.radians(lat0)) / alto)
    vista = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=float(np.clip(min(z_ancho, z_alto), 3.15, 8)))
else:
    vista = pdk.ViewState(latitude=-39.0, longitude=-64.5, zoom=3.15)   # el país entero entra en 520 px de alto
col_mapa, col_barras = st.columns([3, 2])
# Botones de zoom: suman o restan un nivel al zoom de partida (que ya encuadra la provincia elegida o el país).
if st.session_state.get("zoom_prov") != tuple(sel_prov):
    st.session_state["zoom_prov"], st.session_state["dz"] = tuple(sel_prov), 0.0   # al cambiar de provincia, vuelve al encuadre
c_map, c_btn = col_mapa.columns([12, 1.3], gap="small")   # botones al costado, mapa al lado
if c_btn.button("＋", key="z_mas", help="Acercar", width="stretch"):
    st.session_state["dz"] = min(st.session_state["dz"] + 1, 5)
if c_btn.button("－", key="z_menos", help="Alejar", width="stretch"):
    st.session_state["dz"] = max(st.session_state["dz"] - 1, -1.5)
if c_btn.button("⤢", key="z_reset", help="Volver a ver el país entero", width="stretch"):
    st.session_state["dz"] = 0.0
vista.zoom = max(float(vista.zoom) + st.session_state["dz"], 2.2)
# scrollZoom apagado: con la rueda se baja la página; el zoom va con los botones + / − o pellizcando; se arrastra para mover
vista_mapa = pdk.View("MapView", controller={"scrollZoom": False, "dragPan": True, "doubleClickZoom": True, "touchZoom": True})
c_map.pydeck_chart(pdk.Deck(layers=[capa_prov, capa_ciu], views=[vista_mapa], initial_view_state=vista, map_provider=None,
                               tooltip={"html": "<b>{ubicacion_corregida}</b><br>{autos} autos · mediana {mediana_txt}"
                                        "<br><br><b>Más publicados</b><br>{mas_publicados}"
                                        "<br><br><b>Más caro:</b> {mas_caro}<br><b>Más barato:</b> {mas_barato}",
                                        "style": {"backgroundColor": "#22364d", "color": "white", "fontSize": "12px", "whiteSpace": "pre-line"}}),
                      height=480)
# Referencias del mapa: tamaño = cantidad de autos de la ciudad, color = precio mediano de sus autos.
p_lo, p_hi = UBIC_OK["mediana_usd"].min(), UBIC_OK["mediana_usd"].max()
LEYENDA = (
    "<div style='display:flex;flex-wrap:wrap;gap:1.4rem;align-items:center;font-size:.8rem;color:#3d3d3a;margin:.2rem 0 1rem 0'>"
    "<div><b>Cantidad de autos</b><br>"
    "<span style='display:inline-flex;align-items:flex-end;gap:.5rem;margin-top:.2rem'>"
    "<span style='text-align:center'><span style='display:block;width:8px;height:8px;border-radius:50%;background:#d9a07a;margin:0 auto'></span>10</span>"
    "<span style='text-align:center'><span style='display:block;width:16px;height:16px;border-radius:50%;background:#d9a07a;margin:0 auto'></span>100</span>"
    "<span style='text-align:center'><span style='display:block;width:30px;height:30px;border-radius:50%;background:#d9a07a;margin:0 auto'></span>1.000</span></span></div>"
    "<div><b>Precio mediano de la ciudad</b><br>"
    "<span style='display:block;width:170px;height:10px;border-radius:5px;margin-top:.3rem;background:linear-gradient(90deg,rgb(254,215,170),rgb(181,96,58))'></span>"
    f"<span style='display:flex;justify-content:space-between;width:170px'><span>{usd(p_lo)}</span><span>{usd(p_hi)}</span></span></div>"
    "<div><span style='display:inline-block;width:14px;height:10px;background:#eef1f5;border:1px solid #78828f;vertical-align:middle'></span> Provincia</div>"
    "</div>")
# Si se eligió una sola provincia: tabla con sus autos (ordenable haciendo clic en las columnas).
if len(sel_prov) == 1:
    de_prov = AUTOS[AUTOS["provincia"] == sel_prov[0]]
    st.markdown(f"**Autos publicados en {sel_prov[0]}** ({num(len(de_prov))})")
    st.dataframe(de_prov.assign(Ciudad=de_prov["ubicacion"].str.split(", ").str[-1])[
        ["Ciudad", "marca", "modelo", "fabricado_en", "kilometraje", "precio_usd", "tasacion_usd"]]
        .rename(columns={"marca": "Marca", "modelo": "Modelo", "fabricado_en": "Año", "kilometraje": "Km",
                         "precio_usd": "Precio (USD)", "tasacion_usd": "Tasación (USD)"})
        .sort_values("Precio (USD)", ascending=False),
        hide_index=True, width="stretch", height=260)
por_prov = (UBIC.groupby("provincia").agg(autos=("autos", "sum")).reset_index().sort_values("autos", ascending=False).head(8))
barras = alt.Chart(por_prov).mark_bar(color=AZUL).encode(
    y=alt.Y("provincia:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)), x=alt.X("autos:Q", title="Autos publicados"),
    tooltip=[alt.Tooltip("provincia:N", title="Provincia"), alt.Tooltip("autos:Q", title="Autos")])
st_prov = sticker(f"{num(UB['pct_top_provincias'] * 100, 0)} % de los autos: Mendoza, Córdoba y Buenos Aires", px(190), px(150), "#C8602F", size=12)
st_eta = sticker(f"la provincia explica {num(UB['eta2_provincia'] * 100, 1)} % del precio (la marca, {num(UB['eta2_marca'] * 100, 0)} %)", px(190), px(172), AZUL, size=12)
col_barras.altair_chart((barras + st_prov + st_eta).properties(height=250, title="Autos por provincia"), width="stretch")
col_barras.markdown(LEYENDA, unsafe_allow_html=True)
col_barras.caption(f"Cada círculo es una ciudad: pasá el mouse para ver sus autos. {int(UBIC['aproximada'].sum())} ciudades chicas tienen ubicación aproximada (cabecera del departamento o centro de la provincia).")
st.markdown(
    "**Qué mirar:** casi todo el mercado está en Mendoza y Córdoba, y los precios medianos de las ciudades grandes son "
    "parecidos. Lo que cambia el precio es *qué auto es* (marca, modelo, año), no *dónde* se publica.")
with st.expander("Ejemplos: el mismo auto en distintas provincias"):
    st.markdown("Elegí un auto con su año y mirá cuánto cuesta publicado en cada provincia. Si la ubicación importara, las barras "
                "cambiarían mucho de una provincia a otra; si no, quedan parecidas (lo que sí cambia de un aviso a otro es el kilometraje o la versión).")
    ej = AUTOS.dropna(subset=["provincia"]).groupby(["marca", "modelo", "fabricado_en", "provincia"]).agg(
        autos=("precio_usd", "size"), mediana=("precio_usd", "median"), km=("kilometraje", "median")).reset_index()
    ej = ej[ej["autos"] >= 2]
    cobertura = ej.groupby(["marca", "modelo", "fabricado_en"]).agg(provs=("provincia", "nunique"), autos=("autos", "sum")).reset_index()
    cobertura = cobertura[cobertura["provs"] >= 3].sort_values(["provs", "autos"], ascending=False).head(12)
    if cobertura.empty:
        st.info("Todavía no hay un auto con al menos 2 avisos en 3 provincias distintas.")
    else:
        cobertura["etiqueta"] = cobertura.apply(lambda r: f"{r['marca']} {r['modelo']} {int(r['fabricado_en'])} ({int(r['provs'])} provincias, {int(r['autos'])} autos)", axis=1)
        elegida = st.selectbox("Auto de ejemplo", list(cobertura["etiqueta"]))
        c = cobertura[cobertura["etiqueta"] == elegida].iloc[0]
        d_ej = ej[(ej["marca"] == c["marca"]) & (ej["modelo"] == c["modelo"]) & (ej["fabricado_en"] == c["fabricado_en"])].sort_values("mediana")
        gb = alt.Chart(d_ej).mark_bar(color=AZUL).encode(
            y=alt.Y("provincia:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X("mediana:Q", title="Precio mediano publicado (USD)", scale=alt.Scale(domain=[0, float(d_ej["mediana"].max()) * 1.25]),
                    axis=alt.Axis(format="$,.0f")),
            tooltip=[alt.Tooltip("provincia:N", title="Provincia"), alt.Tooltip("autos:Q", title="Avisos"),
                     alt.Tooltip("mediana:Q", title="Mediana (USD)", format=",.0f"), alt.Tooltip("km:Q", title="Km mediano", format=",.0f")])
        tx = gb.mark_text(align="left", dx=4, color=TINTA).encode(text=alt.Text("mediana:Q", format="$,.0f"))
        st.altair_chart((gb + tx).properties(height=40 * len(d_ej) + 40), width="stretch")
        dif = (d_ej["mediana"].max() / d_ej["mediana"].min() - 1) * 100
        st.dataframe(d_ej.rename(columns={"provincia": "Provincia", "autos": "Avisos", "mediana": "Precio mediano (USD)", "km": "Km mediano"})
                     [["Provincia", "Avisos", "Precio mediano (USD)", "Km mediano"]].style.format(
                         {"Precio mediano (USD)": "{:,.0f}", "Km mediano": "{:,.0f}"}), hide_index=True, width="stretch")
        st.caption(f"De la provincia más barata a la más cara hay {num(dif)} % de diferencia en este ejemplo. Son pocos avisos por provincia, así que "
                   "parte de la diferencia es azar o kilometraje: a nivel general, la provincia explica "
                   f"sólo {num(UB['eta2_provincia'] * 100, 1)} % del precio.")
with st.expander("Cómo lo sabemos"):
    st.markdown(f"""
#### Hipótesis H4: descarte de la ubicación geográfica
**Criterio de la cátedra:** una variable con η² menor a 0,05 está en la **zona roja** y no aporta señal.
Provincia ({num(UB['eta2_provincia'] * 100, 1)} %) y ciudad ({num(UB['eta2_ciudad'] * 100, 1)} %) quedan las dos por debajo: 
<span style="background:#c0392b;color:#fff;border-radius:6px;padding:1px 8px;font-size:.8rem;font-weight:600">ZONA ROJA</span>

**Cómo probamos la hipótesis de la ubicación.** Hicimos tres comprobaciones:

1. **¿Cuánto del precio explica cada variable?** Con η² (la parte de la variación del precio que explica un grupo):
   provincia **{num(UB['eta2_provincia'] * 100, 1)} %**, ciudad {num(UB['eta2_ciudad'] * 100, 1)} % y marca
   **{num(UB['eta2_marca'] * 100, 0)} %**. La marca pesa unas 17 veces más que la provincia.
2. **¿Mejora el modelo si la conoce?** Repetimos la misma validación cruzada (agrupada por auto, semilla {M['semilla']}) agregando
   la provincia y después provincia y ciudad. El error pasó de {usd(CPV['referencia (todas)']['mae'])} a
   {usd(c_prov['mae'])} (+{usd(c_prov['delta'])}) con la provincia y a {usd(c_ciu['mae'])} (+{usd(c_ciu['delta'])}) con provincia y ciudad.
   El ruido de la validación es de ± {usd(CPV['referencia (todas)']['desvio'])}: no mejora, incluso empeora un poco.
3. **¿Hay datos suficientes para separar por lugar?** No: el {num(UB['pct_top_provincias'] * 100, 0)} % de los autos está en
   Mendoza, Córdoba y Buenos Aires, y {int(UBIC['aproximada'].sum())} de las {UB['n_ciudades']} ciudades están ubicadas de forma aproximada en el mapa (son departamentos o localidades chicas).

**Gráfico:** el precio mediano de las ciudades con más avisos contra la mediana de todo el mercado (línea naranja).
""", unsafe_allow_html=True)
    med_glob = float(AUTOS["precio_usd"].median())
    top_c = UBIC.nlargest(8, "autos").assign(ciudad=lambda d: d["ubicacion_corregida"].str.replace(", ", " · ", regex=False))
    b_c = alt.Chart(top_c).mark_bar(color=AZUL).encode(
        y=alt.Y("ciudad:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        x=alt.X("mediana_usd:Q", title="Precio mediano de publicación (USD)", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("ciudad:N", title="Ciudad"), alt.Tooltip("autos:Q", title="Autos"),
                 alt.Tooltip("mediana_usd:Q", title="Mediana (USD)", format=",.0f")])
    t_c = b_c.mark_text(align="left", dx=3, color="#3d3d3a").encode(text=alt.Text("mediana_usd:Q", format="$,.0f"))
    r_c = alt.Chart(pd.DataFrame({"m": [med_glob]})).mark_rule(color=NARANJA, strokeDash=[5, 3], size=2).encode(x="m:Q")
    st.altair_chart((b_c + t_c + r_c).properties(height=260, title=f"Precio mediano por ciudad vs. mediana del mercado ({usd(med_glob)})"),
                    width="stretch")
    st.caption("Las ciudades grandes quedan cerca de la mediana global: la localización no introduce diferencias sistemáticas en el precio.")
    st.markdown("**La tabla completa de las pruebas** (error medio de validación cruzada; Δ = cuánto sube respecto de la referencia):")
    tabla_cp = pd.DataFrame(M["columnas_probadas"]).rename(columns={"variante": "Variante", "mae": "MAE (USD)",
                                                                    "desvio": "Desvío", "delta": "Δ vs referencia"})
    tabla_cp["Δ en desvíos"] = tabla_cp["Δ vs referencia"] / CPV["referencia (todas)"]["desvio"]
    st.dataframe(tabla_cp.style.format({"MAE (USD)": "{:,.1f}", "Desvío": "{:,.1f}", "Δ vs referencia": "{:,.1f}", "Δ en desvíos": "{:,.1f}"}),
                 hide_index=True, width="stretch")
    st.markdown(f"""
**Por qué no la dejamos aunque sume algo:** hay {UB['n_ciudades']} ciudades y decenas de ellas aportan sólo 1 o 2 avisos. Con tantas
categorías el modelo memorizaría casos sueltos en lugar de aprender patrones del auto (sobreajuste).

**Cómo se construyó la tabla:** se parte del HistGradientBoosting de referencia y se quita (o agrega) una columna por vez, con la misma
validación cruzada agrupada por vehículo (GroupKFold). Una diferencia menor a un desvío no es evidencia de nada.

**Decisión:** la ubicación no entra al modelo. Las ciudades tienen precios medianos parecidos y lo que cambia el precio es qué auto es.
Las coordenadas del mapa son aproximadas (por ciudad) y están en `app_data/ubicaciones.csv`; el η² y las pruebas con provincia y ciudad se calculan en `build_app.py`.
""")

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
# Barras con el error de cada modelo en validación cruzada (con barra de ± 1 desvío). Gris = piso, azul = elegido.
comp = pd.DataFrame(M["comparacion"])
NOMBRES = {"Piso (mediana)": "Piso: tasar todo a la mediana", "Lineal (Ridge)": "Regresión lineal (Ridge)",
           "Random forest": "Random forest", "HistGradientBoosting": "Boosting sin ajustar",
           "HistGradientBoosting ajustado": "Boosting ajustado (elegido)"}
comp["tipo"] = np.where(comp["modelo"] == "HistGradientBoosting ajustado", "Elegido",
                        np.where(comp["modelo"] == "Piso (mediana)", "Piso", "Probado"))
comp["lo"], comp["hi"] = comp["mae"] - comp["desvio"], comp["mae"] + comp["desvio"]
comp["txt"] = comp["mae"].map(usd)
comp.loc[comp["modelo"] == "HistGradientBoosting ajustado", "txt"] = f"{usd(AJ['mae'])} en validación · {usd(TEST['mae'])} en test"
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
g_c, e_c = st.columns([2.2, 1], gap="large")
g_c.altair_chart((barras + bigotes + textos).properties(height=260), width="stretch")
e_c.markdown(explica(
    "Cada barra es el error promedio, en dólares, de una forma de tasar. Barra más corta = mejor. La rayita negra es la variación entre pruebas.",
    "Para elegir un modelo hay que compararlos a todos con la misma regla. La barra gris es el piso: tasar todos los autos con el precio del medio, sin aprender nada.",
    f"Todos le ganan al piso y el boosting ajustado (azul oscuro) tiene la barra más corta: {usd(AJ['mae'])} en la validación cruzada.",
    f"Elegimos el boosting ajustado. En autos que nunca vio (test) se equivoca {usd(TEST['mae'])}, {num(veces, 1)} veces menos que el piso."), unsafe_allow_html=True)

# Real vs. predicho en el test (como en el Colab): cada punto es un auto que el modelo nunca vio; la diagonal es la predicción perfecta.
test_df = AUTOS[AUTOS["conjunto"] == "test"].copy()
def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
r2_test = _r2(test_df["precio_usd"], test_df["tasacion_usd"])
r2_log = _r2(np.log(test_df["precio_usd"]), np.log(test_df["tasacion_usd"]))
st.markdown("##### Precio real contra precio predicho, en el test")
lim_rp = [float(min(test_df["precio_usd"].min(), test_df["tasacion_usd"].min())) * 0.9,
          float(max(test_df["precio_usd"].max(), test_df["tasacion_usd"].max())) * 1.1]
esc_rp = alt.Scale(type="log", domain=lim_rp)
VAL_RP = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
diag_rp = alt.Chart(pd.DataFrame({"a": lim_rp, "b": lim_rp})).mark_line(color="#555", strokeDash=[5, 4]).encode(x="a:Q", y="b:Q")
mas10 = alt.Chart(pd.DataFrame({"a": lim_rp, "b": [v * 1.1 for v in lim_rp]})).mark_line(color="#999", strokeDash=[2, 3]).encode(x="a:Q", y="b:Q")
menos10 = alt.Chart(pd.DataFrame({"a": lim_rp, "b": [v * 0.9 for v in lim_rp]})).mark_line(color="#999", strokeDash=[2, 3]).encode(x="a:Q", y="b:Q")


def real_vs_predicho(df, color, titulo, texto):
    pts = alt.Chart(df).mark_circle(size=26, opacity=0.6, color=color).encode(
        x=alt.X("precio_usd:Q", scale=esc_rp, title="Precio real (USD, escala log)", axis=alt.Axis(format="$,.0f", values=VAL_RP)),
        y=alt.Y("tasacion_usd:Q", scale=esc_rp, title="Precio predicho (USD, escala log)", axis=alt.Axis(format="$,.0f", values=VAL_RP)),
        tooltip=[alt.Tooltip("auto:N", title="Auto"), alt.Tooltip("fabricado_en:Q", title="Año", format="d"),
                 alt.Tooltip("precio_usd:Q", title="Real", format=",.0f"), alt.Tooltip("tasacion_usd:Q", title="Predicho", format=",.0f")])
    return (pts + diag_rp + mas10 + menos10 + sticker(texto, px(10), px(14), color, size=12)).properties(height=330, title=titulo)


en_pesos, en_dolares = test_df[test_df["moneda"] == "ARS"], test_df[test_df["moneda"] == "USD"]


def resumen_rp(df):
    return (float((df["precio_usd"] - df["tasacion_usd"]).abs().mean()),
            float(((df["tasacion_usd"] / df["precio_usd"] - 1).abs() <= 0.10).mean() * 100), _r2(np.log(df["precio_usd"]), np.log(df["tasacion_usd"])))


mae_p, dentro_p, r2l_p = resumen_rp(en_pesos)
mae_d, dentro_d, r2l_d = resumen_rp(en_dolares)
g_r, e_r = st.columns([2.2, 1], gap="large")
def txt_rp(df):
    return f"{num(len(df))} autos · error USD {num(float((df['precio_usd'] - df['tasacion_usd']).abs().mean()))} · sesgo USD {num(float((df['tasacion_usd'] - df['precio_usd']).mean()))}"


g_r.altair_chart(real_vs_predicho(en_pesos, "#2f6fc0", "Avisos publicados en pesos", txt_rp(en_pesos)), width="stretch")
g_r.altair_chart(real_vs_predicho(en_dolares, "#C8602F", "Avisos publicados en dólares", txt_rp(en_dolares)), width="stretch")
e_r.markdown(explica(
    "Cada punto es un auto del test, que el modelo nunca vio. A la derecha, más caro era de verdad; arriba, más caro lo tasó el modelo. "
    "Arriba están los avisos en pesos y abajo los avisos en dólares, con los mismos ejes.",
    "Si el modelo acertara siempre, todos los puntos caerían sobre la línea punteada oscura. Así se ve de un vistazo cuánto se aleja cada uno. "
    "Las líneas finas marcan ±10 %.",
    f"En pesos, {num(dentro_p)} % de los autos queda dentro de ±10 % y el error promedio es {usd(mae_p)}. "
    f"En dólares, sólo {num(dentro_d)} % queda dentro de ±10 % y el error promedio es {usd(mae_d)}: los puntos se abren mucho más.",
    f"Con avisos en dólares hay más dispersión. Sirve para saber cuándo fiarse menos de la tasación. R² total: {num(r2_test, 2)} en dólares, {num(r2_log, 2)} en escala logarítmica."),
    unsafe_allow_html=True)

st.markdown("##### Dónde se equivoca el modelo, por tramo de precio")
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
g_t, e_t = st.columns([2.2, 1], gap="large")
g_t.altair_chart(alt.layer(g_mae + t_mae, g_rel + t_rel).resolve_scale(y="independent").properties(height=360), width="stretch")
e_t.markdown(explica(
    "Ordenamos los autos del test del más barato al más caro y los partimos en 5 grupos con la misma cantidad. En cada grupo, "
    "la barra azul (eje izquierdo) son los dólares que el modelo se equivoca en promedio, y la naranja (eje derecho) es el porcentaje del precio.",
    "Equivocarse USD 1.000 es poco en un auto de 30.000 y mucho en uno de 3.000. Por eso lo miramos de las dos formas, con un eje para cada una.",
    f"En dólares el error crece con el precio, de {usd(tramo['mae'].iloc[0])} a {usd(tramo['mae'].iloc[-1])}. En porcentaje queda parecido, entre "
    f"{num(tramo['rel_pct'].min(), 1)} % y {num(tramo['rel_pct'].max(), 1)} %.",
    "Los autos caros son los que más se le escapan al modelo en dólares: por eso la app avisa cuando una tasación cae en el 10 % más caro."),
    unsafe_allow_html=True)


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
    imp = imp.reset_index(drop=True)
    notas = alt.Chart(pd.DataFrame({"columna": [imp["columna"].iloc[0]], "v": [imp["sube_mae"].iloc[0]],
                                    "t": [f"+{usd(imp['sube_mae'].iloc[0])}: lo que más pesa"]})).mark_text(
        align="left", dx=6, fontWeight="bold", color="#C8602F").encode(y=alt.Y("columna:N", sort=list(imp["columna"])), x="v:Q", text="t:N")
    st.altair_chart((g_imp + notas).properties(height=26 * len(imp)), width="stretch")
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
st.markdown('<div class="kicker copa">La copa · tu turno</div>', unsafe_allow_html=True)
st.header("Ahora vos")
tab_exp, tab_probar = st.tabs(["Explorá los autos", "Probá el tasador"])

with tab_exp:
    st.markdown(explica(
        "Cada punto es un auto publicado. El color compara su precio con lo que dijo el tasador, que lo tasó <b>sin haber visto ese auto</b>: "
        "verde = publicado más de 10 % por debajo de la tasación, gris = en precio, naranja = más de 10 % por encima.",
        "Un gráfico de puntos deja ver miles de autos a la vez y cambiar qué se compara. Vos elegís qué va en cada eje.",
        "Probá con el año en el eje X: es la caída de valor. Cambiá al kilometraje, la potencia, el motor o la marca y mirá qué cambia. "
        "Pasá el mouse por un punto para ver el auto y hacé clic para abrir su ficha.",
        "Ahora la pregunta es tuya: filtrá, cambiá los ejes y buscá autos que se alejen de la tasación."), unsafe_allow_html=True)
    # Controles que achican el universo. Cada uno filtra el DataFrame `d` que alimenta tanto las 4 métricas
    # de arriba como el gráfico: marca (multiselect), año y precio (sliders) y caja (radio).
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4])
    marcas = alfabetico(AUTOS["marca"].unique())
    sel_marcas = f1.multiselect("Marca", marcas, placeholder="Todas las marcas")
    a_min, a_max = int(AUTOS["fabricado_en"].min()), int(AUTOS["fabricado_en"].max())
    # el slider abarca TODOS los años de los datos (por defecto, el rango completo)
    rango_anio = f2.slider("Año", a_min, a_max, (a_min, a_max))
    pmax = int(np.ceil(AUTOS["precio_usd"].max() / 1000) * 1000)   # hasta el auto más caro: por defecto se ven TODOS los autos
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
    clic = alt.selection_point(name="auto", fields=["idx"], on="click", nearest=True, empty=False)
    dd = d.reset_index(names="idx")
    dd["dif_pct"] = dd["dif"] * 100
    # Variables que se pueden poner en cada eje (selectores al costado del gráfico).
    X_OPC = {"Año de fabricación": "fabricado_en", "Kilometraje": "kilometraje", "Potencia (HP)": "potencia_hp", "Motor (litros)": "motor_lt",
             "Marca": "marca"}
    col_sel, col_graf = st.columns([1, 4], gap="medium")
    nom_x = col_sel.selectbox("Eje horizontal (X)", list(X_OPC), index=0)
    vx, vy = X_OPC[nom_x], "precio_usd"   # el eje vertical es siempre el precio publicado
    ver_todo = col_sel.checkbox("Ver todas las marcas" if vx == "marca" else "Ver todo el rango del eje X", value=False,
                                help=("Por defecto se muestran las 15 marcas con más avisos." if vx == "marca" else
                                      "Por defecto el gráfico se acerca a donde hay más autos. Marcá esto para ver también los extremos."))
    # Zoom a la zona más densa: para el año, la ventana de 10 años con más autos (se calcula de los datos); para el resto, del percentil 2 al 98.
    marcas_x, tick_med = [], None
    if vx == "marca" and len(dd):
        # categórica: se ordenan las marcas por precio mediano (de la más barata a la más cara) y se les da una posición 0..n-1
        top = dd["marca"].value_counts()
        if not ver_todo and not sel_marcas:
            dd = dd[dd["marca"].isin(top.index[:15])]
        med_m = dd.groupby("marca")["precio_usd"].median().sort_values()
        marcas_x = list(med_m.index)
        pos = {m: i for i, m in enumerate(marcas_x)}
        dd["x_graf"] = dd["marca"].map(pos) + (((dd["idx"] * 2654435761) % 1000) / 1000 - 0.5) * 0.7
        tick_med = pd.DataFrame({"x": [pos[m] for m in marcas_x], "med": med_m.values, "marca": marcas_x})
        zoom_x = todo_x = [-0.6, len(marcas_x) - 0.4]
    elif len(dd):
        if vx == "fabricado_en":
            cuenta = dd["fabricado_en"].round().astype(int).value_counts().sort_index()
            anios = range(int(cuenta.index.min()), int(cuenta.index.max()) + 1)
            cuenta = cuenta.reindex(anios, fill_value=0)
            ventana = cuenta.rolling(10).sum().dropna()
            ini = int(ventana.idxmax() - 9) if len(ventana) else int(cuenta.index.min())
            zoom_x = [ini - 0.5, ini + 9.5] if len(ventana) else [dd[vx].min() - 0.5, dd[vx].max() + 0.5]
        else:
            lo, hi = dd[vx].quantile(0.02), dd[vx].quantile(0.98)
            pad = (hi - lo) * 0.03
            zoom_x = [float(lo - pad), float(hi + pad)]
        todo_x = [float(dd[vx].min()) - (0.5 if vx == "fabricado_en" else 0), float(dd[vx].max()) + (0.5 if vx == "fabricado_en" else 0)]
    else:
        zoom_x = todo_x = [0, 1]
    dominio_x = todo_x if ver_todo else zoom_x
    if vx == "marca":
        pass
    elif vx == "fabricado_en":   # desplazamiento horizontal fijo por auto: los puntos no se mueven entre un clic y otro
        dd["x_graf"] = dd[vx] + (((dd["idx"] * 2654435761) % 1000) / 1000 - 0.5) * 0.7
    else:
        dd["x_graf"] = dd[vx]
    y_escala = ESCALA_PRECIO
    y_eje = eje("Precio publicado (USD, escala log)", MARCAS_PRECIO)
    if vx == "marca":
        x_eje = alt.Axis(values=list(range(len(marcas_x))), labelAngle=-45, labelLimit=120, labelOverlap=False,
                         labelExpr="[" + ",".join(json.dumps(m) for m in marcas_x) + "][datum.value]")
    else:
        x_eje = alt.Axis(format="d" if vx == "fabricado_en" else ",.1f" if vx == "motor_lt" else ",.0f")
    disp = alt.Chart(dd).mark_circle(size=36, opacity=0.65, clip=True).encode(
        x=alt.X("x_graf:Q", title=nom_x, axis=x_eje, scale=alt.Scale(domain=dominio_x, nice=False)),
        y=alt.Y(f"{vy}:Q", scale=y_escala, axis=y_eje),
        color=alt.Color("veredicto:N", scale=ESCALA_VEREDICTO, title=None,
                        legend=alt.Legend(orient="top")),
        size=alt.condition(clic, alt.value(160), alt.value(36)),
        tooltip=[alt.Tooltip("auto:N", title="Auto"), alt.Tooltip("version:N", title="Versión"),
                 alt.Tooltip("fabricado_en:Q", title="Año", format="d"),
                 alt.Tooltip("kilometraje:Q", title="Km", format=",.0f"),
                 alt.Tooltip("precio_usd:Q", title="Precio publicado (USD)", format=",.0f"),
                 alt.Tooltip("tasacion_usd:Q", title="Tasación del modelo (USD)", format=",.0f"),
                 alt.Tooltip("dif_txt:N", title="Diferencia vs tasación"),
                 alt.Tooltip("fuente:N", title="Fuente")],
    ).add_params(clic).properties(height=460)
    if tick_med is not None and len(tick_med):   # una rayita con la mediana de cada marca
        disp = alt.layer(disp, alt.Chart(tick_med).mark_tick(color="#22364d", thickness=3, size=34, opacity=0.9).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=dominio_x, nice=False)), y=alt.Y("med:Q", scale=ESCALA_PRECIO),
            tooltip=[alt.Tooltip("marca:N", title="Marca"), alt.Tooltip("med:Q", title="Precio mediano (USD)", format=",.0f")]))
    evento = col_graf.altair_chart(disp, width="stretch", on_select="rerun", key="explora")
    if vx == "marca" and len(dd):
        col_graf.caption("Marcas ordenadas del precio mediano más bajo al más alto; la rayita oscura es la mediana de cada marca. "
                         + ("" if (ver_todo or sel_marcas) else "Se muestran las 15 con más avisos; marcá «Ver todas las marcas» para ver el resto."))
    elif not ver_todo and len(dd):
        col_graf.caption("El gráfico está acercado a donde hay más autos"
                         + (f" (los 10 años con más autos, {int(zoom_x[0] + 0.5)} a {int(zoom_x[1] - 0.5)})" if vx == "fabricado_en" else "")
                         + ". Marcá «Ver todo el rango del eje X» para ver los extremos.")

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
        if abs(elegido["dif"]) >= 0.25:
            st.error(f"TASACIÓN MUY DESFASADA: el modelo tasa este auto {num(abs(elegido['dif']) * 100)} % "
                     f"{'por encima' if elegido['dif'] < 0 else 'por debajo'} del precio publicado. "
                     "Puede ser un precio fuera de mercado o un caso que el modelo no ve bien.", icon=":material/warning:")
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
    # Formulario a la izquierda (izq) y panel de tasación a la derecha (der).
    izq, der = st.columns([1.5, 1])
    c_a, c_b, c_c = izq.columns(3)
    marcas_p = alfabetico(AUTOS["marca"].unique())
    marca = c_a.selectbox("Marca", marcas_p, index=marcas_p.index("Volkswagen") if "Volkswagen" in marcas_p else 0)
    opciones = cuenta[cuenta["marca"] == marca].copy()
    opciones = opciones.set_index("modelo").loc[alfabetico(opciones["modelo"])].reset_index()  # A-Z
    lista_mod = opciones["modelo"].tolist()
    mas_publicado = opciones.sort_values("n", ascending=False)["modelo"].iloc[0]   # la lista va A-Z pero arranca en el más publicado
    modelo = c_b.selectbox("Modelo", lista_mod, index=lista_mod.index(mas_publicado),
                           format_func=lambda m: f"{m}  ({int(opciones.set_index('modelo').loc[m, 'n'])} avisos)")
    ref_m = AUTOS[(AUTOS["marca"] == marca) & (AUTOS["modelo"] == modelo)]

    def tipico(col, por_defecto):
        v = ref_m[col].median()
        return float(v) if pd.notna(v) else por_defecto

    c1, c2, c3 = izq.columns(3)
    anio = c1.slider("Año de fabricación", int(R["fabricado_en"][2]), int(R["fabricado_en"][3]),
                     int(np.clip(round(tipico("fabricado_en", 2018)), R["fabricado_en"][2], R["fabricado_en"][3])))
    km_v = c2.number_input("Kilometraje", min_value=0, max_value=500000,
                           value=int(round(tipico("kilometraje", 80000), -3)), step=5000)
    c4, c5, c6 = izq.columns(3)
    motor = c3.number_input("Motor (litros)", min_value=0.6, max_value=float(R["motor_lt"][3]),
                            value=float(np.clip(round(tipico("motor_lt", 1.6), 1), 0.6, R["motor_lt"][3])), step=0.1, format="%.1f")
    hp = c4.number_input("Potencia (HP)", min_value=40.0, max_value=float(R["potencia_hp"][3]),
                         value=float(np.clip(round(tipico("potencia_hp", 110)), 40, R["potencia_hp"][3])), step=5.0, format="%.0f")
    # Los desplegables de caja, combustible y tracción ofrecen sólo lo que aparece en los avisos de ESE modelo
    # (una Amarok sólo se ofrece diésel, no nafta). Con 20 avisos o más se ignora lo que aparece en menos del 3 %
    # (errores de carga). Si el modelo no tiene dato de alguna, se ofrecen todas.
    def disponibles(col, mapa):
        vc = ref_m[col].dropna().value_counts()
        if len(ref_m) >= 20:
            vc = vc[vc / len(ref_m) >= 0.03]
        pres = [k for k, v in mapa.items() if v in vc.index]
        return (pres or list(mapa)), vc

    CAJAS_P = {"Automática": "automatico", "Manual": "manual"}
    cajas_ok, vc_caja = disponibles("caja", CAJAS_P)
    caja_p = c5.selectbox("Caja", cajas_ok, index=cajas_ok.index(next((k for k, v in CAJAS_P.items() if len(vc_caja) and v == vc_caja.index[0] and k in cajas_ok), cajas_ok[0])))
    comb_opts = {"Diésel": "diesel", "Eléctrico": "electrico", "GNC": "gnc", "Híbrido": "hibrido", "Nafta": "nafta"}  # A-Z
    combs_ok, vc_comb = disponibles("comb", comb_opts)
    comb = c6.selectbox("Combustible", combs_ok, index=combs_ok.index(next((k for k, v in comb_opts.items() if len(vc_comb) and v == vc_comb.index[0] and k in combs_ok), combs_ok[0])))
    TRACS = {"Delantera": "delantera", "Trasera": "trasera", "4x4": "4x4"}
    tracs_ok, vc_trac = disponibles("trac", TRACS)
    trac = c_c.selectbox("Tracción", tracs_ok, index=tracs_ok.index(next((k for k, v in TRACS.items() if len(vc_trac) and v == vc_trac.index[0] and k in tracs_ok), tracs_ok[0])))
    izq.caption("Caja, combustible y tracción: sólo las que aparecen en los avisos de este modelo.")

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
    if n_mod >= 3 and not (ref_m["fabricado_en"].min() <= anio <= ref_m["fabricado_en"].max()):
        avisos.append(f"No hay {marca} {modelo} de {anio} en los datos: los autos de este modelo van de "
                      f"{int(ref_m['fabricado_en'].min())} a {int(ref_m['fabricado_en'].max())}. La tasación es una extrapolación.")
    for col, v, nombre in [("fabricado_en", anio, "El año"), ("kilometraje", km_v, "El kilometraje"),
                           ("motor_lt", motor, "El motor"), ("potencia_hp", hp, "La potencia")]:
        p01, p99 = REAL[col]
        if v < p01 or v > p99:
            avisos.append(f"{nombre} ({num(v, 1 if col == 'motor_lt' else 0)}) está fuera del rango donde está el 98 % "
                          f"de los autos que vio el modelo ({num(p01, 1 if col == 'motor_lt' else 0)} a "
                          f"{num(p99, 1 if col == 'motor_lt' else 0)}).")
    # Autos caros: el tramo de precio más alto es donde más se equivoca el modelo (ver el gráfico de "El modelo").
    alto = TEST["por_tramo"][-1]
    if precio > p90:
        avisos.append(f"Esta tasación está en el 10 % más caro del mercado (más de {usd(p90)}). Ahí el modelo se equivoca "
                      f"más: en el tramo más caro del test el error fue de {usd(alto['mae'])} en promedio "
                      f"({num(alto['rel'] * 100, 1)} %). Tomalo como orientación.")
    if anio >= 2025 and km_v > 60000:
        avisos.append("Un auto tan nuevo con tantos kilómetros es raro en los datos.")

    # ---- Panel de tasación (a la derecha del formulario) ----
    # Percentil: de todos los autos publicados, qué porcentaje cuesta menos que esta tasación.
    pct = float((AUTOS["precio_usd"] < precio).mean() * 100)
    p_min, p_max = float(AUTOS["precio_usd"].min()), float(AUTOS["precio_usd"].max())
    # Posición en la barra: escala logarítmica entre el auto más barato y el más caro publicados.
    pos = float(np.clip((np.log(precio) - np.log(p_min)) / (np.log(p_max) - np.log(p_min)), 0, 1)) * 100
    lo, hi, tc = precio * (1 - margen), precio * (1 + margen), M["tipo_cambio"]
    # Comparables: los 3 avisos del mismo modelo más parecidos en año y km (con su precio real y la tasación del modelo).
    comp_html = ""
    alerta_html = ""
    if len(ref_m):
        cerca = ref_m.assign(dist=(ref_m["fabricado_en"] - anio).abs() * 10000 + (ref_m["kilometraje"] - km_v).abs() / 10).nsmallest(3, "dist")
        filas = "".join(
            f"<tr><td>{r.marca} {r.modelo} · {int(r.fabricado_en)} · {num(r.kilometraje)} km</td>"
            f"<td style='text-align:right'>{usd(r.precio_usd)} real<br>{usd(r.tasacion_usd)} tasado</td></tr>"
            for r in cerca.itertuples())
        comp_html = f"<div class='r' style='margin-top:.5rem'>Comparables en los datos</div><table>{filas}</table>"
        # Alerta de desfase: la tasación contra el precio real mediano de los comparables más parecidos.
        med_comp = float(cerca["precio_usd"].median())
        desf = precio / med_comp - 1
        if abs(desf) >= 0.25:
            sentido = "más CARA" if desf > 0 else "más BARATA"
            alerta_html = (f"<div class='alerta'><b>TASACIÓN MUY DESFASADA</b><br>Es {num(abs(desf) * 100)} % {sentido} que "
                           f"los {len(cerca)} avisos más parecidos (mediana {usd(med_comp)}). Revisá los datos cargados "
                           "(año, kilometraje, versión) o no te fíes de este número.</div>")
    der.markdown(alerta_html + 
        f"<div class='panel'><div class='t'>Tasación estimada</div><div class='v'>{usd(precio)}</div>"
        f"<div class='s'>≈ ARS {num(precio * tc)} (cotización de referencia {num(tc)} ARS/USD)</div>"
        f"<div class='r'>Rango sugerido: {usd(lo)} a {usd(hi)} (± {num(margen * 100, 1)} %)</div>"
        f"<div class='s'>En pesos: ARS {num(lo * tc)} a ARS {num(hi * tc)}</div>"
        f"<div class='nota'>Margen proporcional al valor: el error mediano en test para modelos con {tramo_te} avisos. "
        f"{marca} {modelo} tiene {n_train_mod} en entrenamiento.</div>"
        f"<div class='barra'><div class='punto' style='left:{pos:.1f}%'></div></div>"
        f"<div class='ext'><span>{usd(p_min)}</span><span>{usd(p_max)}</span></div>"
        f"<div class='r'>Percentil {num(pct)} del mercado ({num(len(AUTOS))} autos)</div>{comp_html}</div>",
        unsafe_allow_html=True)
    for a_ in avisos:
        izq.warning(a_)
    r2 = st.container()
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

# Resaltado del índice según la parte de la página donde estás (scroll-spy). Un bloque st.html corre un
# poco de JavaScript (st.html con unsafe_allow_javascript) en la página principal: mira qué ancla (`ancla("...")`) quedó más cerca del borde de arriba y
# marca con la clase "act" el link del índice que le corresponde.
st.html("""
<script>
const doc = document;
function marcar() {
  const links = [...doc.querySelectorAll('a.idx')];
  let actual = null;
  for (const l of links) {
    const el = doc.getElementById(l.getAttribute('href').slice(1));
    if (el && el.getBoundingClientRect().top <= 140) actual = l;   // la última ancla que ya pasó el borde de arriba
  }
  const sc = doc.querySelector('[data-testid="stMain"]');
  if (sc && sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 5) actual = links[links.length - 1];  // al final de todo: el último
  links.forEach(l => l.classList.toggle('act', l === (actual || links[0])));
}
if (!window.__idxSpy) {
  window.__idxSpy = true;
  doc.addEventListener('scroll', marcar, true);
}
setInterval(marcar, 400);   // también al volver a dibujarse la página
marcar();
</script>""", unsafe_allow_javascript=True)
