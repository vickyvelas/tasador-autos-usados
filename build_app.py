"""Construye los datos de la app a partir del CSV que produce el pipeline.

    python build_app.py                       # usa data/dataset.csv
    python build_app.py --datos final_2026-12-07.csv   # una cosecha nueva de Airflow

Hace lo mismo que el Colab de la Entrega 3 (misma limpieza, misma partición,
misma semilla, misma grilla) y guarda en app_data/:
  modelo.joblib    el modelo elegido (entrenado con train, como en el notebook)
  autos.csv        cada auto limpio con su tasación (fuera de muestra) para explorar
  metricas.json    todos los números que muestra la app
Después: streamlit run app.py
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit, KFold,
                                     cross_val_predict, cross_validate)

import coordenadas as geo_coords
import tasador as T

AQUI = Path(__file__).parent
SALIDA = AQUI / "app_data"

def cv_mae(pipe, X, y, cv, groups=None):
    r = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring="neg_mean_absolute_error", n_jobs=-1)
    return float(-r["test_score"].mean()), float(r["test_score"].std())


def main(ruta_datos: str, sin_api: bool = False):
    SALIDA.mkdir(exist_ok=True)
    # PASO 1 · datos de entrada: el CSV de la capa plata que deja el pipeline de Airflow
    # (data/dataset.csv por defecto, o el que se pase con --datos).
    df_raw = pd.read_csv(ruta_datos, encoding="utf-8")
    fecha_datos = str(pd.to_datetime(df_raw["fecha_ingesta"], errors="coerce").dt.date.max())
    # PASO 2 · limpieza del notebook (tasador.limpiar): precio en dólares, filtros y clave de vehículo.
    # `pasos` guarda cuántas filas quedan en cada filtro: es la tabla de "Cómo lo sabemos" de la app.
    df, pasos = T.limpiar(df_raw)
    # Copia con TODAS las columnas (versión, url, fuente, moneda...) para mostrar en la app.
    # El modelo, en cambio, recibe sólo las columnas de abajo (sin las COLUMNAS_SIN_USO).
    info = df.copy()  # guarda versión, fuente y moneda para mostrar en la app
    # Nombres para mostrar: se unifican CITROEN/Citroen/Citroën, Eco sport/EcoSport, Leapmotors/Leapmotor...
    # con tasador.claves_marca_modelo(unificar=True) (el modelo NO las unifica, como el notebook); acá se elige la escritura más frecuente de cada marca y modelo para mostrar.
    clave_m, clave_mo = T.claves_marca_modelo(info, unificar=True)
    info["marca"] = info.groupby(clave_m)["marca"].transform(lambda s: s.value_counts().index[0])
    info["modelo"] = info.groupby(clave_m + "|" + clave_mo)["modelo"].transform(lambda s: s.value_counts().index[0])
    # Ubicación: se corrige la codificación y se saca la provincia. Sólo sirve para probar la hipótesis y para el mapa.
    info["ubicacion_c"] = info["ubicacion"].map(T.arreglar_codificacion).str.replace(r"\s+", " ", regex=True).str.strip()
    info["provincia"] = info["ubicacion_c"].map(T.provincia_de)
    df = df.drop(columns=T.COLUMNAS_SIN_USO)

    # PASO 3 · partición 80/20 por vehículo (GroupShuffleSplit con la clave) y semilla 42, igual que el notebook.
    # X = columnas del modelo, y = precio en dólares, grupos = el vehículo (para que un mismo auto no quede en train y test).
    X, y, grupos = df.drop(columns=["precio_usd"]), df["precio_usd"], df["clave"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=T.SEMILLA)
    idx_tr, idx_te = next(gss.split(X, y, groups=grupos))
    X_train, X_test = X.iloc[idx_tr], X.iloc[idx_te]
    y_train, y_test = y.iloc[idx_tr], y.iloc[idx_te]
    g_train = grupos.iloc[idx_tr]
    # Validación cruzada de 5 pedazos, también agrupada por vehículo. Sólo se usa con el train.
    cv = GroupKFold(n_splits=5)
    print(f"train {len(X_train)} · test {len(X_test)}")

    # PASO 4 · el piso (tasar todo con la mediana) y los 3 modelos a comparar. Cada uno va dentro del mismo
    # Pipeline (tasador.pipeline), con media y desvío de la validación cruzada: es la tabla de la Entrega 3.
    ### piso y comparación de modelos
    piso = cv_mae(T.pipeline(DummyRegressor(strategy="median")), X_train, y_train, cv, g_train)
    modelos = {
        "Lineal (Ridge)": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=T.SEMILLA, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=T.SEMILLA),
    }
    comparacion = [{"modelo": "Piso (mediana)", "mae": piso[0], "desvio": piso[1]}]
    for nombre, est in modelos.items():
        m, s = cv_mae(T.pipeline(est), X_train, y_train, cv, g_train)
        comparacion.append({"modelo": nombre, "mae": m, "desvio": s})
        print(f"{nombre}: {m:,.0f} ± {s:,.0f}")

    # PASO 5 · ajuste del elegido: GridSearchCV con 2 x 4 x 2 x 2 = 32 combinaciones de hiperparámetros.
    # La mejor se reentrena sobre todo el train (busqueda.best_estimator_) y es el modelo que usa la app.
    ### grilla (32 combinaciones, igual que el notebook)
    grilla = {
        "modelo__regressor__learning_rate": [0.05, 0.1],
        "modelo__regressor__max_iter": [100, 200, 400, 600],
        "modelo__regressor__max_leaf_nodes": [15, 31],
        "modelo__regressor__min_samples_leaf": [10, 30],
    }
    busqueda = GridSearchCV(T.pipeline_hgb(), grilla, cv=cv, scoring="neg_mean_absolute_error",
                            n_jobs=-1, return_train_score=True)
    busqueda.fit(X_train, y_train, groups=g_train)
    modelo_final = busqueda.best_estimator_
    mejores = {k.replace("modelo__regressor__", ""): v for k, v in busqueda.best_params_.items()}
    r = pd.DataFrame(busqueda.cv_results_)
    p = "param_modelo__regressor__"
    tabla_grilla = pd.DataFrame({
        "learning_rate": r[p + "learning_rate"], "max_iter": r[p + "max_iter"],
        "max_leaf_nodes": r[p + "max_leaf_nodes"], "min_samples_leaf": r[p + "min_samples_leaf"],
        "mae_val": -r["mean_test_score"], "desvio_val": r["std_test_score"], "mae_train": -r["mean_train_score"],
    }).sort_values("mae_val")
    ajustado = (float(-busqueda.best_score_), float(tabla_grilla["desvio_val"].iloc[0]))
    comparacion.append({"modelo": "HistGradientBoosting ajustado", "mae": ajustado[0], "desvio": ajustado[1]})
    print(f"ajustado: {ajustado[0]:,.0f} ± {ajustado[1]:,.0f} {mejores}")

    # PASO 6 · test: el 20 % que el modelo nunca vio. Se mide UNA sola vez, con el modelo ya elegido.
    # rel = error relativo de cada auto; "por tramo" divide los autos de test en 5 grupos de igual cantidad según su precio.
    ### test, una sola vez
    pred_test = modelo_final.predict(X_test)
    err = pred_test - y_test
    rel = np.abs(err) / y_test
    mae_test = float(mean_absolute_error(y_test, pred_test))
    tramos = pd.qcut(y_test, 5)
    por_tramo = (pd.DataFrame({"abs": err.abs(), "rel": rel, "tramo": tramos})
                 .groupby("tramo", observed=True).agg(mae=("abs", "mean"), rel=("rel", "median")))
    por_tramo = [{"tramo": f"{iv.left/1000:,.0f}k–{iv.right/1000:,.0f}k", "orden": i,
                  "mae": float(f.mae), "rel": float(f.rel)} for i, (iv, f) in enumerate(por_tramo.iterrows())]
    sesgo_moneda = info.loc[X_test.index].assign(e=err).groupby("moneda")["e"].mean().to_dict()
    print(f"test: {mae_test:,.0f} · rel mediano {rel.median():.1%}")

    # PASO 7 · en qué se apoya: se desordena una columna por vez en el test y se mide cuánto sube el error.
    ### importancia por permutación (test)
    imp = permutation_importance(modelo_final, X_test, y_test, scoring="neg_mean_absolute_error",
                                 n_repeats=5, random_state=T.SEMILLA, n_jobs=-1)
    importancia = sorted([{"columna": c, "sube_mae": float(m), "desvio": float(s)}
                          for c, m, s in zip(X_test.columns, imp.importances_mean, imp.importances_std)],
                         key=lambda d: -d["sube_mae"])

    # PASO 8 · columnas probadas: se vuelve a entrenar el mismo boosting sacando una columna (o grupo) por vez.
    # De acá salen los números de las hipótesis ("sin año sube el error USD X") de "Lo que encontramos".
    ### columnas probadas (sacar una por vez) — sólo las que se citan en la app
    def sin(lista, col):
        return [c for c in lista if c != col]

    def pipe_cols(num, cat, te):
        from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
        tr = [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
              ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)]
        if te:
            tr.append(("te", TargetEncoder(target_type="continuous", random_state=T.SEMILLA), te))
        return Pipeline([("ingenieria", T.IngenieriaAutos()), ("prep", ColumnTransformer(tr)),
                         ("modelo", TransformedTargetRegressor(
                             regressor=HistGradientBoostingRegressor(random_state=T.SEMILLA),
                             func=np.log1p, inverse_func=np.expm1))])

    N, C, E = T.NUM_COLS, T.CAT_COLS, T.TE_COLS
    variantes = [
        ("referencia (todas)", N, C, E), ("sin modelo (sólo marca)", N, C, []),
        ("sin marca ni modelo", N, sin(C, "marca_n"), []), ("sin fabricado_en", sin(N, "fabricado_en"), C, E),
        ("sin potencia_hp", sin(N, "potencia_hp"), C, E), ("sin kilometraje", sin(N, "kilometraje"), C, E),
        ("sin motor_lt", sin(N, "motor_lt"), C, E), ("sin consumo_lt_100km", sin(N, "consumo_lt_100km"), C, E),
        ("sin transmision", N, sin(C, "transmision"), E),
        ("sin traccion ni combustible", N, [c for c in C if c not in ("traccion", "combustible")], E),
        # Hipótesis descartada: ¿mejora el modelo si le damos la ubicación? (provincia como categoría;
        # ciudad con TargetEncoder, igual que el modelo). Las columnas extra se agregan sólo para esta prueba.
        ("con provincia", N, C + ["provincia"], E),
        ("con provincia y ciudad", N, C + ["provincia"], E + ["ubicacion_c"]),
    ]
    X_train_u = X_train.assign(provincia=info.loc[X_train.index, "provincia"], ubicacion_c=info.loc[X_train.index, "ubicacion_c"])
    columnas = []
    for nombre, n, c, t in variantes:
        m, s = cv_mae(pipe_cols(n, c, t), X_train_u, y_train, cv, g_train)
        columnas.append({"variante": nombre, "mae": m, "desvio": s})
    for fila in columnas:
        fila["delta"] = fila["mae"] - columnas[0]["mae"]
    print("columnas:", [(f["variante"], round(f["delta"])) for f in columnas])

    # PASO 9 · chequeos de fuga: validar al azar contra validar por vehículo, correlación de Spearman del año con el
    # precio, y qué tan bien tasa los modelos que el TargetEncoder casi no vio.
    ### fuga: al azar vs agrupado, Spearman, TargetEncoder
    hgb = next(c for c in comparacion if c["modelo"] == "HistGradientBoosting")
    azar = cv_mae(T.pipeline_hgb(), X_train, y_train, KFold(n_splits=5, shuffle=True, random_state=T.SEMILLA))
    spearman = float(X_train["fabricado_en"].corr(y_train, method="spearman"))
    mod_train = T.IngenieriaAutos().transform(X_train)["modelo_n"].value_counts()
    frec = T.IngenieriaAutos().transform(X_test)["modelo_n"].map(mod_train).fillna(0)
    cortes = pd.cut(frec, [-1, 0, 9, 49, 100_000], labels=["0 (nunca visto)", "1–9", "10–49", "50+"])
    tabla_te = (pd.DataFrame({"abs": np.abs(err).values, "rel": rel.values, "frec": cortes.values})
                .groupby("frec", observed=True).agg(avisos=("abs", "size"), mae=("abs", "mean"), rel=("rel", "median")))
    tabla_te = [{"avisos_en_train": str(k), "autos_test": int(f.avisos), "mae": float(f.mae), "rel": float(f.rel)}
                for k, f in tabla_te.iterrows()]

    # PASO 10 · tasación de cada auto para la pestaña Explorá. Siempre "sin haberlo visto":
    # - autos de train: cross_val_predict (cada auto se tasa con un modelo entrenado sin su pedazo)
    # - autos de test: el modelo final, que no los vio
    ### tasación de cada auto, siempre fuera de muestra:
    ### train -> cross_val_predict con el modelo elegido; test -> modelo_final
    pred_oof = cross_val_predict(T.pipeline_hgb(**mejores), X_train, y_train, cv=cv, groups=g_train, n_jobs=-1)
    autos = info[["fuente", "marca", "modelo", "version", "fabricado_en", "kilometraje", "precio", "moneda",
                  "motor_lt", "potencia_hp", "transmision", "traccion", "combustible", "precio_usd", "url"]].copy()
    autos["ubicacion"] = info["ubicacion_c"]   # ciudad y provincia (ya corregidas) para el mapa y la tabla de la app
    autos["provincia"] = info["provincia"]
    ing = T.IngenieriaAutos().transform(X)
    autos["caja"] = ing["transmision"]
    autos["comb"] = ing["combustible"]
    autos["conjunto"] = "train"
    autos.loc[X_test.index, "conjunto"] = "test"
    autos["tasacion_usd"] = np.nan
    autos.loc[X_train.index, "tasacion_usd"] = pred_oof
    autos.loc[X_test.index, "tasacion_usd"] = pred_test
    autos["avisos_modelo"] = ing["modelo_n"].map(mod_train).fillna(0).astype(int)
    autos.round({"precio_usd": 0, "tasacion_usd": 0}).to_csv(SALIDA / "autos.csv", index=False)

    # PASO 11 · hallazgo del análisis: se ajusta una recta al logaritmo de la mediana de precio por año
    # (años con 30 autos o más). Si la pendiente es b, el auto pierde 1 - e^b de su valor por año.
    ### depreciación: mediana de precio por año (para el hallazgo del EDA)
    por_anio = (info[info["fabricado_en"].between(2000, 2026)].groupby("fabricado_en")["precio_usd"]
                .agg(["median", "size"]).reset_index())
    base = por_anio[por_anio["size"] >= 30]
    pendiente = np.polyfit(base["fabricado_en"], np.log(base["median"]), 1)[0]

    # PASO 12 · rangos de lo que vio el modelo (percentiles 1 y 99, mínimo y máximo): la app avisa cuando
    # alguien carga un valor fuera de ahí.
    ### rangos vistos (para avisar en "Probá el modelo")
    rangos = {c: [float(X_train[c].quantile(0.01)), float(X_train[c].quantile(0.99)),
                  float(X_train[c].min()), float(X_train[c].max())] for c in T.NUM_COLS}

    # PASO 13 · todo lo que muestra la app se guarda en metricas.json: ningún número de la app está escrito a mano.
    metricas = {
        "fecha_datos": fecha_datos, "tipo_cambio": T.TIPO_CAMBIO, "semilla": T.SEMILLA,
        "pasos": [{"paso": a, "filas": int(b), "por_que": c} for a, b, c in pasos],
        "por_fuente": info["fuente"].value_counts().to_dict(),
        "columnas_descartadas": T.COLUMNAS_SIN_USO,
        "n_train": len(X_train), "n_test": len(X_test), "vehiculos_compartidos": 0,
        "comparacion": comparacion, "mejores_parametros": mejores,
        "grilla": tabla_grilla.round({"mae_val": 1, "desvio_val": 1, "mae_train": 1}).to_dict(orient="records"),
        "test": {"mae": mae_test, "rel_mediano": float(rel.median()), "por_tramo": por_tramo,
                 "sesgo_moneda": {k: float(v) for k, v in sesgo_moneda.items()}},
        "importancia": importancia, "columnas_probadas": columnas,
        "fuga": {"mae_agrupado": hgb["mae"], "desvio_agrupado": hgb["desvio"], "mae_azar": azar[0],
                 "desvio_azar": azar[1], "spearman_anio": spearman,
                 "repetidos": int((g_train.value_counts() > 1).sum()), "vehiculos_train": int(g_train.nunique())},
        "target_encoder": tabla_te,
        "depreciacion_anual": float(1 - np.exp(-pendiente)),
        "mediana_por_anio": por_anio.to_dict(orient="records"),
        "rangos": rangos,
        "precio_mediano": float(info["precio_usd"].median()),
        "precio_p10_p90": [float(info["precio_usd"].quantile(0.1)), float(info["precio_usd"].quantile(0.9))],
    }
    # Mapa de la ubicación: un punto por ciudad (coordenadas de data/ubicaciones_coordenadas.csv, geocodificadas una vez)
    # con cuántos autos hay y el precio mediano. También las provincias simplificadas para dibujar el contorno.
    ub = (info.groupby("ubicacion_c").agg(autos=("precio_usd", "size"), mediana_usd=("precio_usd", "median"))
          .reset_index().rename(columns={"ubicacion_c": "ubicacion_corregida"}))
    # Coordenadas: coordenadas.py usa las de data/ubicaciones_coordenadas.csv y, para las ubicaciones nuevas, las pide a la
    # API oficial Georef (departamento -> localidad cabecera) y las capitales a Wikidata; sin red, el centro de la provincia.
    coords = geo_coords.obtener(ub["ubicacion_corregida"], T.provincia_de, usar_api=not sin_api)
    ub = ub.merge(coords, on="ubicacion_corregida", how="left")
    ub["mediana_usd"] = ub["mediana_usd"].round(0)
    n_sin_coordenadas = int(ub["latitud"].isna().sum())
    ub["aproximada"] = ub["aproximada"].fillna(False).astype(bool)
    # Qué autos hay en cada ciudad (para el cartel del mapa): los 3 modelos más publicados con su cantidad y precio
    # mediano, y el auto más caro y el más barato. Los nombres son los de mostrar (marca y modelo unificados).
    nombre = info["marca"] + " " + info["modelo"]

    def resumen_ciudad(idx):
        g = info.loc[idx.index].assign(nombre=nombre.loc[idx.index])
        top = g.groupby("nombre")["precio_usd"].agg(["size", "median"]).sort_values("size", ascending=False).head(3)
        txt = "\n".join(f"{n} ({int(r['size'])}) · USD {r['median']:,.0f}".replace(",", ".") for n, r in top.iterrows())
        caro, barato = g.loc[g["precio_usd"].idxmax()], g.loc[g["precio_usd"].idxmin()]
        fmt = lambda r: f"{r['nombre']} {int(r['fabricado_en'])} · USD {r['precio_usd']:,.0f}".replace(",", ".")
        return pd.Series({"mas_publicados": txt, "mas_caro": fmt(caro), "mas_barato": fmt(barato)})

    ub = ub.merge(info.groupby("ubicacion_c").apply(resumen_ciudad, include_groups=False).reset_index()
                  .rename(columns={"ubicacion_c": "ubicacion_corregida"}), on="ubicacion_corregida", how="left")
    ub.to_csv(SALIDA / "ubicaciones.csv", index=False, encoding="utf-8")
    (SALIDA / "provincias.geojson").write_bytes((AQUI / "data" / "provincias.geojson").read_bytes())
    por_prov = info["provincia"].value_counts()
    ubicacion = {"eta2_provincia": T.eta_cuadrado(info["precio_usd"], info["provincia"]),
                 "eta2_ciudad": T.eta_cuadrado(info["precio_usd"], info["ubicacion_c"]),
                 "eta2_marca": T.eta_cuadrado(info["precio_usd"], info["marca"]),
                 "n_ciudades": int(info["ubicacion_c"].nunique()), "n_provincias": int(info["provincia"].nunique()),
                 "sin_coordenadas": n_sin_coordenadas,
                 "top_provincias": {k: int(v) for k, v in por_prov.head(3).items()},
                 "pct_top_provincias": float(por_prov.head(3).sum() / por_prov.sum())}
    metricas["ubicacion"] = ubicacion
    (SALIDA / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    joblib.dump(modelo_final, SALIDA / "modelo.joblib", compress=3)
    print("listo:", SALIDA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default=str(AQUI / "data" / "dataset.csv"))
    ap.add_argument("--sin-api", action="store_true",
                    help="no consulta la API de coordenadas: las ciudades nuevas van al centro de su provincia")
    args = ap.parse_args()
    main(args.datos, args.sin_api)
