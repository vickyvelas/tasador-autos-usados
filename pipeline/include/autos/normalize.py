"""Funciones de limpieza compartidas por los recolectores.

Regla general: si un valor no se puede leer, se devuelve None (queda vacío / NaN
en el CSV), nunca 0. Un 0 se confunde con un dato real (por ejemplo 0 km).
"""
import re
import unicodedata

_NUM = re.compile(r"(\d+(\.\d+)?)")


def remove_accents(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _vacio(val) -> bool:
    return val is None or (isinstance(val, str) and val.strip() == "")


def as_number(val) -> float | None:
    """Precios y km: el punto es separador de miles y la coma es decimal.
    "$ 12.500.000" -> 12500000.0 | "15,5" -> 15.5 | "" -> None"""
    if _vacio(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(".", "").replace(",", ".").strip()
    match = _NUM.search(s)
    return float(match.group(1)) if match else None


def as_int(val) -> int | None:
    num = as_number(val)
    return int(num) if num is not None else None


def parse_tecnico(val) -> float | None:
    """Motor, consumo y HP: el punto sí es decimal. "1,6" -> 1.6 | "120 CV" -> 120.0"""
    if _vacio(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", ".").strip()
    match = _NUM.search(s)
    return float(match.group(1)) if match else None


def parse_consumo(val) -> float | None:
    return parse_tecnico(val)


def parse_motor(val) -> float | None:
    """Cilindrada en litros: 1600 (cc) -> 1.6 | "1.7 lts" -> 1.7 | 0 o vacío -> None"""
    num = parse_tecnico(val)
    if not num:
        return None
    if num > 100:
        return round(num / 1000, 1)
    return num


def clean_price_and_currency(text: str) -> tuple[float | None, str]:
    """"U$S 15.000" -> (15000.0, "USD") | "$ 21.500.000" -> (21500000.0, "ARS")"""
    if not text:
        return None, "ARS"
    t = text.upper()
    currency = "USD" if ("U$" in t or "USD" in t) else "ARS"
    precio = as_number(t)
    return (precio if precio else None), currency


def _slug(text: str) -> str:
    return remove_accents(text).replace(" ", "-")


def format_consumption_carone(val) -> float | None:
    return parse_tecnico(val)
