"""Recolector de CarOne: API GraphQL pública del sitio (datos estructurados)."""
import logging
from datetime import datetime

import requests

from ..normalize import as_int, as_number, parse_motor, parse_tecnico

logger = logging.getLogger(__name__)

_GRAPHQL_URL = "https://carone.com.ar/api/graphql"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
    "Content-Type": "application/json",
    "x-v6-country": "ar",
    "Origin": "https://carone.com.ar",
    "Referer": "https://carone.com.ar/comprar?carOptions=usados",
}

PAGE_SIZE = 20
MAX_PAGINAS = 500  # tope de seguridad: 500 x 20 = 10.000 autos

_QUERY_PRODUCTOS = """
query GetProductsCard($q: String!, $pageSize: Int!, $currentPage: Int!, $filter: ProductAttributeFilterInput) {
  products(search: $q, pageSize: $pageSize, currentPage: $currentPage, filter: $filter) {
    total_count
    items {
      sku name url_key carone_year carone_mileage carone_potency
      carone_cylinder_capacity carone_consumption
      carone_marca_data { label } carone_modelo_data { label }
      carone_transmission_data { label } carone_traction_data { label }
      carone_fuel_data { label } carone_dealer_id
      price_range { maximum_price { final_price { currency value } } }
    }
  }
}
"""


def get_available_brands() -> list[str]:
    """Lista de marcas del catálogo. Se usa para saber si la fuente está viva."""
    payload = {
        "operationName": "CatalogFilters",
        "variables": {"filters": {}},
        "query": "query CatalogFilters($filters: CatalogFiltersInput) { catalogFilters(filters: $filters) { brands { default { label } others { label } } } }",
    }
    try:
        resp = requests.post(_GRAPHQL_URL, json=payload, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = ((resp.json().get("data") or {}).get("catalogFilters") or {}).get("brands") or {}
        all_brands = (data.get("default") or []) + (data.get("others") or [])
        return [b["label"] for b in all_brands if b.get("label")]
    except Exception as e:
        logger.error(f"[carone] Error descubriendo marcas: {e}")
        return []


def _a_registro(item: dict) -> dict:
    price_info = ((item.get("price_range") or {}).get("maximum_price") or {}).get("final_price") or {}
    return {
        "fuente": "carone",
        "id_publicacion": item.get("sku"),
        "marca": (item.get("carone_marca_data") or {}).get("label"),
        "modelo": (item.get("carone_modelo_data") or {}).get("label"),
        "version": item.get("name"),
        "fabricado_en": as_int(item.get("carone_year")),
        "kilometraje": as_int(item.get("carone_mileage")),
        "precio": as_number(price_info.get("value")) or None,
        "moneda": price_info.get("currency") or "ARS",
        "motor_lt": parse_motor(item.get("carone_cylinder_capacity")),
        "potencia_hp": parse_tecnico(item.get("carone_potency")) or None,
        "transmision": (item.get("carone_transmission_data") or {}).get("label"),
        "traccion": (item.get("carone_traction_data") or {}).get("label"),
        "combustible": (item.get("carone_fuel_data") or {}).get("label"),
        "consumo_lt_100km": parse_tecnico(item.get("carone_consumption")) or None,
        # La API sólo da el id del concesionario, no la ciudad
        "ubicacion": item.get("carone_dealer_id"),
        "url": f"https://carone.com.ar/comprar/usados/{item.get('url_key')}",
        "fecha_ingesta": datetime.now().isoformat(),
    }


def search(marca: str = None, modelo: str = None) -> list[dict]:
    """Todos los autos usados en stock, paginando de a 20.

    Si una página falla se lanza la excepción: Airflow reintenta la tarea en vez
    de guardar un bronce incompleto sin avisar (antes había un `except: break`)."""
    results = []
    filters = {"stock_status": {"eq": "IN_STOCK"}, "carone_tags_arg": {"in": [2]}}
    if marca:
        filters["carone_marca_label"] = {"eq": marca}

    for current_page in range(1, MAX_PAGINAS + 1):
        payload = {
            "operationName": "GetProductsCard",
            "variables": {"q": "", "pageSize": PAGE_SIZE, "currentPage": current_page,
                          "sort": {"created_at": "DESC"}, "filter": filters},
            "query": _QUERY_PRODUCTOS,
        }
        resp = requests.post(_GRAPHQL_URL, json=payload, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = (resp.json().get("data") or {}).get("products") or {}
        items = data.get("items") or []
        total_disponible = data.get("total_count") or 0
        if not items:
            break
        results.extend(_a_registro(item) for item in items)
        logger.info(f"[carone] página {current_page}: {len(results)}/{total_disponible} autos")
        if len(results) >= total_disponible:
            break
    return results
