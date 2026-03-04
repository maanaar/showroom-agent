import json
import math
from typing import Optional
from langchain_core.tools import tool
from services.data_service import get_vehicles, get_vehicle_by_name

HELMET_TYPE = "خوذة"


def _clean(val):
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _to_json(items: list) -> str:
    clean = [{k: _clean(v) for k, v in item.items()} for item in items]
    return json.dumps(clean, ensure_ascii=False)


@tool
def search_helmets(
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    company: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Search available helmets by filters. Returns a JSON array of matching helmets."""
    filters = {"type": HELMET_TYPE}
    if max_price:
        filters["max_price"] = max_price
    if min_price:
        filters["min_price"] = min_price
    if company:
        filters["company"] = company
    items = get_vehicles(filters, limit=limit)
    return _to_json(items)


@tool
def helmet_details(name: str) -> str:
    """Get full details of a specific helmet by name. Returns a JSON object."""
    v = get_vehicle_by_name(name)
    if not v:
        return json.dumps({"error": "هذا الموديل غير متوفر"}, ensure_ascii=False)
    return _to_json([v])


@tool
def cheapest_helmets(limit: int = 5) -> str:
    """Get the cheapest available helmets sorted by price. Returns a JSON array."""
    items = get_vehicles({"type": HELMET_TYPE}, limit=limit, sort_by="price", ascending=True)
    return _to_json(items)
