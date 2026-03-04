import json
import math
from typing import Optional
from langchain_core.tools import tool
from services.data_service import get_vehicles, get_vehicle_by_name, get_catalog_summary

MOTO_TYPE = "موتوسيكل"


def _clean(val):
    """Convert NaN to None so JSON serialization works cleanly."""
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _to_json(vehicles: list) -> str:
    clean = [{k: _clean(v) for k, v in vehicle.items()} for vehicle in vehicles]
    return json.dumps(clean, ensure_ascii=False)


@tool
def search_motorcycles(
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    company: Optional[str] = None,
    transmission: Optional[str] = None,
    limit: int = 3,
) -> str:
    """Search available motorcycles by filters. Returns a JSON array of matching motorcycles."""
    filters = {"type": MOTO_TYPE}
    if max_price:
        filters["max_price"] = max_price
    if min_price:
        filters["min_price"] = min_price
    if company:
        filters["company"] = company
    if transmission:
        filters["transmission"] = transmission
    vehicles = get_vehicles(filters, limit=limit)
    return _to_json(vehicles)


@tool
def motorcycle_details(vehicle_name: str) -> str:
    """Get full details of a specific motorcycle by name. Returns a JSON object."""
    v = get_vehicle_by_name(vehicle_name)
    if not v:
        return json.dumps({"error": "الموديل غير متوفر"}, ensure_ascii=False)
    return _to_json([v])


@tool
def cheapest_motorcycles(limit: int = 3) -> str:
    """Get the cheapest available motorcycles sorted by price. Returns a JSON array."""
    vehicles = get_vehicles({"type": MOTO_TYPE}, limit=limit, sort_by="price", ascending=True)
    return _to_json(vehicles)


@tool
def motorcycle_installments(vehicle_name: str) -> str:
    """Get all installment plan options (6, 12, 18, 24 months) for a motorcycle. Returns JSON."""
    v = get_vehicle_by_name(vehicle_name)
    if not v:
        return json.dumps({"error": "الموديل غير متوفر"}, ensure_ascii=False)
    result = {
        "name_ar": v["name_ar"],
        "name_en": v["name_en"],
        "price": _clean(v["price"]),
        "min_down": _clean(v["min_down"]),
        "installment_6": _clean(v["installment_6"]),
        "installment_12": _clean(v["installment_12"]),
        "installment_18": _clean(v["installment_18"]),
        "installment_24": _clean(v["installment_24"]),
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def motorcycle_by_monthly_budget(max_monthly: float, months: int = 12, limit: int = 3) -> str:
    """Find motorcycles that fit a monthly installment budget. months must be 6,12,18 or 24. Returns JSON."""
    if months not in (6, 12, 18, 24):
        return json.dumps({"error": "مدة التقسيط يجب أن تكون 6 أو 12 أو 18 أو 24 شهراً"}, ensure_ascii=False)
    filters = {f"max_installment_{months}": max_monthly, "type": MOTO_TYPE}
    vehicles = get_vehicles(filters, limit=limit, sort_by=f"installment_{months}", ascending=True)
    return _to_json(vehicles)


@tool
def motorcycle_catalog_summary() -> str:
    """Get a summary of all available motorcycles: total count, companies, price range. Returns JSON."""
    s = get_catalog_summary()
    moto_types = {k: v for k, v in s["types"].items() if MOTO_TYPE in k}
    result = {
        "total": s["total"],
        "types": moto_types or s["types"],
        "companies": s["companies"],
        "price_min": _clean(s["price_min"]),
        "price_max": _clean(s["price_max"]),
    }
    return json.dumps(result, ensure_ascii=False)
