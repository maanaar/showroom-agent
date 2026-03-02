import json
import math
from langchain_core.tools import tool
from services.data_service import get_vehicles, get_vehicle_by_name

SCOOTER_TYPE = "اسكوتر"


def _clean(val):
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _to_json(vehicles: list) -> str:
    clean = [{k: _clean(v) for k, v in vehicle.items()} for vehicle in vehicles]
    return json.dumps(clean, ensure_ascii=False)


@tool
def search_scooters(
    max_price: float = None,
    min_price: float = None,
    company: str = None,
    transmission: str = None,
    limit: int = 3,
) -> str:
    """Search available scooters by filters. Returns a JSON array of matching scooters."""
    filters = {"type": SCOOTER_TYPE}
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
def scooter_details(vehicle_name: str) -> str:
    """Get full details of a specific scooter by name. Returns a JSON object."""
    v = get_vehicle_by_name(vehicle_name)
    if not v:
        return json.dumps({"error": "الموديل غير متوفر"}, ensure_ascii=False)
    return _to_json([v])


@tool
def cheapest_scooters(limit: int = 3) -> str:
    """Get the cheapest available scooters sorted by price. Returns a JSON array."""
    vehicles = get_vehicles({"type": SCOOTER_TYPE}, limit=limit, sort_by="price", ascending=True)
    return _to_json(vehicles)


@tool
def scooter_installments(vehicle_name: str) -> str:
    """Get all installment plan options for a scooter. Returns JSON."""
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
def scooter_by_monthly_budget(max_monthly: float, months: int = 12, limit: int = 3) -> str:
    """Find scooters that fit a monthly installment budget. months must be 6,12,18 or 24. Returns JSON."""
    if months not in (6, 12, 18, 24):
        return json.dumps({"error": "مدة التقسيط يجب أن تكون 6 أو 12 أو 18 أو 24 شهراً"}, ensure_ascii=False)
    filters = {f"max_installment_{months}": max_monthly, "type": SCOOTER_TYPE}
    vehicles = get_vehicles(filters, limit=limit, sort_by=f"installment_{months}", ascending=True)
    return _to_json(vehicles)
