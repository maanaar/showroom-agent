import json
from typing import Optional
from langchain_core.tools import tool
from services.data_service import get_vehicles, get_vehicle_by_name, calculate_custom_installment

SCOOTER_TYPE = "اسكوتر"


def _to_json(vehicles: list) -> str:
    return json.dumps(vehicles, ensure_ascii=False, default=str)


@tool
def search_scooters(
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    company: Optional[str] = None,
    transmission: Optional[str] = None,
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
    return _to_json(get_vehicles(filters, limit=limit))


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
    return _to_json(get_vehicles({"type": SCOOTER_TYPE}, limit=limit, sort_by="price", ascending=True))


@tool
def scooter_installments(vehicle_name: str) -> str:
    """Get installment plan options (6, 12, 18, 24 months) for a scooter. Returns JSON."""
    v = get_vehicle_by_name(vehicle_name)
    if not v:
        return json.dumps({"error": "الموديل غير متوفر"}, ensure_ascii=False)

    plans = {}
    for months in (6, 12, 18, 24):
        calc = calculate_custom_installment(v, months)
        if "error" not in calc:
            plans[str(months)] = {
                "monthly_payment":   calc["monthly_payment"],
                "total_repayment":   calc["total_repayment"],
                "interest_rate_pct": calc["interest_rate_pct"],
            }

    return json.dumps({
        "name_ar": v["name_ar"],
        "name_en": v["name_en"],
        "price":   v["price"],
        "plans":   plans,
    }, ensure_ascii=False, default=str)


@tool
def scooter_by_monthly_budget(max_monthly: float, months: int = 12, limit: int = 3) -> str:
    """Find scooters that fit a monthly installment budget. Returns JSON."""
    from services.db_service import get_installment_rate
    rate_data = get_installment_rate(months, 0)
    if not rate_data:
        return json.dumps({"error": "لا تتوفر بيانات تقسيط لهذه المدة"}, ensure_ascii=False)

    rate = rate_data["percentage_per_month"]
    max_price = max_monthly * months / (1 + rate / 100 * months)
    return _to_json(get_vehicles({"type": SCOOTER_TYPE, "max_price": max_price}, limit=limit, sort_by="price", ascending=True))
