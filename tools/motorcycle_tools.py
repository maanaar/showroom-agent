import json
from typing import Optional
from langchain_core.tools import tool
from services.data_service import get_vehicles, get_vehicle_by_name, get_catalog_summary, calculate_custom_installment

MOTO_TYPE = "موتوسيكل"


def _to_json(vehicles: list) -> str:
    return json.dumps(vehicles, ensure_ascii=False, default=str)


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
    return _to_json(get_vehicles(filters, limit=limit))


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
    return _to_json(get_vehicles({"type": MOTO_TYPE}, limit=limit, sort_by="price", ascending=True))


@tool
def motorcycle_installments(vehicle_name: str) -> str:
    """Get installment plan options (6, 12, 18, 24 months) for a motorcycle. Returns JSON."""
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
def motorcycle_by_monthly_budget(max_monthly: float, months: int = 12, limit: int = 3) -> str:
    """Find motorcycles that fit a monthly installment budget. Returns JSON."""
    from services.db_service import get_installment_rate
    rate_data = get_installment_rate(months, 0)
    if not rate_data:
        return json.dumps({"error": "لا تتوفر بيانات تقسيط لهذه المدة"}, ensure_ascii=False)

    # monthly = price * (1 + rate_per_month/100 * months) / months
    # → max_price = max_monthly * months / (1 + rate_per_month/100 * months)
    rate = rate_data["percentage_per_month"]
    max_price = max_monthly * months / (1 + rate / 100 * months)
    return _to_json(get_vehicles({"type": MOTO_TYPE, "max_price": max_price}, limit=limit, sort_by="price", ascending=True))


@tool
def motorcycle_catalog_summary() -> str:
    """Get a summary of all available motorcycles: total count, companies, price range. Returns JSON."""
    s = get_catalog_summary()
    moto_types = {k: v for k, v in s["types"].items() if MOTO_TYPE in k}
    return json.dumps({
        "total":     s["total"],
        "types":     moto_types or s["types"],
        "companies": s["companies"],
        "price_min": s["price_min"],
        "price_max": s["price_max"],
    }, ensure_ascii=False, default=str)
