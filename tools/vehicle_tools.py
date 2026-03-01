from langchain_core.tools import tool
from services.data_service import (
    get_vehicles,
    get_vehicle_by_name,
    get_catalog_summary,
    format_vehicle_arabic,
    _fmt_price,
    _safe,
)


@tool
def search_vehicles(
    type: str = None,
    max_price: float = None,
    company: str = None,
    transmission: str = None,
):
    """Search motorcycles using filters."""

    filters = {
        "type": type,
        "max_price": max_price,
        "company": company,
        "transmission": transmission,
    }

    vehicles = get_vehicles(filters)

    if not vehicles:
        return "لا توجد نتائج حالياً."

    return "\n\n".join(
        format_vehicle_arabic(v)
        for v in vehicles
    )


@tool
def vehicle_details(vehicle_name: str):
    """Get details for a specific motorcycle."""

    vehicle = get_vehicle_by_name(vehicle_name)

    if not vehicle:
        return "هذا الموديل غير متوفر."

    return format_vehicle_arabic(vehicle)


@tool
def catalog_summary():
    """Get an overview of available motorcycles: total count, types, companies, and price range."""

    s = get_catalog_summary()
    types = " | ".join(f"{k} ({v})" for k, v in s["types"].items())
    companies = " | ".join(f"{k} ({v})" for k, v in s["companies"].items())
    return (
        f"إجمالي الموتوسيكلات المتاحة: {s['total']}\n"
        f"الأنواع: {types}\n"
        f"الشركات: {companies}\n"
        f"نطاق الأسعار: من {_fmt_price(s['price_min'])} إلى {_fmt_price(s['price_max'])}"
    )


@tool
def get_installment_plans(vehicle_name: str):
    """Get all installment plan options (6, 12, 18, 24 months) for a specific motorcycle."""

    v = get_vehicle_by_name(vehicle_name)
    if not v:
        return "هذا الموديل غير متوفر."

    lines = [f"خطط التقسيط لـ {_safe(v['name_ar'])} ({_safe(v['name_en'])})"]
    lines.append(f"   السعر الكامل: {_fmt_price(v['price'])}")
    lines.append(f"   أقل مقدم: {_fmt_price(v['min_down'])}")
    for months, key in ((6, "installment_6"), (12, "installment_12"),
                        (18, "installment_18"), (24, "installment_24")):
        lines.append(f"   قسط {months} شهر: {_fmt_price(v[key])}/شهر")
    return "\n".join(lines)


@tool
def compare_vehicles(vehicle_name_1: str, vehicle_name_2: str):
    """Compare two motorcycles side by side."""

    v1 = get_vehicle_by_name(vehicle_name_1)
    v2 = get_vehicle_by_name(vehicle_name_2)

    if not v1:
        return f"الموديل '{vehicle_name_1}' غير متوفر."
    if not v2:
        return f"الموديل '{vehicle_name_2}' غير متوفر."

    def row(label, k):
        return f"   {label}: {_safe(v1[k])}  |  {_safe(v2[k])}"

    lines = [
        f"مقارنة: {_safe(v1['name_ar'])}  VS  {_safe(v2['name_ar'])}",
        row("الشركة", "company"),
        row("النوع", "type"),
        f"   السعر: {_fmt_price(v1['price'])}  |  {_fmt_price(v2['price'])}",
        row("المحرك", "engine_cc"),
        row("نوع المحرك", "engine_type"),
        row("ناقل الحركة", "transmission"),
        row("السرعة القصوى", "max_speed"),
        f"   قسط 12 شهر: {_fmt_price(v1['installment_12'])}/شهر  |  {_fmt_price(v2['installment_12'])}/شهر",
    ]
    return "\n".join(lines)


@tool
def cheapest_vehicles(
    type: str = None,
    company: str = None,
    limit: int = 5,
):
    """Get the cheapest available motorcycles sorted by price, with optional type/company filter."""

    filters = {"type": type, "company": company}
    vehicles = get_vehicles(filters, limit=limit, sort_by="price", ascending=True)

    if not vehicles:
        return "لا توجد نتائج حالياً."

    return "\n\n".join(format_vehicle_arabic(v) for v in vehicles)


@tool
def similar_vehicles(query: str, k: int = 5):
    """Find motorcycles similar to a natural language description using vector similarity search.
    Use this as a fallback when filters return no results, or for vague queries like
    'موتوسيكل رياضي سريع' or 'مناسب للمدينة واقتصادي في البنزين'.
    Always returns results ranked by semantic similarity."""

    from services.vector_service import search_similar
    try:
        vehicles = search_similar(query, k=k)
    except Exception as e:
        return f"خطأ في البحث المتشابه: {e}"

    if not vehicles:
        return "لا توجد نتائج."

    return "\n\n".join(format_vehicle_arabic(v) for v in vehicles)


@tool
def search_by_monthly_budget(
    max_monthly: float,
    months: int = 12,
    type: str = None,
    company: str = None,
):
    """Find motorcycles where the monthly installment fits within a budget.
    months must be one of: 6, 12, 18, 24."""

    if months not in (6, 12, 18, 24):
        return "مدة التقسيط يجب أن تكون 6 أو 12 أو 18 أو 24 شهراً."

    filters = {
        f"max_installment_{months}": max_monthly,
        "type": type,
        "company": company,
    }
    vehicles = get_vehicles(filters, sort_by=f"installment_{months}", ascending=True)

    if not vehicles:
        return f"لا توجد موتوسيكلات بقسط أقل من {_fmt_price(max_monthly)} على {months} شهر."

    return "\n\n".join(format_vehicle_arabic(v) for v in vehicles)