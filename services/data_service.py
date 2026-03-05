import re
import math
from typing import Optional, List


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn():
    from services.db_service import _get_conn as _db_conn
    return _db_conn()


def _motor_to_dict(row) -> dict:
    return {
        "name_en":       row["english_name"],
        "name_ar":       row["arabic_name"],
        "company":       row["company"],
        "agent":         row["agency_name"],
        "type":          row["moto_type"],
        "price":         row["price"],
        "engine_cc":     row["engin_capacity"],
        "engine_type":   row["engin_type"],
        "transmission":  row["transmission_type"],
        "max_speed":     row["max_speed"],
        "fuel_capacity": row["fule_capacity"],
        "brakes":        row["brake_type"],
        "notes":         row["notes"],
        "color":         row["colors"],
        "available":     row["is_available"],
        "condition":     row["status"],
        "img_url":       row["img_url"],
    }


# ---------------------------------------------------------------------------
# Vehicle queries
# ---------------------------------------------------------------------------

def get_vehicles(
    filters: dict = None,
    limit: int = 6,
    sort_by: str = None,
    ascending: bool = True,
) -> List[dict]:
    conn = _get_conn()
    query = "SELECT * FROM motors WHERE is_available = 1"
    params = []

    if filters:
        vtype = filters.get("type")
        if vtype:
            query += " AND moto_type LIKE ?"
            params.append(f"%{vtype}%")

        max_price = filters.get("max_price")
        if max_price:
            query += " AND price <= ?"
            params.append(float(max_price))

        min_price = filters.get("min_price")
        if min_price:
            query += " AND price >= ?"
            params.append(float(min_price))

        company = filters.get("company")
        if company:
            query += " AND company LIKE ?"
            params.append(f"%{company}%")

        transmission = filters.get("transmission")
        if transmission:
            query += " AND transmission_type LIKE ?"
            params.append(f"%{transmission}%")

        condition = filters.get("condition")
        if condition:
            query += " AND status LIKE ?"
            params.append(f"%{condition}%")

    col_map = {"price": "price", "engine_cc": "engin_capacity"}
    if sort_by and sort_by in col_map:
        direction = "ASC" if ascending else "DESC"
        query += f" ORDER BY {col_map[sort_by]} {direction}"

    query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [_motor_to_dict(r) for r in rows]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06ff]", "", s.lower())


def get_vehicle_by_name(name: str) -> Optional[dict]:
    """Four-tier search (case-insensitive):
    1. SQL LIKE — consecutive substring in English or Arabic name.
    2. ALL query words are whole tokens in English name + company (exact multi-word match).
    3. MAJORITY (≥2/3) of words match across English + Arabic + company tokens
       — handles Arabic transliteration like "سيم اس تي" → "Symphony ST".
    4. Normalized match (strip punctuation/spaces then substring check).
    """
    conn = _get_conn()
    q = name.lower().strip()

    # Tier 1 — SQL LIKE
    rows = conn.execute(
        "SELECT * FROM motors WHERE LOWER(english_name) LIKE ? OR arabic_name LIKE ?",
        (f"%{q}%", f"%{name.strip()}%"),
    ).fetchall()
    if rows:
        return _motor_to_dict(rows[0])

    # Tiers 2–4 load all rows once
    all_rows = conn.execute("SELECT * FROM motors").fetchall()

    words = q.split()
    if words:
        def _all_tokens(row):
            return (
                set(re.split(r'\W+', str(row["english_name"]).lower()))
                | set(re.split(r'\W+', str(row["arabic_name"])))
                | set(re.split(r'\W+', str(row["company"]).lower()))
            )

        # Tier 2 — ALL words match (precise)
        for row in all_rows:
            if all(w in _all_tokens(row) for w in words):
                return _motor_to_dict(row)

        # Tier 3 — MAJORITY match (≥2/3) for transliterated Arabic queries
        threshold = max(1, math.ceil(len(words) * 2 / 3))
        for row in all_rows:
            toks = _all_tokens(row)
            if sum(1 for w in words if w in toks) >= threshold:
                return _motor_to_dict(row)

    # Tier 4 — normalized substring
    q_norm = _normalize(q)
    if q_norm:
        for row in all_rows:
            if q_norm in _normalize(str(row["english_name"])):
                return _motor_to_dict(row)

    return None


def get_catalog_summary() -> dict:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT moto_type, company, price FROM motors WHERE is_available = 1"
    ).fetchall()

    types     = {}
    companies = {}
    prices    = []
    for r in rows:
        types[r["moto_type"]]   = types.get(r["moto_type"], 0) + 1
        companies[r["company"]] = companies.get(r["company"], 0) + 1
        if r["price"] is not None:
            prices.append(r["price"])

    return {
        "total":     len(rows),
        "types":     types,
        "companies": companies,
        "price_min": min(prices) if prices else 0,
        "price_max": max(prices) if prices else 0,
    }


def get_price_spread(filters: dict = None, count: int = 5) -> List[dict]:
    """Return `count` vehicles evenly spread across the price range."""
    conn = _get_conn()
    query  = "SELECT * FROM motors WHERE is_available = 1 AND price IS NOT NULL"
    params = []

    if filters:
        vtype = filters.get("type")
        if vtype:
            query += " AND moto_type LIKE ?"
            params.append(f"%{vtype}%")
        company = filters.get("company")
        if company:
            query += " AND company LIKE ?"
            params.append(f"%{company}%")
        max_price = filters.get("max_price")
        if max_price is not None:
            query += " AND price <= ?"
            params.append(float(max_price))
        min_price = filters.get("min_price")
        if min_price is not None:
            query += " AND price >= ?"
            params.append(float(min_price))

    query += " ORDER BY price"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        return []
    if len(rows) <= count:
        return [_motor_to_dict(r) for r in rows]

    step    = (len(rows) - 1) / (count - 1)
    indices = sorted({round(i * step) for i in range(count)})
    return [_motor_to_dict(rows[i]) for i in indices]


def get_similar_vehicles(vehicle: dict, count: int = 3) -> List[dict]:
    """Return up to `count` vehicles of the same type within ±40% of the given price,
    excluding the vehicle itself."""
    price    = vehicle.get("price")
    vtype    = vehicle.get("type")
    name_en  = vehicle.get("name_en", "")
    if not price or not vtype:
        return []
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM motors "
        "WHERE is_available = 1 AND price IS NOT NULL "
        "  AND moto_type LIKE ? "
        "  AND price BETWEEN ? AND ? "
        "  AND LOWER(english_name) != LOWER(?) "
        "ORDER BY ABS(price - ?) "
        "LIMIT ?",
        (f"%{vtype}%", price * 0.6, price * 1.4, name_en, price, count),
    ).fetchall()
    return [_motor_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Installment calculation (rates from instalments table, not per-vehicle)
# ---------------------------------------------------------------------------

def calculate_custom_installment(vehicle: dict, months: int, down_payment: float = 0) -> dict:
    """Calculate monthly installment using the range-based instalments table.

    Rates are universal (not per vehicle):
      total_interest = price * (percentage_per_month / 100) * months

    Args:
        vehicle:      vehicle dict with at least 'price', 'name_ar', 'name_en'
        months:       number of installment months requested
        down_payment: down payment amount in EGP (default 0 = no down payment)
    """
    from services.db_service import get_installment_rate

    price   = vehicle.get("price")
    name_en = vehicle.get("name_en") or ""

    if not price:
        return {"error": "سعر المنتج غير متوفر", "name_ar": vehicle.get("name_ar"), "name_en": name_en}

    down_pct  = (down_payment / price * 100) if down_payment and price else 0
    rate_data = get_installment_rate(months, down_pct)

    if not rate_data:
        return {
            "error": "لا تتوفر خطة تقسيط لهذا الخيار",
            "name_ar": vehicle.get("name_ar"),
            "name_en": name_en,
        }

    monthly_rate_pct = rate_data["percentage_per_month"]
    financed_amount  = price - down_payment          # interest applies only to financed amount
    total_interest   = financed_amount * monthly_rate_pct / 100 * months
    total_repayment  = price + total_interest        # down_payment + financed + interest
    monthly_payment  = (financed_amount + total_interest) / months

    return {
        "name_ar":           vehicle.get("name_ar"),
        "name_en":           name_en,
        "price":             price,
        "down_payment":      round(down_payment),
        "financed_amount":   round(financed_amount),
        "months":            months,
        "interest_rate_pct": round(monthly_rate_pct * months, 2),
        "total_interest":    round(total_interest),
        "total_repayment":   round(total_repayment),
        "monthly_payment":   round(monthly_payment),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_price(val) -> str:
    try:
        return f"{int(val):,} جنيه"
    except (ValueError, TypeError):
        return "غير محدد"


def _safe(val, default="غير محدد") -> str:
    if val is None:
        return default
    try:
        if math.isnan(float(val)):
            return default
    except (TypeError, ValueError):
        pass
    return str(val)


def _has_value(val) -> bool:
    if val is None:
        return False
    try:
        return not math.isnan(float(val))
    except (TypeError, ValueError):
        return bool(val)


def format_vehicle_arabic(v: dict) -> str:
    lines = [
        f"* {_safe(v.get('name_ar'))} ({_safe(v.get('name_en'))})",
        f"   الشركة: {_safe(v.get('company'))} | الوكيل: {_safe(v.get('agent'))}",
        f"   النوع: {_safe(v.get('type'))} | اللون: {_safe(v.get('color'))}",
        f"   السعر: {_fmt_price(v.get('price'))}",
        f"   المحرك: {_safe(v.get('engine_cc'))} | {_safe(v.get('engine_type'))} | {_safe(v.get('transmission'))}",
        f"   السرعة القصوى: {_safe(v.get('max_speed'))}",
    ]
    if _has_value(v.get("notes")):
        lines.append(f"   ملاحظات: {v['notes']}")
    return "\n".join(lines)
