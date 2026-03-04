import re
import math
import pandas as pd
from pathlib import Path
from typing import Optional, List

DATA_PATH = Path(__file__).parent.parent / "data" / "vehicles.xlsx"

# Arabic column names
COL_NAME_EN = "الاسم بالانجليزى"
COL_NAME_AR = "الاسم بالعربي"
COL_COMPANY = "الشركة"
COL_AGENT = "وكيل مصر"
COL_TYPE = "نوع المركبة"
COL_PRICE = "سعر البيع"
COL_ENGINE_CC = "سعة المحرك / القدرة"
COL_ENGINE_TYPE = "نوع المحرك"
COL_TRANSMISSION = "ناقل الحركة"
COL_MAX_SPEED = "السرعة القصوى"
COL_FUEL = "سعة الوقود/البطارية"
COL_BRAKES = "الفرامل"
COL_NOTES = "ملاحظات إضافية"
COL_DOWN = "اقل مقدم"
COL_INST_6 = "قسط على 6 شهور"
COL_INST_12 = "قسط على سنه"
COL_INST_18 = "قسط على 18 شهر"
COL_INST_24 = "قسط على سنتين"
COL_COLOR = "الون"
COL_AVAILABLE = "متاح/غير متاح"
COL_CONDITION = "الحاله"

_df: Optional[pd.DataFrame] = None


def _load() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_excel(DATA_PATH)
        # Coerce numeric columns
        for col in (COL_PRICE, COL_DOWN, COL_INST_6, COL_INST_12, COL_INST_18, COL_INST_24):
            _df[col] = pd.to_numeric(_df[col], errors="coerce")
    return _df


def _vehicle_to_dict(row: pd.Series) -> dict:
    return {
        "name_en": row.get(COL_NAME_EN),
        "name_ar": row.get(COL_NAME_AR),
        "company": row.get(COL_COMPANY),
        "agent": row.get(COL_AGENT),
        "type": row.get(COL_TYPE),
        "price": row.get(COL_PRICE),
        "engine_cc": row.get(COL_ENGINE_CC),
        "engine_type": row.get(COL_ENGINE_TYPE),
        "transmission": row.get(COL_TRANSMISSION),
        "max_speed": row.get(COL_MAX_SPEED),
        "fuel_capacity": row.get(COL_FUEL),
        "brakes": row.get(COL_BRAKES),
        "notes": row.get(COL_NOTES),
        "min_down": row.get(COL_DOWN),
        "installment_6": row.get(COL_INST_6),
        "installment_12": row.get(COL_INST_12),
        "installment_18": row.get(COL_INST_18),
        "installment_24": row.get(COL_INST_24),
        "color": row.get(COL_COLOR),
        "available": row.get(COL_AVAILABLE),
        "condition": row.get(COL_CONDITION),
    }


_INST_COL = {6: COL_INST_6, 12: COL_INST_12, 18: COL_INST_18, 24: COL_INST_24}


def get_vehicles(
    filters: dict = None,
    limit: int = 6,
    sort_by: str = None,
    ascending: bool = True,
) -> List[dict]:
    df = _load().copy()
    df = df[df[COL_AVAILABLE] == "متاح"]

    if filters:
        vtype = filters.get("type")
        if vtype:
            df = df[df[COL_TYPE].str.contains(vtype, na=False, case=False)]

        max_price = filters.get("max_price")
        if max_price:
            df = df[df[COL_PRICE] <= float(max_price)]

        min_price = filters.get("min_price")
        if min_price:
            df = df[df[COL_PRICE] >= float(min_price)]

        company = filters.get("company")
        if company:
            df = df[df[COL_COMPANY].str.contains(company, na=False, case=False)]

        transmission = filters.get("transmission")
        if transmission:
            df = df[df[COL_TRANSMISSION].str.contains(transmission, na=False, case=False)]

        condition = filters.get("condition")
        if condition:
            df = df[df[COL_CONDITION].str.contains(condition, na=False, case=False)]

        for months in (6, 12, 18, 24):
            max_inst = filters.get(f"max_installment_{months}")
            if max_inst:
                df = df[df[_INST_COL[months]] <= float(max_inst)]

    col_map = {
        "price": COL_PRICE,
        "installment_6": COL_INST_6,
        "installment_12": COL_INST_12,
        "installment_18": COL_INST_18,
        "installment_24": COL_INST_24,
        "engine_cc": COL_ENGINE_CC,
    }
    if sort_by and sort_by in col_map:
        df = df.sort_values(col_map[sort_by], ascending=ascending, na_position="last")

    return [_vehicle_to_dict(row) for _, row in df.head(limit).iterrows()]


def _normalize(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9\u0600-\u06ff]", "", s.lower())


def get_vehicle_by_name(name: str) -> Optional[dict]:
    """Three-tier search (case-insensitive):
    1. Exact consecutive substring in English or Arabic name.
    2. Every word in the query appears somewhere in the English name.
    3. Normalized match (strip punctuation/spaces then substring check).
    """
    df = _load()
    q = name.lower().strip()

    # Tier 1 — consecutive substring (handles "jet 14" → "jet 14")
    mask = (
        df[COL_NAME_EN].str.lower().str.contains(q, na=False, regex=False)
        | df[COL_NAME_AR].str.contains(name.strip(), na=False, regex=False)
    )

    # Tier 2 — every query word appears as a whole token in the English name OR company column
    # Uses token-level matching so "st" won't match inside "rooster"
    if not mask.any():
        words = q.split()
        if words:
            def _token_match(row):
                name_tokens = set(re.split(r'\W+', str(row[COL_NAME_EN]).lower()))
                company_tokens = set(re.split(r'\W+', str(row[COL_COMPANY]).lower()))
                all_tokens = name_tokens | company_tokens
                return all(w in all_tokens for w in words)
            mask = df.apply(_token_match, axis=1)

    # Tier 3 — normalized (strip spaces/punctuation: "jet14" → "jet14" in "jet 14")
    if not mask.any():
        q_norm = _normalize(q)
        if q_norm:
            mask = df[COL_NAME_EN].apply(
                lambda cell: q_norm in _normalize(str(cell)) if cell else False
            )

    result = df[mask]
    if not result.empty:
        return _vehicle_to_dict(result.iloc[0])
    return None


def get_catalog_summary() -> dict:
    df = _load()
    available = df[df[COL_AVAILABLE] == "متاح"]
    return {
        "total": int(len(available)),
        "types": available[COL_TYPE].value_counts().to_dict(),
        "companies": available[COL_COMPANY].value_counts().to_dict(),
        "price_min": float(available[COL_PRICE].min()),
        "price_max": float(available[COL_PRICE].max()),
    }


def _clean_val(val):
    """Replace NaN floats with None so dicts are JSON-safe."""
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def get_price_spread(filters: dict = None, count: int = 5) -> List[dict]:
    """Return `count` vehicles evenly spread across the price range.

    Always includes the cheapest and the most expensive so the caller
    gets a representative sample rather than the same top-N rows every time.
    """
    df = _load().copy()
    df = df[df[COL_AVAILABLE] == "متاح"]

    if filters:
        vtype = filters.get("type")
        if vtype:
            df = df[df[COL_TYPE].str.contains(vtype, na=False, case=False)]
        company = filters.get("company")
        if company:
            df = df[df[COL_COMPANY].str.contains(company, na=False, case=False)]
        max_price = filters.get("max_price")
        if max_price is not None:
            df = df[df[COL_PRICE] <= float(max_price)]
        min_price = filters.get("min_price")
        if min_price is not None:
            df = df[df[COL_PRICE] >= float(min_price)]

    df = df.dropna(subset=[COL_PRICE]).sort_values(COL_PRICE).reset_index(drop=True)

    if df.empty:
        return []
    if len(df) <= count:
        return [
            {k: _clean_val(v) for k, v in _vehicle_to_dict(df.iloc[i]).items()}
            for i in range(len(df))
        ]

    # Evenly spaced indices: 0, ..., len-1  → cheapest + spread + most expensive
    step = (len(df) - 1) / (count - 1)
    indices = sorted({round(i * step) for i in range(count)})
    return [
        {k: _clean_val(v) for k, v in _vehicle_to_dict(df.iloc[i]).items()}
        for i in indices
    ]


def _fmt_price(val) -> str:
    try:
        return f"{int(val):,} جنيه"
    except (ValueError, TypeError):
        return "غير محدد"


def _safe(val, default="غير محدد") -> str:
    """Return string for val, or default for None/NaN."""
    if val is None:
        return default
    try:
        import math
        if math.isnan(float(val)):
            return default
    except (TypeError, ValueError):
        pass
    return str(val)


def _has_value(val) -> bool:
    """True only for non-None, non-NaN values."""
    if val is None:
        return False
    try:
        import math
        return not math.isnan(float(val))
    except (TypeError, ValueError):
        return bool(val)


def format_vehicle_arabic(v: dict) -> str:
    lines = [
        f"* {_safe(v['name_ar'])} ({_safe(v['name_en'])})",
        f"   الشركة: {_safe(v['company'])} | الوكيل: {_safe(v['agent'])}",
        f"   النوع: {_safe(v['type'])} | اللون: {_safe(v['color'])}",
        f"   السعر: {_fmt_price(v['price'])}",
        f"   المحرك: {_safe(v['engine_cc'])} | {_safe(v['engine_type'])} | {_safe(v['transmission'])}",
        f"   السرعة القصوى: {_safe(v['max_speed'])}",
    ]
    if _has_value(v.get("min_down")):
        lines.append(
            f"   أقل مقدم: {_fmt_price(v['min_down'])} | قسط سنة: {_fmt_price(v['installment_12'])}/شهر"
        )
    if _has_value(v.get("notes")):
        lines.append(f"   ملاحظات: {v['notes']}")
    return "\n".join(lines)
