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


def get_vehicle_by_name(name: str) -> Optional[dict]:
    df = _load()
    mask = (
        df[COL_NAME_EN].str.lower().str.contains(name.lower(), na=False)
        | df[COL_NAME_AR].str.contains(name, na=False)
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
