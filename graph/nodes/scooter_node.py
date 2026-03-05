"""
Scooter Node: calls scooter tools and stores results in state.
No LLM — deterministic data fetching based on intent + filters.
"""
import json
from graph.state import AgentState
from services.data_service import get_price_spread, get_vehicle_by_name, calculate_custom_installment, get_similar_vehicles
from tools.scooter_tools import (
    search_scooters,
    scooter_details,
    cheapest_scooters,
    scooter_installments,
    scooter_by_monthly_budget,
)

SCOOTER_TYPE = "اسكوتر"


def scooter_node(state: AgentState) -> dict:
    intent = state.get("intent", "browse")
    filters = state.get("filters", {})

    vehicles = []
    ask_clarification = None

    def _resolve_name(filters: dict) -> str:
        """Combine company + vehicle_name for a richer search query."""
        parts = [filters.get("company", ""), filters.get("vehicle_name", "")]
        return " ".join(p for p in parts if p).strip()

    if intent == "installment" and not filters.get("vehicle_name") and not filters.get("max_installment_12"):
        ask_clarification = "vehicle_name"
        return {"vehicles": [], "ask_clarification": ask_clarification}

    if intent == "details":
        name = _resolve_name(filters)
        v = get_vehicle_by_name(name)
        if v:
            vehicles = [v] + get_similar_vehicles(v, count=3)

    elif intent == "installment" and filters.get("vehicle_name"):
        if "down_payment" not in filters:
            ask_clarification = "down_payment"
            return {"vehicles": [], "ask_clarification": ask_clarification}
        months = filters.get("months")
        name   = _resolve_name(filters)
        v = get_vehicle_by_name(name)
        down_payment = filters.get("down_payment", 0)
        if months and v:
            months = int(months)
            calc = calculate_custom_installment(v, months, down_payment=down_payment)
            vehicles = [calc]
        elif v:
            raw = scooter_installments.invoke({"vehicle_name": v["name_en"]})
            result = json.loads(raw)
            if isinstance(result, dict) and "error" not in result:
                vehicles = [result]

    elif intent == "installment" and filters.get("max_installment_12"):
        raw = scooter_by_monthly_budget.invoke({
            "max_monthly": filters["max_installment_12"],
            "months": 12,
        })
        vehicles = json.loads(raw)
        if vehicles:
            seen = {v.get("name_en") for v in vehicles}
            for extra in get_similar_vehicles(vehicles[0], count=3):
                if extra.get("name_en") not in seen:
                    vehicles.append(extra)
                    seen.add(extra.get("name_en"))

    elif intent == "filter":
        has_filters = any(
            filters.get(k)
            for k in ("max_price", "min_price", "company", "transmission",
                      "max_installment_12", "max_installment_6",
                      "max_installment_18", "max_installment_24")
        )
        if has_filters:
            raw = search_scooters.invoke({
                "max_price": filters.get("max_price"),
                "min_price": filters.get("min_price"),
                "company": filters.get("company"),
                "transmission": filters.get("transmission"),
                "limit": 5,
            })
            vehicles = json.loads(raw)
        else:
            # No actual filter values — treat as browse (price spread)
            vehicles = get_price_spread({"type": SCOOTER_TYPE}, count=5)

    else:
        # browse — return a price-spread sample so different price points are visible
        vehicles = get_price_spread({"type": SCOOTER_TYPE}, count=5)

    return {"vehicles": vehicles, "ask_clarification": ask_clarification}
