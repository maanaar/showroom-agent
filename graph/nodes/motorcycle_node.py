"""
Motorcycle Node: calls motorcycle tools and stores results in state.
No LLM — deterministic data fetching based on intent + filters.
"""
import json
from graph.state import AgentState
from services.data_service import get_price_spread
from tools.motorcycle_tools import (
    search_motorcycles,
    motorcycle_details,
    cheapest_motorcycles,
    motorcycle_installments,
    motorcycle_by_monthly_budget,
    motorcycle_catalog_summary,
)

MOTO_TYPE = "موتوسيكل"


def motorcycle_node(state: AgentState) -> dict:
    intent = state.get("intent", "browse")
    filters = state.get("filters", {})

    vehicles = []

    if intent == "details":
        name = filters.get("vehicle_name", "")
        raw = motorcycle_details.invoke({"vehicle_name": name})
        result = json.loads(raw)
        if isinstance(result, list):
            vehicles = result
        elif isinstance(result, dict) and "error" not in result:
            vehicles = [result]

    elif intent == "installment" and filters.get("vehicle_name"):
        raw = motorcycle_installments.invoke({"vehicle_name": filters["vehicle_name"]})
        result = json.loads(raw)
        if isinstance(result, dict) and "error" not in result:
            vehicles = [result]

    elif intent == "installment" and filters.get("max_installment_12"):
        raw = motorcycle_by_monthly_budget.invoke({
            "max_monthly": filters["max_installment_12"],
            "months": 12,
        })
        vehicles = json.loads(raw)

    elif intent == "filter":
        has_filters = any(
            filters.get(k)
            for k in ("max_price", "min_price", "company", "transmission",
                      "max_installment_12", "max_installment_6",
                      "max_installment_18", "max_installment_24")
        )
        if has_filters:
            raw = search_motorcycles.invoke({
                "max_price": filters.get("max_price"),
                "min_price": filters.get("min_price"),
                "company": filters.get("company"),
                "transmission": filters.get("transmission"),
                "limit": 5,
            })
            vehicles = json.loads(raw)
        else:
            # No actual filter values — treat as browse (price spread)
            vehicles = get_price_spread({"type": MOTO_TYPE}, count=5)

    else:
        # browse — return a price-spread sample so different price points are visible
        vehicles = get_price_spread({"type": MOTO_TYPE}, count=5)

    return {"vehicles": vehicles}
