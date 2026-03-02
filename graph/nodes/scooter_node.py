"""
Scooter Node: calls scooter tools and stores results in state.
No LLM — deterministic data fetching based on intent + filters.
"""
import json
from graph.state import AgentState
from tools.scooter_tools import (
    search_scooters,
    scooter_details,
    cheapest_scooters,
    scooter_installments,
    scooter_by_monthly_budget,
)


def scooter_node(state: AgentState) -> dict:
    intent = state.get("intent", "browse")
    filters = state.get("filters", {})

    vehicles = []

    if intent == "details":
        name = filters.get("vehicle_name", "")
        raw = scooter_details.invoke({"vehicle_name": name})
        result = json.loads(raw)
        if isinstance(result, list):
            vehicles = result
        elif isinstance(result, dict) and "error" not in result:
            vehicles = [result]

    elif intent == "installment" and filters.get("vehicle_name"):
        raw = scooter_installments.invoke({"vehicle_name": filters["vehicle_name"]})
        result = json.loads(raw)
        if isinstance(result, dict) and "error" not in result:
            vehicles = [result]

    elif intent == "installment" and filters.get("max_installment_12"):
        raw = scooter_by_monthly_budget.invoke({
            "max_monthly": filters["max_installment_12"],
            "months": 12,
        })
        vehicles = json.loads(raw)

    elif intent == "filter":
        raw = search_scooters.invoke({
            "max_price": filters.get("max_price"),
            "min_price": filters.get("min_price"),
            "company": filters.get("company"),
            "transmission": filters.get("transmission"),
            "limit": 3,
        })
        vehicles = json.loads(raw)

    else:
        # browse
        raw = search_scooters.invoke({"limit": 3})
        vehicles = json.loads(raw)

    return {"vehicles": vehicles}
