"""
Helmet Node: calls helmet tools and stores results in state.
No LLM — deterministic data fetching based on intent + filters.
"""
import json
from graph.state import AgentState
from tools.helmet_tools import (
    search_helmets,
    helmet_details,
    cheapest_helmets,
)


def helmet_node(state: AgentState) -> dict:
    intent = state.get("intent", "browse")
    filters = state.get("filters", {})

    vehicles = []

    if intent == "details":
        name = filters.get("vehicle_name", "")
        raw = helmet_details.invoke({"name": name})
        result = json.loads(raw)
        if isinstance(result, list):
            vehicles = result
        elif isinstance(result, dict) and "error" not in result:
            vehicles = [result]

    elif intent == "filter":
        raw = search_helmets.invoke({
            "max_price": filters.get("max_price"),
            "min_price": filters.get("min_price"),
            "company": filters.get("company"),
            "limit": 5,
        })
        vehicles = json.loads(raw)

    else:
        # browse
        raw = search_helmets.invoke({"limit": 5})
        vehicles = json.loads(raw)

    return {"vehicles": vehicles}
