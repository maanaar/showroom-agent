"""
Data Node: fetches vehicles from the Excel database based on the detected intent/filters.
"""
from graph.state import AgentState
from services.data_service import get_vehicles, get_vehicle_by_name

INTENTS_NEEDING_DATA = {"browse", "filter", "details", "installment"}


def data_node(state: AgentState) -> dict:
    intent = state.get("intent", "other")
    filters = state.get("filters", {})
    vehicles = []

    if intent == "details":
        vehicle_name = filters.get("vehicle_name", "")
        if vehicle_name:
            vehicle = get_vehicle_by_name(vehicle_name)
            if vehicle:
                vehicles = [vehicle]
        if not vehicles:
            # Fall back to filtered search
            vehicles = get_vehicles(filters, limit=1)

    elif intent in ("browse", "filter", "installment"):
        vehicles = get_vehicles(filters, limit=6)

    return {"vehicles": vehicles}
