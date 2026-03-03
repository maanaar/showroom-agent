"""
Compare Node: fetches both vehicles by name and stores them in state.
No LLM — deterministic lookup; Gemini in response_node does the comparison text.
"""
import math
from graph.state import AgentState
from services.data_service import get_vehicle_by_name


def _clean(v: dict) -> dict:
    return {
        k: (None if isinstance(val, float) and math.isnan(val) else val)
        for k, val in v.items()
    }


def compare_node(state: AgentState) -> dict:
    filters = state.get("filters", {})
    name1 = filters.get("vehicle_name", "")
    name2 = filters.get("vehicle_name_2", "")

    vehicles = []
    for name in (name1, name2):
        if name:
            v = get_vehicle_by_name(name)
            if v:
                vehicles.append(_clean(v))

    return {"vehicles": vehicles}
