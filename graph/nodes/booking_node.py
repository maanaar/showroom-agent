"""
Booking Node: saves the appointment/booking request to the DB.
"""
from graph.state import AgentState
from services.db_service import save_booking


def booking_node(state: AgentState) -> dict:
    user_id = state.get("user_id", "unknown")
    lead    = state.get("lead", {})
    filters = state.get("filters", {})

    name             = lead.get("name")
    phone            = lead.get("phone") or user_id
    vehicle_interest = filters.get("vehicle_name") or filters.get("company")

    try:
        save_booking(phone_number=phone, name=name, vehicle_interest=vehicle_interest)
        booking_saved = True
    except Exception:
        booking_saved = False

    return {"booking_saved": booking_saved}
