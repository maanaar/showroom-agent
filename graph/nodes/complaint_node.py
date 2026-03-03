"""
Complaint Node: saves the complaint to the SQLite DB and sets a flag
so the response_node can generate an appropriate acknowledgement.
"""
from graph.state import AgentState
from services.db_service import save_complaint


def complaint_node(state: AgentState) -> dict:
    user_id = state.get("user_id", "unknown")
    message = state.get("current_message", "")

    try:
        save_complaint(phone_number=user_id, message_text=message)
        complaint_saved = True
    except Exception:
        complaint_saved = False

    return {"complaint_saved": complaint_saved}
