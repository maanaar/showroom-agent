from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from graph.agent_graph import get_agent
from graph.state import AgentState
from services.db_service import upsert_client

router = APIRouter()

# In-memory session store: user_id → partial AgentState
_sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = None
    vehicles: list = []
    usage: dict = {}


def _get_or_create_session(user_id: str) -> dict:
    if user_id not in _sessions:
        _sessions[user_id] = {
            "user_id": user_id,
            "current_message": "",
            "conversation_history": [],
            "intent": None,
            "filters": {},
            "vehicles": [],
            "lead": {},
            "booking_stage": None,
            "response": None,
            "recommendations": [],
            "complaint_saved": None,
            "ask_clarification": None,
            "intent_usage": None,
            "usage": None,
        }
    return _sessions[user_id]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = _get_or_create_session(request.user_id)
    session["current_message"] = request.message

    agent = get_agent()
    result: AgentState = agent.invoke(session)

    # Persist updated state for this user
    _sessions[request.user_id] = result

    # Persist client record to SQLite (fire-and-forget; errors are non-fatal)
    try:
        lead = result.get("lead", {})
        phone = lead.get("phone") or request.user_id
        upsert_client(
            phone_number=phone,
            name=lead.get("name"),
            last_user_reply=request.message,
            last_bot_reply=result.get("response", ""),
            last_bot_reply_type=result.get("intent", "other"),
            has_purchased=result.get("booking_stage") == "confirmed",
        )
    except Exception:
        pass

    return ChatResponse(
        response=result.get("response", ""),
        intent=result.get("intent"),
        vehicles=result.get("vehicles", []),
        usage=result.get("usage") or {},
    )


@router.delete("/chat/{user_id}")
async def reset_session(user_id: str):
    """Clear conversation history for a user."""
    _sessions.pop(user_id, None)
    return {"status": "ok", "message": f"Session for {user_id} cleared."}


