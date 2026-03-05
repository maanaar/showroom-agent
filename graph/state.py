from typing import TypedDict, List, Dict, Optional, Any


class AgentState(TypedDict):
    user_id: str
    current_message: str
    conversation_history: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    intent: Optional[str]                        # browse | filter | details | installment | booking | greeting | other
    product_type: Optional[str]                  # motorcycle | scooter | helmet | None
    filters: Dict[str, Any]                      # extracted filters from user message
    vehicles: List[Dict[str, Any]]               # fetched vehicle/product records (as raw dicts)
    lead: Dict[str, Any]                         # lead info: name, phone
    booking_stage: Optional[str]                 # None | collecting_info | confirmed
    response: Optional[str]                      # final response to send back
    recommendations: List[str]                   # suggested follow-up questions
    complaint_saved: Optional[bool]              # True if complaint was persisted to DB
    ask_clarification: Optional[str]             # "down_payment" | "vehicle_name" | None — ask user for missing info
    intent_usage: Optional[Dict[str, int]]       # token counts from intent_node LLM call
    usage: Optional[Dict[str, int]]              # token counts from response_node LLM call
