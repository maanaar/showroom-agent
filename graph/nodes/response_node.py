"""
Response Node: Gemini Flash (via OpenRouter) generates the final Arabic response.
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graph.state import AgentState
from llm.gemini import get_gemini
from services.data_service import _fmt_price, _safe, _has_value
from services.db_service import update_client_turn

SHOWROOM_INFO = """Showroom info:
- Name: ibyco — motorcycles & accessories showroom
- Address: 12 Street 302, off Bahaa El-Din El-Ghatoury, Smouha, Alexandria
- Working hours: every day 2 PM to 11 PM — Friday is closed
- WhatsApp / page messages: 01505989502 — 01505989506"""

SYSTEM_PROMPT = """You are a professional sales assistant at "ibyco" motorcycles & accessories showroom.
Always reply in the SAME LANGUAGE the customer uses. If they write in Egyptian Arabic, reply in professional Egyptian Arabic. If English, reply in English.
Your tone should be professional yet friendly and natural — like a skilled salesperson talking to an important customer.
When presenting products, organise the information clearly and neatly.
NEVER mention information that is not in the data provided to you. Do NOT make up or hallucinate products.
If nothing matches the customer's request, let them know politely.

Always encourage the customer to visit the showroom or contact us — end every reply with a polite invitation to visit or get in touch.

""" + SHOWROOM_INFO + """

Important notes:
- Installment plans are available from 1 to 24 months. If you receive calculated installment data, present it directly without hesitation or apology.
- Oils and accessories are NOT in the electronic catalog — direct the customer to visit the showroom or call for enquiries.
- If someone asks about a scooter suitable for kids, use engine size and top speed (50cc/40km/h for children, 150cc+ for youth).
- If asked about "latest model", show the latest products and confirm the showroom updates its catalog regularly.
- Keep your replies concise and useful."""

BOOKING_PROMPT = """You are a professional sales assistant at "ibyco" showroom. The customer wants to book an appointment or visit.
Always reply in the SAME LANGUAGE the customer uses.
Your tone should be professional yet friendly and natural.
If they haven't provided their name and phone number yet, ask politely so we can coordinate the appointment.
If they have provided them, reassure them that the showroom team will contact them shortly to confirm the appointment or complete the deal.

""" + SHOWROOM_INFO


def _format_vehicle(v: dict) -> str:
    """Format a vehicle dict into readable text for the LLM context."""
    lines = [
        f"* {_safe(v.get('name_ar'))} ({_safe(v.get('name_en'))})",
        f"   Type: {_safe(v.get('type'))} | Color: {_safe(v.get('color'))}",
        f"   Brand: {_safe(v.get('company'))} | Agent: {_safe(v.get('agent'))}",
        f"   Price: {_fmt_price(v.get('price'))}",
    ]
    if v.get("engine_cc"):
        lines.append(
            f"   Engine: {_safe(v.get('engine_cc'))} | {_safe(v.get('engine_type'))} | {_safe(v.get('transmission'))}"
        )
    if _has_value(v.get("min_down")):
        lines.append(
            f"   Min down payment: {_fmt_price(v.get('min_down'))} | 12-month installment: {_fmt_price(v.get('installment_12'))}/month"
        )
    if v.get("notes"):
        lines.append(f"   Notes: {v['notes']}")
    return "\n".join(lines)


def _build_context(state: AgentState) -> str:
    intent = state.get("intent", "other")
    product_type = state.get("product_type") or "motorcycle"
    vehicles = state.get("vehicles", [])
    lead = state.get("lead", {})
    ask_clarification = state.get("ask_clarification")

    # Clarification needed before we can calculate
    if ask_clarification == "vehicle_name":
        return "The customer wants installment info but didn't specify the model. Ask them which model or product they want installment for."
    if ask_clarification == "down_payment":
        return "The customer wants installment info but didn't mention the down payment. Ask them how much down payment they will pay."

    product_label = {"motorcycle": "motorcycles", "scooter": "scooters", "helmet": "helmets"}.get(
        product_type, "products"
    )

    parts = []

    if intent == "compare":
        if vehicles:
            parts.append("The two products to compare:")
            for v in vehicles:
                parts.append(_format_vehicle(v))
            parts.append("Compare these two products in detail: price, engine, speed, installment, and features.")
        else:
            parts.append("Neither of the two products requested for comparison was found in the catalog.")
    elif intent == "installment" and vehicles and vehicles[0].get("monthly_payment") is not None:
        v = vehicles[0]
        parts.append(f"Installment options for {v.get('name_ar')} ({v.get('name_en')}):")
        parts.append(f"   Product price: {_fmt_price(v.get('price'))}")
        if v.get("down_payment"):
            parts.append(f"   Down payment: {_fmt_price(v.get('down_payment'))}")
        for plan in vehicles:
            if plan.get("monthly_payment") is not None:
                parts.append(f"   {plan['months']} months → {_fmt_price(plan['monthly_payment'])}/month")
    elif vehicles:
        parts.append(f"Available {product_label} matching the request:")
        for v in vehicles:
            parts.append(_format_vehicle(v))
    elif intent in ("browse", "filter", "details"):
        parts.append(f"No {product_label} currently available matching these criteria.")

    if intent == "complaint":
        if state.get("complaint_saved"):
            parts.append("The customer's complaint has been received and logged successfully. They will be contacted shortly.")
        else:
            parts.append("The customer wants to file a complaint.")

    if intent == "booking":
        name = lead.get("name")
        phone = lead.get("phone")
        if name and phone:
            parts.append(f"Customer info: Name: {name}, Phone: {phone}")
        else:
            parts.append("The customer wants to book but hasn't provided their details yet.")

    return "\n".join(parts)


def response_node(state: AgentState) -> dict:
    llm = get_gemini()
    message = state["current_message"]
    history = state.get("conversation_history", [])
    intent = state.get("intent", "other")

    try:
        context = _build_context(state)
    except Exception:
        import traceback
        traceback.print_exc()
        context = ""

    sys_content = BOOKING_PROMPT if intent == "booking" else SYSTEM_PROMPT
    if context:
        sys_content += f"\n\nAvailable data:\n{context}"

    messages = [SystemMessage(content=sys_content)]

    for turn in history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=message))

    # Gemini 2.0 Flash pricing via OpenRouter
    _INPUT_COST_PER_M  = 0.10   # $ per 1M input tokens
    _OUTPUT_COST_PER_M = 0.40   # $ per 1M output tokens

    usage = {}
    try:
        result = llm.invoke(messages)
        response_text = result.content.strip()
        meta = getattr(result, "usage_metadata", None) or getattr(result, "response_metadata", {}).get("token_usage", {})
        if meta:
            usage = {
                "input_tokens":    meta.get("input_tokens")  or meta.get("prompt_tokens", 0),
                "output_tokens":   meta.get("output_tokens") or meta.get("completion_tokens", 0),
                "total_tokens":    meta.get("total_tokens", 0),
                "thinking_tokens": meta.get("output_token_details", {}).get("reasoning", 0),
            }
    except Exception as e:
        response_text = f"Sorry, an error occurred: {e}"

    # Aggregate intent + response tokens and compute cost
    intent_usage = state.get("intent_usage") or {}
    total_input    = (intent_usage.get("input_tokens", 0)    + usage.get("input_tokens", 0))
    total_output   = (intent_usage.get("output_tokens", 0)   + usage.get("output_tokens", 0))
    total_thinking = (intent_usage.get("thinking_tokens", 0) + usage.get("thinking_tokens", 0))
    total_all      = total_input + total_output
    cost_usd = (total_input * _INPUT_COST_PER_M + total_output * _OUTPUT_COST_PER_M) / 1_000_000

    print(
        f"\n── Token usage ──────────────────────────────\n"
        f"  Intent node  → in: {intent_usage.get('input_tokens', 0):>6} | "
        f"out: {intent_usage.get('output_tokens', 0):>5} | "
        f"think: {intent_usage.get('thinking_tokens', 0):>5}\n"
        f"  Response node→ in: {usage.get('input_tokens', 0):>6} | "
        f"out: {usage.get('output_tokens', 0):>5} | "
        f"think: {usage.get('thinking_tokens', 0):>5}\n"
        f"  TOTAL        → in: {total_input:>6} | out: {total_output:>5} | "
        f"think: {total_thinking:>5} | all: {total_all:>6}\n"
        f"  Cost         → ${cost_usd:.6f}\n"
        f"─────────────────────────────────────────────"
    )

    usage["intent_input_tokens"]    = intent_usage.get("input_tokens", 0)
    usage["intent_output_tokens"]   = intent_usage.get("output_tokens", 0)
    usage["intent_thinking_tokens"] = intent_usage.get("thinking_tokens", 0)
    usage["total_all_tokens"]       = total_all
    usage["cost_usd"]               = round(cost_usd, 6)

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text},
    ]

    # Persist turn summary to DB
    user_id = state.get("user_id", "unknown")
    try:
        update_client_turn(
            phone_number=user_id,
            user_message=message,
            bot_response=response_text,
            intent=intent,
            filters=state.get("filters", {}),
            lead=state.get("lead", {}),
        )
    except Exception:
        pass  # never crash the response over a DB write

    booking_stage = state.get("booking_stage")
    lead = state.get("lead", {})
    if intent == "booking":
        if lead.get("name") and lead.get("phone"):
            booking_stage = "confirmed"
        else:
            booking_stage = "collecting_info"

    return {
        "response": response_text,
        "conversation_history": updated_history,
        "booking_stage": booking_stage,
        "usage": usage,
    }
