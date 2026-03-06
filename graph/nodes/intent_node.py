"""
Intent Node: Gemini Flash (via OpenRouter) classifies the user message into intent, product_type, and filters.
"""
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from llm.gemini import get_gemini

SYSTEM_PROMPT = """You are an intelligent assistant for "ibyco" motorcycle and accessories showroom.
Your task is to analyse the customer's message and extract intent and structured information.
The customer may write in Arabic (Egyptian dialect), English, or mixed — you must understand all.

Return JSON only (no extra text) in this format:
{
  "intent": "browse | filter | details | installment | compare | booking | complaint | greeting | other",
  "product_type": "motorcycle | scooter | helmet | null",
  "filters": {
    "max_price": <number or null>,
    "min_price": <number or null>,
    "company": "<brand name or null>",
    "vehicle_name": "<first model name or null>",
    "vehicle_name_2": "<second model name for comparison or null>",
    "months": <number of installment months or null — e.g. 9 or 36>,
    "max_installment_12": <max acceptable monthly payment as number or null>,
    "transmission": "يدوي | أوتوماتيك | null",
    "down_payment": <down payment amount in EGP as number or null — e.g. 5000>
  },
  "lead_info": {
    "name": "<customer name or null>",
    "phone": "<phone number or null>"
  }
}

Intent definitions:
- browse: wants to see products generally, asks "what do you have", "what's available", "latest model", "price ranges" without specifying a number
- filter: wants to filter by a specific price number, brand, installment, or criteria (e.g. "suitable for kids", "cheapest", "budget of X EGP")
- details: asks about a specific model BY NAME (e.g. "tell me about jet x", "specs of haojue k4")
- installment: asks about installment plans, payment methods, or monthly payments
- compare: wants to compare two models (extract vehicle_name and vehicle_name_2)
- booking: wants to book an appointment, visit, or buy (e.g. "I want to buy", "book a test ride")
- complaint: complaining about a product, service, or bad experience
- greeting: general greeting or small talk
- other: general questions not directly about a specific product (e.g. offers, working hours, branches, oils, accessories, after-sales service)

Important rules:
- If the customer mentions a product NOT in the catalog (oils, accessories, spare parts) → intent = "other" even if a model name is mentioned
- If asking about offers/discounts without mentioning a product → intent = "other"
- If asking about working hours, branches, or address → intent = "other"
- "latest model" or "newest release" without a specific name = browse, NOT details
- details ONLY when the model name is explicitly mentioned

Product type definitions:
- motorcycle: موتوسيكل / motorbike / دراجة نارية
- scooter: اسكوتر / سكوتر / scooter
- helmet: خوذة / هيلمت / helmet
- null: unspecified (default: motorcycle)"""


def intent_node(state: AgentState) -> dict:
    llm = get_gemini()
    message = state["current_message"]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]

    intent_usage = {}
    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        meta = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage", {})
        if meta:
            intent_usage = {
                "input_tokens":    meta.get("input_tokens")  or meta.get("prompt_tokens", 0),
                "output_tokens":   meta.get("output_tokens") or meta.get("completion_tokens", 0),
                "total_tokens":    meta.get("total_tokens", 0),
                "thinking_tokens": meta.get("output_token_details", {}).get("reasoning", 0),
            }
    except Exception:
        data = {}

    intent = data.get("intent", "other")
    product_type = data.get("product_type") or None
    if product_type == "null":
        product_type = None

    filters = data.get("filters", {})
    filters = {k: v for k, v in filters.items() if v is not None and v != "null"}

    lead_info = data.get("lead_info", {})
    existing_lead = state.get("lead", {})
    if lead_info.get("name"):
        existing_lead["name"] = lead_info["name"]
    if lead_info.get("phone"):
        existing_lead["phone"] = lead_info["phone"]

    return {
        "intent": intent,
        "product_type": product_type,
        "filters": filters,
        "lead": existing_lead,
        "intent_usage": intent_usage,
        "ask_clarification": None,   # reset each turn; nodes set it if needed
    }
