"""
Response Node: Gemini Flash (via OpenRouter) generates the final Arabic response.
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graph.state import AgentState
from llm.gemini import get_gemini
from services.data_service import _fmt_price, _safe, _has_value

SYSTEM_PROMPT = """أنت مساعد مبيعات ودود ومحترف في معرض "أيمن بدر" للموتوسيكلات والاكسسوارات.
تحدث دائماً بالعربية. كن ودياً ومفيداً ومختصراً.
عند عرض المنتجات، رتب المعلومات بشكل واضح.
لا تخترع معلومات غير موجودة في البيانات المقدمة لك.
إذا لم تجد ما يناسب العميل، أخبره بلطف."""

BOOKING_PROMPT = """أنت مساعد مبيعات في معرض "أيمن بدر". العميل يريد الحجز.
إذا لم يعطِ اسمه ورقم هاتفه بعد، اطلب منه ذلك بأدب.
إذا أعطى الاسم والهاتف، أكد له أن طلبه سيُراجع وسيتواصل معه أحد المندوبين."""


def _format_vehicle(v: dict) -> str:
    """Format a vehicle dict (from JSON tools) into readable Arabic text for the LLM context."""
    lines = [
        f"* {_safe(v.get('name_ar'))} ({_safe(v.get('name_en'))})",
        f"   النوع: {_safe(v.get('type'))} | اللون: {_safe(v.get('color'))}",
        f"   الشركة: {_safe(v.get('company'))} | الوكيل: {_safe(v.get('agent'))}",
        f"   السعر: {_fmt_price(v.get('price'))}",
    ]
    if v.get("engine_cc"):
        lines.append(
            f"   المحرك: {_safe(v.get('engine_cc'))} | {_safe(v.get('engine_type'))} | {_safe(v.get('transmission'))}"
        )
    if _has_value(v.get("min_down")):
        lines.append(
            f"   أقل مقدم: {_fmt_price(v.get('min_down'))} | قسط 12 شهر: {_fmt_price(v.get('installment_12'))}/شهر"
        )
    if v.get("notes"):
        lines.append(f"   ملاحظات: {v['notes']}")
    return "\n".join(lines)


def _build_context(state: AgentState) -> str:
    intent = state.get("intent", "other")
    product_type = state.get("product_type") or "motorcycle"
    vehicles = state.get("vehicles", [])
    lead = state.get("lead", {})

    product_label = {"motorcycle": "موتوسيكلات", "scooter": "اسكوتر", "helmet": "خوذات"}.get(
        product_type, "منتجات"
    )

    parts = []

    if vehicles:
        parts.append(f"المتاح من {product_label} التي تطابق الطلب:")
        for v in vehicles:
            parts.append(_format_vehicle(v))
    elif intent in ("browse", "filter", "details"):
        parts.append(f"لا توجد {product_label} متاحة تطابق هذه المعايير حالياً.")

    if intent == "booking":
        name = lead.get("name")
        phone = lead.get("phone")
        if name and phone:
            parts.append(f"بيانات العميل: الاسم: {name}، الهاتف: {phone}")
        else:
            parts.append("العميل يريد الحجز لكن لم يعطِ بياناته بعد.")

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
        sys_content += f"\n\nمعلومات متاحة:\n{context}"

    messages = [SystemMessage(content=sys_content)]

    for turn in history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=message))

    try:
        result = llm.invoke(messages)
        response_text = result.content.strip()
    except Exception as e:
        response_text = f"عذراً، حدث خطأ: {e}"

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text},
    ]

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
    }
