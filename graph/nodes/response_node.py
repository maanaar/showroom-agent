"""
Response Node: generates the final Arabic response using the LLM.
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graph.state import AgentState
from llm.qwen import get_llm
from services.data_service import format_vehicle_arabic, get_catalog_summary

SYSTEM_PROMPT = """/no_think
أنت مساعد مبيعات ودود ومحترف في معرض "أيمن بدر" للموتوسيكلات والدراجات النارية.
تحدث دائماً بالعربية. كن ودياً ومفيداً ومختصراً.
عند عرض الموتوسيكلات، رتب المعلومات بشكل واضح.
لا تخترع معلومات غير موجودة في البيانات المقدمة لك.
إذا لم تجد ما يناسب العميل، أخبره بلطف."""

BOOKING_PROMPT = """/no_think
أنت مساعد مبيعات في معرض "أيمن بدر". العميل يريد الحجز.
إذا لم يعطِ اسمه ورقم هاتفه بعد، اطلب منه ذلك بأدب.
إذا أعطى الاسم والهاتف، أكد له أن طلبه سيُراجع وسيتواصل معه أحد المندوبين."""


def _build_context(state: AgentState) -> str:
    intent = state.get("intent", "other")
    vehicles = state.get("vehicles", [])
    filters = state.get("filters", {})
    lead = state.get("lead", {})

    parts = []

    if vehicles:
        parts.append("الموتوسيكلات المتاحة التي تطابق الطلب:")
        for v in vehicles:
            parts.append(format_vehicle_arabic(v))
    elif intent in ("browse", "filter") and not vehicles:
        parts.append("لا توجد موتوسيكلات متاحة تطابق هذه المعايير حالياً.")

    if intent == "browse" and not filters and not vehicles:
        summary = get_catalog_summary()
        from services.data_service import _fmt_price
        parts.append(
            f"المعرض يضم {summary['total']} موتوسيكل متاح. "
            f"الأسعار تبدأ من {_fmt_price(summary['price_min'])} حتى {_fmt_price(summary['price_max'])}."
        )

    if intent == "booking":
        name = lead.get("name")
        phone = lead.get("phone")
        if name and phone:
            parts.append(f"بيانات العميل: الاسم: {name}، الهاتف: {phone}")
        else:
            parts.append("العميل يريد الحجز لكن لم يعطِ بياناته بعد.")

    return "\n".join(parts)


def response_node(state: AgentState) -> dict:
    llm = get_llm()
    message = state["current_message"]
    history = state.get("conversation_history", [])
    intent = state.get("intent", "other")

    # Build context from fetched data
    try:
        context = _build_context(state)
    except Exception as ctx_err:
        import traceback
        traceback.print_exc()
        context = ""

    # System message
    sys_content = BOOKING_PROMPT if intent == "booking" else SYSTEM_PROMPT
    if context:
        sys_content += f"\n\nمعلومات متاحة:\n{context}"

    messages = [SystemMessage(content=sys_content)]

    # Add conversation history (last 6 turns for context)
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

    # Update conversation history
    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response_text},
    ]

    # Update booking stage if needed
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
