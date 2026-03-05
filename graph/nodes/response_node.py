"""
Response Node: Gemini Flash (via OpenRouter) generates the final Arabic response.
"""
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graph.state import AgentState
from llm.gemini import get_gemini
from services.data_service import _fmt_price, _safe, _has_value
from services.db_service import update_client_turn

SHOWROOM_INFO = """معلومات المعرض:
- الاسم: آي بايكو (ibyco) للموتوسيكلات والإكسسوارات
- العنوان: ١٢ش ٣٠٢ متفرع من بهاء الدين الغتورى، سموحه، الإسكندرية
- المواعيد: كل يوم من ٢ ظهراً إلى ١١ مساءً — عدا الجمعة إجازة رسمية
- واتساب / رسايل الصفحة: 01505989502 — 01505989506"""

SYSTEM_PROMPT = """أنت مساعد مبيعات محترف في معرض "آي بايكو" (ibyco) للموتوسيكلات والإكسسوارات.
اتكلم بالمصري — لغة مهنية ومحترمة لكن طبيعية ومش متكلفة. زي موظف مبيعات شاطر بيكلم عميل مهم.
عند عرض المنتجات، رتب المعلومات بشكل واضح ومنظم.
ماتذكرش أي معلومات مش موجودة في البيانات اللي قدامك.
لو مفيش حاجة تناسب العميل، وضحله ده بأسلوب لطيف.

هدفك دايماً تشجع العميل يزور المعرض أو يتواصل معانا — اختم كل رد بدعوة مهذبة للزيارة أو التواصل.

""" + SHOWROOM_INFO + """

معلومات مهمة:
- أنظمة التقسيط متاحة من 1 لحد 24 شهر. لو وصلتك بيانات تقسيط محسوبة، قدّمها مباشرة بدون أي تحفظات أو اعتذار.
- الزيوت والاكسسوارات مش موجودة في الكتالوج الإلكتروني — وجّه العميل يزور المعرض أو يتصل للاستفسار.
- لو حد سأل عن اسكوتر مناسب لسن صغير، اعتمد على حجم المحرك والسرعة القصوى (50cc/40كم-س للأطفال، 150cc+ للشباب).
- لو سأل عن "أحدث موديل" اعرضله آخر المنتجات وأكدله إن المعرض بيحدّث الكتالوج باستمرار.
- خلي ردودك مختصرة ومفيدة."""

BOOKING_PROMPT = """أنت مساعد مبيعات محترف في معرض "آي بايكو" (ibyco). العميل عايز يحجز أو يزور المعرض.
اتكلم بالمصري — لغة مهنية ومحترمة لكن طبيعية.
لو ماذكرش اسمه ورقم تليفونه لسه، اطلبهم منه بأسلوب مهذب عشان ننسقله الموعد.
لو ذكرهم، طمّنه إن فريق المعرض هيتواصل معاه في أقرب وقت لتحديد الموعد أو إتمام الصفقة.

""" + SHOWROOM_INFO


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
    ask_clarification = state.get("ask_clarification")

    # Clarification needed before we can calculate
    if ask_clarification == "vehicle_name":
        return "العميل يريد التقسيط لكن لم يحدد الموديل. اسأله عن الموديل أو المنتج اللي يريد التقسيط عليه."
    if ask_clarification == "down_payment":
        return "العميل يريد التقسيط لكن لم يذكر المقدم. اسأله عن مبلغ المقدم اللي هيدفعه."

    product_label = {"motorcycle": "موتوسيكلات", "scooter": "اسكوتر", "helmet": "خوذات"}.get(
        product_type, "منتجات"
    )

    parts = []

    if intent == "compare":
        if vehicles:
            parts.append("المنتجان المطلوب مقارنتهما:")
            for v in vehicles:
                parts.append(_format_vehicle(v))
            parts.append("قارن بين هذين المنتجين بشكل مفصل: السعر، المحرك، السرعة، التقسيط، والمميزات.")
        else:
            parts.append("لم يتم العثور على أي من المنتجين المطلوب مقارنتهما في الكتالوج.")
    elif intent == "installment" and vehicles and vehicles[0].get("monthly_payment") is not None:
        v = vehicles[0]
        parts.append(f"خيارات تقسيط {v.get('name_ar')} ({v.get('name_en')}):")
        parts.append(f"   سعر المنتج:  {_fmt_price(v.get('price'))}")
        if v.get("down_payment"):
            parts.append(f"   المقدم:      {_fmt_price(v.get('down_payment'))}")
        for plan in vehicles:
            if plan.get("monthly_payment") is not None:
                parts.append(f"   {plan['months']} شهر  →  {_fmt_price(plan['monthly_payment'])}/شهر")
    elif vehicles:
        parts.append(f"المتاح من {product_label} التي تطابق الطلب:")
        for v in vehicles:
            parts.append(_format_vehicle(v))
    elif intent in ("browse", "filter", "details"):
        parts.append(f"لا توجد {product_label} متاحة تطابق هذه المعايير حالياً.")

    if intent == "complaint":
        if state.get("complaint_saved"):
            parts.append("تم استلام شكوى العميل وتسجيلها بنجاح. سيتم التواصل معه في أقرب وقت.")
        else:
            parts.append("العميل يريد تقديم شكوى.")

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
        response_text = f"عذراً، حدث خطأ: {e}"

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
