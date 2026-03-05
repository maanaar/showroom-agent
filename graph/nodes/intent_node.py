"""
Intent Node: Gemini Flash (via OpenRouter) classifies the user message into intent, product_type, and filters.
"""
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from llm.gemini import get_gemini

SYSTEM_PROMPT = """أنت مساعد ذكي في معرض "أيمن بدر" للموتوسيكلات والاكسسوارات.
مهمتك تحليل رسالة العميل واستخراج النية والمعلومات المطلوبة.

أرجع JSON فقط (بدون أي نص إضافي) بالشكل التالي:
{
  "intent": "browse | filter | details | installment | compare | booking | complaint | greeting | other",
  "product_type": "motorcycle | scooter | helmet | null",
  "filters": {
    "max_price": <رقم أو null>,
    "min_price": <رقم أو null>,
    "company": "<اسم الشركة أو null>",
    "vehicle_name": "<اسم الموديل الأول أو null>",
    "vehicle_name_2": "<اسم الموديل الثاني للمقارنة أو null>",
    "months": <عدد الأشهر المطلوبة للتقسيط كرقم أو null — مثلاً 9 أو 36>,
    "max_installment_12": <أقصى قسط شهري مقبول كرقم أو null>,
    "transmission": "يدوي | أوتوماتيك | null",
    "down_payment": <مبلغ المقدم بالجنيه كرقم أو null — مثلاً 5000>
  },
  "lead_info": {
    "name": "<الاسم أو null>",
    "phone": "<رقم الهاتف أو null>"
  }
}

تعريف النوايا:
- browse: يريد تصفح المنتجات عموماً أو يسأل "عندكم ايه" أو "ايه الموجود" أو "آخر حاجة نزلت" أو "أحدث موديل" أو "فئات الأسعار" أو "نطاق الأسعار" بدون تحديد رقم
- filter: يريد تصفية حسب سعر محدد بالأرقام أو شركة أو قسط أو يحدد معايير معينة (مثلاً "مناسب للسن الصغير" أو "أرخص حاجة" أو "بميزانية X جنيه")
- details: يسأل عن موديل بعينه بالاسم تحديداً (مثل "كلمني عن jet x" أو "مواصفات هاوجي k4")
- installment: يسأل عن التقسيط أو طرق الدفع أو الأقساط
- compare: يريد مقارنة بين موديلين (استخرج vehicle_name و vehicle_name_2)
- booking: يريد الحجز أو الشراء أو يقول "عايز أشتري"
- complaint: يشتكي من منتج أو خدمة أو تجربة سيئة
- greeting: تحية أو كلام عام
- other: أسئلة عامة ليس لها علاقة مباشرة بمنتج محدد (مثل: عروض، مواعيد، فروع، زيوت، اكسسوارات، خدمة ما بعد البيع)

قواعد مهمة:
- إذا ذكر العميل منتج غير متوفر في الكتالوج (زيوت، اكسسوارات، قطع غيار) → intent = "other" حتى لو ذكر اسم موديل
- إذا سأل عن عروض أو تخفيضات بدون ذكر منتج → intent = "other"
- إذا سأل عن مواعيد أو فروع أو عنوان → intent = "other"
- "آخر حاجة نزلت" أو "أحدث موديل" بدون ذكر اسم = browse وليس details
- details فقط لما يذكر اسم الموديل صراحة

تعريف نوع المنتج:
- motorcycle: موتوسيكل / دراجة نارية
- scooter: اسكوتر / سكوتر
- helmet: خوذة / هيلمت
- null: غير محدد (افتراضي: motorcycle)"""


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
    }
