"""
Intent Node: Qwen (local Ollama) classifies the user message into intent, product_type, and filters.
"""
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from llm.qwen import get_llm

SYSTEM_PROMPT = """أنت مساعد ذكي في معرض "أيمن بدر" للموتوسيكلات والاكسسوارات.
مهمتك تحليل رسالة العميل واستخراج النية والمعلومات المطلوبة.

أرجع JSON فقط (بدون أي نص إضافي) بالشكل التالي:
{
  "intent": "browse | filter | details | installment | booking | greeting | other",
  "product_type": "motorcycle | scooter | helmet | null",
  "filters": {
    "max_price": <رقم أو null>,
    "min_price": <رقم أو null>,
    "company": "<اسم الشركة أو null>",
    "vehicle_name": "<اسم الموديل أو null>",
    "max_installment_12": <رقم أو null>,
    "transmission": "يدوي | أوتوماتيك | null"
  },
  "lead_info": {
    "name": "<الاسم أو null>",
    "phone": "<رقم الهاتف أو null>"
  }
}

تعريف النوايا:
- browse: يريد تصفح المنتجات عموماً
- filter: يريد تصفية حسب سعر/شركة/قسط
- details: يسأل عن موديل بعينه
- installment: يسأل عن التقسيط أو طرق الدفع
- booking: يريد الحجز أو الشراء
- greeting: تحية أو كلام عام
- other: أسئلة أخرى

تعريف نوع المنتج:
- motorcycle: موتوسيكل / دراجة نارية
- scooter: اسكوتر / سكوتر
- helmet: خوذة / هيلمت
- null: غير محدد (افتراضي: motorcycle)"""


def intent_node(state: AgentState) -> dict:
    llm = get_llm()
    message = state["current_message"]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=message),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
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
    }
