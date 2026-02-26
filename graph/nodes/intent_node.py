"""
Intent Node: uses the LLM to parse the user message into a structured intent + filters.
"""
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from llm.qwen import get_llm

SYSTEM_PROMPT = """/no_think
أنت مساعد ذكي في معرض "أيمن بدر" للموتوسيكلات.
مهمتك تحليل رسالة العميل واستخراج النية والمعلومات المطلوبة.

أرجع JSON فقط (بدون أي نص إضافي) بالشكل التالي:
{
  "intent": "browse | filter | details | installment | booking | greeting | other",
  "filters": {
    "type": "موتوسيكل | اسكوتر | null",
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
- browse: يريد تصفح الموتوسيكلات عموماً
- filter: يريد تصفية حسب سعر/نوع/شركة/قسط
- details: يسأل عن موديل بعينه
- installment: يسأل عن التقسيط أو طرق الدفع
- booking: يريد الحجز أو الشراء
- greeting: تحية أو كلام عام
- other: أسئلة أخرى"""


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
    filters = data.get("filters", {})
    lead_info = data.get("lead_info", {})

    filters = {k: v for k, v in filters.items() if v is not None and v != "null"}

    existing_lead = state.get("lead", {})
    if lead_info.get("name"):
        existing_lead["name"] = lead_info["name"]
    if lead_info.get("phone"):
        existing_lead["phone"] = lead_info["phone"]

    return {
        "intent": intent,
        "filters": filters,
        "lead": existing_lead,
    }
