from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from tools.vehicle_tools import (
    search_vehicles,
    vehicle_details,
    catalog_summary,
    get_installment_plans,
    compare_vehicles,
    cheapest_vehicles,
    similar_vehicles,
    search_by_monthly_budget,
)

_SYSTEM = """/no_think
أنت مساعد مبيعات في معرض أيمن بدر للموتوسيكلات.
استخدم الأدوات للحصول على البيانات الحقيقية.
لا تخترع معلومات.
تحدث بالعربية بطريقة ودودة ومختصرة.
"""

tools = [
    search_vehicles,
    vehicle_details,
    catalog_summary,
    get_installment_plans,
    compare_vehicles,
    cheapest_vehicles,
    similar_vehicles,
    search_by_monthly_budget,
]

_tools_map = {t.name: t for t in tools}

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        base = ChatOllama(
            model="qwen3:4b",
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=1024,
            think=False,
        )
        _llm = base.bind_tools(tools)
    return _llm


def run_agent(message: str, history: list = None) -> str:
    """Run the tool-calling loop and return the final text response."""
    llm = _get_llm()

    msgs = [SystemMessage(content=_SYSTEM)]
    for turn in (history or [])[-6:]:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        else:
            msgs.append(AIMessage(content=turn["content"]))
    msgs.append(HumanMessage(content=message))

    for _ in range(5):  # max tool-call rounds
        response = llm.invoke(msgs)

        if not response.tool_calls:
            return response.content.strip()

        msgs.append(response)
        for tc in response.tool_calls:
            tool = _tools_map.get(tc["name"])
            result = tool.invoke(tc["args"]) if tool else f"أداة غير معروفة: {tc['name']}"
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return response.content.strip()
