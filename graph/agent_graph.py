from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes.intent_node import intent_node
from graph.nodes.motorcycle_node import motorcycle_node
from graph.nodes.scooter_node import scooter_node
from graph.nodes.helmet_node import helmet_node
from graph.nodes.complaint_node import complaint_node
from graph.nodes.booking_node import booking_node
from graph.nodes.compare_node import compare_node
from graph.nodes.response_node import response_node

# Intents that require data fetching before response
DATA_INTENTS = {"browse", "filter", "details", "installment"}


def _route_after_intent(state: AgentState) -> str:
    """Route to the correct category node, or straight to response for greeting/booking/other."""
    intent = state.get("intent", "other")
    product_type = state.get("product_type")  # motorcycle | scooter | helmet | None

    if intent == "complaint":
        return "complaint"

    if intent == "booking":
        return "booking"

    if intent == "compare":
        return "compare"

    if intent not in DATA_INTENTS:
        return "response"

    if product_type == "scooter":
        return "scooter"
    if product_type == "helmet":
        return "helmet"

    # Default to motorcycle (covers None / explicit "motorcycle")
    return "motorcycle"


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("intent", intent_node)
    graph.add_node("motorcycle", motorcycle_node)
    graph.add_node("scooter", scooter_node)
    graph.add_node("helmet", helmet_node)
    graph.add_node("complaint", complaint_node)
    graph.add_node("booking", booking_node)
    graph.add_node("compare", compare_node)
    graph.add_node("response", response_node)

    # Entry
    graph.set_entry_point("intent")

    # Intent → conditional branch
    graph.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "motorcycle": "motorcycle",
            "scooter": "scooter",
            "helmet": "helmet",
            "complaint": "complaint",
            "booking": "booking",
            "compare": "compare",
            "response": "response",
        },
    )

    # All category nodes feed into response
    graph.add_edge("motorcycle", "response")
    graph.add_edge("scooter", "response")
    graph.add_edge("helmet", "response")
    graph.add_edge("complaint", "response")
    graph.add_edge("booking", "response")
    graph.add_edge("compare", "response")

    # Response is terminal
    graph.add_edge("response", END)

    return graph.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent
