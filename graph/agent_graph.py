"""
LangGraph agent workflow for the Ayman Badr Motorcycle Showroom agent.

Flow:
  START → intent_node → route → data_node (if needed) → response_node → END
"""
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes.intent_node import intent_node
from graph.nodes.data_node import data_node, INTENTS_NEEDING_DATA
from graph.nodes.response_node import response_node


def _route_after_intent(state: AgentState) -> str:
    """After detecting intent, decide whether to fetch data or go straight to response."""
    if state.get("intent") in INTENTS_NEEDING_DATA:
        return "data_node"
    return "response_node"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("intent_node", intent_node)
    graph.add_node("data_node", data_node)
    graph.add_node("response_node", response_node)

    graph.set_entry_point("intent_node")

    graph.add_conditional_edges(
        "intent_node",
        _route_after_intent,
        {"data_node": "data_node", "response_node": "response_node"},
    )

    graph.add_edge("data_node", "response_node")
    graph.add_edge("response_node", END)

    return graph.compile()


# Singleton compiled graph
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent
