from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes.agent_node import agent_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)

    graph.set_entry_point("agent")

    graph.add_edge("agent", END)

    return graph.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent