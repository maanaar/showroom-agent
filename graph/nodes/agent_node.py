from graph.state import AgentState
from llm.agent import run_agent


def agent_node(state: AgentState):
    message = state["current_message"]
    history = state.get("conversation_history", [])

    response = run_agent(message, history)

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]

    return {
        "response": response,
        "conversation_history": updated_history,
    }
