"""ReAct agents built as explicit LangGraph state machines.

ReAct (Yao et al., 2022) = interleave Reasoning and Acting: the model decides on
an action (a tool call), observes the result, and repeats until it can answer.
Here the "action" step uses the model's native tool-calling API instead of
parsing 'Thought/Action:' text, which is how ReAct is usually built today.

Graph (same shape for both agents):

    START -> agent --(has tool calls?)--yes--> tools --> agent
                     +--no, but no FINAL ANSWER yet--> nudge --> agent  (at most 2x)
                     +--no--> END

``build_react_agent`` is the generic loop; ``build_rag_agent`` is the rebuilt
2025 notebook agent (one retriever tool over a document collection).
"""

from __future__ import annotations

import functools
import json
from typing import Annotated, Any, Callable, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import tools as T

REACT_SYSTEM_PROMPT = """You are a careful operations assistant for an online store.
Use the tools to look up every fact you need; never guess product data, rates or fees.
Use the calculator for arithmetic instead of doing it in your head.
Call one tool at a time if a later step depends on an earlier result.
When you have the result, reply briefly and end with one final line in this format:
FINAL ANSWER: <the number you computed>
Write the number in plain decimal form and keep the decimal point.
Do not add units, currency symbols or thousands separators."""


class AgentState(TypedDict):
    # add_messages appends new messages (and replaces ones with the same id).
    messages: Annotated[list[BaseMessage], add_messages]


def store_tools() -> list[BaseTool]:
    def lookup_product(name: str):
        """Look up a product by name. Returns price_usd, weight_kg (per unit), stock and warehouse."""
        return T.lookup_product(name)

    def get_warehouse(name: str):
        """Look up a warehouse by city name. Returns its region and storage_fee_usd_per_unit_per_day."""
        return T.get_warehouse(name)

    def shipping_cost(region: str, weight_kg: float):
        """Shipping cost in USD to a region (north, south, east, west or international). weight_kg is the TOTAL weight of the shipment: the unit weight_kg from lookup_product multiplied by the number of units."""
        return T.shipping_cost(region, weight_kg)

    def convert_currency(amount: float, from_currency: str, to_currency: str):
        """Convert money between USD, INR, EUR, GBP and JPY using the company's fixed rates."""
        return T.convert_currency(amount, from_currency, to_currency)

    def customer_discount(tier: str):
        """Discount fraction for a customer tier (bronze, silver, gold). 0.05 means 5% off."""
        return T.customer_discount(tier)

    def days_between(start_date: str, end_date: str):
        """Number of days from start_date to end_date, both YYYY-MM-DD."""
        return T.days_between(start_date, end_date)

    def calculator(expression: str):
        """Evaluate an arithmetic expression, e.g. '3 * 34.5 + 2.75'. Supports + - * / // % ** round floor ceil min max abs sqrt."""
        return T.calculator(expression)

    fns = [lookup_product, get_warehouse, shipping_cost, convert_currency,
           customer_discount, days_between, calculator]
    return [
        StructuredTool.from_function(func=_jsonify(f), name=f.__name__, description=f.__doc__)
        for f in fns
    ]


def _jsonify(f: Callable[..., Any]) -> Callable[..., str]:
    """Wrap a pure tool so the model always receives JSON text and readable errors.

    functools.wraps keeps f's signature, so LangChain can build the argument schema.
    """

    @functools.wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        try:
            return json.dumps({"result": f(*args, **kwargs)})
        except T.ToolError as e:
            return json.dumps({"error": str(e)})

    return wrapped


NUDGE = ("You stopped without finishing. Continue: call the tools you still need, "
         "then end with the line 'FINAL ANSWER: <number>'.")


def has_final_answer(text: str) -> bool:
    return "final answer" in (text or "").lower()


def build_react_agent(
    llm: Any,
    tools: Sequence[BaseTool],
    system_prompt: str = REACT_SYSTEM_PROMPT,
    is_finished: Callable[[str], bool] | None = has_final_answer,
    max_nudges: int = 2,
):
    """Compile the agent graph. ``llm`` needs ``bind_tools`` and ``invoke`` (any LangChain chat model).

    If ``is_finished`` is given and the model stops (no tool call) with a reply that
    fails it, a 'nudge' message sends it back to the agent node, at most ``max_nudges``
    times. Small models sometimes stop mid-task with an empty reply; this recovers them.
    """
    by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(list(tools))

    def agent_node(state: AgentState) -> dict:
        reply = llm_with_tools.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [reply]}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                content = json.dumps({"error": f"unknown tool {call['name']!r}"})
            else:
                try:
                    # Pass the FULL argument dict (the 2025 notebook only passed 'query').
                    content = str(tool.invoke(call["args"]))
                except Exception as e:  # bad arguments from the model
                    content = json.dumps({"error": f"{type(e).__name__}: {e}"})
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def nudge_node(state: AgentState) -> dict:
        return {"messages": [HumanMessage(content=NUDGE)]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        if is_finished is not None and not is_finished(str(last.content)):
            used = sum(1 for m in state["messages"] if isinstance(m, HumanMessage) and m.content == NUDGE)
            if used < max_nudges:
                return "nudge"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("nudge", nudge_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "nudge": "nudge", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("nudge", "agent")
    return g.compile()


RAG_SYSTEM_PROMPT = """You answer questions using a document collection.
Always call the search_documents tool before answering, even if you think you know.
Answer only from the retrieved passages and cite the passage ids you used in [brackets].
If the passages do not contain the answer, say so."""


def retriever_tool(search: Callable[[str], list[tuple[str, str]]]) -> BaseTool:
    """Wrap ``search(query) -> [(doc_id, text), ...]`` as a tool returning cited passages."""

    def search_documents(query: str) -> str:
        """Search the document collection and return the most relevant passages with their ids."""
        hits = search(query)
        if not hits:
            return "No relevant passages found."
        return "\n---\n".join(f"[{doc_id}] {text}" for doc_id, text in hits)

    return StructuredTool.from_function(func=search_documents)


def build_rag_agent(llm: Any, search: Callable[[str], list[tuple[str, str]]]):
    return build_react_agent(llm, [retriever_tool(search)], RAG_SYSTEM_PROMPT, is_finished=None)


def count_tool_calls(messages: Sequence[BaseMessage]) -> int:
    return sum(len(m.tool_calls) for m in messages if isinstance(m, AIMessage))
