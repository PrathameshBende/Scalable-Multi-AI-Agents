"""The LangGraph loop, driven by a scripted fake chat model (no real LLM needed)."""

import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from scalable_agents.agent import (  # noqa: E402
    build_rag_agent, build_react_agent, count_tool_calls, store_tools,
)


class ScriptedLLM:
    """Returns the scripted AIMessages in order; records what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.bound = None

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def call(name, args, cid):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


def test_react_loop_runs_tools_and_stops():
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[call("lookup_product", {"name": "aurora desk lamp"}, "c1")]),
        AIMessage(content="", tool_calls=[call("calculator", {"expression": "3 * 34.5"}, "c2")]),
        AIMessage(content="FINAL ANSWER: 103.5"),
    ])
    agent = build_react_agent(llm, store_tools())
    out = agent.invoke({"messages": [HumanMessage(content="3 lamps?")]})
    msgs = out["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert json.loads(tool_msgs[0].content)["result"]["price_usd"] == 34.5
    assert json.loads(tool_msgs[1].content)["result"] == 103.5
    assert msgs[-1].content == "FINAL ANSWER: 103.5"
    assert count_tool_calls(msgs) == 2
    assert "calculator" in llm.bound and len(llm.bound) == 7
    # The model saw the tool result before its final answer (the ReAct observation step).
    assert any(isinstance(m, ToolMessage) for m in llm.seen[-1])


def test_multi_argument_tool_gets_all_arguments():
    # The 2025 notebook passed only args['query']; a 2-argument tool would have broken.
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[call("shipping_cost", {"region": "east", "weight_kg": 6.8}, "c1")]),
        AIMessage(content="FINAL ANSWER: 35.7"),
    ])
    out = build_react_agent(llm, store_tools()).invoke({"messages": [HumanMessage(content="q")]})
    assert json.loads(out["messages"][2].content)["result"] == 35.7


def test_bad_tool_name_and_bad_input_become_error_observations():
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[call("teleport", {}, "c1"),
                                          call("lookup_product", {"name": "toaster"}, "c2")]),
        AIMessage(content="FINAL ANSWER: 0"),
    ])
    out = build_react_agent(llm, store_tools()).invoke({"messages": [HumanMessage(content="q")]})
    errors = [json.loads(m.content)["error"] for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "unknown tool" in errors[0] and "unknown product" in errors[1]


def test_step_limit():
    from langgraph.errors import GraphRecursionError

    loop = [AIMessage(content="", tool_calls=[call("calculator", {"expression": "1+1"}, f"c{i}")])
            for i in range(50)]
    agent = build_react_agent(ScriptedLLM(loop), store_tools())
    with pytest.raises(GraphRecursionError):
        agent.invoke({"messages": [HumanMessage(content="q")]}, config={"recursion_limit": 5})


def test_rag_agent_returns_cited_passages():
    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[call("search_documents", {"query": "vitamin d"}, "c1")]),
        AIMessage(content="Vitamin D helps bones [d1]."),
    ])
    agent = build_rag_agent(llm, lambda q: [("d1", "vitamin d improves bone density")])
    msgs = agent.invoke({"messages": [HumanMessage(content="vitamin d?")]})["messages"]
    assert msgs[2].content.startswith("[d1] vitamin d")
    assert msgs[-1].content.endswith("[d1].")


def test_nudge_recovers_an_agent_that_stops_early():
    llm = ScriptedLLM([
        AIMessage(content=""),  # stops with nothing
        AIMessage(content="", tool_calls=[call("calculator", {"expression": "2*3"}, "c1")]),
        AIMessage(content="FINAL ANSWER: 6"),
    ])
    msgs = build_react_agent(llm, store_tools()).invoke({"messages": [HumanMessage(content="q")]})["messages"]
    assert msgs[-1].content == "FINAL ANSWER: 6"
    assert sum(isinstance(m, HumanMessage) for m in msgs) == 2  # question + one nudge


def test_nudges_are_capped():
    llm = ScriptedLLM([AIMessage(content="dunno") for _ in range(5)])
    msgs = build_react_agent(llm, store_tools(), max_nudges=2).invoke(
        {"messages": [HumanMessage(content="q")]})["messages"]
    assert len(llm.replies) == 2  # 1 first try + 2 nudged tries used
    assert msgs[-1].content == "dunno"


def test_product_shipping_tool_variant():
    import pytest

    llm = ScriptedLLM([
        AIMessage(content="", tool_calls=[call("shipping_cost", {"region": "south", "product": "nimbus kettle", "quantity": 4}, "c1")]),
        AIMessage(content="FINAL ANSWER: 25.2"),
    ])
    tools = store_tools(shipping="product")
    assert len(tools) == 7 and "quantity" in next(t for t in tools if t.name == "shipping_cost").args
    out = build_react_agent(llm, tools).invoke({"messages": [HumanMessage(content="q")]})
    assert json.loads(out["messages"][2].content)["result"] == 25.2
    with pytest.raises(ValueError):
        store_tools(shipping="drone")
