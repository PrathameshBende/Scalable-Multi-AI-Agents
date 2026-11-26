"""Scalable LLM and agent systems: RAG (HyDE, Fusion), ReAct agents, model compression.

Every module here is deliberately free of heavy imports at module level, so the
unit tests run on a plain CPU machine with only numpy, rank-bm25 and pytest.
"""

__version__ = "2.0.0"
