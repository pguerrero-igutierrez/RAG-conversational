"""
07_pipeline.py

End-to-End RAG pipeline orchestrator.
"""

import importlib
import time

router_module = importlib.import_module("router")
retriever_module = importlib.import_module("retriever")
generator_module = importlib.import_module("generator")