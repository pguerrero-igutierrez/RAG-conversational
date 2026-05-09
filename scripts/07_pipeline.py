"""
07_pipeline.py

End-to-End RAG pipeline orchestrator.
"""

import importlib
import time

router_module = importlib.import_module("05_router")
retriever_module = importlib.import_module("03_retriever")
generator_module = importlib.import_module("06_generator")