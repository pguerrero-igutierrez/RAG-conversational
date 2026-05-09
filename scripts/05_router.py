"""
05_router.py

Trains and uses a SetFit few-shot classifier to route queries.
1 = RAG needed (factual, specific questions).
0 = No RAG needed (conversational, greetings, general logic).
"""

import os
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

base_dir = "./rag_project"
router_path = f"{base_dir}/router_model"