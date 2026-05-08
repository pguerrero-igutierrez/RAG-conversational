"""
Loads the Mixtral-8x7B-Instruct model with 4-bit quantization via bitsandbytes,
exposes an llm_chat() helper used by the router, generator and feedback modules.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

hf_token = os.environ.get("HF_TOKEN", "")
hf_cache = "./hf_cache"
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

os.makedirs(hf_cache, exist_ok=True)
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_4bit = device == "cuda"

print(f"Device : {device}")
print(f"4-bit  : {use_4bit}")
print(f"Model  : {model_name}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=hf_cache, use_fast=True, token=hf_token
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if use_4bit:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=hf_cache,
        quantization_config=bnb_cfg,
        device_map="auto",
        token=hf_token,
    )
else:
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=hf_cache,
        torch_dtype=torch.float32,
        token=hf_token,
    )

llm.eval()

llm_pipe = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tokenizer,
    device_map="auto",
)

if device == "cuda":
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM used: {used:.1f} / {total:.1f} GB")
else:
    print("Model loaded on CPU.")


def llm_chat(system, user, temperature=0.1, max_new_tokens=512):
    prompt = f"[INST] {system}\n\n{user} [/INST]"
    out = llm_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()
