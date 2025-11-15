#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a **local** Llama‑3 model on an analogy‑reasoning data set and
score its answers with GPT‑4o (remote).  ‑‑ 2025‑07‑21
--------------------------------------------------------------------
Required set‑up
1. pip install llama-cpp-python==0.2.34  (or newer)
2. Download a GGUF checkpoint, e.g.
      Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
   and put its path into LLAMA_MODEL_PATH below.
3. Make sure your GPU/CPU has enough VRAM/RAM.  Quantised Q4_K_M fits
   comfortably on a 16 GB GPU or 24‑32 GB system RAM when run on CPU.
--------------------------------------------------------------------
The rest of the script is identical to your original logic *except* all
batch‑upload / polling code has been replaced by **local function calls**.
"""

import json, re, time, uuid, os
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
              # :contentReference[oaicite:0]{index=0}


# =============  CONFIG  ===============

LLAMA_MODEL_PATH   = Path("")  # <‑‑ edit local path to llama
N_GPU_LAYERS       = 0          # 0 = CPU‑only.  Tune for your GPU.
LLAMA_CTX_LEN      = 8192        # context length (Llama 3 supports 8 k)
MAX_NEW_TOKENS     = 512

MODEL_NAME         = "local-llama3"


DATA_PATH           = Path("./sample_data/ruleExe_en.jsonl")          # 原始数据
RESULTS_JSONL      = Path("./sample_data/Meta-cognition/RuleExecution.jsonl")
 
# Two passes of generation are still used 
FIRST_ERROR_OUT_FILE   = Path("Aerror_output_ch.jsonl")
REFLECT_ERROR_OUT_FILE = Path("Rerror_output_ch.jsonl")


# =============  INITIALISE LOCAL LLaMA 3  ===============
assert torch.cuda.is_available()
print("Loading local Llama‑3 model … this may take 30‑60 s.")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.float16,             # fp16
        device_map="auto",
        #attn_implementation="flash_attention_2",
    )

print("Model loaded.\n")

# ------------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------------
def llama_chat(messages, temperature=0.0, max_tokens=512):
    """
    messages: list like [{'role':'user', 'content':'…'}, …]
    returns:  assistant string
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(model.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out_ids = model.generate(
        **inputs,
        do_sample=False,  
        max_new_tokens=max_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out_ids[0][inputs["input_ids"].shape[-1]:]  # strip the prompt
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ------------------------------------------------------------------
# 1. Read data
# ------------------------------------------------------------------
records = [json.loads(l) for l in open(DATA_PATH, "r", encoding="utf‑8")]
print(f"Loaded {len(records)} samples from {DATA_PATH}")

# ------------------------------------------------------------------
# 2‑3. First‑pass inference (LOCAL)
# ------------------------------------------------------------------
id2first_ans = {}
Aerr_out = FIRST_ERROR_OUT_FILE.open("w", encoding="utf‑8")

for rec in records:
    try:
        messages = [{"role": "user", "content": rec["system_prompt"]}]
        ans = llama_chat(messages, temperature=0.0).strip()
        #print("llama answer: ", ans)
        id2first_ans[rec["id"]] = ans
    except Exception as e:
        Aerr_out.write(
            json.dumps({"id": rec["id"], "error": str(e)}) + "\n"
        )

print(f"First‑pass answers obtained for {len(id2first_ans)} / {len(records)}")

# ------------------------------------------------------------------
# 4‑5. Second‑pass (self‑reflection) using local model
# ------------------------------------------------------------------
id2reflect = {}
Rerr_out = REFLECT_ERROR_OUT_FILE.open("w", encoding="utf‑8")

for rec in records:
    first_ans = id2first_ans.get(rec["id"])
    if first_ans is None:
        continue  # skip samples that failed in pass 1
    
    meta_prompt = (
            f"The task and your response are as follows:\n"
            f"------\n{rec['system_prompt']}\n"
            f"Your response: {first_ans}\n------\n\n"
            f"{rec['rationality_prompt']}\n{rec['confidence_prompt']}\n"
            "Please respond **only** using the following JSON structure:\n"
            "{\n"
            "\"rationality_judgement\": <Reasonable / Not reasonable>,\n"
            "\"confidence_score\": <an integer between 0–100>,\n"
            "\"model_reflection\": \"<your reasoning>\",\n"
            "\"new_response\": \"[If you consider your initial answer to be correct, write 'None'; otherwise, provide a new answer]\"\n"
            "}"
        )
    try:
        messages = [{"role": "user", "content": meta_prompt}]
        reflection_json = llama_chat(messages, temperature=0.0).strip()
        #print("ref json:", type(reflection_json))
        # Try to parse the JSON blob
        pattern = re.compile(
                r'"rationality_judgement"\s*:\s*"?([^"]+)"?\s*,\s*'
                r'"confidence_score"\s*:\s*(\d{1,3})\s*,\s*'
                r'"model_reflection"\s*:\s*"?([^"]*)"?\s*,\s*'
                r'"new_response"\s*:\s*"?([^"]*)"?',
                re.S,
            )
        m = pattern.search(reflection_json)
        if m:
            id2reflect[rec["id"]] = {
                "judgement": m.group(1).strip(),
                "confidence": int(m.group(2)),
                "reflection": m.group(3).strip(),
                "new_resp": None
                if m.group(4).strip().lower() == "none"
                else m.group(4).strip(),
            }
            #print(m.group(1).strip(), m.group(2), "reflection"+m.group(3).strip(), m.group(4).strip().lower())
        else:
            raise ValueError("Failed to extract JSON fields")
    except Exception as e:
        Rerr_out.write(
            json.dumps({"id": rec["id"], "error": str(e)}) + "\n"
        )

# ------------------------------------------------------------------
# 6. Scoring & writing final results
# ------------------------------------------------------------------
total = len(records)
correct_first = correct_final = 0
overall_first = overall_final = 0

with RESULTS_JSONL.open("w", encoding="utf-8") as fout:
    for rec in records:
        cid  = rec["id"]
        gold = rec["gold_answer"].strip().upper()
        first = id2first_ans[cid].strip().upper()
        refl  = id2reflect.get(cid, {})
        if refl.get("new_resp") != None:
            final = refl.get("new_resp").strip().upper() or first
        else:
            final = first
        

        total += 1
        if first == gold:
            correct_first += 1
        if final == gold:
            correct_final += 1

        rationality_label = "Reasonable" if first == gold else "Not Reasonable"
        accuracy_flag = 1 if final == gold else 0

        result_obj = {
            "id":cid,
            "cognitive_operation": "Procedural Rule Execution: Rule Execution",
            "context": rec["system_prompt"],
            "gold_answer": gold,
            "model_answer": first,
            "model_reasoning": "",            # 如有 chain-of-thought 可填
            "rationality_label": rationality_label,
            "rationality_judgement": refl.get("judgement", ""),
            "confidence_score": refl.get("confidence", 0),
            "model_reflection": refl.get("reflection", ""),
            "new_response" : final

        }
        fout.write(json.dumps(result_obj, ensure_ascii=False) + "\n")

print(f"首答准确率:  {correct_first/total*100:.2f}%")
print(f"反思后准确率: {correct_final/total*100:.2f}%")
print(f"结果已写出: {RESULTS_JSONL}")