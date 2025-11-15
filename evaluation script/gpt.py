#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch调用过程：上传测试数据，自动转换为batch接受的格式（batch_input file)
             open AI platform 接收input file确认后开始答题，最后将答案存储于output file返回
             解析output file里的数据获得回答
batch 会在24小时后自动断掉

"""

import json, time, uuid, re, tempfile, os
from pathlib import Path
from statistics import mean
from openai import OpenAI

# ============= 配置 ==============
MODEL_NAME          = "gpt-4.1"
DATA_PATH           = Path("./sample_data/ruleExe_en.jsonl")          # 原始数据
FIRST_BATCH_FILE    = Path("first_batch_en.jsonl")      # 第一次推理请求
REFLECT_BATCH_FILE  = Path("reflect_batch_en.jsonl")    # 二次反思请求
FIRST_OUTPUT_FILE   = Path("first_output_en.jsonl")     # Batch 输出
REFLECT_OUTPUT_FILE = Path("reflect_output_en.jsonl")
FIRST_ERROR_OUT_FILE = Path("Aerror_output_en.jsonl") # 存储batch output file解析中的错误
REFLECT_ERROR_OUT_FILE = Path("Rerror_output_en.jsonl") # 存储batch output file解析中的错误
RESULTS_JSONL       = Path("./sample_data/Meta-cognition/RuleExecution.jsonl")
POLL_INTERVAL       = 5   # seconds


# =================================

client = OpenAI(api_key="")

def meta_prompt_choice(language, system_prompt, first_answer, rationality_prompt, confidence_prompt):
    """
    根据语言选择prompt
    """
    if language == "Spanish":
        meta_prompt = (
                f"Las tareas y respuestas que recibiste anteriormente son:\n"
                f"------\n{system_prompt}\n"
                f"Tu respuesta: {first_answer}\n------\n\n"
                f"{rationality_prompt}\n{confidence_prompt}\n"
                "Por favor responde **solo** con la siguiente estructura JSON:\n"
                "{\n"
                " \"rationality_judgement\": <Razonable / No razonable>,\n"
                " \"confidence_score\": <un número entero de 0-100>,\n"
                " \"model_reflection\": \"<tu razonamiento>,\n"
                " \"new_response\": \"<Si consideras que tu respuesta inicial es correcta, escribe \"None\". De lo contrario, proporciona una nueva respuesta.>\"\n"
                "}"
            )
    elif language == "Chinese":
        meta_prompt = (
            f"你之前收到的任务与你的回答如下：\n"
            f"------\n{system_prompt}\n"
            f"你的回答：{first_answer}\n------\n\n"
            f"{rationality_prompt}\n{confidence_prompt}\n"
            "请你仅按照以下 JSON 结构作答：\n"
            "{\n"
            "\"rationality_judgement\": \"合理 / 不合理\",\n"
            "\"confidence_score\": <0-100>,\n"
            "\"model_reflection\": \"<你的理由>\",\n"
            "\"new_response\": \"<如果你认为原答案正确填 None，否则给出新答案>\"\n"
            "}"
        )
    elif language == "English":
        meta_prompt = (
            f"The task and your response are as follows:\n"
            f"------\n{system_prompt}\n"
            f"Your response: {first_answer}\n------\n\n"
            f"{rationality_prompt}\n{confidence_prompt}\n"
            "Please respond **only** using the following JSON structure:\n"
            "{\n"
            "\"rationality_judgement\": <Reasonable / Not reasonable>,\n"
            "\"confidence_score\": <an integer between 0–100>,\n"
            "\"model_reflection\": \"<your reasoning>\",\n"
            "\"new_response\": \"[If you consider your initial answer to be correct, write 'None'; otherwise, provide a new answer]\"\n"
            "}"
        )
        
    else:
        print("No language found, cannot decide the meta prompt.")
        return None
    print(meta_prompt)
    return meta_prompt

# ---------- 1. 读取数据 ----------
records = [json.loads(l) for l in open(DATA_PATH, 'r', encoding='utf-8')]
print(f"Loaded {len(records)} samples.")


# ---------- 2. 生成第一次推理 Batch 输入 ----------
with FIRST_BATCH_FILE.open("w", encoding="utf-8") as fout:
    for rec in records:
        req = {
            "custom_id": rec["id"],               # 方便后续对齐
            "method"   : "POST",
            "url"      : "/v1/chat/completions",
            "body"     : {
                "model": MODEL_NAME,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": rec["system_prompt"]}
                ]
            }
        }
        fout.write(json.dumps(req, ensure_ascii=False) + "\n")
print(f"First batch file written: {FIRST_BATCH_FILE}")


# ---------- 3. 上传文件 & 创建 Batch ----------
file_resp = client.files.create(file=open(FIRST_BATCH_FILE, "rb"), purpose="batch")
batch_resp = client.batches.create(
    input_file_id=file_resp.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"   # 处理窗口，可选 1h/2h/24h/...
)
batch_id = batch_resp.id
print(f"First batch created: {batch_id}")

# ---------- 4. 轮询等待批处理完成 ----------
while True:
    batch = client.batches.retrieve(batch_id)

    # —— 取字段方式 A：直接用属性 ——  
    counts     = batch.request_counts
    completed  = counts.completed   or 0
    total      = counts.total       or 0
    failed     = counts.failed      or 0
    in_prog   = max(total - completed - failed, 0)
    # 播报batch处理进度
    print(f"[…] Status: {batch.status} | "
            f"Completed: {completed}/{total} "
            f"(in‑progress={in_prog}, failed={failed}), completed={completed}")

    if failed and getattr(batch, "failed_file", None):
        # batch中途出错，从platform下载错误报告，查看原因
        import requests, json, itertools
        ln = next(iter(requests.get(batch.failed_file.url, stream=True).iter_lines()))
        print("Example error:", json.loads(ln.decode())["error"])
        # 只打印一次即可
        batch.failed_file = None  

    if batch.status in ("completed", "failed", "expired"):
        #batch结束
        break
    time.sleep(POLL_INTERVAL)

# ---------- 5. 下载 Batch 输出 ----------
url = client.batches.retrieve(batch_id).output_file_id
content = client.files.content(url)
content_bytes = content.read() 
FIRST_OUTPUT_FILE.write_bytes(content_bytes)
print(f"First batch output saved to {FIRST_OUTPUT_FILE}")

# ---------- 6. 解析第一次回答 ----------
Aerr_out = FIRST_ERROR_OUT_FILE.open("w", encoding="utf-8")
id2first_ans = {}
for line in open(FIRST_OUTPUT_FILE, 'r', encoding='utf-8'):
    obj = json.loads(line)
    cid  = obj["custom_id"]

    # -------- 兼容两种成功结构, 有时候输出文件choices位置变化 --------
    resp_obj = obj.get("response", {})
    if "choices" in resp_obj:                     # 老格式
        choices = resp_obj["choices"]
    elif "body" in resp_obj and "choices" in resp_obj["body"]:  # 新格式
        choices = resp_obj["body"]["choices"]
    else:
        # 无法解析，写入错误文件
        Aerr_out.write(line)
        continue
    
    ans  = choices[0]["message"]["content"].strip()
    id2first_ans[cid] = ans
    print(id2first_ans)

# ---------- 7. 构建反思 Batch 输入 ----------
with REFLECT_BATCH_FILE.open("w", encoding="utf-8") as fout:
    for rec in records:
        try:
            first_ans = id2first_ans[rec["id"]]
            meta_prompt = meta_prompt_choice(rec['language'], rec['system_prompt'], first_ans, rec['rationality_prompt'], rec['confidence_prompt'])
            if meta_prompt == None:
                meta_prompt = (
                    f"Las tareas y respuestas que recibiste anteriormente son:\n"
                    f"------\n{rec['system_prompt']}\n"
                    f"Tu respuesta: {first_ans}\n------\n\n"
                    f"{rec['rationality_prompt']}\n{rec['confidence_prompt']}\n"
                    "Por favor responde **solo** con la siguiente estructura JSON:\n"
                    "{\n"
                    " \"rationality_judgement\": <Razonable / No razonable>,\n"
                    " \"confidence_score\": <un número entero de 0-100>,\n"
                    " \"model_reflection\": \"<tu razonamiento>,\n"
                    " \"new Response\": \"<Si consideras que tu respuesta inicial es correcta, escribe \"None\". De lo contrario, proporciona una nueva respuesta.>\"\n"
                    "}"
                )
                print("back up meta_prompt")
            
            req = {
                "custom_id": rec["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": meta_prompt}]
                }
            }
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
        except KeyError:
            print(f"{rec["id"]} not answered.")
            continue

print(f"Reflect batch file written: {REFLECT_BATCH_FILE}")

# ---------- 8. 创建反思 Batch ----------
file2 = client.files.create(file=open(REFLECT_BATCH_FILE, "rb"), purpose="batch")
batch2 = client.batches.create(input_file_id=file2.id, endpoint="/v1/chat/completions", completion_window="24h")
batch2_id = batch2.id
print(f"Reflect batch created: {batch2_id}")

# ---------- 9. 轮询等待 ----------
while True:
    st_batch = client.batches.retrieve(batch2_id)
    # —— 取字段方式 A：直接用属性 ——  
    counts     = st_batch.request_counts
    completed  = counts.completed   or 0
    total      = counts.total       or 0
    failed     = counts.failed      or 0
    in_prog   = max(total - completed - failed, 0)
    # 播报batch处理进度
    print(f"[…] Status: {st_batch.status} | "
            f"Completed: {completed}/{total} "
            f"(in‑progress={in_prog}, failed={failed}), completed={completed}")

    if failed and getattr(st_batch, "failed_file", None):
        # batch中途出错，从platform下载错误报告，查看原因
        import requests, json, itertools
        ln = next(iter(requests.get(st_batch.failed_file.url, stream=True).iter_lines()))
        print("Example error:", json.loads(ln.decode())["error"])
        # 只打印一次即可
        st_batch.failed_file = None  

    if st_batch.status in ("completed", "failed", "expired"):
        #batch结束
        break
    time.sleep(POLL_INTERVAL)

# 下载第二次的output batch file 获取里面的回答
content2 = client.files.content(client.batches.retrieve(batch2_id).output_file_id)
content2_bytes = content2.read() 
REFLECT_OUTPUT_FILE.write_bytes(content2_bytes)
print(f"Reflect output saved: {REFLECT_OUTPUT_FILE}")

# ---------- 10. 解析反思答案 ----------
pattern = re.compile(
    r'"rationality_judgement"\s*:\s*"([^"]+)"\s*,\s*'
    r'"confidence_score"\s*:\s*(\d{1,3})\s*,\s*'
    r'"model_reflection"\s*:\s*"([^"]*)"\s*,\s*'
    r'"new_response"\s*:\s*"([^"]*)"', re.S
)

id2reflect = {}
for line in open(REFLECT_OUTPUT_FILE, 'r', encoding='utf-8'):
    obj = json.loads(line)
    cid = obj["custom_id"]
    text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
    m = pattern.search(text)
    if m:
        id2reflect[cid] = {
            "judgement": m.group(1).strip(),
            "confidence": int(m.group(2)),
            "reflection": m.group(3).strip(),
            "new_resp": None if m.group(4).strip().lower() == "none" else m.group(4).strip()
        }
    else:
        id2reflect[cid] = {
            "judgement": "",
            "confidence": 0,
            "reflection": "解析失败",
            "new_resp": None
        }

# ---------- 11. 计算准确率 & 写结果 ----------
total, correct_first, correct_final = 0, 0, 0
with RESULTS_JSONL.open("w", encoding="utf-8") as fout:
    
    for rec in records:
        try:
            cid  = rec["id"]
            gold = rec["gold_answer"].strip()
            first = id2first_ans[cid]
            refl  = id2reflect.get(cid, {})
            final = refl.get("new_resp") or first

            total += 1
            if first == gold:
                correct_first += 1
            if final == gold:
                correct_final += 1

            rationality_label = "合理" if first == gold else "不合理"
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
        except KeyError:
            print(f"Question {rec["id"]} not answered.")
            continue

print(f"首答准确率:  {correct_first/total*100:.2f}%")
print(f"反思后准确率: {correct_final/total*100:.2f}%")
print(f"结果已写出: {RESULTS_JSONL}")
