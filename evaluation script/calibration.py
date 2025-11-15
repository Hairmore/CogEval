
# ──────────────────────────────────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────────────────────────────────
from pathlib import Path
import json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import re

############################################################################
# 1)  CONFIGURE THESE TWO LINES ONLY
############################################################################
ROOT      = Path("")   # <-- adjust to your mount-point
LANG_CODE = "ch"                                 # e.g. "ch", "en", …

############################################################################
# 2)  CONSTANTS / HELPERS   ← keep CORRECT_LABELS and brier_score as before
############################################################################
############################################################################
# 2)  CONSTANTS / HELPERS
############################################################################
import re

CORRECT_SET   = {"合理", "Reasonable", "Razonable"}      # ground-truth “right”
INCORRECT_SET = {"不合理", "Not reasonable", "No razonable"}

MODEL_ORDER = [
    "gpt-4.1mini",
    "qwen-plus-notk",
    "qwen-plus-tk",
    "qwen3-32b-notk",
    "qwen3-32b-tk",
    "llama3",
]

def canonical_model(raw: str) -> str | None:
    s = raw.lower().replace("local-", "").replace("-mini", "mini")
    table = {
        "gpt-4.1mini"    : {"gpt-4.1mini", "gpt-4.1-mini", "gpt4.1mini"},
        "qwen-plus-notk" : {"qwen-plus-notk", "qwen3plus-notk","qwenplus-notk"},
        "qwen-plus-tk"   : {"qwen-plus-tk","qwen3plus_tk","qwenplus-tk","qwen3plus-tk"},
        "qwen3-32b-notk" : {"qwen3-32b-notk"},
        "qwen3-32b-tk"   : {"qwen3-32b-tk","qwen32b-tk"},
        "llama3"         : {"llama3", "llama-3"},
    }
    for canon, variants in table.items():
        if s in variants:
            return canon
    return None
def is_reasonable(text) -> bool | None:
    """
    Map a label to  True  (reasonable / correct),
                    False (unreasonable / incorrect),
                    None  (missing / unrecognised).
    """
    if text is None:                # ←——  catches null / missing values
        return None

    t = str(text).strip().lower()

    if t in {"合理", "reasonable", "razonable"}:
        return True
    if t in {"不合理", "unreasonable", "irrazonable"}:
        return False

    return None       

def load_language(root, lang_code):
    rows = []
    lang_dir = root / lang_code

    for capability_dir in sorted(lang_dir.iterdir()):
        if not capability_dir.is_dir():
            continue

        for fp in capability_dir.glob("*.jsonl"):
            # pull canonical model name
            m = re.search(r"\(([^)]+)\)", fp.stem)
            model = canonical_model(m.group(1) if m else fp.stem)
            if model is None:
                print(f"[SKIP] {fp.name}: unknown model id")
                continue

            correct, conf = [], []

            with fp.open(encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)

                    # ----- confidence -------------------------------------------------
                    try:
                        conf_val = float(obj.get("confidence_score")) / 100.0
                    except (TypeError, ValueError):
                        conf_val = float(0) / 100.0
                        #continue          # skip if confidence missing / bad

                    # ----- correctness: judgement vs. label --------------------------
                    gt  = is_reasonable(obj.get("rationality_label", ""))
                    mdl = is_reasonable(obj.get("rationality_judgement", ""))
                    if gt is None or mdl is None:
                        print(mdl)
                        continue          # skip if wording un-recognised

                    correct.append(gt == mdl)
                    conf.append(conf_val)

            if not conf:
                print(f"[WARN] {fp.name}: no usable rows")
                continue

            # Brier & CQS
            c   = np.asarray(conf)
            crt = np.asarray(correct, dtype=float)
            brier = np.mean((c - crt) ** 2)
            rows.append({
                "capability": capability_dir.name,
                "model"     : model,
                "brier"     : brier,
                "cqs"       : 1.0 - brier,
            })

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"],
                                 categories=MODEL_ORDER,
                                 ordered=True)
    return df
# 3)  LOAD + SHAPE THE DATA
############################################################################
df = load_language(ROOT, LANG_CODE)



pivot = (df.pivot(index="capability", columns="model", values="cqs")
           .sort_index())
           
# save the “long” tidy frame
out_path = Path("calibration.csv")
print(df)
try:
    df.to_csv(out_path, index=False)        # ← saves alongside the script
    print("✅  文件保存成功：", out_path)
except FileNotFoundError as e:
    print("❌  目录不存在：", e.filename)
except PermissionError as e:
    print("❌  没有写权限或文件被占用：", e.filename)
except OSError as e:
    if e.errno == errno.EROFS:            # 30
        print("❌  文件系统只读（只读挂载、U 盘写保护或 macOS NTFS）")
    else:
        print("❌  其它 OS 错误：", e)

# (optional) if you also want the pivoted wide table you plotted
pivot = (df.pivot(index="capability", columns="model", values="cqs")
           .reindex(columns=MODEL_ORDER)
           .sort_index())
pivot.to_csv("calibration_wide.csv") 

############################################################################
# 4)  RADAR PLOT
############################################################################
labels  = pivot.index.tolist()
angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# ── spokes & limits ──────────────────────────────────────────────────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1)

# ── one trace per model ──────────────────────────────────────────────────
for model in pivot.columns:
    values = pivot[model].fillna(0).tolist() + pivot[model][:1].tolist()
    ax.plot(angles, values, linewidth=1, label=model)
    ax.fill(angles, values, alpha=0.10)

ax.set_title(f"Metacognitive Calibration (CQS) – language = {LANG_CODE}", pad=20)

# ── LEGEND OUTSIDE, ON THE RIGHT ─────────────────────────────────────────
#        (x-offset 1.15     y-centre 0.5)
ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5),
          frameon=False, fontsize=7)      # tweak fontsize / frame to taste

# bump the right margin so the legend isn’t clipped
fig.subplots_adjust(right=0.70)           # 0.70 leaves ~30 % of the width for legend

plt.tight_layout()
plt.show()
