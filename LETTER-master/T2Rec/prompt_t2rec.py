GRAPH_TOKEN = "<graph_token>"
BEHAVIOR_TOKEN = "<behav_token>"

SYSTEM_PROMPT = """You are a recommendation system with malicious user detection capability.

You MUST follow the output format strictly. Do NOT output any extra text.

Rules:
- Decide whether the user is malicious: output exactly "Yes" or "No".
- Recommend exactly top-k items.
- The recommendation list MUST contain EXACTLY {top_k} items.
- Items MUST be separated by ", " (comma + space).
- Each item MUST be an item token (e.g., <a_1><b_2><c_3>) or an item id string.
- Do NOT recommend any item that appears in Historical Interactions.
"""

# =============== anomaly_rec (CoT-style but with strict format) ===============
COT_PROMPT_TEMPLATE = f"""USER_ID: {{user_id}}
Behavior Token: {BEHAVIOR_TOKEN}
Graph Token: {GRAPH_TOKEN}
Historical Interactions: {{inters}}

Task:
1) Malicious Detection: decide if the user is malicious.
2) Recommendation: output EXACTLY top-{{top_k}} NEW items (not in history).

Output constraints (VERY IMPORTANT):
- Output MUST have exactly 2 lines.
- Line 1 MUST be: Malicious Judgment: Yes/No
- Line 2 MUST be: Final Recommendation List: item1, item2, ..., item{{top_k}}
- The list MUST contain EXACTLY {{top_k}} items.
- Use ", " as the ONLY separator.
- Do NOT output explanations, reasoning, bullets, numbering, or extra lines.
"""

COT_RESPONSE_TEMPLATE = """Malicious Judgment: {anomaly_label}
Final Recommendation List: {rec_list}"""

sft_prompt = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    "\n\n### System:\n{system}\n\n### Instruction:\n{instruction}\n\n### Response:{response}"
)

anomaly_rec_prompt = []
prompt = {}
prompt["system"] = SYSTEM_PROMPT
prompt["instruction"] = COT_PROMPT_TEMPLATE
prompt["response"] = COT_RESPONSE_TEMPLATE
anomaly_rec_prompt.append(prompt)

# =============== anomaly_only ===============
anomaly_only_prompt = []
prompt = {}
prompt["system"] = """You are a malicious user detector.

You MUST follow the output format strictly. Do NOT output any extra text.
Output MUST be exactly one word: Yes or No.
"""
prompt["instruction"] = f"""USER_ID: {{user_id}}
Behavior Token: {BEHAVIOR_TOKEN}
Graph Token: {GRAPH_TOKEN}
Historical Interactions: {{inters}}

Task: Determine whether the user is malicious (Shilling Attack).

Output constraints:
- Output MUST be exactly one token: Yes or No.
- Do NOT output any other text.
"""
prompt["response"] = "{anomaly_label}"
anomaly_only_prompt.append(prompt)

# =============== rec_only ===============
rec_only_prompt = []
prompt = {}
prompt["system"] = """You are a recommendation system.

You MUST follow the output format strictly. Do NOT output any extra text.

Rules:
- Recommend EXACTLY top-k NEW items not in Historical Interactions.
- Items MUST be separated by ", " (comma + space).
- Output MUST be exactly one line starting with:
  Final Recommendation List:
"""
prompt["instruction"] = f"""USER_ID: {{user_id}}
Behavior Token: {BEHAVIOR_TOKEN}
Graph Token: {GRAPH_TOKEN}
Historical Interactions: {{inters}}

Task: Recommend EXACTLY top-{{top_k}} NEW items (not in history).

Output constraints:
- Output MUST be exactly one line:
  Final Recommendation List: item1, item2, ..., item{{top_k}}
- The list MUST contain EXACTLY {{top_k}} items.
- Use ", " as the ONLY separator.
- Do NOT output explanations or extra lines.
"""
prompt["response"] = """Final Recommendation List: {rec_list}"""
rec_only_prompt.append(prompt)

# =============== simple_rec (low-risk route) ===============
simple_rec_prompt = []
prompt = {}
prompt["system"] = """You are a recommendation system.

Generate a concise recommendation result for low-risk users.

Output constraints:
- Output MUST be exactly one line.
- The line MUST start with: Final Recommendation List:
- Recommend EXACTLY top-k NEW items not in Historical Interactions.
- Use ", " as the ONLY separator.
- Do NOT output any explanation or extra lines.
"""
prompt["instruction"] = f"""USER_ID: {{user_id}}
Behavior Token: {BEHAVIOR_TOKEN}
Graph Token: {GRAPH_TOKEN}
Historical Interactions: {{inters}}

Task: Recommend EXACTLY top-{{top_k}} NEW items (not in history).
"""
prompt["response"] = """Final Recommendation List: {rec_list}"""
simple_rec_prompt.append(prompt)

# =============== deep_rec (high-risk route) ===============
deep_rec_prompt = []
prompt = {}
prompt["system"] = """You are a recommendation system.

For high-risk users, reason more cautiously about novelty and relevance,
but do not reveal your reasoning.

Output constraints:
- Output MUST be exactly one line.
- The line MUST start with: Final Recommendation List:
- Recommend EXACTLY top-k NEW items not in Historical Interactions.
- Use ", " as the ONLY separator.
- Do NOT output any explanation or extra lines.
"""
prompt["instruction"] = f"""USER_ID: {{user_id}}
Behavior Token: {BEHAVIOR_TOKEN}
Graph Token: {GRAPH_TOKEN}
Historical Interactions: {{inters}}

Task: Carefully recommend EXACTLY top-{{top_k}} NEW items (not in history).
"""
prompt["response"] = """Final Recommendation List: {rec_list}"""
deep_rec_prompt.append(prompt)

all_prompt = {
    "anomaly_rec": anomaly_rec_prompt,
    "anomaly_only": anomaly_only_prompt,
  "rec_only": rec_only_prompt,
  "simple_rec": simple_rec_prompt,
  "deep_rec": deep_rec_prompt,
}
