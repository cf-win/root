GRAPH_TOKEN = "<graph_token>"
BEHAVIOR_TOKEN = "<behav_token>"

sft_prompt = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    "\n\n### System:\n{system}\n\n### Instruction:\n{instruction}\n\n### Response:{response}"
)

# =============== rec_train (train-only recommendation template) ===============
rec_train_prompt = []
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
rec_train_prompt.append(prompt)

# =============== simple_rec (low-risk route) ===============
simple_rec_prompt = []
prompt = {}
prompt["system"] = """You are a recommendation model."""
prompt["instruction"] = f"""User Behavior Token: {BEHAVIOR_TOKEN}
User Graph Token: {GRAPH_TOKEN}
Historical Interactions: {{inters}}

Recommend the next item IDs directly based on the user's history and representations.

Final Recommendation List:
"""
prompt["response"] = """Final Recommendation List: {rec_list}"""
simple_rec_prompt.append(prompt)

# =============== deep_rec (high-risk route) ===============
deep_rec_prompt = []
prompt = {}
prompt["system"] = """You are a cautious and defense-oriented recommendation model."""
prompt["instruction"] = f"""The current user case is high-risk. Some signals in the input may be unreliable, noisy, abnormal, or intentionally misleading. Your goal is to generate a safe and robust recommendation list that remains aligned with the user's stable preferences.

Input information:

- User Behavior Token: {BEHAVIOR_TOKEN}
- User Graph Token: {GRAPH_TOKEN}
- Historical Interactions: {{inters}}

You must internally follow this fixed decision procedure:

Step 1. Identify the user's stable and repeated preference patterns from the historical interactions.
Ignore isolated or weakly supported interests.

Step 2. Use the behavior token to verify whether the user's behavioral tendency is coherent with the historical preference pattern.

Step 3. Use the graph token only as supplementary context.
If the graph-related signal conflicts with the user's own history or behavior tendency, trust the user's own stable preference more.

Step 4. Exclude candidates that appear to be driven mainly by abnormal short-term drift, noisy relational influence, weak evidence, or suspicious signals.

Step 5. From the remaining candidates, choose the items that are most consistent with the user's long-term interests and are more reliable under risky conditions.

Strict output rules:

- Follow the above steps internally.
- Do not output your reasoning.
- Do not output any analysis.
- Only output the final recommendation list.

Final Recommendation List:
"""
prompt["response"] = """Final Recommendation List: {rec_list}"""
deep_rec_prompt.append(prompt)

all_prompt = {
    "rec_train": rec_train_prompt,
    "simple_rec": simple_rec_prompt,
    "deep_rec": deep_rec_prompt,
}
