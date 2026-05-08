import json
import re
from pathlib import Path
from openai import OpenAI


BASE_DIR = Path("/home/thskadud/appworld")
OUTPUT_DIR = BASE_DIR / "diagnosis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/gemma-4-26B-A4B-it"
BASE_URL = "http://localhost:8002/v1"
API_KEY = "api_key"

MAX_TOKENS = 60000


SYSTEM_PROMPT = """
You are a researcher diagnosing the reflection of a personal assistant agent.
Your goal is to identify the critical error that occurred in the reflection.

Task Information:
Task ID: {task_id}
Question: {question}
Failure log: {error_log}
Task execution trajectory: {trajectory}

Mistake categories: {mistake_category}

Your diagnosis must be in the following JSON format:
{{
    "mistake_step": <step_number>,
    "mistake_category": <mistake category>,
    "reason": <detailed explanation>,
    "suggested_fix": {{
        "need_fix": <bool>,
        "fix_direction": <string>
    }}
}}

Instructions:
1. Identify the root cause of the failure.
2. "mistake_step" must correspond to an actual step in the trajectory.
3. If the mistake does not fit existing categories, create a new one with:
   {{
       "name": <category name>,
       "description": <category description>
   }}
4. The "reason" must clearly explain why the step is incorrect.
5. The "suggested_fix" should provide guidance, not the full solution.
6. Carefully analyze the full trajectory before making a decision.
7. Ensure the output is valid JSON and contains all required fields.
8. Return only the diagnosis JSON. Do not include markdown fences or extra text.
"""


def load_jsonl_file(file_path):
    data = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def normalize_text(text):
    if not isinstance(text, str):
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text


def normalize_obj(obj):
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [normalize_obj(v) for v in obj]

    if isinstance(obj, str):
        return normalize_text(obj)

    return obj


def compact_json(obj):
    obj = normalize_obj(obj)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def extract_json_from_response(text):
    if text is None:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_diagnosis(data):
    if not isinstance(data, dict):
        return False, "response is not a dict"

    required_keys = {
        "mistake_step",
        "mistake_category",
        "reason",
        "suggested_fix",
    }

    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        return False, f"missing keys: {missing_keys}"

    if not isinstance(data["suggested_fix"], dict):
        return False, "suggested_fix must be a dict"

    if "need_fix" not in data["suggested_fix"]:
        return False, "suggested_fix.need_fix is missing"

    if "fix_direction" not in data["suggested_fix"]:
        return False, "suggested_fix.fix_direction is missing"

    return True, None


def get_mistake_category(task_id, id_failure_category, global_failure_category):
    task_specific_categories = id_failure_category.get(task_id, {})

    return {
        "global_categories": global_failure_category,
        "task_specific_categories": task_specific_categories,
    }


def call_vllm(client, messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
            },
            "skip_special_tokens": False,
        },
    )

    message = response.choices[0].message

    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        print(f"🤖 Reasoning:\n{reasoning}\n")

    return message.content


def save_result(task_id, result):
    output_path = OUTPUT_DIR / f"{task_id.replace('/', '_')}_diagnosis.json"

    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return output_path


def run_diagnosis(client, item, id_failure_category, global_failure_category):
    task_id = item["task_id"]
    source_path = item["source_path"]

    source = load_json(BASE_DIR / source_path)

    question_path = BASE_DIR / "data" / "tasks" / task_id / "specs.json"
    question = load_json(question_path)["instruction"]

    error_log = source["root_trajectory_summary"]["evaluation"]

    trajectory_path = source["root_trajectory_path"]
    trajectory_temp = load_json(BASE_DIR / trajectory_path)["messages"]
    trajectory = [
        {
            "role": msg["role"],
            "content": msg["content"],
        }
        for msg in trajectory_temp if msg["content"] is not None
    ]

    mistake_category = get_mistake_category(
        task_id=task_id,
        id_failure_category=id_failure_category,
        global_failure_category=global_failure_category,
    )

    prompt = SYSTEM_PROMPT.format(
        task_id=task_id,
        question=compact_json(question),
        error_log=compact_json(error_log),
        trajectory=compact_json(trajectory),
        mistake_category=compact_json(mistake_category),
    )
    
    print(prompt)
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": "Analyze the trajectory and return only the diagnosis JSON.",
        },
    ]

    raw_response = call_vllm(client, messages)
    parsed_response = extract_json_from_response(raw_response)

    is_valid, error = validate_diagnosis(parsed_response)

    result = {
        "task_id": task_id,
        "source_path": source_path,
        "question": question,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "is_valid": is_valid,
        "error": error,
    }

    output_path = save_result(task_id, result)

    print(
        f"[DONE] {task_id} | "
        f"valid={is_valid} | "
        f"error={error} | "
        f"saved={output_path}"
    )

    return result


def split_dataset(items):
    random_pair_based = []
    single_based = []

    for item in items:
        source_path = item["source_path"]

        if "/analysis_random_pairs/" in source_path:
            random_pair_based.append(item)
        elif "/analysis_root/" in source_path:
            single_based.append(item)

    return single_based, random_pair_based


def main():
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    naive_pair_based = load_jsonl_file(
        "dataset/gemma-4-26b-a4b-it_all_inclusive_sft.jsonl"
    )

    single_based, random_pair_based = split_dataset(naive_pair_based)

    global_failure_category = {}
    id_failure_category = {}

    print(f"single_based: {len(single_based)}")
    print(f"random_pair_based: {len(random_pair_based)}")

    all_results = []

    for item in single_based:
        task_id = item.get("task_id", "unknown")

        try:
            result = run_diagnosis(
                client=client,
                item=item,
                id_failure_category=id_failure_category,
                global_failure_category=global_failure_category,
            )

        except Exception as e:
            print(f"[ERROR] {task_id}: {e}")

            result = {
                "task_id": task_id,
                "source_path": item.get("source_path"),
                "raw_response": None,
                "parsed_response": None,
                "is_valid": False,
                "error": str(e),
            }

            save_result(task_id, result)

        all_results.append(result)
        break  # --- REMOVE THIS TO RUN ON FULL DATASET ---
    summary_path = OUTPUT_DIR / "diagnosis_summary.json"

    with open(summary_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[SUMMARY SAVED] {summary_path}")


if __name__ == "__main__":
    main()