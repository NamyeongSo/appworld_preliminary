# Insight 파일을 지정해 특정 AppWorld task에서 ReAct 에이전트 실행하기

이 문서는 `/home/thskadud/appworld` 레포에서 **insight/reflection 파일을 ReAct agent prompt에 주입한 뒤, 특정 task 하나에 대해 추론 실행**하는 방법을 정리한다.

## 요약

기본 `appworld run` CLI에는 `--insight-file` 옵션이 직접 존재하지 않는다. 따라서 insight를 쓰려면 아래 둘 중 하나를 사용한다.

1. `hypothesis_1` 실험 runner의 reflection-guided ReAct agent 사용
2. insight를 포함한 prompt 파일을 직접 만들고 `appworld run --override`로 `prompt_file_path` 지정

추천은 목적에 따라 다르다.

- 기존 `hypothesis_1` 실험/분석 흐름을 유지하고 싶으면 **방법 1**
- insight 파일 하나로 특정 task를 빠르게 한 번 돌려보고 싶으면 **방법 2**

---

## 관련 코드 위치

### 기본 ReAct agent

```text
experiments/code/simplified/react_code_agent.py
```

등록 이름:

```python
simplified_react_code_agent
```

이 agent는 `prompt_file_path`로 지정된 prompt template을 읽고, AppWorld task instruction을 렌더링한 뒤 ReAct 방식으로 Python code block을 생성/실행한다.

### 기본 ReAct prompt

```text
experiments/prompts/react_code_agent/instructions.txt
```

### AppWorld experiment 실행 CLI

```bash
appworld run <experiment_name> --task-id <TASK_ID>
```

`--task-id`를 넘기면 해당 task 하나만 실행한다.

### insight-guided ReAct agent

```text
hypothesis_1/code/guided_agent.py
hypothesis_1/code/guidance.py
hypothesis_1/code/runner.py
```

`hypothesis_1/code/guided_agent.py`에는 아래 agent가 등록되어 있다.

```python
hypothesis_1_reflection_react_code_agent
```

이 agent는 기존 `SimplifiedReActCodeAgent`를 상속하고, task instruction 뒤에 reflection guidance를 붙여 실행한다.

---

## 방법 1: `hypothesis_1` runner로 insight 지정하기

이 방법은 insight/reflection을 manifest에 넣고, `hypothesis_1.code.runner`가 해당 guidance를 ReAct prompt에 주입하도록 하는 방식이다.

### 1. insight 파일 준비

예시:

```text
/path/to/insight.txt
```

파일 안에는 plain text 또는 아래처럼 `<guidance>` 태그가 있어도 된다.

```xml
<guidance>
이 task에서는 Spotify 추천곡 전체를 queue에 추가한 뒤, queue 길이와 추천곡 수가 같은지 확인하고 shuffle/play 해야 한다.
</guidance>
```

### 2. 단일 insight manifest 생성

아래 예시는 task `8749218_1`에 insight 파일 하나를 연결한다.

```bash
python - <<'PY'
import json
from pathlib import Path
from hypothesis_1.code.guidance import extract_guidance_text

task_id = "8749218_1"
insight_path = Path("/path/to/insight.txt")
out_path = Path(".tmp/manual_reflection_manifest.json")

text = insight_path.read_text(encoding="utf-8")
guidance = extract_guidance_text(text, require_tags=False)

manifest = {
    "schema_version": 1,
    "unique_task_ids": [task_id],
    "samples": [
        {
            "sample_id": f"manual_{task_id}",
            "task_id": task_id,
            "reflections": {
                "analysis": {
                    "guidance": guidance,
                    "json_path": str(insight_path),
                    "guidance_chars": len(guidance),
                }
            },
        }
    ],
}

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out_path)
PY
```

생성 파일:

```text
.tmp/manual_reflection_manifest.json
```

### 3. prompt 주입 확인 dry-run

```bash
python -m hypothesis_1.code.runner \
  --manifest .tmp/manual_reflection_manifest.json \
  --condition reflection_analysis_pass1 \
  --task-id 8749218_1 \
  --run-id manual_8749218_1__analysis \
  --dry-run
```

prompt preview는 보통 아래 경로에 생긴다.

```text
hypothesis_1/outputs/prompt_previews/reflection_analysis_pass1/
```

여기서 실제 prompt에 guidance가 들어갔는지 확인할 수 있다.

### 4. 실제 ReAct 실행

로컬 vLLM/OpenAI-compatible 서버를 쓰는 경우 기본적으로 `MODEL_SERVER_URL`을 본다.

```bash
MODEL_SERVER_URL=http://localhost:8002 \
python -m hypothesis_1.code.runner \
  --manifest .tmp/manual_reflection_manifest.json \
  --condition reflection_analysis_pass1 \
  --task-id 8749218_1 \
  --run-id manual_8749218_1__analysis \
  --rerun-existing
```

### 5. 결과 확인

출력은 대략 아래에 저장된다.

```text
hypothesis_1/appworld_root/experiments/outputs/hypothesis_1/reflection_analysis_pass1/manual_8749218_1__analysis/tasks/8749218_1/
```

주요 파일:

```text
misc/run_record.json
misc/evaluation.json
logs/lm_calls.jsonl
logs/environment_io.md
evaluation/report.md
```

---

## 방법 2: insight 포함 prompt를 만들고 `appworld run --override` 사용하기

이 방법은 가장 단순하다. 기본 ReAct prompt에 insight를 삽입한 새 prompt 파일을 만든 뒤, `appworld run`에서 해당 prompt를 사용하게 한다.

### 1. insight 포함 prompt 생성

```bash
python - <<'PY'
from pathlib import Path
from appworld_agents.code.context_engineering.policy_context import inject_policy_context_prompt_template

orig = Path("experiments/prompts/react_code_agent/instructions.txt").read_text(encoding="utf-8")
insight = Path("/path/to/insight.txt").read_text(encoding="utf-8")

context_prompt = "# Policy Context\n\n## Insights\n\n1. " + insight.strip() + "\n"

injected = inject_policy_context_prompt_template(
    original_prompt_template=orig,
    context_prompt=context_prompt,
)

out = Path(".tmp/react_with_insight_prompt.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(injected, encoding="utf-8")
print(out.resolve())
PY
```

생성 파일 예시:

```text
/home/thskadud/appworld/.tmp/react_with_insight_prompt.txt
```

### 2. 특정 task에 대해 ReAct 실행

예를 들어 기존 config `simplified_react_code_agent/openai/gpt-4o-2024-05-13/test_normal`를 사용한다면:

```bash
appworld run simplified_react_code_agent/openai/gpt-4o-2024-05-13/test_normal \
  --task-id 8749218_1 \
  --root . \
  --override '{"config":{"agent":{"prompt_file_path":"/home/thskadud/appworld/.tmp/react_with_insight_prompt.txt","skip_if_finished":false}}}'
```

`skip_if_finished=false`를 넣는 이유는 같은 experiment/task output에 `misc/finished`가 이미 있으면 agent가 실행을 건너뛸 수 있기 때문이다.

### 3. 결과 확인

출력 위치:

```text
experiments/outputs/simplified_react_code_agent/openai/gpt-4o-2024-05-13/test_normal/tasks/8749218_1/
```

주요 파일:

```text
logs/environment_io.md
logs/lm_calls.jsonl
misc/usage.json
evaluation/report.md
```

---

## 모델/서버 설정 주의

### OpenAI API config를 쓰는 경우

`experiments/configs/simplified_react_code_agent/openai/...` config는 OpenAI API key 환경변수가 필요할 수 있다.

### 로컬 vLLM 서버를 쓰는 경우

`hypothesis_1/code/config.py`의 기본 모델 설정은 OpenAI-compatible local server를 전제로 한다.

기본값:

```text
MODEL_SERVER_URL=http://localhost:8002
model=google/gemma-4-26B-A4B-it
```

실행 전 서버가 떠 있어야 한다.

---

## 빠른 체크리스트

1. task id 확인

```text
8749218_1
```

2. insight 파일 준비

```text
/path/to/insight.txt
```

3. 가장 간단한 방식 선택

- `hypothesis_1` 분석 흐름 유지: 방법 1
- 단발성 실행: 방법 2

4. dry-run 또는 prompt preview로 insight가 들어갔는지 확인

5. 실제 실행 후 아래 파일 확인

```text
logs/environment_io.md
misc/evaluation.json 또는 evaluation/report.md
```

---

## 최소 명령 템플릿

### 방법 1 템플릿

```bash
python -m hypothesis_1.code.runner \
  --manifest <MANIFEST_JSON> \
  --condition reflection_analysis_pass1 \
  --task-id <TASK_ID> \
  --run-id <SAMPLE_ID>__analysis \
  --rerun-existing
```

### 방법 2 템플릿

```bash
appworld run <REACT_EXPERIMENT_NAME> \
  --task-id <TASK_ID> \
  --root . \
  --override '{"config":{"agent":{"prompt_file_path":"<INSIGHT_INJECTED_PROMPT>","skip_if_finished":false}}}'
```

