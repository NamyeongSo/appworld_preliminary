import os
from datetime import datetime
from openai import OpenAI
import re
import json


def add(a: float, b: float):
    return a + b


def get_current_time():
    return datetime.now().isoformat()


def count_characters(text: str):
    return len(text)


def ask_user(reaction: str):
    print(f"🤖 {reaction}")
    return input("👨 ")


os.environ["OPENAI_API_KEY"] = "NONE"
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="http://localhost:8002/v1"
)


def call_vllm(messages):
    response = client.chat.completions.create(
        # model="Qwen/Qwen2.5-1.5B-Instruct",
        model = "google/gemma-4-26B-A4B-it",
        messages=messages,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True
            },
            "skip_special_tokens": False
        }
    )
    print(f"🤖 Raw response:\n{response}\n")
    print(f"🤖 Reasoning:\n{response.choices[0].message.reasoning}\n")
    return response.choices[0].message.content


def thinking_parser(response):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        content = re.sub(pattern, "", response, flags=re.DOTALL).strip()
        return thinking, content

    return None, response


def tool_call_parser(response):
    pattern = r"<response>(.*?)</response>"
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return None

    payload = match.group(1).strip()

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    return parsed.get("tool_call")


def execute_tool(tool_call):
    function_name = tool_call["name"]
    arguments = tool_call.get("arguments", {})

    if function_name not in globals():
        raise ValueError(f"Unknown function: {function_name}")

    function = globals()[function_name]
    result = function(**arguments)

    return function_name, arguments, result


def generate_conversation(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"👨 {user_prompt}")

    while True:
        response = call_vllm(messages)
        think, content = thinking_parser(response)

        messages.append({
            "role": "assistant",
            "content": response
        })

        tool_calls = tool_call_parser(response)

        if tool_calls is None:
            raise ValueError(f"LLM format error:\n{response}")

        print(f"🤖 tool_call: {tool_calls}")

        if len(tool_calls) == 0:
            break

        for tool_call in tool_calls:
            function_name, arguments, result = execute_tool(tool_call)

            messages.append({
                "role": "assistant",
                "content": f"{function_name}({arguments}) = {result}"
            })

            print(f"🤖 tool_call result: {result}")

            if function_name == "ask_user":
                if result == "":
                    return messages

                messages.append({
                    "role": "user",
                    "content": result
                })

                break

    return messages


system_prompt = """
You are a tool-using assistant.

Your job:
- Read the user's request.
- Decide which tool should be called.
- You must always output exactly one tool call.

Available tools:
1. add
   - Description: Add two numbers.
   - Arguments:
     - a: float
     - b: float

2. get_current_time
   - Description: Get the current time.
   - Arguments: none

3. count_characters
   - Description: Count the number of characters in a string.
   - Arguments:
     - text: string

4. ask_user
   - Description: Send a message to the user and optionally receive the user's next reply.
   - Arguments:
     - reaction: string

Rules:
- Never return an empty tool_call list.
- Always output exactly one tool call.
- If the user greets you, chats casually, asks a general question, or needs a final answer, use ask_user.
- If you need to ask the user a clarification question, use ask_user.
- If the user's request can be handled by add, get_current_time, or count_characters, call that tool first.
- After a non-ask_user tool result is available, use ask_user in the next turn to report the result to the user.
- Do not provide natural language outside the required XML format.
- The content inside <response> must be valid JSON.
- JSON keys and string values must use double quotes.
- The "tool_call" value must always be a list containing exactly one object.

Output format:
<response>
{
  "tool_call": [
    {
      "name": "function_name",
      "arguments": {
        "argument_name": "argument_value"
      }
    }
  ]
}
</response>
"""

user_prompt = input("Enter your prompt: ")
conversation = generate_conversation(system_prompt, user_prompt)
print(conversation)