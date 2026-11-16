#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai",
# ]
# ///
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

model = client.models.list().data[0].id
messages = []

while True:
    try:
        prompt = input("User: ")
    except (KeyboardInterrupt, EOFError):
        break

    if prompt.lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    print("Assistant: ", end="", flush=True)
    answer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            answer += delta.content

    print("\n")
    messages.append({"role": "assistant", "content": answer})
