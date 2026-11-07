"""Minimal agent harness: one model, one shell tool, a loop.

Usage: python harness.py   (needs ANTHROPIC_API_KEY; DEBUG=1 logs raw requests/responses)
Each session is also written to logs/<timestamp>.log.
"""
import datetime
import json
import os
import subprocess
import anthropic

client = anthropic.Anthropic()
DEBUG = os.environ.get("DEBUG")
MODEL = "claude-sonnet-5-5"
SYSTEM = "You are a coding agent. Use the shell tool to inspect and change files in the current directory."
TOOLS = [{
    "name": "shell",
    "description": "Run a shell command in the current directory. Returns stdout and stderr.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

os.makedirs("logs", exist_ok=True)
LOG_FILE = f"logs/{datetime.datetime.now():%Y%m%d-%H%M%S}.log"


def log(text, screen=True):
    if screen:
        print(text)
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


def dump(obj):
    return json.dumps(obj, indent=2, default=lambda o: o.model_dump())


def run(cmd):
    log(f"\n$ {cmd}")
    if input("run? [y/N] ").strip().lower() != "y":
        log("(declined)")
        return "User declined to run this command."
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        out = (p.stdout + p.stderr)[-20000:] + f"[exit code {p.returncode}]"
    except subprocess.TimeoutExpired:
        out = "Command timed out after 120s."
    log(out)
    return out


if DEBUG:
    log(">>> SETUP\n" + dump(dict(model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS)))

messages = []
sent = 0  # messages already logged, so each debug entry shows only what's new
while True:
    try:
        text = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        break
    log(f"\n> {text}", screen=False)
    messages.append({"role": "user", "content": text})
    while True:
        if DEBUG:
            log(">>> REQUEST (new messages)\n" + dump(messages[sent:]))
        resp = client.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
        )
        if DEBUG:
            log("<<< RESPONSE\n" + resp.model_dump_json(indent=2))
        messages.append({"role": "assistant", "content": resp.content})
        sent = len(messages)
        for block in resp.content:
            if block.type == "text":
                log(block.text)
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id, "content": run(b.input["command"])}
            for b in resp.content if b.type == "tool_use"
        ]})
