#!/usr/bin/env python3
"""
Check whether the vLLM OpenAI-compatible server is reachable from wherever
habitat-llm runs, before launching a full planner_demo experiment.

Run this in the SAME env/process/container that will run planner_demo.py
(not in the vLLM server's own venv) -- the whole point is to test the
network path habitat_llm/llm/vllm_model.py actually uses.

Usage:
    python scripts/check_vllm_connection.py
    python scripts/check_vllm_connection.py --host localhost --port 8000 \
        --model Meta-Llama-3.1-70B-Instruct
"""

import argparse
import socket
import sys
import time


def check_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    print(f"[1/4] TCP connect to {host}:{port} ...", end=" ", flush=True)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print("OK")
            return True
    except OSError as e:
        print(f"FAILED ({e})")
        if isinstance(e, ConnectionRefusedError):
            print(
                "      -> Connection refused: nothing is listening on that "
                "port. The vLLM server process is not up (crashed, not "
                "started yet, or bound to a different port)."
            )
        elif isinstance(e, TimeoutError):
            print(
                "      -> Timed out: something in between (firewall, "
                "wrong host, different container/network namespace) is "
                "dropping the connection rather than refusing it."
            )
        else:
            print(f"      -> {type(e).__name__}: check host/port are correct.")
        return False


def check_health(base_url: str, timeout: float = 5.0) -> bool:
    import urllib.error
    import urllib.request

    print(f"[2/4] GET {base_url}/health ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as resp:
            print(f"OK (status {resp.status})")
            return True
    except urllib.error.URLError as e:
        print(f"FAILED ({e})")
        return False


def check_models(base_url: str, expected_model: str, timeout: float = 5.0) -> bool:
    import json
    import urllib.error
    import urllib.request

    print(f"[3/4] GET {base_url}/v1/models ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as resp:
            data = json.loads(resp.read())
            served_names = [m["id"] for m in data.get("data", [])]
            print(f"OK, served model(s): {served_names}")
            if expected_model not in served_names:
                print(
                    f"      -> WARNING: '{expected_model}' "
                    f"(from generation_params.engine) is not in that list. "
                    "It must match --served-model-name exactly, or every "
                    "completion request will fail."
                )
                return False
            return True
    except urllib.error.URLError as e:
        print(f"FAILED ({e})")
        return False


def check_completion(base_url: str, model: str, timeout: float = 60.0) -> bool:
    print(
        f"[4/4] POST {base_url}/v1/completions (model={model}, a real "
        "generation call, may take a while on first request) ...",
        end=" ",
        flush=True,
    )
    try:
        from openai import OpenAI
    except ImportError:
        print("SKIPPED (openai package not installed in this env)")
        return True

    try:
        client = OpenAI(api_key="not-used", base_url=base_url, timeout=timeout)
        start = time.time()
        response = client.completions.create(
            model=model, prompt="Say OK.", max_tokens=5, temperature=0.0
        )
        elapsed = time.time() - start
        print(f"OK ({elapsed:.1f}s): {response.choices[0].text!r}")
        return True
    except Exception as e:  # noqa: BLE001 -- deliberately broad for a diagnostic
        print(f"FAILED ({type(e).__name__}: {e})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default="Meta-Llama-3.1-70B-Instruct",
        help="Must match --served-model-name / generation_params.engine",
    )
    parser.add_argument(
        "--skip-completion",
        action="store_true",
        help="Skip the real generation call (steps 1-3 only, instant)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    plain_base_url = f"http://{args.host}:{args.port}"

    ok = check_tcp(args.host, args.port)
    if not ok:
        print("\nStopping here -- fix the TCP connection before checking further.")
        return 1

    ok &= check_health(plain_base_url)
    ok &= check_models(plain_base_url, args.model)
    if not args.skip_completion:
        ok &= check_completion(base_url, args.model)

    print()
    print("All checks passed." if ok else "Some checks failed -- see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
