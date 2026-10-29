"""Small audited DeepSeek transport. Secrets never enter reports or prompts."""
import os
import time
import hashlib
import json

import httpx


class ModelError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(self, key=None, model=None, *, thinking=False, max_tokens=1800):
        self.key = key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.base = "https://api.deepseek.com"
        self.calls = []
        self.thinking = thinking
        self.max_tokens = max_tokens

    def complete(self, messages, *, tools=None, json_output=False):
        if not self.key:
            raise ModelError("服务器尚未配置 DEEPSEEK_API_KEY。")
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "thinking": {"type": "enabled" if self.thinking else "disabled"},
                   "temperature": 0, "max_tokens": self.max_tokens}
        if self.thinking:
            payload["reasoning_effort"] = "high"
        if tools:
            payload["tools"] = tools
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(2):
            started = time.monotonic()
            record = {"requested_model": self.model, "attempt": attempt + 1,
                      "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      "settings": {k: payload[k] for k in ["thinking", "temperature", "max_tokens"]}}
            try:
                with httpx.Client(timeout=httpx.Timeout(100, connect=15), trust_env=False) as client:
                    response = client.post(self.base + "/chat/completions", json=payload,
                                           headers={"Authorization": "Bearer " + self.key})
                record["http_status"] = response.status_code
                if response.status_code != 200:
                    # Do not log raw HTTP bodies, which may echo request credentials.
                    raise ModelError(f"DeepSeek HTTP {response.status_code}")
                data = response.json()
                record.update({"response_model": data.get("model"),
                               "system_fingerprint": data.get("system_fingerprint"),
                               "usage": data.get("usage", {})})
                choice = data["choices"][0]
                record["response_message"] = {k: v for k, v in choice["message"].items() if k != "reasoning_content"}
                record["finish_reason"] = choice.get("finish_reason")
                if choice.get("finish_reason") == "length":
                    raise ModelError("DeepSeek response reached its configured token limit.")
                return choice["message"]
            except (httpx.HTTPError, ValueError, KeyError, ModelError) as error:
                record["error"] = type(error).__name__ + (": " + str(error) if isinstance(error, ModelError) else "")
                retryable = isinstance(error, httpx.TransportError) or record.get("http_status") in {429, 500, 502, 503, 504}
                if not retryable or attempt == 1:
                    raise ModelError(record["error"]) from None
                time.sleep(1)
            finally:
                record["seconds"] = round(time.monotonic() - started, 3)
                self.calls.append(record)

    def metrics(self):
        return {"api_calls": len(self.calls),
                "input_tokens": sum(x.get("usage", {}).get("prompt_tokens", 0) for x in self.calls),
                "output_tokens": sum(x.get("usage", {}).get("completion_tokens", 0) for x in self.calls),
                "api_seconds": round(sum(x["seconds"] for x in self.calls), 3),
                "errors": sum("error" in x for x in self.calls)}
