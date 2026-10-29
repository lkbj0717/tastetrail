"""OpenAI-compatible GLM/DeepSeek transport; never records hidden reasoning or secrets."""
import hashlib
import json
import time
import httpx
from .config import settings
from .llm import ModelError


class ModelClient:
    def __init__(self, *, provider=None, max_tokens=8192, transport=None, reasoning_effort=None):
        cfg = settings()
        provider = provider or cfg.get("TASTETRAIL_MODEL_PROVIDER", "glm")
        if provider not in {"glm", "deepseek"}:
            raise ValueError("Unsupported model provider")
        prefix = provider.upper()
        self.provider = provider
        self.key = cfg.get(prefix + "_API_KEY", "")
        self.base = cfg.get(prefix + "_BASE_URL", "https://openclaw-v2.top/llm/v1" if provider == "glm" else "https://api.deepseek.com").rstrip("/")
        self.model = cfg.get(prefix + "_MODEL", "glm-5.3-flash" if provider == "glm" else "deepseek-flash")
        self.calls, self.max_tokens, self.transport = [], max_tokens, transport
        if reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("Invalid reasoning effort")
        self.reasoning_effort = reasoning_effort

    def complete(self, messages, *, tools=None, json_output=False):
        if not self.key:
            raise ModelError("Model credentials are not configured on the server.")
        payload = dict(model=self.model, messages=messages, stream=False, temperature=0,
                       max_tokens=self.max_tokens)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        if tools:
            payload["tools"] = tools
        # This self-hosted GLM endpoint times out with guided JSON decoding or
        # native tool schemas (capability probe 2026-10-20). Enforce JSON locally.
        if tools and self.provider == "glm":
            raise ModelError("This GLM deployment requires the typed JSON workflow, not native tools.")
        if json_output and self.provider != "glm":
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(2):
            started = time.monotonic()
            record = {"requested_model": self.model, "attempt": attempt + 1,
                      "max_tokens": self.max_tokens, "temperature": 0,
                      "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
            try:
                with httpx.Client(timeout=httpx.Timeout(150, connect=15), trust_env=False, transport=self.transport) as http:
                    response = http.post(self.base + "/chat/completions", json=payload,
                                         headers={"Authorization": "Bearer " + self.key})
                record["http_status"] = response.status_code
                if response.status_code != 200:
                    raise ModelError(f"{self.provider} HTTP {response.status_code}")
                data = response.json()
                choice = data["choices"][0]
                record.update(response_model=data.get("model"), usage=data.get("usage") or {},
                              finish_reason=choice.get("finish_reason"))
                if choice.get("finish_reason") == "length":
                    raise ModelError("Model response exceeded token budget.")
                message = {k: v for k, v in choice["message"].items()
                           if k in {"role", "content", "tool_calls"} and v is not None}
                record["response_message"] = message
                return message
            except (httpx.HTTPError, ValueError, KeyError, IndexError, ModelError) as error:
                record["error"] = str(error) if isinstance(error, ModelError) else type(error).__name__
                retry = isinstance(error, httpx.TransportError) or record.get("http_status") in {429, 500, 502, 503, 504}
                if not retry or attempt:
                    raise ModelError(record["error"]) from None
                time.sleep(1)
            finally:
                record["seconds"] = round(time.monotonic() - started, 3)
                self.calls.append(record)

    def metrics(self):
        return {"api_calls": len(self.calls), "api_seconds": round(sum(x["seconds"] for x in self.calls), 3),
                "input_tokens": sum(x.get("usage", {}).get("prompt_tokens", 0) for x in self.calls),
                "output_tokens": sum(x.get("usage", {}).get("completion_tokens", 0) for x in self.calls),
                "errors": sum("error" in x for x in self.calls)}
