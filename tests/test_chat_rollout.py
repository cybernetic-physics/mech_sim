from __future__ import annotations

from typing import Any

from rl import chat_rollout


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "bad request"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeRequests:
    calls: list[dict[str, Any]] = []

    @classmethod
    def post(
        cls,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
        headers: dict[str, str],
    ) -> FakeResponse:
        cls.calls.append(json)
        if len(cls.calls) == 1:
            return FakeResponse(400)
        return FakeResponse(200, {"ok": True})


def test_chat_completion_retries_without_seed_on_bad_request(monkeypatch) -> None:
    FakeRequests.calls = []
    monkeypatch.setattr(chat_rollout, "requests", FakeRequests)

    result = chat_rollout._chat_completion(
        base_url="http://localhost:30000",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        timeout_s=1.0,
        seed=123,
    )

    assert result == {"ok": True}
    assert FakeRequests.calls[0]["seed"] == 123
    assert "seed" not in FakeRequests.calls[1]
