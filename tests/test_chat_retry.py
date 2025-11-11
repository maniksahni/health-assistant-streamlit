import types

from chat.client import chat_completion


class DummyStream:
    def __iter__(self):
        return iter([
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": ", world!"}}]},
        ])


def test_chat_completion_streams(monkeypatch):
    # Patch openai.ChatCompletion.create to return streaming first
    import openai

    def fake_create(*args, **kwargs):
        if kwargs.get("stream"):
            return DummyStream()
        class R:
            choices = [types.SimpleNamespace(message={"content": "fallback"})]
        return R()

    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)
    out = chat_completion([{"role": "user", "content": "hi"}], retries=1)
    assert "Hello, world!" in out


def test_chat_completion_retries(monkeypatch):
    # First call raises, second returns non-stream response
    import openai

    calls = {"n": 0}

    def fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        class R:
            choices = [types.SimpleNamespace(message={"content": "ok"})]
        return R()

    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)
    out = chat_completion([{"role": "user", "content": "hi"}], retries=2, request_timeout=1)
    assert out == "ok"
