from __future__ import annotations

import json

import pytest

from swa.speech_out_bridge import SpeechOutputBridge, get_tts_endpoint


def test_tts_endpoint_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWA_TTS_HOST", raising=False)
    monkeypatch.delenv("SWA_TTS_PORT", raising=False)

    assert get_tts_endpoint() == ("127.0.0.1", 9100)


def test_tts_endpoint_can_target_lan_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWA_TTS_HOST", "192.168.1.30")
    monkeypatch.setenv("SWA_TTS_PORT", "9191")

    assert get_tts_endpoint() == ("192.168.1.30", 9191)


def test_tts_port_must_be_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWA_TTS_PORT", "70000")

    with pytest.raises(ValueError, match="SWA_TTS_PORT"):
        get_tts_endpoint()


def test_speech_bridge_sends_to_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[bytes, tuple[str, int]]] = []

    class FakeSocket:
        def setsockopt(self, *_args: object) -> None:
            pass

        def sendto(self, payload: bytes, endpoint: tuple[str, int]) -> None:
            sent.append((payload, endpoint))

    monkeypatch.setenv("SWA_TTS_HOST", "192.168.1.30")
    monkeypatch.setenv("SWA_TTS_PORT", "9191")
    monkeypatch.setattr("swa.speech_out_bridge.socket.socket", lambda *_args: FakeSocket())

    SpeechOutputBridge().broadcast("hello")

    payload, endpoint = sent[0]
    assert endpoint == ("192.168.1.30", 9191)
    assert json.loads(payload)["token"] == "hello"
