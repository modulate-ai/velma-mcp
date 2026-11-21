import asyncio

import httpx
import pytest
import respx
from mcp.server.fastmcp.exceptions import ToolError

from velma_mcp import client, server
from conftest import BASE, KEY, MP3, load, multipart_body

SVD = BASE + "/api/velma-2-synthetic-voice-detection-batch"
MUSIC = BASE + "/api/velma-2-ai-music-detection-batch"
REDACT = BASE + "/api/velma-2-pii-phi-redaction-batch"
STT = BASE + "/api/velma-2-stt-batch"
TRIAGE = BASE + "/api/velma-2-batch"


@pytest.fixture
def api():
    with respx.mock(assert_all_called=False) as router:
        yield router


async def test_original_tools_registered():
    names = {t.name for t in await server.mcp.list_tools()}
    assert names >= {
        "detect_synthetic_voice", "detect_ai_music", "redact_audio",
        "transcribe_audio", "analyze_conversation",
    }  # fmt: skip


async def test_synthetic_summary(api, audio):
    route = api.post(SVD).respond(200, json=load("synthetic.json"))
    out = await server.detect_synthetic_voice("clip.wav")
    s = out["summary"]
    assert s["frame_counts"] == {"synthetic": 2, "non-synthetic": 1, "no-content": 1}
    assert s["synthetic_share_of_non_silent_frames"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["mean_confidence_synthetic_frames"] == pytest.approx((0.9732 + 0.8) / 2, abs=1e-3)
    assert s["mean_confidence_non_synthetic_frames"] == 0.6
    assert s["majority_verdict"] == "synthetic"
    assert "overall_label" not in s
    assert out["frames"] == load("synthetic.json")["frames"]
    assert out["duration_ms"] == 34500 and out["filename"] == "recording.wav"
    req = route.calls.last.request
    assert req.headers["x-api-key"] == KEY
    assert b'name="upload_file"; filename="clip.wav"' in req.content


async def test_synthetic_all_silent(api, audio):
    data = {"filename": None, "duration_ms": 1000, "frames": [
        {"start_time_ms": 0, "end_time_ms": 1000, "verdict": "no-content", "confidence": 1.0}]}
    api.post(SVD).respond(200, json=data)
    s = (await server.detect_synthetic_voice("clip.wav"))["summary"]
    assert s["majority_verdict"] is None
    assert s["synthetic_share_of_non_silent_frames"] is None
    assert s["mean_confidence_synthetic_frames"] is None


async def test_music(api, audio):
    api.post(MUSIC).respond(200, json=load("music.json"))
    out = await server.detect_ai_music("clip.wav")
    src = load("music.json")
    assert out["primary_verdict"] == "ai-vocal-music"
    for k in ("duration_s", "latency_ms", "vocal_ai_percentage", "instrumental_ai_confidence"):
        assert out[k] == src[k]
    assert out["windows"] == src["windows"]


async def test_transcribe_sends_only_set_fields(api, audio):
    route = api.post(STT).respond(200, json=load("stt.json"))
    out = await server.transcribe_audio("clip.wav", emotion_signal=True, speaker_diarization=False)
    assert out["text"].startswith("Hello, how are you?")
    assert len(out["utterances"]) == 2 and out["duration_ms"] == 5000
    body = route.calls.last.request.content.decode("latin-1")
    assert 'name="emotion_signal"\r\n\r\ntrue' in body
    assert 'name="speaker_diarization"\r\n\r\nfalse' in body
    for absent in ("accent_signal", "deepfake_signal", "pii_phi_tagging", "language", "config"):
        assert f'name="{absent}"' not in body


async def test_transcribe_rejects_bad_language(api, audio):
    route = api.post(STT).respond(200, json=load("stt.json"))
    with pytest.raises(ToolError, match="language"):
        await server.transcribe_audio("clip.wav", language="english!")
    assert not route.called
    await server.transcribe_audio("clip.wav", language="en-US")
    assert route.called


async def test_analyze_conversation(api, audio):
    route = api.post(TRIAGE).respond(200, json=load("batch.json"))
    out = await server.analyze_conversation("clip.wav")
    assert out["summary"] == "Hello everyone. Welcome to the meeting."
    assert out["topics"] == ["welcome"]
    assert out["conversation_type"] == {"name": "Meeting", "confidence": 0.9}
    assert out["behaviors_count"] == 2 and out["behaviors_detected_count"] == 1
    assert out["raw"] == load("batch.json")
    assert b'name="config"' not in route.calls.last.request.content


async def test_analyze_conversation_config(api, audio):
    route = api.post(TRIAGE).respond(200, json=load("batch.json"))
    await server.analyze_conversation("clip.wav", config="default")
    assert b"\r\n\r\ndefault\r\n" in route.calls.last.request.content
    await server.analyze_conversation("clip.wav", config='{"stt": {}}')
    assert b'{"stt": {}}' in route.calls.last.request.content
    route.reset()
    for bad in ("{not json", "[1]"):
        with pytest.raises(ToolError, match="config"):
            await server.analyze_conversation("clip.wav", config=bad)
    assert not route.called


def redaction_route(api, boundary="velmaBOUNDARY42"):
    body, ctype = multipart_body(load("redaction_metadata.json"), MP3, boundary)
    return api.post(REDACT).respond(200, content=body, headers={"content-type": ctype})


async def test_redact_files_mode_default(api, audio, env):
    route = redaction_route(api)
    out = await server.redact_audio("clip.wav", language="en", speaker_diarization=True)
    meta = load("redaction_metadata.json")
    assert out["text"] == meta["text"] and out["redaction_ranges"] == [[1600, 2400]]
    assert out["utterances"] == meta["utterances"] and out["duration_ms"] == 5000
    assert "redacted_audio_resource" not in out
    saved = (env / "velma-output").glob("clip-redacted-*.mp3")
    [path] = list(saved)
    assert str(path) == out["redacted_audio_path"]
    assert path.read_bytes() == MP3
    body = route.calls.last.request.content.decode("latin-1")
    assert 'name="language"\r\n\r\nen' in body
    assert 'name="speaker_diarization"\r\n\r\ntrue' in body


async def test_redact_never_overwrites(api, audio):
    redaction_route(api)
    a = await server.redact_audio("clip.wav")
    b = await server.redact_audio("clip.wav")
    assert a["redacted_audio_path"] != b["redacted_audio_path"]


async def test_redact_resources_mode(api, audio, env, monkeypatch):
    monkeypatch.setenv("VELMA_MCP_OUTPUT_MODE", "resources")
    redaction_route(api)
    out = await server.redact_audio("clip.wav")
    assert "redacted_audio_path" not in out
    assert not (env / "velma-output").exists()
    uri = out["redacted_audio_resource"]
    assert uri.startswith("velma://redacted/")
    [content] = await server.mcp.read_resource(uri)
    assert content.mime_type == "audio/mpeg"
    assert content.content == MP3
    templates = await server.mcp.list_resource_templates()
    assert [t.uriTemplate for t in templates] == ["velma://redacted/{id}"]


async def test_redact_both_mode_and_resource_bound(api, audio, env, monkeypatch):
    monkeypatch.setenv("VELMA_MCP_OUTPUT_MODE", "both")
    redaction_route(api)
    first = await server.redact_audio("clip.wav")
    assert first["redacted_audio_path"] and first["redacted_audio_resource"]
    for _ in range(server.MAX_RESOURCES):
        await server.redact_audio("clip.wav")
    assert len(server._redacted) == server.MAX_RESOURCES
    with pytest.raises(Exception):
        await server.mcp.read_resource(first["redacted_audio_resource"])


@pytest.mark.parametrize("missing", ["metadata", "audio"])
async def test_redact_missing_part(api, audio, missing):
    body, ctype = multipart_body(load("redaction_metadata.json"), MP3)
    body = body.replace(f'name="{missing}"'.encode(), b'name="other"')
    api.post(REDACT).respond(200, content=body, headers={"content-type": ctype})
    with pytest.raises(ToolError, match=missing):
        await server.redact_audio("clip.wav")


async def test_redact_quoted_boundary_and_case(api, audio):
    body, _ = multipart_body(load("redaction_metadata.json"), MP3, "b-1")
    body = body.replace(b"Content-Disposition", b"content-disposition")
    api.post(REDACT).respond(
        200, content=body, headers={"content-type": 'multipart/form-data; boundary="b-1"'}
    )
    out = await server.redact_audio("clip.wav")
    assert out["text"]


async def test_invalid_output_mode_at_startup(monkeypatch):
    monkeypatch.setenv("VELMA_MCP_OUTPUT_MODE", "nope")
    with pytest.raises(SystemExit):
        server.main()


# --- errors -------------------------------------------------------------------------

E = load("errors.json")
CASES = [
    (400, E["400"], "400"), (401, E["401"], "API key"), (403, E["403"], "403"),
    (413, E["413"], "100 MB"), (422, E["422_string"], "too short"),
    (422, {"detail": E["422_list"]}, "body.upload_file: Field required"),
    (429, E["429_concurrency"], "try again"), (429, E["429_credits"], "Do not retry"),
    (429, E["429_monthly"], "Do not retry"), (500, E["500"], "500"),
    (502, E["502"], "502"), (504, E["504"], "504"),
]  # fmt: skip


@pytest.mark.parametrize("status,detail,expect", CASES)
async def test_error_mapping(api, audio, status, detail, expect):
    payload = detail if isinstance(detail, dict) else {"detail": detail}
    api.post(STT).respond(status, json=payload)
    with pytest.raises(ToolError) as exc:
        await server.transcribe_audio("clip.wav")
    msg = str(exc.value)
    assert f"({status})" in msg and expect in msg


async def test_429_kinds_are_distinguished(api, audio):
    msgs = {}
    for k in ("429_concurrency", "429_credits", "429_monthly"):
        api.post(STT).respond(429, json={"detail": E[k]})
        with pytest.raises(ToolError) as exc:
            await server.transcribe_audio("clip.wav")
        msgs[k] = str(exc.value)
    assert "Do not retry" not in msgs["429_concurrency"]
    assert "Do not retry" in msgs["429_credits"] and "Do not retry" in msgs["429_monthly"]


async def test_non_json_error_body(api, audio):
    api.post(STT).respond(502, text="<html>bad gateway</html>")
    with pytest.raises(ToolError, match="bad gateway"):
        await server.transcribe_audio("clip.wav")


@pytest.mark.parametrize("status", [400, 401, 403, 413, 422, 429, 500, 502, 504])
async def test_key_never_in_errors(api, audio, status):
    api.post(STT).respond(status, json={"detail": f"echo {KEY} here"})
    with pytest.raises(ToolError) as exc:
        await server.transcribe_audio("clip.wav")
    assert KEY not in str(exc.value)
    assert "***" in str(exc.value)


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("t"), httpx.ConnectTimeout("t"), httpx.ConnectError(f"boom {KEY}")]
)
async def test_transport_errors_are_readable(api, audio, error):
    api.post(STT).mock(side_effect=error)
    with pytest.raises(ToolError) as exc:
        await server.transcribe_audio("clip.wav")
    assert KEY not in str(exc.value)
    assert "Velma" in str(exc.value)


async def test_missing_key(api, audio, monkeypatch):
    monkeypatch.delenv("VELMA_API_KEY")
    with pytest.raises(ToolError, match="VELMA_API_KEY is not set"):
        await server.transcribe_audio("clip.wav")


async def test_base_url_override(api, audio, monkeypatch):
    monkeypatch.setenv("VELMA_API_BASE_URL", "https://example.test/")
    route = api.post("https://example.test/api/velma-2-stt-batch").respond(200, json=load("stt.json"))
    await server.transcribe_audio("clip.wav")
    assert route.called


async def test_concurrency_limit_is_three(api, audio):
    state = {"now": 0, "max": 0}

    async def handler(request):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return httpx.Response(200, json=load("stt.json"))

    api.post(STT).mock(side_effect=handler)
    await asyncio.gather(*(server.transcribe_audio("clip.wav") for _ in range(8)))
    assert state["max"] == client.MAX_CONCURRENCY == 3


# --- seven more batch tools ---------------------------------------------------------

EMOTION = BASE + "/api/velma-2-emotion-batch"
ACCENT = BASE + "/api/velma-2-accent-batch"
LANGUAGE = BASE + "/api/velma-2-language-detection-batch"
EVENTS = BASE + "/api/velma-2-audio-event-classifier"
MUSIC_SPEECH = BASE + "/api/velma-2-music-detection-batch"
STT_EN_FAST = BASE + "/api/velma-2-stt-batch-english-vfast"
STT_ML_FAST = BASE + "/api/velma-2-stt-batch-multilingual-vfast"

# (tool name, url, call args, bad extension for that tool)
NEW_TOOLS = [
    ("detect_emotion", EMOTION, {}, "clip.m4a"),
    ("detect_accent", ACCENT, {}, "clip.m4a"),
    ("detect_language", LANGUAGE, {}, "clip.txt"),
    ("detect_audio_events", EVENTS, {}, "clip.m4a"),
    ("classify_music_speech", MUSIC_SPEECH, {}, "clip.txt"),
    ("transcribe_english_fast", STT_EN_FAST, {}, "clip.txt"),
    ("transcribe_multilingual_fast", STT_ML_FAST, {}, "clip.m4a"),
]  # fmt: skip


async def test_twelve_tools_registered():
    names = {t.name for t in await server.mcp.list_tools()}
    assert {n for n, *_ in NEW_TOOLS} <= names
    assert len(names) == 12


async def test_emotion(api, audio):
    route = api.post(EMOTION).respond(200, json=load("emotion.json"))
    out = await server.detect_emotion("clip.wav")
    assert out["emotion"] == "Happy"
    assert out["time_series"] == load("emotion.json")["time_series"]
    assert "not proof" in out["note"]
    assert route.calls.last.request.headers["x-api-key"] == KEY
    assert b'name="upload_file"; filename="clip.wav"' in route.calls.last.request.content


async def test_accent(api, audio):
    api.post(ACCENT).respond(200, json=load("accent.json"))
    out = await server.detect_accent("clip.wav")
    assert out["accent"] == "British"
    assert len(out["time_series"]) == 2


async def test_language(api, audio):
    api.post(LANGUAGE).respond(200, json=load("language.json"))
    out = await server.detect_language("clip.wav")
    assert out["predicted_language"] == "English"
    assert out["predicted_language_code"] == "en"
    assert out["confidence"] == 0.9847
    assert out["duration_ms"] == 14253


async def test_audio_events(api, audio):
    api.post(EVENTS).respond(200, json=load("audio_events.json"))
    out = await server.detect_audio_events("clip.wav")
    assert out["probs"] == {"Applause": 0.998, "Laughter": 0.946}
    assert out["top_event"] == "Applause"
    assert out["duration_ms"] == 2000


async def test_audio_events_none_detected(api, audio):
    api.post(EVENTS).respond(200, json={"probs": {}, "duration_ms": 5400})
    out = await server.detect_audio_events("clip.wav")
    assert out["probs"] == {} and out["top_event"] is None


async def test_classify_music_speech(api, audio):
    api.post(MUSIC_SPEECH).respond(200, json=load("music_speech.json"))
    out = await server.classify_music_speech("clip.wav")
    src = load("music_speech.json")
    assert out["primary_label"] == "speech"
    assert out["speech_pct"] == 86.7 and out["music_pct"] == 0.0
    assert out["frames"] == src["frames"]


async def test_english_fast_sends_only_set_fields(api, audio):
    route = api.post(STT_EN_FAST).respond(200, json=load("stt_english_fast_plain.json"))
    out = await server.transcribe_english_fast("clip.wav", time_stamps=True)
    body = route.calls.last.request.content.decode("latin-1")
    assert 'name="time_stamps"\r\n\r\ntrue' in body
    for absent in ("speaker_diarization", "medical_domain"):
        assert f'name="{absent}"' not in body
    assert out == {"text": "Hello, how are you doing today?", "duration_ms": 2000}


async def test_english_fast_all_options(api, audio):
    route = api.post(STT_EN_FAST).respond(200, json=load("stt_english_fast.json"))
    out = await server.transcribe_english_fast(
        "clip.wav", speaker_diarization=True, time_stamps=True, medical_domain=False
    )
    body = route.calls.last.request.content.decode("latin-1")
    assert 'name="speaker_diarization"\r\n\r\ntrue' in body
    assert 'name="medical_domain"\r\n\r\nfalse' in body
    src = load("stt_english_fast.json")
    assert out["words"] == src["words"] and out["utterances"] == src["utterances"]
    assert "words_unavailable" not in out


async def test_multilingual_fast(api, audio):
    route = api.post(STT_ML_FAST).respond(200, json=load("stt_multilingual_fast.json"))
    out = await server.transcribe_multilingual_fast("clip.wav", language="es")
    assert out == {
        "text": "Hola a todos. Bienvenidos a la junta semanal.",
        "duration_ms": 4200,
        "language": "es",
    }
    assert 'name="language"\r\n\r\nes' in route.calls.last.request.content.decode("latin-1")


async def test_multilingual_fast_no_language_field_by_default(api, audio):
    route = api.post(STT_ML_FAST).respond(200, json=load("stt_multilingual_fast.json"))
    await server.transcribe_multilingual_fast("clip.wav")
    assert b'name="language"' not in route.calls.last.request.content


async def test_multilingual_fast_rejects_bad_language(api, audio):
    route = api.post(STT_ML_FAST).respond(200, json=load("stt_multilingual_fast.json"))
    with pytest.raises(ToolError, match="Invalid language"):
        await server.transcribe_multilingual_fast("clip.wav", language="not a code!")
    assert not route.called


@pytest.mark.parametrize("name,url,kwargs,bad", NEW_TOOLS)
async def test_new_tools_reject_bad_extension(api, env, name, url, kwargs, bad):
    (env / bad).write_bytes(b"data")
    route = api.post(url).respond(200, json={})
    with pytest.raises(ToolError, match="Unsupported file extension"):
        await getattr(server, name)(bad, **kwargs)
    assert not route.called


@pytest.mark.parametrize("name,url,kwargs,bad", NEW_TOOLS)
@pytest.mark.parametrize("status", [400, 401, 403, 413, 422, 429, 500, 502, 504])
async def test_new_tools_never_leak_key(api, audio, name, url, kwargs, bad, status):
    api.post(url).respond(status, json={"detail": f"echo {KEY} here"})
    with pytest.raises(ToolError) as exc:
        await getattr(server, name)("clip.wav", **kwargs)
    assert KEY not in str(exc.value)


@pytest.mark.parametrize("name,url,kwargs,bad", NEW_TOOLS)
async def test_new_tools_transport_error_hides_key(api, audio, name, url, kwargs, bad):
    api.post(url).mock(side_effect=httpx.ConnectError(f"boom {KEY}"))
    with pytest.raises(ToolError) as exc:
        await getattr(server, name)("clip.wav", **kwargs)
    assert KEY not in str(exc.value)
