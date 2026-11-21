# Changelog

## 0.2.0

- Moved out of the combined `velma-agent-tools` repo into its own repository, with the package at the repo root.
- Seven new tools, so all twelve Velma batch endpoints are covered: `detect_emotion`, `detect_accent`, `detect_language`, `detect_audio_events`, `classify_music_speech`, `transcribe_english_fast` (with `speaker_diarization`, `time_stamps`, `medical_domain`) and `transcribe_multilingual_fast`.
- Added `CONTRIBUTING.md`.

## 0.1.0

First release.

- Five tools: `detect_synthetic_voice`, `detect_ai_music`, `redact_audio`, `transcribe_audio`, `analyze_conversation`.
- Input files limited to `VELMA_MCP_BASE_PATH`. Path escapes and symlink escapes are rejected.
- Redacted audio saved as a file, served as an MCP resource, or both (`VELMA_MCP_OUTPUT_MODE`).
- HTTP errors mapped to short messages. The API key is never included in errors or logs.

Batch endpoints only. Streaming is not covered.
