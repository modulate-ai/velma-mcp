# Contributing

Fixes and new tools are welcome.

## Set up

```bash
uv sync
uv run pytest        # live tests are skipped without VELMA_API_KEY
```

Live tests run against the real API and spend credits, so they only run when `VELMA_API_KEY` is set.

## Add a tool

1. Check the endpoint in the public docs at https://docs.modulate.ai. Copy field names, accepted file extensions and limits from its OpenAPI spec. Do not guess them.
2. Add the tool to `src/velma_mcp/server.py` next to the others. Use `resolve_input` for the file path, `post_upload` for the request and `json_body` for the reply, so the folder sandbox, error mapping and key scrubbing apply.
3. Write the docstring for an agent reader: what it returns, accepted formats, the 100 MB limit, and that the audio is uploaded to Modulate.
4. Add a mocked test in `tests/test_tools.py` with realistic response data in `tests/data/`, a test for a rejected extension, and a live test in `tests/test_live.py`.
5. Add the tool to the table in `README.md` and a line to `CHANGELOG.md`.

## Rules

- Never log or return the API key, and never read files outside `VELMA_MCP_BASE_PATH`.
- Detection results are probability signals, not proof. Say so in the tool output.
- Do not add automatic retries on 429. Credit and monthly limits must not be retried.
- Test clips must be recorded by you or licensed for redistribution. Record the source in `fixtures/SOURCES.md`.
