# Baseline Test Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three stale backend test failures by updating their test boundaries and expectations to the current architecture without changing production behavior.

**Architecture:** Treat the existing failures as the RED state for a test-suite migration. Patch the current `StreamingOrchestrator` boundary in the consultation test, and restrict static `Settings` tests to fields still owned by that model. Production modules remain untouched.

**Tech Stack:** Python 3.12, pytest, `unittest.mock`, FastAPI test utilities, Pydantic Settings, uv.

## Global Constraints

- Modify tests only; do not edit files under `backend/src`.
- Do not restore `default_llm_provider`, `default_llm_model`, `embedding_model`, `embedding_dim`, or required startup PaddleOCR credentials.
- Preserve both success and failure event-order assertions for background consultations.
- Explicitly unset provider credentials and private-network opt-in variables during the full suite.
- Do not call real model providers or external services.

---

### Task 1: Align consultation event testing with StreamingOrchestrator

**Files:**
- Modify: `backend/tests/test_api_consultation.py:89-148`
- Test: `backend/tests/test_api_consultation.py`

**Interfaces:**
- Consumes: `run_consultation_task(consultation_id: str, store: ConsultationStore) -> None`, `StreamingOrchestrator.run(...) -> MDTState | dict`, and `EventBus` queue events.
- Produces: A deterministic test of consultation status transitions and terminal WebSocket events at the current orchestration boundary.

- [ ] **Step 1: Verify the current RED state**

Run:

```bash
cd backend
uv run pytest -q tests/test_api_consultation.py::test_run_consultation_task_publishes_events -vv
```

Expected: FAIL because stale graph mocks do not intercept `StreamingOrchestrator`, allowing a `MagicMock` LLM response into `json.loads()`.

- [ ] **Step 2: Replace obsolete graph mocks with the current orchestration boundary**

In `test_run_consultation_task_publishes_events`, replace the `build_mdt_graph` and `run_consultation` patches with:

```python
patch("medmdt.mdt.streaming.StreamingOrchestrator") as mock_orchestrator
```

Configure the actual mocked call boundary:

```python
mock_orchestrator.return_value.run.return_value = {
    "final_report": "report",
    "discussion_rounds": [],
    "consensus": {},
    "divergences": [],
}
```

Remove the unused `mock_run.return_value` assignment. Retain the existing infrastructure, expert, and moderator patches so no real dependencies are built.

- [ ] **Step 3: Assert persisted state as well as event order**

After the success event assertions, add:

```python
assert store.get(cid)["status"] == ConsultationStatus.COMPLETED
```

After the failure event assertions, add:

```python
assert store.get(cid2)["status"] == ConsultationStatus.FAILED
```

- [ ] **Step 4: Verify GREEN for the consultation module**

Run:

```bash
cd backend
uv run pytest -q tests/test_api_consultation.py -vv
```

Expected: all consultation API tests PASS; the success case emits only `running` then completed `done`, and the infrastructure failure emits `running`, `error`, then failed `done`.

- [ ] **Step 5: Commit the consultation test migration**

```bash
git add backend/tests/test_api_consultation.py
git commit -m "test: align consultation event mocks"
```

---

### Task 2: Align static Settings tests with current ownership

**Files:**
- Modify: `backend/tests/test_config.py:1-30`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: `Settings()` for infrastructure configuration and the cached `get_settings() -> Settings` accessor.
- Produces: Tests that document static infrastructure defaults, optional runtime-managed PaddleOCR credentials, and deterministic cache behavior.

- [ ] **Step 1: Verify the two current RED cases**

Run:

```bash
cd backend
uv run pytest -q \
  tests/test_config.py::test_settings_loads_defaults \
  tests/test_config.py::test_settings_requires_paddleocr_token -vv
```

Expected: two failures: removed `default_llm_provider` is absent, and `Settings()` correctly does not raise without a PaddleOCR token.

- [ ] **Step 2: Restrict the defaults test to fields owned by Settings**

Replace the test body with:

```python
def test_settings_loads_defaults():
    settings = Settings()

    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.milvus_host == "localhost"
    assert settings.milvus_port == 19530
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.paddleocr_api_url == (
        "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    )
    assert settings.mdt_max_rounds == 3
    assert settings.mdt_consensus_threshold == 0.8
```

Do not assert runtime-owned LLM or embedding fields on `Settings`.

- [ ] **Step 3: Replace obsolete token-required behavior with optional startup behavior**

Replace `test_settings_requires_paddleocr_token` with:

```python
def test_settings_starts_without_paddleocr_token():
    settings = Settings()

    assert isinstance(settings, Settings)
```

This records that static application settings can be created before a PaddleOCR token is configured through runtime settings.

- [ ] **Step 4: Remove environment side effects from the singleton test**

Remove unused `os` and `pytest` imports. Replace the singleton test with:

```python
def test_get_settings_returns_singleton():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
```

- [ ] **Step 5: Verify GREEN for configuration tests**

Run:

```bash
cd backend
uv run pytest -q tests/test_config.py -vv
```

Expected: all configuration tests PASS with no environment leakage.

- [ ] **Step 6: Commit the configuration test migration**

```bash
git add backend/tests/test_config.py
git commit -m "test: align static settings expectations"
```

---

### Task 3: Verify the complete backend suite

**Files:**
- Verify only: `backend/tests/`
- Verify only: `backend/src/`

**Interfaces:**
- Consumes: the two test-only commits from Tasks 1 and 2.
- Produces: evidence that all backend tests pass offline and no production file changed.

- [ ] **Step 1: Re-run the three formerly failing tests together**

Run:

```bash
cd backend
uv run pytest -q \
  tests/test_api_consultation.py::test_run_consultation_task_publishes_events \
  tests/test_config.py::test_settings_loads_defaults \
  tests/test_config.py::test_settings_starts_without_paddleocr_token -vv
```

Expected: `3 passed`.

- [ ] **Step 2: Run the full backend suite offline**

Run:

```bash
cd backend
env \
  -u OPENAI_API_KEY -u OPENAI_VISION_MODEL \
  -u ANTHROPIC_API_KEY -u ANTHROPIC_VISION_MODEL \
  -u DASHSCOPE_API_KEY -u QWEN_VISION_MODEL \
  -u ZHIPUAI_API_KEY -u ZHIPU_VISION_MODEL \
  -u MOONSHOT_API_KEY -u MOONSHOT_VISION_MODEL \
  -u CUSTOM_VISION_API_KEY -u CUSTOM_VISION_MODEL \
  -u CUSTOM_VISION_BASE_URL \
  -u MEDMDT_ALLOW_PRIVATE_BASE_URLS \
  -u MEDMDT_ALLOWED_BASE_URL_HOSTS \
  -u MEDMDT_ALLOWED_BASE_URL_CIDRS \
  uv run pytest -q
```

Expected: all non-contract backend tests PASS and six opt-in provider contract tests SKIP.

- [ ] **Step 3: Verify syntax, formatting, and scope**

Run:

```bash
cd backend
uv run python -m compileall -q src scripts tests
cd ..
git diff --check
git status --short
git diff --name-only d7ceb3f..HEAD
```

Expected: compile and diff checks succeed. Since `d7ceb3f`, changed implementation artifacts are limited to the approved design/plan documents and the two backend test files; no `backend/src` file appears.

- [ ] **Step 4: Record final verification without adding production changes**

If verification exposes a new failure, stop and return to systematic root-cause investigation. Otherwise, report the exact pass/skip counts and current commits; do not push without explicit user authorization.
