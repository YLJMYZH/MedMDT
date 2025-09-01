# Baseline Test Alignment Design

## Goal

Eliminate the three accepted backend baseline failures by aligning stale tests with the current streaming-consultation and runtime-settings architecture, without restoring removed production configuration fields or changing production behavior.

## Root Causes

### Consultation event test

`test_run_consultation_task_publishes_events` still patches the removed graph execution boundary (`build_mdt_graph` and `run_consultation`). Since `run_consultation_task` now constructs `StreamingOrchestrator` directly, the stale mocks do not intercept execution. A `MagicMock` LLM then reaches expert selection and produces the observed JSON type error.

### Static settings tests

`test_settings_loads_defaults` still expects `default_llm_provider`, `embedding_model`, and `embedding_dim` on the infrastructure `Settings` model. These fields deliberately moved to the runtime settings model and JSON-backed settings workflow.

`test_settings_requires_paddleocr_token` still expects process startup to require a PaddleOCR token. The token is now optional at startup and is configured through runtime settings, so the old assertion contradicts the current product behavior.

## Chosen Approach

Use a test-only migration. Do not add compatibility aliases to `Settings`, restore startup validation, or refactor production dependency injection.

### Consultation test boundary

Patch the actual `StreamingOrchestrator` class used by `run_consultation_task`. Configure its instance `run()` method to return the successful consultation result. Keep the existing assertions for the `running` and completed `done` events, and keep the separate infrastructure-failure assertions for `running`, `error`, and failed `done` events. Also assert the stored consultation status for both paths so the test covers the state transition as well as event publication.

### Settings test boundaries

Keep `Settings` tests focused on infrastructure defaults such as Neo4j, Milvus, Elasticsearch, PaddleOCR endpoint options, and MDT limits. Remove assertions for fields owned by runtime settings.

Replace the obsolete required-token test with an assertion that `Settings()` succeeds without a PaddleOCR token. Runtime token behavior remains covered by runtime/settings API tests, where the field is actually owned.

Make the cached-settings singleton test deterministic by clearing `get_settings` before and after the assertion instead of mutating an environment variable that the static settings model no longer consumes.

## Scope

Expected modified files:

- `backend/tests/test_api_consultation.py`
- `backend/tests/test_config.py`

No production source file should change. No legacy field or validation behavior should be restored.

## Verification

1. Re-run the three currently failing tests and require all three to pass.
2. Run the consultation and configuration test modules together.
3. Run the complete backend test suite with provider and network opt-in environment variables unset. Provider contract tests must remain skipped rather than contacting real services.
4. Run Python `compileall` and `git diff --check`.
5. Confirm the worktree is clean after committing the test migration.

## Success Criteria

- The full backend suite has no failures.
- The six opt-in real-provider contract cases remain skipped when credentials are absent.
- Consultation success and failure event contracts remain explicitly tested.
- Static settings tests describe only fields owned by `Settings`.
- No production behavior or public API changes.
