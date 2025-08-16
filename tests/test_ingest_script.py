# tests/test_ingest_script.py
from unittest.mock import patch, MagicMock
import pytest


def test_ingest_script_imports():
    from scripts.ingest import build_agent, run_ingest


@patch("scripts.ingest.build_agent")
def test_run_ingest_single_file(mock_build, tmp_path):
    from scripts.ingest import run_ingest

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    mock_agent = MagicMock()
    mock_report = MagicMock()
    mock_report.entities_count = 5
    mock_report.relations_count = 3
    mock_report.chunks_count = 2
    mock_agent.process_file.return_value = [mock_report]
    mock_build.return_value = mock_agent

    results = run_ingest(str(pdf))

    assert len(results) == 1
    mock_agent.process_file.assert_called_once_with(str(pdf))


@patch("scripts.ingest.build_agent")
def test_run_ingest_directory(mock_build, tmp_path):
    from scripts.ingest import run_ingest

    (tmp_path / "a.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "b.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "c.txt").write_text("not a pdf")

    mock_agent = MagicMock()
    mock_report = MagicMock()
    mock_report.entities_count = 1
    mock_report.relations_count = 0
    mock_report.chunks_count = 1
    mock_agent.process_file.return_value = [mock_report]
    mock_build.return_value = mock_agent

    results = run_ingest(str(tmp_path))

    assert len(results) == 2
    assert mock_agent.process_file.call_count == 2
