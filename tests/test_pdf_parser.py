# tests/test_pdf_parser.py
from unittest.mock import patch, MagicMock
import json
import pytest
from medmdt.config.settings import Settings
from medmdt.extractor.parsers.pdf_parser import PaddleOCRClient
from medmdt.extractor.schemas import ParsedPage


@pytest.fixture
def settings():
    return Settings(paddleocr_token="test-token")


@pytest.fixture
def client(settings):
    return PaddleOCRClient(settings)


def _mock_submit_response(job_id="job-123"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"jobId": job_id}}
    return resp


def _mock_done_response(result_url="https://example.com/result.jsonl"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {
            "state": "done",
            "extractProgress": {
                "extractedPages": 1,
                "startTime": "2026-01-01T00:00:00",
                "endTime": "2026-01-01T00:01:00",
            },
            "resultUrl": {"jsonUrl": result_url},
        }
    }
    return resp


def _mock_jsonl_response():
    page_result = {
        "result": {
            "layoutParsingResults": [
                {
                    "markdown": {
                        "text": "# 糖尿病诊疗指南\n\n二甲双胍是一线用药。",
                        "images": {},
                    },
                    "outputImages": {},
                }
            ]
        }
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps(page_result)
    resp.raise_for_status = MagicMock()
    return resp


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_local_file(mock_requests, client, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")

    mock_requests.post.return_value = _mock_submit_response()
    mock_requests.get.side_effect = [_mock_done_response(), _mock_jsonl_response()]

    pages = client.parse(str(pdf_file))

    assert len(pages) == 1
    assert isinstance(pages[0], ParsedPage)
    assert "糖尿病" in pages[0].markdown
    assert pages[0].page_num == 0


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_url(mock_requests, client):
    mock_requests.post.return_value = _mock_submit_response()
    mock_requests.get.side_effect = [_mock_done_response(), _mock_jsonl_response()]

    pages = client.parse("https://example.com/doc.pdf")

    assert len(pages) == 1
    post_kwargs = mock_requests.post.call_args
    assert "json" in post_kwargs.kwargs or (len(post_kwargs.args) > 1)


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_handles_failed_job(mock_requests, client, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    mock_requests.post.return_value = _mock_submit_response()
    failed_resp = MagicMock()
    failed_resp.status_code = 200
    failed_resp.json.return_value = {
        "data": {"state": "failed", "errorMsg": "Unsupported format"}
    }
    mock_requests.get.return_value = failed_resp

    with pytest.raises(RuntimeError, match="Unsupported format"):
        client.parse(str(pdf_file))


@patch("medmdt.extractor.parsers.pdf_parser.requests")
def test_parse_with_embedded_images(mock_requests, client, tmp_path):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    mock_requests.post.return_value = _mock_submit_response()

    page_result = {
        "result": {
            "layoutParsingResults": [
                {
                    "markdown": {
                        "text": "# Page with image\n\n![xray](img_0.png)",
                        "images": {"img_0.png": "https://example.com/img_0.png"},
                    },
                    "outputImages": {},
                }
            ]
        }
    }
    jsonl_resp = MagicMock()
    jsonl_resp.status_code = 200
    jsonl_resp.text = json.dumps(page_result)
    jsonl_resp.raise_for_status = MagicMock()

    img_resp = MagicMock()
    img_resp.status_code = 200
    img_resp.content = b"fake-png-bytes"

    mock_requests.get.side_effect = [_mock_done_response(), jsonl_resp, img_resp]

    pages = client.parse(str(pdf_file))

    assert len(pages) == 1
    assert len(pages[0].images) == 1
    assert pages[0].images[0] == b"fake-png-bytes"
