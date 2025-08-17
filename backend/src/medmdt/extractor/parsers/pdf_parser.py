# src/medmdt/extractor/parsers/pdf_parser.py
import json
import time
import requests
from pathlib import Path

from medmdt.config.settings import Settings
from medmdt.extractor.schemas import ParsedPage


class PaddleOCRClient:
    def __init__(self, settings: Settings):
        self._api_url = settings.paddleocr_api_url
        self._headers = {"Authorization": f"bearer {settings.paddleocr_token}"}
        self._optional_payload = {
            "useDocOrientationClassify": settings.paddleocr_use_doc_orientation_classify,
            "useDocUnwarping": settings.paddleocr_use_doc_unwarping,
            "useChartRecognition": settings.paddleocr_use_chart_recognition,
        }

    def parse(self, file_path: str, poll_interval: float = 5.0) -> list[ParsedPage]:
        job_id = self._submit(file_path)
        jsonl_url = self._poll_until_done(job_id, poll_interval)
        return self._fetch_results(jsonl_url)

    def _submit(self, file_path: str) -> str:
        if file_path.startswith("http"):
            headers = {**self._headers, "Content-Type": "application/json"}
            payload = {
                "fileUrl": file_path,
                "model": "PaddleOCR-VL-1.6",
                "optionalPayload": self._optional_payload,
            }
            resp = requests.post(self._api_url, json=payload, headers=headers)
        else:
            data = {
                "model": "PaddleOCR-VL-1.6",
                "optionalPayload": json.dumps(self._optional_payload),
            }
            with open(file_path, "rb") as f:
                resp = requests.post(
                    self._api_url, headers=self._headers, data=data, files={"file": f},
                )

        resp.raise_for_status()
        return resp.json()["data"]["jobId"]

    def _poll_until_done(self, job_id: str, poll_interval: float) -> str:
        while True:
            resp = requests.get(f"{self._api_url}/{job_id}", headers=self._headers)
            resp.raise_for_status()
            data = resp.json()["data"]
            state = data["state"]

            if state == "done":
                return data["resultUrl"]["jsonUrl"]
            elif state == "failed":
                raise RuntimeError(f"PaddleOCR job failed: {data.get('errorMsg', 'unknown')}")

            time.sleep(poll_interval)

    def _fetch_results(self, jsonl_url: str) -> list[ParsedPage]:
        resp = requests.get(jsonl_url)
        resp.raise_for_status()

        pages: list[ParsedPage] = []
        page_num = 0

        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)["result"]

            for layout in result["layoutParsingResults"]:
                md_text = layout["markdown"]["text"]
                md_images = layout["markdown"].get("images", {})

                image_bytes_list: list[bytes] = []
                for _img_name, img_url in md_images.items():
                    img_resp = requests.get(img_url)
                    if img_resp.status_code == 200:
                        image_bytes_list.append(img_resp.content)

                pages.append(ParsedPage(
                    page_num=page_num,
                    markdown=md_text,
                    images=image_bytes_list,
                ))
                page_num += 1

        return pages
