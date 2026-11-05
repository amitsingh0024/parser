import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ParseResult:
    job_id: str
    filename: str
    pages_parsed: int
    selected_pages: List[int]
    latency_sec: float
    tokens: int
    tokens_per_sec: float
    markdown: str
    raw_layout: str
    annotated_image_url: Optional[str]
    annotated_image_urls: List[str]

    def save_markdown(self, output_path: str):
        """Save parsed markdown directly to a local file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.markdown)


class JobHandle:
    """Handle to a submitted asynchronous parsing job."""

    def __init__(self, client: "IndicOCRClient", job_id: str):
        self.client = client
        self.job_id = job_id
        self._last_state: Optional[Dict[str, Any]] = None

    def status(self) -> Dict[str, Any]:
        """Fetch the current job status and progress."""
        self._last_state = self.client.get_job_status(self.job_id)
        return self._last_state

    def wait(self, timeout: float = 600.0, poll_interval: float = 2.0) -> ParseResult:
        """Wait until the job completes and return the ParseResult."""
        t_start = time.time()
        while time.time() - t_start < timeout:
            st = self.status()
            current_status = st.get("status")

            if current_status == "completed":
                data = st.get("result", {})
                return ParseResult(
                    job_id=self.job_id,
                    filename=data.get("filename", ""),
                    pages_parsed=data.get("pages_parsed", 0),
                    selected_pages=data.get("selected_pages", []),
                    latency_sec=data.get("latency_sec", 0.0),
                    tokens=data.get("tokens", 0),
                    tokens_per_sec=data.get("tokens_per_sec", 0.0),
                    markdown=data.get("markdown", ""),
                    raw_layout=data.get("raw_layout", ""),
                    annotated_image_url=data.get("annotated_image_url"),
                    annotated_image_urls=data.get("annotated_image_urls", []),
                )
            elif current_status == "failed":
                err = st.get("error", "Unknown error")
                raise RuntimeError(f"OCR Job {self.job_id} failed: {err}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {self.job_id} did not complete within {timeout}s.")


class IndicOCRClient:
    """Lightweight Python SDK for Baidu Unlimited-OCR-2 Indic Microservice."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = {"User-Agent": "IndicOCR-Client/1.0"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def health(self) -> Dict[str, Any]:
        """Check microservice health, GPU status, and VRAM utilization."""
        req = urllib.request.Request(f"{self.base_url}/healthz", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def models(self) -> List[Dict[str, Any]]:
        """List available model checkpoints."""
        req = urllib.request.Request(f"{self.base_url}/v1/models", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8")).get("checkpoints", [])

    def parse(
        self,
        file_path: str,
        pages: str = "1",
        fast_mode: bool = True,
        max_tokens: int = 2048,
        no_repeat_ngram_size: int = 15,
        temperature: float = 0.0,
        checkpoint: str = "latest",
    ) -> ParseResult:
        """Synchronously parse a document (image or PDF) and block until completed."""
        fields = {
            "pages": pages,
            "fast_mode": "true" if fast_mode else "false",
            "max_tokens": str(max_tokens),
            "no_repeat_ngram_size": str(no_repeat_ngram_size),
            "temperature": str(temperature),
            "checkpoint": checkpoint,
        }
        body, content_type = self._encode_multipart(fields, file_path)
        req = urllib.request.Request(
            f"{self.base_url}/v1/parse",
            data=body,
            headers=self._headers(content_type),
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return ParseResult(
                job_id=data.get("request_id", ""),
                filename=data.get("filename", ""),
                pages_parsed=data.get("pages_parsed", 0),
                selected_pages=data.get("selected_pages", []),
                latency_sec=data.get("latency_sec", 0.0),
                tokens=data.get("tokens", 0),
                tokens_per_sec=data.get("tokens_per_sec", 0.0),
                markdown=data.get("markdown", ""),
                raw_layout=data.get("raw_layout", ""),
                annotated_image_url=data.get("annotated_image_url"),
                annotated_image_urls=data.get("annotated_image_urls", []),
            )

    def submit_job(
        self,
        file_path: str,
        pages: str = "1",
        fast_mode: bool = True,
        max_tokens: int = 2048,
        no_repeat_ngram_size: int = 15,
        temperature: float = 0.0,
        checkpoint: str = "latest",
        callback_url: Optional[str] = None,
    ) -> JobHandle:
        """Submit an asynchronous job to the task queue and return a JobHandle."""
        fields = {
            "pages": pages,
            "fast_mode": "true" if fast_mode else "false",
            "max_tokens": str(max_tokens),
            "no_repeat_ngram_size": str(no_repeat_ngram_size),
            "temperature": str(temperature),
            "checkpoint": checkpoint,
        }
        if callback_url:
            fields["callback_url"] = callback_url

        body, content_type = self._encode_multipart(fields, file_path)
        req = urllib.request.Request(
            f"{self.base_url}/v1/jobs",
            data=body,
            headers=self._headers(content_type),
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return JobHandle(self, data["job_id"])

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Fetch raw job metadata and progress."""
        req = urllib.request.Request(f"{self.base_url}/v1/jobs/{job_id}", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recently submitted jobs."""
        req = urllib.request.Request(f"{self.base_url}/v1/jobs?limit={limit}", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8")).get("jobs", [])

    def _encode_multipart(self, fields: Dict[str, str], file_path: str):
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()
        for k, v in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            body.extend(f"{v}\r\n".encode("utf-8"))

        fname = os.path.basename(file_path)
        ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"))
        with open(file_path, "rb") as f:
            body.extend(f.read())
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        return bytes(body), f"multipart/form-data; boundary={boundary}"
