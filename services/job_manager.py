import asyncio
import datetime
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional


class JobManager:
    """Manages asynchronous OCR parsing jobs with persistent state and background GPU queuing."""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._load_existing_jobs()

    def _load_existing_jobs(self):
        try:
            for fname in os.listdir(self.storage_dir):
                if fname.endswith(".json"):
                    fpath = os.path.join(self.storage_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.jobs[data["job_id"]] = data
        except Exception as e:
            print(f"[WARNING] Could not load existing jobs from disk: {e}")

    def _save_job_to_disk(self, job_id: str):
        try:
            job_data = self.jobs.get(job_id)
            if job_data:
                fpath = os.path.join(self.storage_dir, f"{job_id}.json")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(job_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to persist job {job_id} to disk: {e}")

    def create_job(
        self,
        job_id: str,
        filename: str,
        input_file_path: str,
        pages: str = "1",
        fast_mode: bool = True,
        max_tokens: int = 2048,
        no_repeat_ngram_size: int = 15,
        temperature: float = 0.0,
        checkpoint: str = "latest",
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "started_at": None,
            "completed_at": None,
            "progress": {
                "current_page": 0,
                "total_pages": 0,
                "percentage": 0,
            },
            "parameters": {
                "filename": filename,
                "input_file_path": input_file_path,
                "pages": pages,
                "fast_mode": fast_mode,
                "max_tokens": max_tokens,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "temperature": temperature,
                "checkpoint": checkpoint,
                "callback_url": callback_url,
            },
            "result": None,
            "error": None,
        }
        self.jobs[job_id] = job
        self._save_job_to_disk(job_id)
        self.queue.put_nowait(job_id)
        return job

    def update_progress(self, job_id: str, current_page: int, total_pages: int):
        if job_id in self.jobs:
            pct = round((current_page / max(1, total_pages)) * 100, 1)
            self.jobs[job_id]["progress"] = {
                "current_page": current_page,
                "total_pages": total_pages,
                "percentage": pct,
            }
            self.jobs[job_id]["status"] = "processing"
            self._save_job_to_disk(job_id)

    def mark_completed(self, job_id: str, result: Dict[str, Any]):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["progress"]["percentage"] = 100.0
            self._save_job_to_disk(job_id)
            self._dispatch_webhook(job_id)

    def mark_failed(self, job_id: str, error_msg: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            self.jobs[job_id]["error"] = error_msg
            self._save_job_to_disk(job_id)
            self._dispatch_webhook(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        sorted_jobs = sorted(
            self.jobs.values(),
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
        return sorted_jobs[:limit]

    def _dispatch_webhook(self, job_id: str):
        callback_url = self.jobs[job_id]["parameters"].get("callback_url")
        if not callback_url:
            return

        def _send():
            try:
                payload = json.dumps(self.jobs[job_id], ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    callback_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "Baidu-OCR-Microservice/1.0"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    print(f"[INFO] Webhook delivered to {callback_url} (HTTP {resp.status})")
            except Exception as e:
                print(f"[WARNING] Failed to deliver webhook for job {job_id} to {callback_url}: {e}")

        asyncio.get_event_loop().run_in_executor(None, _send)
