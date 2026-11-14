import asyncio
import datetime
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional


class JobManager:
    """Manages asynchronous OCR parsing jobs and batch workflows with persistent state and background GPU queuing."""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.batches_dir = os.path.join(storage_dir, "batches")
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.batches_dir, exist_ok=True)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.batches: Dict[str, Dict[str, Any]] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._load_existing_jobs()
        self._load_existing_batches()

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

    def _load_existing_batches(self):
        try:
            for fname in os.listdir(self.batches_dir):
                if fname.endswith(".json"):
                    fpath = os.path.join(self.batches_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.batches[data["batch_id"]] = data
        except Exception as e:
            print(f"[WARNING] Could not load existing batches from disk: {e}")

    def _save_job_to_disk(self, job_id: str):
        try:
            job_data = self.jobs.get(job_id)
            if job_data:
                fpath = os.path.join(self.storage_dir, f"{job_id}.json")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(job_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to persist job {job_id} to disk: {e}")

    def _save_batch_to_disk(self, batch_id: str):
        try:
            batch_data = self.batches.get(batch_id)
            if batch_data:
                fpath = os.path.join(self.batches_dir, f"{batch_id}.json")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(batch_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to persist batch {batch_id} to disk: {e}")

    def create_job(
        self,
        job_id: str,
        filename: str,
        input_file_path: str,
        pages: str = "1",
        fast_mode: bool = True,
        max_tokens: int = 1024,
        no_repeat_ngram_size: int = 7,
        temperature: float = 0.0,
        checkpoint: str = "latest",
        script: str = "auto",
        callback_url: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        job = {
            "job_id": job_id,
            "batch_id": batch_id,
            "status": "queued",
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "started_at": None,
            "completed_at": None,
            "progress": {
                "pages_done": 0,
                "pages_total": 0,
                "current_page": 0,
                "total_pages": 0,
                "percentage": 0.0,
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
                "script": script,
                "callback_url": callback_url,
            },
            "result": None,
            "error": None,
        }
        self.jobs[job_id] = job
        self._save_job_to_disk(job_id)

        if batch_id and batch_id in self.batches:
            self.batches[batch_id]["job_ids"].append(job_id)
            self._save_batch_to_disk(batch_id)

        self.queue.put_nowait(job_id)
        return job

    def create_batch(self, batch_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        batch = {
            "batch_id": batch_id,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "job_ids": [],
            "metadata": metadata or {},
        }
        self.batches[batch_id] = batch
        self._save_batch_to_disk(batch_id)
        return batch

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        batch = self.batches.get(batch_id)
        if not batch:
            return None
        # Aggregate current statuses
        jobs_status = []
        completed_count = 0
        failed_count = 0
        queued_count = 0
        processing_count = 0

        for j_id in batch.get("job_ids", []):
            j = self.jobs.get(j_id)
            if j:
                st = j.get("status", "unknown")
                if st == "completed":
                    completed_count += 1
                elif st == "failed":
                    failed_count += 1
                elif st == "processing":
                    processing_count += 1
                elif st == "queued":
                    queued_count += 1
                jobs_status.append({
                    "job_id": j_id,
                    "filename": j.get("parameters", {}).get("filename"),
                    "status": st,
                    "progress": j.get("progress"),
                    "error": j.get("error"),
                    "result_summary": {
                        "tokens": j["result"].get("tokens") if j.get("result") else None,
                        "latency_sec": j["result"].get("latency_sec") if j.get("result") else None,
                        "degenerated": j["result"].get("degenerated", False) if j.get("result") else False,
                        "confidence": j["result"].get("confidence") if j.get("result") else None,
                    } if j.get("result") else None,
                })

        total = len(batch.get("job_ids", []))
        is_done = total > 0 and (completed_count + failed_count == total)

        return {
            "batch_id": batch_id,
            "created_at": batch.get("created_at"),
            "total_jobs": total,
            "completed": completed_count,
            "failed": failed_count,
            "processing": processing_count,
            "queued": queued_count,
            "is_completed": is_done,
            "jobs": jobs_status,
        }

    def update_progress(self, job_id: str, current_page: int, total_pages: int):
        if job_id in self.jobs:
            pct = round((current_page / max(1, total_pages)) * 100, 1)
            self.jobs[job_id]["progress"] = {
                "pages_done": current_page,
                "pages_total": total_pages,
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
