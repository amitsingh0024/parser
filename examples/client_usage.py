#!/usr/bin/env python3
"""Example: Using the Indic OCR Microservice Client in Python."""

import os
import sys

# Add parent directory to path so client package can be imported directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import IndicOCRClient


def main():
    # Initialize client (points to microservice)
    client = IndicOCRClient(base_url="http://localhost:8000")

    # 1. Health check & GPU verification
    print("--- 1. Health Check ---")
    try:
        health = client.health()
        print(f"Status: {health['status']}")
        print(f"GPU: {health.get('gpu_name')} | VRAM: {health.get('vram_used_gb')}GB / {health.get('vram_total_gb')}GB")
    except Exception as e:
        print(f"Error connecting to microservice: {e}")
        return

    # Sample file to test
    sample_file = "outputs/web_results/6e296c57/original_general-hindi-amp-gk-model-papers-8c0010f1.pdf"
    if not os.path.exists(sample_file):
        sample_file = "data/sample.jpg"  # fallback

    if not os.path.exists(sample_file):
        print("No sample file found to test.")
        return

    # 2. Synchronous Parsing (Ideal for 1-2 pages or images)
    print("\n--- 2. Synchronous Parsing (Page 1) ---")
    result = client.parse(sample_file, pages="1", fast_mode=True)
    print(f"Parsed in: {result.latency_sec}s ({result.tokens_per_sec} tok/s)")
    print(f"Markdown preview (first 200 chars):\n{result.markdown[:200]}...")

    # 3. Asynchronous Job Pipeline (Ideal for multi-page documents)
    print("\n--- 3. Asynchronous Job Pipeline (Pages 1-2) ---")
    job = client.submit_job(sample_file, pages="1-2", fast_mode=True)
    print(f"Job submitted! Job ID: {job.job_id}")

    print("Waiting for job completion...")
    job_result = job.wait(poll_interval=2.0)
    print(f"Job completed in: {job_result.latency_sec}s!")
    print(f"Pages parsed: {job_result.pages_parsed}")
    print(f"Tokens: {job_result.tokens}")

    # Save to file
    out_md = "outputs/client_parsed_sample.md"
    job_result.save_markdown(out_md)
    print(f"Saved parsed result to {out_md}")


if __name__ == "__main__":
    main()
