"""
Generate story embeddings using Google Gemini Batch Embedding API.
Outputs: outputs/gemini_embeddings.npy + metadata JSON.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from config import Config


def main():
    config = Config
    config.validate()

    # Load API key
    env_path = config.BASE_DIR / ".env"
    if not env_path.exists():
        print(f".env file not found at {env_path}")
        print("Create it with: GEMINI_API_KEY=your_key_here")
        return

    load_dotenv(env_path, override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)
    print(f"Gemini client initialized | Model: {config.GEMINI_MODEL} | Dim: {config.GEMINI_DIM}")

    # Load stories from input file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSONL file with stories")
    parser.add_argument("--output", type=str, default="outputs/gemini_embeddings.npy")
    args = parser.parse_args()

    print(f"Loading stories from {args.input}...")
    stories = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Support multiple JSONL formats
                if "anchor_text" in item:
                    stories.extend([item["anchor_text"], item["text_a"], item["text_b"]])
                elif "anchor_story" in item:
                    stories.extend([item["anchor_story"], item["similar_story"], item["dissimilar_story"]])
                elif "text" in item:
                    stories.append(item["text"])

    print(f"Loaded {len(stories)} stories")

    # Submit batch jobs
    batch_size = config.GEMINI_BATCH_SIZE
    checkpoint_file = Path(args.output).with_suffix(".checkpoint.json")
    batch_jobs = []
    start_idx = 0

    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            ckpt = json.load(f)
            batch_jobs = ckpt.get("batch_jobs", [])
            if batch_jobs:
                start_idx = batch_jobs[-1]["end_idx"]
                print(f"Resuming from index {start_idx}")

    temp_files = []
    for i in tqdm(range(start_idx, len(stories), batch_size), desc="Submitting batches"):
        batch = stories[i : i + batch_size]
        batch_file = f"_batch_temp_{i}.jsonl"
        temp_files.append(batch_file)

        with open(batch_file, "w") as f:
            for j, story in enumerate(batch):
                f.write(json.dumps({
                    "key": f"story_{i+j}",
                    "request": {
                        "output_dimensionality": config.GEMINI_DIM,
                        "content": {"parts": [{"text": story}]},
                    },
                }) + "\n")

        uploaded = client.files.upload(
            file=batch_file,
            config=types.UploadFileConfig(
                mime_type="application/x-jsonlines",
                display_name=f"batch_{i}",
            ),
        )

        job = client.batches.create_embeddings(
            model=config.GEMINI_MODEL,
            src={"file_name": uploaded.name},
            config={"display_name": f"emb_{i}_{i+batch_size}"},
        )

        batch_jobs.append({
            "job_name": job.name,
            "start_idx": i,
            "end_idx": i + batch_size,
            "batch_size": len(batch),
            "state": "SUBMITTED",
        })

        # Save checkpoint
        with open(checkpoint_file, "w") as f:
            json.dump({"total": len(stories), "batch_jobs": batch_jobs}, f, indent=2)

        if os.path.exists(batch_file):
            os.remove(batch_file)
        time.sleep(0.5)

    print(f"Submitted {len(batch_jobs)} batches. Polling for completion...")

    # Poll for completion
    done_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    pending = [i for i, j in enumerate(batch_jobs) if j.get("state") != "COMPLETED"]

    with tqdm(total=len(batch_jobs), initial=len(batch_jobs) - len(pending), desc="Waiting") as pbar:
        while pending:
            time.sleep(10)
            for idx in pending[:]:
                try:
                    job = client.batches.get(name=batch_jobs[idx]["job_name"])
                    if job.state.name in done_states:
                        pending.remove(idx)
                        batch_jobs[idx]["state"] = "COMPLETED"
                        batch_jobs[idx]["final_state"] = job.state.name
                        if job.state.name == "JOB_STATE_SUCCEEDED":
                            batch_jobs[idx]["dest_file"] = job.dest.file_name
                        pbar.update(1)
                except Exception as e:
                    print(f"Error checking batch {idx}: {e}")

    # Download embeddings
    print("Downloading embeddings...")
    all_embeddings = []
    for job_info in sorted(batch_jobs, key=lambda x: x["start_idx"]):
        if job_info.get("final_state") == "JOB_STATE_SUCCEEDED":
            try:
                content = client.files.download(file=job_info["dest_file"]).decode("utf-8")
                for line in content.splitlines():
                    if line.strip():
                        parsed = json.loads(line)
                        if "response" in parsed and "embedding" in parsed["response"]:
                            all_embeddings.append(parsed["response"]["embedding"]["values"])
            except Exception as e:
                print(f"Error downloading batch {job_info['start_idx']}: {e}")

    if all_embeddings:
        arr = np.array(all_embeddings, dtype=np.float32)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.save(args.output, arr)
        print(f"Saved {arr.shape[0]} embeddings ({arr.shape}) to {args.output}")

        meta = {"model": config.GEMINI_MODEL, "dim": config.GEMINI_DIM, "count": arr.shape[0]}
        with open(Path(args.output).with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Cleanup checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    else:
        print("No embeddings downloaded.")


if __name__ == "__main__":
    main()
