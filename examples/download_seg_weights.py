#!/usr/bin/env python3
"""预下载 RoboEngine 分割相关权重到指定目录。

用法示例：
python examples/download_seg_weights.py --output-dir /data/haoxiang/roboengine
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODELS = [
    ("michaelyuanqwq/roboengine-sam", "roboengine-sam"),
    ("YxZhang/evf-sam2-multitask", "evf-sam2-multitask"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RoboEngine segmentation weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="本地保存根目录，例如 /data/haoxiang/roboengine",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token（可选）。未提供时会读取环境变量 HF_TOKEN",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行下载线程数，默认 8",
    )
    return parser.parse_args()


def download_repo(repo_id: str, local_dir: Path, token: str | None, max_workers: int) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Download] {repo_id}")
    print(f"[Target  ] {local_dir}")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
        max_workers=max_workers,
        resume_download=True,
    )
    print(f"[Done    ] {repo_id} -> {path}")
    return path


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN")

    print("=" * 70)
    print("RoboEngine Segmentation Weights Downloader")
    print(f"Output root: {output_root}")
    print("=" * 70)

    downloaded = {}
    for repo_id, subdir in DEFAULT_MODELS:
        target_dir = output_root / subdir
        downloaded[repo_id] = download_repo(
            repo_id=repo_id,
            local_dir=target_dir,
            token=token,
            max_workers=args.max_workers,
        )

    print("\n" + "=" * 70)
    print("下载完成，可在代码中使用以下本地路径：")
    print(f"sam_versoin         = {output_root / 'roboengine-sam'}")
    print(f"sam_tokenizer_version = {output_root / 'evf-sam2-multitask'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
