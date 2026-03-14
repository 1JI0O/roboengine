#!/usr/bin/env python3
"""在一个 color 目录上运行 RoboEngine 分割，并导出逐帧 mask。

示例：
python examples/run_roboseg_on_color_dir.py \
  --color-dir /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/color \
  --output-dir /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/sam_mask_out \
  --device cuda \
  --anchor-frequency 8

如果要额外做对象分割并导出 merged：
python examples/run_roboseg_on_color_dir.py \
  --color-dir /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/color \
  --output-dir /data/haoxiang/data/airexo2/task_0012/train/scene_0001/cam_105422061350/sam_mask_out \
  --instruction "pick up the basket" \
  --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from robo_engine.infer_engine import RoboEngineObjectSegmentation
from robo_engine.infer_engine import RoboEngineRobotSegmentation

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboSeg on one color directory")
    parser.add_argument("--color-dir", type=str, required=True, help="输入帧目录（如 .../cam_xxx/color）")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--sam-version",
        type=str,
        default="/data/haoxiang/roboengine/roboengine-sam",
        help="机器人分割权重路径（sam_versoin）",
    )
    parser.add_argument(
        "--sam-tokenizer-version",
        type=str,
        default="/data/haoxiang/roboengine/evf-sam2-multitask",
        help="tokenizer 权重路径（sam_tokenizer_version）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--anchor-frequency", type=int, default=8, help="视频锚点间隔")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="可选：对象分割文本指令；提供后会同时输出 object 和 merged",
    )
    return parser.parse_args()


def list_frames(color_dir: Path) -> list[Path]:
    frame_paths = [p for p in color_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    frame_paths.sort(key=lambda p: p.name)
    return frame_paths


def load_frames(frame_paths: list[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for fp in tqdm(frame_paths, desc="Loading frames"):
        frames.append(np.array(Image.open(fp).convert("RGB")))
    return frames


def save_masks(mask_np: np.ndarray, frame_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(tqdm(frame_paths, desc=f"Saving {out_dir.name} masks")):
        mask = mask_np[i]
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        mask_u8 = ((mask > 0).astype(np.uint8) * 255)
        Image.fromarray(mask_u8).save(out_dir / fp.name)


def main() -> None:
    args = parse_args()

    color_dir = Path(args.color_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not color_dir.is_dir():
        raise FileNotFoundError(f"color dir not found: {color_dir}")

    frame_paths = list_frames(color_dir)
    if len(frame_paths) == 0:
        raise RuntimeError(f"No image frames found in: {color_dir}")

    print("=" * 80)
    print(f"Color dir   : {color_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"#Frames     : {len(frame_paths)}")
    print(f"Device      : {args.device}")
    print(f"Anchor freq : {args.anchor_frequency}")
    print("=" * 80)

    image_np_list = load_frames(frame_paths)

    # 1) Robot segmentation
    engine_robo = RoboEngineRobotSegmentation(
        sam_versoin=args.sam_version,
        sam_tokenizer_version=args.sam_tokenizer_version,
        device=args.device,
    )
    robo_masks = engine_robo.gen_video(
        image_np_list=image_np_list,
        anchor_frequency=args.anchor_frequency,
    )
    save_masks(robo_masks, frame_paths, output_dir / "robot")

    # 2) Optional object segmentation + merged
    if args.instruction is not None and args.instruction.strip() != "":
        engine_obj = RoboEngineObjectSegmentation(
            sam_versoin=args.sam_tokenizer_version,
            sam_tokenizer_version=args.sam_tokenizer_version,
            device=args.device,
        )
        obj_masks = engine_obj.gen_video(
            image_np_list=image_np_list,
            instruction=args.instruction,
            anchor_frequency=args.anchor_frequency,
        )
        save_masks(obj_masks, frame_paths, output_dir / "object")

        merged_masks = ((robo_masks + obj_masks) > 0).astype(np.float32)
        save_masks(merged_masks, frame_paths, output_dir / "merged")

    print("Done.")
    print(f"Robot masks : {output_dir / 'robot'}")
    if args.instruction is not None and args.instruction.strip() != "":
        print(f"Object masks: {output_dir / 'object'}")
        print(f"Merged masks: {output_dir / 'merged'}")


if __name__ == "__main__":
    main()
