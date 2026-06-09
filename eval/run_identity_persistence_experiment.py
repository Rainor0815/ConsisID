import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_prompt_suite(path: Path) -> Dict[str, List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run(command: List[str], cwd: Path):
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _mp4_set(output_dir: Path) -> set:
    if not output_dir.exists():
        return set()
    return {path.resolve() for path in output_dir.glob("*.mp4")}


def _new_video(output_dir: Path, before: set) -> Optional[Path]:
    candidates = [path for path in _mp4_set(output_dir) if path not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _build_infer_command(
    args,
    mode: str,
    prompt: str,
    output_dir: Path,
    seed: int,
) -> List[str]:
    command = [
        sys.executable,
        "infer.py",
        "--model_path",
        args.model_path,
        "--prompt",
        prompt,
        "--output_path",
        str(output_dir),
        "--seed",
        str(seed),
        "--num_frames",
        str(args.num_frames),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--dtype",
        args.dtype,
    ]

    if args.identity_memory_path:
        command.extend(["--load_identity_memory", args.identity_memory_path])
    else:
        command.extend(["--identity_image_path", args.identity_image_path])

    if mode == "episodic":
        command.extend(
            [
                "--episodic_identity_memory_path",
                args.episodic_identity_memory_path,
                "--episodic_top_k",
                str(args.episodic_top_k),
                "--episodic_min_similarity",
                str(args.episodic_min_similarity),
                "--episodic_base_weight",
                str(args.episodic_base_weight),
            ]
        )

    return command


def _build_eval_command(args, video_path: Path, output_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "eval/arcface_identity_stability.py",
        "--model_path",
        args.model_path,
        "--video_path",
        str(video_path),
        "--chunk_size",
        str(args.chunk_size),
        "--sample_stride",
        str(args.sample_stride),
        "--output_dir",
        str(output_dir),
        "--device",
        args.eval_device,
    ]
    if args.identity_memory_path:
        command.extend(["--identity_memory_path", args.identity_memory_path])
    else:
        command.extend(["--identity_image_path", args.identity_image_path])
    return command


def _read_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_aggregate(rows: List[Dict], output_dir: Path):
    if not rows:
        return
    csv_path = output_dir / "identity_persistence_aggregate.csv"
    json_path = output_dir / "identity_persistence_aggregate.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Aggregate CSV: {csv_path}")
    print(f"Aggregate JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline vs episodic identity persistence experiments.")
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--identity_image_path", type=str, default=None)
    parser.add_argument("--identity_memory_path", type=str, default=None)
    parser.add_argument("--episodic_identity_memory_path", type=str, default="identity_memory_bank.pt")
    parser.add_argument("--prompt_suite", type=str, default="eval/identity_prompt_suite.json")
    parser.add_argument("--output_dir", type=str, default="output/identity_persistence")
    parser.add_argument("--modes", nargs="+", default=["baseline", "episodic"], choices=["baseline", "episodic"])
    parser.add_argument("--categories", nargs="+", default=["realistic", "stylized"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--episodic_top_k", type=int, default=3)
    parser.add_argument("--episodic_min_similarity", type=float, default=0.45)
    parser.add_argument("--episodic_base_weight", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=49)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--eval_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only print them.")
    args = parser.parse_args()

    if not args.identity_image_path and not args.identity_memory_path:
        raise ValueError("Provide --identity_image_path or --identity_memory_path.")

    suite = _load_prompt_suite(PROJECT_ROOT / args.prompt_suite)
    output_root = PROJECT_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []

    for category in args.categories:
        for prompt_case in suite[category]:
            for mode in args.modes:
                for seed in args.seeds:
                    run_name = f"{category}_{prompt_case['name']}_{mode}_seed{seed}"
                    video_output_dir = output_root / "videos" / run_name
                    eval_output_dir = output_root / "scores" / run_name
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    eval_output_dir.mkdir(parents=True, exist_ok=True)

                    infer_command = _build_infer_command(args, mode, prompt_case["prompt"], video_output_dir, seed)
                    before = _mp4_set(video_output_dir)
                    if args.execute:
                        _run(infer_command, PROJECT_ROOT)
                        video_path = _new_video(video_output_dir, before)
                        if video_path is None:
                            raise RuntimeError(f"No new video found for run: {run_name}")
                        eval_command = _build_eval_command(args, video_path, eval_output_dir)
                        _run(eval_command, PROJECT_ROOT)
                        summary = _read_summary(eval_output_dir / "summary.json")
                        aggregate_rows.append(
                            {
                                "run_name": run_name,
                                "category": category,
                                "prompt_name": prompt_case["name"],
                                "mode": mode,
                                "seed": seed,
                                "video_path": str(video_path),
                                "mean_similarity": summary["mean_similarity"],
                                "min_similarity": summary["min_similarity"],
                                "detection_rate": summary["detection_rate"],
                                "first_chunk_mean_similarity": summary["first_chunk_mean_similarity"],
                                "last_chunk_mean_similarity": summary["last_chunk_mean_similarity"],
                                "chunk_similarity_decay": summary["chunk_similarity_decay"],
                                "chunk_mean_similarity_slope": summary["chunk_mean_similarity_slope"],
                            }
                        )
                    else:
                        print(" ".join(infer_command))
                        print("After generation, evaluate the produced MP4 with:")
                        print(" ".join(_build_eval_command(args, video_output_dir / "<generated>.mp4", eval_output_dir)))

    if args.execute:
        _write_aggregate(aggregate_rows, output_root)


if __name__ == "__main__":
    main()
