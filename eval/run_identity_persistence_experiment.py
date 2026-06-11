import argparse
import csv
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.face_recovery import (
    RecoveryProfile,
    build_recovery_profiles,
    format_recovery_reason,
    rank_segment_attempt,
    segment_passes_face_guard,
)


def _load_prompt_suite(path: Path) -> Dict[str, List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run(command: List[str], cwd: Path):
    print(shlex.join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mp4_set(output_dir: Path) -> set:
    if not output_dir.exists():
        return set()
    return {path.resolve() for path in output_dir.glob("*.mp4")}


def _new_video(output_dir: Path, before: set) -> Optional[Path]:
    candidates = [path for path in _mp4_set(output_dir) if path not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _quote_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace("'", "\\'")


def _concat_videos(video_paths: List[Path], output_path: Path) -> Path:
    if len(video_paths) == 1:
        return video_paths[0]

    list_path = output_path.with_suffix(".concat.txt")
    with list_path.open("w", encoding="utf-8") as handle:
        for video_path in video_paths:
            handle.write(f"file '{_quote_concat_path(video_path)}'\n")

    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ],
        PROJECT_ROOT,
    )
    return output_path


def _build_infer_command(
    args,
    mode: str,
    prompt: str,
    output_dir: Path,
    seed: int,
    negative_prompt: Optional[str] = None,
    episodic_identity_memory_path: Optional[str] = None,
    episodic_top_k: Optional[int] = None,
    episodic_base_weight: Optional[float] = None,
    local_face_scale: Optional[float] = None,
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
    if negative_prompt:
        command.extend(["--negative_prompt", negative_prompt])
    if local_face_scale is not None:
        command.extend(["--local_face_scale", str(local_face_scale)])

    if args.identity_memory_path:
        command.extend(["--load_identity_memory", args.identity_memory_path])
    else:
        command.extend(["--identity_image_path", args.identity_image_path])

    if mode == "episodic":
        command.extend(
            [
                "--episodic_identity_memory_path",
                episodic_identity_memory_path or args.episodic_identity_memory_path,
                "--episodic_top_k",
                str(episodic_top_k if episodic_top_k is not None else args.episodic_top_k),
                "--episodic_min_similarity",
                str(args.episodic_min_similarity),
                "--episodic_base_weight",
                str(episodic_base_weight if episodic_base_weight is not None else args.episodic_base_weight),
                "--episodic_exclude_exact_match",
                "--episodic_exact_match_epsilon",
                str(args.episodic_exact_match_epsilon),
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


def _build_guarded_update_command(args, video_path: Path, eval_output_dir: Path, bank_path: Path, output_dir: Path, label: str) -> List[str]:
    return [
        sys.executable,
        "tools/guarded_memory_update_from_video.py",
        "--model_path",
        args.model_path,
        "--video_path",
        str(video_path),
        "--summary_json",
        str(eval_output_dir / "summary.json"),
        "--frame_scores_csv",
        str(eval_output_dir / "frame_scores.csv"),
        "--bank_path",
        str(bank_path),
        "--accepted_frame_dir",
        str(output_dir / "accepted_online_frames"),
        "--report_csv",
        str(output_dir / "guarded_memory_update.csv"),
        "--min_segment_detection_rate",
        str(args.online_update_min_detection_rate),
        "--min_segment_mean_similarity",
        str(args.online_update_min_mean_similarity),
        "--min_frame_similarity",
        str(args.online_update_min_frame_similarity),
        "--max_frames",
        str(args.online_update_max_frames_per_segment),
        "--crop_margin",
        str(args.online_update_crop_margin),
        "--max_episodes",
        str(args.episodic_memory_max_episodes),
        "--dtype",
        args.dtype,
        "--source_label",
        label,
    ]


def _read_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_video(args, video_path: Path, eval_output_dir: Path) -> Dict:
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    _run(_build_eval_command(args, video_path, eval_output_dir), PROJECT_ROOT)
    return _read_summary(eval_output_dir / "summary.json")


def _summary_metric(value, fallback: float = -1.0) -> float:
    return fallback if value is None else float(value)


def _segment_quality_key(summary: Dict):
    return (
        float(summary.get("detection_rate") or 0.0),
        _summary_metric(summary.get("mean_similarity")),
        _summary_metric(summary.get("min_similarity")),
    )


def _segment_passes_active_guard(summary: Dict, args) -> bool:
    if args.face_recovery:
        return segment_passes_face_guard(
            summary,
            args.face_recovery_min_detection_rate,
            args.face_recovery_min_mean_similarity,
            args.face_recovery_min_frame_similarity,
        )
    return not _segment_needs_rerun(summary, args)


def _active_guard_reason(summary: Dict, args) -> str:
    if args.face_recovery:
        return format_recovery_reason(
            summary,
            args.face_recovery_min_detection_rate,
            args.face_recovery_min_mean_similarity,
            args.face_recovery_min_frame_similarity,
        )
    return "failed_guard" if _segment_needs_rerun(summary, args) else ""


def _active_attempt_key(summary: Dict, args):
    if args.face_recovery:
        return rank_segment_attempt(summary)
    return _segment_quality_key(summary)


def _build_attempt_profiles(args, prompt: str, negative_prompt: Optional[str], segment_index: int, mode: str) -> List[RecoveryProfile]:
    if mode == "episodic" and args.face_recovery:
        return build_recovery_profiles(prompt, negative_prompt, segment_index, args)

    max_attempts = 1 + (args.rerun_max_attempts if args.rerun_failed_segments and mode == "episodic" else 0)
    profiles = [
        RecoveryProfile(
            attempt_index=0,
            name="initial",
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed_offset=0,
            episodic_top_k=None,
            episodic_base_weight=None,
            local_face_scale=None,
        )
    ]
    for attempt_index in range(1, max_attempts):
        profiles.append(
            RecoveryProfile(
                attempt_index=attempt_index,
                name=f"rerun_{attempt_index:02d}",
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed_offset=0,
                episodic_top_k=args.rerun_top_k,
                episodic_base_weight=args.rerun_base_weight,
                local_face_scale=None,
            )
        )
    return profiles


def _segment_needs_rerun(summary: Dict, args) -> bool:
    if float(summary.get("detection_rate") or 0.0) < args.rerun_min_detection_rate:
        return True
    if _summary_metric(summary.get("mean_similarity")) < args.rerun_min_mean_similarity:
        return True
    if _summary_metric(summary.get("min_similarity")) < args.rerun_min_frame_similarity:
        return True
    return False


def _scheduled_base_weight(args, segment_index: int) -> float:
    if args.episodic_base_weight_late is None or args.segments <= 1:
        return args.episodic_base_weight
    progress = segment_index / max(args.segments - 1, 1)
    return args.episodic_base_weight + (args.episodic_base_weight_late - args.episodic_base_weight) * progress


def _scheduled_top_k(args, segment_index: int) -> int:
    if args.episodic_top_k_late is None or args.segments <= 1:
        return args.episodic_top_k
    progress = segment_index / max(args.segments - 1, 1)
    value = round(args.episodic_top_k + (args.episodic_top_k_late - args.episodic_top_k) * progress)
    return max(1, int(value))


def _prepare_online_memory_bank(args, output_root: Path, run_name: str) -> Optional[Path]:
    if not args.online_memory_update:
        return None
    source = PROJECT_ROOT / args.episodic_identity_memory_path
    if args.online_memory_path:
        target = PROJECT_ROOT / args.online_memory_path
    else:
        target = output_root / "online_memory_banks" / f"{run_name}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _safe_torch_load(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.detach().to(torch.float32) - right.detach().to(torch.float32)).abs().max().item())


def _episode_max_abs_diff(base_payload: Dict, episode: Dict) -> float:
    diffs = [_tensor_max_abs_diff(base_payload["id_cond"], episode["id_cond"])]
    base_hidden = base_payload.get("id_vit_hidden", [])
    episode_hidden = episode.get("id_vit_hidden", [])
    if len(base_hidden) != len(episode_hidden):
        return float("inf")
    for left, right in zip(base_hidden, episode_hidden):
        diffs.append(_tensor_max_abs_diff(left, right))
    return max(diffs)


def _validate_episodic_bank(args):
    if "episodic" not in args.modes:
        return

    bank_path = PROJECT_ROOT / args.episodic_identity_memory_path
    if not bank_path.exists():
        raise FileNotFoundError(f"Episodic identity memory bank not found: {bank_path}")

    bank_payload = _safe_torch_load(bank_path)
    episodes = bank_payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"Episodic identity memory bank has no episodes: {bank_path}")

    if not args.identity_memory_path:
        return

    base_path = PROJECT_ROOT / args.identity_memory_path
    if not base_path.exists():
        raise FileNotFoundError(f"Base identity memory not found: {base_path}")

    base_payload = _safe_torch_load(base_path)
    diffs = [_episode_max_abs_diff(base_payload, episode) for episode in episodes]
    has_distinct_episode = any(diff > args.memory_distinct_epsilon for diff in diffs)
    if not has_distinct_episode and not args.allow_self_only_memory_bank:
        raise ValueError(
            "Episodic memory bank is self-only: every episode is numerically identical to "
            f"{base_path}. Build a bank from distinct realistic reference episodes, or pass "
            "--allow_self_only_memory_bank only for debugging no-op behavior. "
            f"max_abs_diffs={diffs}"
        )


def _validate_temporal_coverage(args):
    if args.num_frames < 1:
        raise ValueError("--num_frames must be >= 1")
    if args.segments < 1:
        raise ValueError("--segments must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be >= 1")
    if args.min_num_chunks < 1:
        raise ValueError("--min_num_chunks must be >= 1")

    total_frames = args.num_frames * args.segments
    required_frames = args.chunk_size * args.min_num_chunks
    if total_frames < required_frames:
        raise ValueError(
            f"Identity persistence requires at least {args.min_num_chunks} chunks. "
            f"Set --num_frames * --segments >= {required_frames} for --chunk_size {args.chunk_size}; "
            f"current total is {args.num_frames} * {args.segments} = {total_frames}."
        )


def _write_segment_attempts(rows: List[Dict], output_dir: Path):
    if not rows:
        return
    csv_path = output_dir / "segment_attempts.csv"
    json_path = output_dir / "segment_attempts.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Segment attempts CSV: {csv_path}")
    print(f"Segment attempts JSON: {json_path}")


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


def _run_quality_checks(rows: List[Dict], args, output_dir: Path) -> List[Dict]:
    rows_by_key = {(row["category"], row["prompt_name"], row["seed"], row["mode"]): row for row in rows}
    validation_rows = []

    for row in rows:
        reasons = []
        if row["num_chunks"] < args.min_num_chunks:
            reasons.append(f"num_chunks {row['num_chunks']} < {args.min_num_chunks}")
        if row["detection_rate"] < args.min_detection_rate:
            reasons.append(f"detection_rate {row['detection_rate']:.4f} < {args.min_detection_rate:.4f}")
        if row["mean_similarity"] is None or row["mean_similarity"] < args.min_mean_similarity:
            reasons.append(f"mean_similarity {row['mean_similarity']} < {args.min_mean_similarity:.4f}")
        if row["min_similarity"] is None or row["min_similarity"] < args.min_frame_similarity:
            reasons.append(f"min_similarity {row['min_similarity']} < {args.min_frame_similarity:.4f}")
        if row["chunk_similarity_decay"] is not None and row["chunk_similarity_decay"] < -args.max_allowed_chunk_decay:
            reasons.append(
                f"chunk_similarity_decay {row['chunk_similarity_decay']:.4f} < {-args.max_allowed_chunk_decay:.4f}"
            )
        validation_rows.append(
            {
                "check_type": "single_run",
                "category": row["category"],
                "prompt_name": row["prompt_name"],
                "seed": row["seed"],
                "mode": row["mode"],
                "passed": not reasons,
                "reasons": "; ".join(reasons),
                "mean_similarity": row["mean_similarity"],
                "min_similarity": row["min_similarity"],
                "detection_rate": row["detection_rate"],
                "chunk_similarity_decay": row["chunk_similarity_decay"],
                "baseline_video_md5": None,
                "episodic_video_md5": None,
                "mean_delta": None,
                "min_delta": None,
            }
        )

    comparison_keys = sorted(
        {
            (row["category"], row["prompt_name"], row["seed"])
            for row in rows
            if row["mode"] in {"baseline", "episodic"}
        }
    )
    for category, prompt_name, seed in comparison_keys:
        baseline = rows_by_key.get((category, prompt_name, seed, "baseline"))
        episodic = rows_by_key.get((category, prompt_name, seed, "episodic"))
        if baseline is None or episodic is None:
            continue

        baseline_md5 = _file_md5(Path(baseline["video_path"]))
        episodic_md5 = _file_md5(Path(episodic["video_path"]))
        mean_delta = episodic["mean_similarity"] - baseline["mean_similarity"]
        min_delta = episodic["min_similarity"] - baseline["min_similarity"]
        reasons = []
        if baseline_md5 == episodic_md5:
            reasons.append("baseline and episodic videos are byte-identical")
        if mean_delta < args.min_episodic_mean_delta:
            reasons.append(f"mean_delta {mean_delta:.4f} < {args.min_episodic_mean_delta:.4f}")
        if min_delta < args.min_episodic_min_delta:
            reasons.append(f"min_delta {min_delta:.4f} < {args.min_episodic_min_delta:.4f}")

        validation_rows.append(
            {
                "check_type": "baseline_vs_episodic",
                "category": category,
                "prompt_name": prompt_name,
                "seed": seed,
                "mode": "comparison",
                "passed": not reasons,
                "reasons": "; ".join(reasons),
                "mean_similarity": episodic["mean_similarity"],
                "min_similarity": episodic["min_similarity"],
                "detection_rate": episodic["detection_rate"],
                "chunk_similarity_decay": episodic["chunk_similarity_decay"],
                "baseline_video_md5": baseline_md5,
                "episodic_video_md5": episodic_md5,
                "mean_delta": mean_delta,
                "min_delta": min_delta,
            }
        )

    if not validation_rows:
        return validation_rows

    csv_path = output_dir / "identity_persistence_validation.csv"
    json_path = output_dir / "identity_persistence_validation.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(validation_rows, handle, indent=2)
    print(f"Validation CSV: {csv_path}")
    print(f"Validation JSON: {json_path}")
    return validation_rows


def main():
    parser = argparse.ArgumentParser(description="Run baseline vs episodic identity persistence experiments.")
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--identity_image_path", type=str, default=None)
    parser.add_argument("--identity_memory_path", type=str, default=None)
    parser.add_argument("--episodic_identity_memory_path", type=str, default="identity_memory_bank.pt")
    parser.add_argument("--prompt_suite", type=str, default="eval/identity_prompt_suite.json")
    parser.add_argument("--output_dir", type=str, default="output/identity_persistence")
    parser.add_argument("--modes", nargs="+", default=["baseline", "episodic"], choices=["baseline", "episodic"])
    parser.add_argument("--categories", nargs="+", default=["realistic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--episodic_top_k", type=int, default=3)
    parser.add_argument("--episodic_min_similarity", type=float, default=0.45)
    parser.add_argument("--episodic_base_weight", type=float, default=1.0)
    parser.add_argument("--episodic_top_k_late", type=int, default=None)
    parser.add_argument("--episodic_base_weight_late", type=float, default=None)
    parser.add_argument("--episodic_exact_match_epsilon", type=float, default=0.0)
    parser.add_argument("--episodic_memory_max_episodes", type=int, default=96)
    parser.add_argument("--chunk_size", type=int, default=49)
    parser.add_argument("--min_num_chunks", type=int, default=3)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--eval_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--memory_distinct_epsilon", type=float, default=1e-6)
    parser.add_argument("--allow_self_only_memory_bank", action="store_true")
    parser.add_argument("--min_detection_rate", type=float, default=0.95)
    parser.add_argument("--min_mean_similarity", type=float, default=0.55)
    parser.add_argument("--min_frame_similarity", type=float, default=0.35)
    parser.add_argument("--max_allowed_chunk_decay", type=float, default=0.05)
    parser.add_argument("--min_episodic_mean_delta", type=float, default=0.02)
    parser.add_argument("--min_episodic_min_delta", type=float, default=0.02)
    parser.add_argument("--online_memory_update", action="store_true")
    parser.add_argument("--online_memory_path", type=str, default=None)
    parser.add_argument("--online_update_min_detection_rate", type=float, default=0.9)
    parser.add_argument("--online_update_min_mean_similarity", type=float, default=0.45)
    parser.add_argument("--online_update_min_frame_similarity", type=float, default=0.55)
    parser.add_argument("--online_update_max_frames_per_segment", type=int, default=2)
    parser.add_argument("--online_update_crop_margin", type=float, default=0.35)
    parser.add_argument("--rerun_failed_segments", action="store_true")
    parser.add_argument("--rerun_max_attempts", type=int, default=1)
    parser.add_argument("--rerun_min_detection_rate", type=float, default=0.9)
    parser.add_argument("--rerun_min_mean_similarity", type=float, default=0.45)
    parser.add_argument("--rerun_min_frame_similarity", type=float, default=0.35)
    parser.add_argument("--rerun_base_weight", type=float, default=None)
    parser.add_argument("--rerun_top_k", type=int, default=None)
    parser.add_argument("--face_recovery", action="store_true")
    parser.add_argument("--face_recovery_max_attempts", type=int, default=3)
    parser.add_argument("--face_recovery_min_detection_rate", type=float, default=0.5)
    parser.add_argument("--face_recovery_min_mean_similarity", type=float, default=0.45)
    parser.add_argument("--face_recovery_min_frame_similarity", type=float, default=0.35)
    parser.add_argument("--face_recovery_seed_stride", type=int, default=101)
    parser.add_argument("--face_recovery_prompt_suffix", type=str, default=None)
    parser.add_argument("--face_recovery_fallback_prompt", type=str, default=None)
    parser.add_argument("--face_recovery_negative_prompt", type=str, default=None)
    parser.add_argument("--face_recovery_local_face_scales", type=str, default="default,1.25,1.5")
    parser.add_argument("--face_recovery_top_ks", type=str, default="6,6")
    parser.add_argument("--face_recovery_base_weights", type=str, default="0.25,0.2")
    parser.add_argument("--fail_on_validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only print them.")
    args = parser.parse_args()

    if not args.identity_image_path and not args.identity_memory_path:
        raise ValueError("Provide --identity_image_path or --identity_memory_path.")
    _validate_temporal_coverage(args)
    _validate_episodic_bank(args)

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
                    segment_eval_root = output_root / "segment_scores" / run_name
                    advanced_segment_controls = (
                        mode == "episodic"
                        and (args.online_memory_update or args.rerun_failed_segments or args.face_recovery)
                    )
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    eval_output_dir.mkdir(parents=True, exist_ok=True)
                    segment_eval_root.mkdir(parents=True, exist_ok=True)
                    online_memory_bank = _prepare_online_memory_bank(args, output_root, run_name) if mode == "episodic" else None
                    memory_path_for_generation = (
                        str(online_memory_bank.relative_to(PROJECT_ROOT))
                        if online_memory_bank is not None and online_memory_bank.is_relative_to(PROJECT_ROOT)
                        else str(online_memory_bank)
                        if online_memory_bank is not None
                        else args.episodic_identity_memory_path
                    )
                    segment_attempt_rows = []
                    base_negative_prompt = prompt_case.get("negative_prompt") or args.negative_prompt

                    if args.execute:
                        segment_paths = []
                        for segment_index in range(args.segments):
                            segment_dir = video_output_dir / f"segment_{segment_index:02d}"
                            segment_dir.mkdir(parents=True, exist_ok=True)
                            segment_seed = seed + segment_index
                            attempt_summaries = []
                            attempt_profiles = _build_attempt_profiles(
                                args,
                                prompt_case["prompt"],
                                base_negative_prompt,
                                segment_index,
                                mode,
                            )
                            for profile in attempt_profiles:
                                attempt_index = profile.attempt_index
                                attempt_dir = (
                                    segment_dir / f"attempt_{attempt_index:02d}"
                                    if advanced_segment_controls
                                    else segment_dir
                                )
                                attempt_dir.mkdir(parents=True, exist_ok=True)
                                attempt_seed = segment_seed + profile.seed_offset
                                attempt_base_weight = _scheduled_base_weight(args, segment_index)
                                attempt_top_k = _scheduled_top_k(args, segment_index)
                                if profile.episodic_base_weight is not None:
                                    attempt_base_weight = profile.episodic_base_weight
                                if profile.episodic_top_k is not None:
                                    attempt_top_k = profile.episodic_top_k

                                infer_command = _build_infer_command(
                                    args,
                                    mode,
                                    profile.prompt,
                                    attempt_dir,
                                    attempt_seed,
                                    negative_prompt=profile.negative_prompt,
                                    episodic_identity_memory_path=memory_path_for_generation,
                                    episodic_top_k=attempt_top_k,
                                    episodic_base_weight=attempt_base_weight,
                                    local_face_scale=profile.local_face_scale,
                                )
                                before = _mp4_set(attempt_dir)
                                _run(infer_command, PROJECT_ROOT)
                                attempt_path = _new_video(attempt_dir, before)
                                if attempt_path is None:
                                    raise RuntimeError(
                                        f"No new video found for run: {run_name} segment {segment_index} attempt {attempt_index}"
                                    )

                                attempt_summary = None
                                if advanced_segment_controls:
                                    attempt_eval_dir = segment_eval_root / f"segment_{segment_index:02d}_attempt_{attempt_index:02d}"
                                    attempt_summary = _evaluate_video(args, attempt_path, attempt_eval_dir)
                                    attempt_summaries.append((attempt_path, attempt_eval_dir, attempt_summary))
                                    segment_attempt_rows.append(
                                        {
                                            "run_name": run_name,
                                            "segment_index": segment_index,
                                            "attempt_index": attempt_index,
                                            "recovery_profile": profile.name,
                                            "video_path": str(attempt_path),
                                            "attempt_seed": attempt_seed,
                                            "attempt_seed_offset": profile.seed_offset,
                                            "negative_prompt": profile.negative_prompt or "",
                                            "local_face_scale": profile.local_face_scale if profile.local_face_scale is not None else "",
                                            "episodic_memory_path": memory_path_for_generation,
                                            "episodic_top_k": attempt_top_k,
                                            "episodic_base_weight": attempt_base_weight,
                                            "detection_rate": attempt_summary["detection_rate"],
                                            "mean_similarity": attempt_summary["mean_similarity"],
                                            "min_similarity": attempt_summary["min_similarity"],
                                            "zero_face_chunks": attempt_summary.get("zero_face_chunks"),
                                            "max_consecutive_missing_frames": attempt_summary.get("max_consecutive_missing_frames"),
                                            "mean_face_area_ratio": attempt_summary.get("mean_face_area_ratio"),
                                            "min_face_area_ratio": attempt_summary.get("min_face_area_ratio"),
                                            "face_guard_passed": _segment_passes_active_guard(attempt_summary, args),
                                            "accepted_attempt": False,
                                            "rerun_reason": _active_guard_reason(attempt_summary, args),
                                        }
                                    )
                                    if _segment_passes_active_guard(attempt_summary, args):
                                        break
                                else:
                                    attempt_summaries.append((attempt_path, None, attempt_summary))
                                    break

                            if advanced_segment_controls:
                                best_path, best_eval_dir, best_summary = max(
                                    attempt_summaries,
                                    key=lambda item: _active_attempt_key(item[2], args),
                                )
                                for row in segment_attempt_rows:
                                    if row["run_name"] == run_name and row["segment_index"] == segment_index:
                                        row["accepted_attempt"] = row["video_path"] == str(best_path)
                                if args.online_memory_update and mode == "episodic":
                                    guarded_dir = segment_eval_root / f"segment_{segment_index:02d}_guarded_update"
                                    guarded_dir.mkdir(parents=True, exist_ok=True)
                                    _run(
                                        _build_guarded_update_command(
                                            args,
                                            best_path,
                                            best_eval_dir,
                                            online_memory_bank,
                                            guarded_dir,
                                            f"{run_name}:segment_{segment_index:02d}",
                                        ),
                                        PROJECT_ROOT,
                                    )
                                segment_paths.append(best_path)
                            else:
                                segment_paths.append(attempt_summaries[-1][0])

                        video_path = _concat_videos(
                            segment_paths,
                            video_output_dir / f"{seed}_joined_{args.segments}x{args.num_frames}.mp4",
                        )
                        summary = _evaluate_video(args, video_path, eval_output_dir)
                        aggregate_rows.append(
                            {
                                "run_name": run_name,
                                "category": category,
                                "prompt_name": prompt_case["name"],
                                "mode": mode,
                                "seed": seed,
                                "video_path": str(video_path),
                                "segments": args.segments,
                                "segment_frames": args.num_frames,
                                "sampled_frames": summary["sampled_frames"],
                                "detected_frames": summary["detected_frames"],
                                "missing_frames": summary["missing_frames"],
                                "num_chunks": summary["num_chunks"],
                                "mean_similarity": summary["mean_similarity"],
                                "min_similarity": summary["min_similarity"],
                                "detection_rate": summary["detection_rate"],
                                "first_chunk_mean_similarity": summary["first_chunk_mean_similarity"],
                                "last_chunk_mean_similarity": summary["last_chunk_mean_similarity"],
                                "chunk_similarity_decay": summary["chunk_similarity_decay"],
                                "chunk_mean_similarity_slope": summary["chunk_mean_similarity_slope"],
                                "zero_face_chunks": summary.get("zero_face_chunks"),
                                "max_consecutive_missing_frames": summary.get("max_consecutive_missing_frames"),
                                "mean_face_area_ratio": summary.get("mean_face_area_ratio"),
                                "min_face_area_ratio": summary.get("min_face_area_ratio"),
                                "face_center_std_x": summary.get("face_center_std_x"),
                                "face_center_std_y": summary.get("face_center_std_y"),
                            }
                        )
                        if segment_attempt_rows:
                            _write_segment_attempts(segment_attempt_rows, segment_eval_root)
                    else:
                        for segment_index in range(args.segments):
                            segment_dir = video_output_dir / f"segment_{segment_index:02d}"
                            segment_seed = seed + segment_index
                            attempt_profiles = _build_attempt_profiles(
                                args,
                                prompt_case["prompt"],
                                base_negative_prompt,
                                segment_index,
                                mode,
                            )
                            for profile in attempt_profiles:
                                attempt_dir = (
                                    segment_dir / f"attempt_{profile.attempt_index:02d}"
                                    if advanced_segment_controls
                                    else segment_dir
                                )
                                infer_command = _build_infer_command(
                                    args,
                                    mode,
                                    profile.prompt,
                                    attempt_dir,
                                    segment_seed + profile.seed_offset,
                                    negative_prompt=profile.negative_prompt,
                                    episodic_identity_memory_path=memory_path_for_generation,
                                    episodic_top_k=profile.episodic_top_k or _scheduled_top_k(args, segment_index),
                                    episodic_base_weight=profile.episodic_base_weight
                                    if profile.episodic_base_weight is not None
                                    else _scheduled_base_weight(args, segment_index),
                                    local_face_scale=profile.local_face_scale,
                                )
                                print(shlex.join(infer_command))
                            if advanced_segment_controls:
                                print(
                                    "After this segment, evaluate attempts, optionally run recovery profiles, "
                                    "and run guarded_memory_update_from_video.py for accepted segments."
                                )
                        joined_placeholder = video_output_dir / f"{seed}_joined_{args.segments}x{args.num_frames}.mp4"
                        print(f"Concatenate generated segments to: {joined_placeholder}")
                        print("After generation, evaluate the produced MP4 with:")
                        print(shlex.join(_build_eval_command(args, joined_placeholder, eval_output_dir)))

    if args.execute:
        _write_aggregate(aggregate_rows, output_root)
        validation_rows = _run_quality_checks(aggregate_rows, args, output_root)
        failed_rows = [row for row in validation_rows if not row["passed"]]
        if failed_rows and args.fail_on_validation:
            raise RuntimeError(f"Identity persistence validation failed ({len(failed_rows)} failed checks).")


if __name__ == "__main__":
    main()
