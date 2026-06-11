import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.consisid_utils import prepare_face_models, process_face_embeddings_infer
from util.identity_memory import append_identity_episode, make_identity_episode


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_summary(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _float_value(value, default=None):
    if value in (None, ""):
        return default
    return float(value)


def _crop_with_margin(frame, bbox, margin: float):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    x1 = int(max(x1 - bw * margin, 0))
    y1 = int(max(y1 - bh * margin, 0))
    x2 = int(min(x2 + bw * margin, width))
    y2 = int(min(y2 + bh * margin, height))
    return frame[y1:y2, x1:x2]


def _extract_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _write_report(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Append generated-frame identity episodes only when segment and frame scores pass "
            "strict guardrails. This avoids poisoning identity memory with drifted faces."
        )
    )
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--frame_scores_csv", type=str, required=True)
    parser.add_argument("--bank_path", type=str, required=True)
    parser.add_argument("--accepted_frame_dir", type=str, required=True)
    parser.add_argument("--report_csv", type=str, required=True)
    parser.add_argument("--min_segment_detection_rate", type=float, default=0.9)
    parser.add_argument("--min_segment_mean_similarity", type=float, default=0.45)
    parser.add_argument("--min_frame_similarity", type=float, default=0.55)
    parser.add_argument("--max_frames", type=int, default=2)
    parser.add_argument("--crop_margin", type=float, default=0.35)
    parser.add_argument("--max_episodes", type=int, default=96)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--source_label", type=str, default=None)
    args = parser.parse_args()

    summary = _read_summary(Path(args.summary_json))
    rows = _read_csv(Path(args.frame_scores_csv))
    report_rows = []

    segment_detection = float(summary.get("detection_rate") or 0.0)
    segment_mean = summary.get("mean_similarity")
    segment_mean = float(segment_mean) if segment_mean is not None else None
    segment_passed = (
        segment_detection >= args.min_segment_detection_rate
        and segment_mean is not None
        and segment_mean >= args.min_segment_mean_similarity
    )

    if not segment_passed:
        _write_report(
            Path(args.report_csv),
            [
                {
                    "frame_index": "",
                    "accepted": False,
                    "reason": "segment_guard_failed",
                    "arcface_similarity": segment_mean if segment_mean is not None else "",
                    "face_det_score": "",
                    "bbox_area_ratio": "",
                    "crop_path": "",
                }
            ],
        )
        print(
            "Guarded memory update skipped: "
            f"detection_rate={segment_detection:.3f}, mean_similarity={segment_mean}"
        )
        return

    candidate_rows = []
    for row in rows:
        if row.get("face_detected") != "True":
            continue
        similarity = _float_value(row.get("arcface_similarity"))
        if similarity is None or similarity < args.min_frame_similarity:
            continue
        candidate_rows.append((similarity, row))
    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    candidate_rows = candidate_rows[: max(args.max_frames, 0)]

    if not candidate_rows:
        _write_report(
            Path(args.report_csv),
            [
                {
                    "frame_index": "",
                    "accepted": False,
                    "reason": "no_frame_passed_similarity_guard",
                    "arcface_similarity": "",
                    "face_det_score": "",
                    "bbox_area_ratio": "",
                    "crop_path": "",
                }
            ],
        )
        print("Guarded memory update skipped: no frame passed frame-level similarity guard.")
        return

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"
    face_models = prepare_face_models(args.model_path, device, dtype)
    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_mean, eva_std = face_models
    accepted_dir = Path(args.accepted_frame_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    appended = 0
    for similarity, row in candidate_rows:
        frame_index = int(row["frame_index"])
        bbox = [
            _float_value(row["bbox_x1"], 0.0),
            _float_value(row["bbox_y1"], 0.0),
            _float_value(row["bbox_x2"], 0.0),
            _float_value(row["bbox_y2"], 0.0),
        ]
        try:
            frame = _extract_frame(Path(args.video_path), frame_index)
            crop = _crop_with_margin(frame, bbox, args.crop_margin)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_path = accepted_dir / f"frame_{frame_index:05d}_sim_{similarity:.3f}.png"
            Image.fromarray(crop_rgb).save(crop_path)
            id_cond, id_vit_hidden, aligned_image, face_kps = process_face_embeddings_infer(
                face_helper_1,
                face_clip_model,
                face_helper_2,
                eva_mean,
                eva_std,
                face_main_model,
                device,
                dtype,
                str(crop_path),
                is_align_face=True,
            )
            episode = make_identity_episode(
                id_cond,
                id_vit_hidden,
                aligned_image,
                face_kps,
                source=str(crop_path),
                metadata={
                    "role": "guarded_generated_frame",
                    "source_video": str(args.video_path),
                    "source_label": args.source_label,
                    "frame_index": frame_index,
                    "arcface_similarity_to_reference": similarity,
                    "segment_detection_rate": segment_detection,
                    "segment_mean_similarity": segment_mean,
                },
            )
            count = append_identity_episode(args.bank_path, episode, max_episodes=args.max_episodes)
            appended += 1
            report_rows.append(
                {
                    "frame_index": frame_index,
                    "accepted": True,
                    "reason": "accepted",
                    "arcface_similarity": similarity,
                    "face_det_score": row.get("face_det_score", ""),
                    "bbox_area_ratio": row.get("bbox_area_ratio", ""),
                    "crop_path": str(crop_path),
                }
            )
        except Exception as exc:
            report_rows.append(
                {
                    "frame_index": frame_index,
                    "accepted": False,
                    "reason": f"extract_failed:{type(exc).__name__}:{exc}",
                    "arcface_similarity": similarity,
                    "face_det_score": row.get("face_det_score", ""),
                    "bbox_area_ratio": row.get("bbox_area_ratio", ""),
                    "crop_path": "",
                }
            )

    _write_report(Path(args.report_csv), report_rows)
    print(f"Guarded memory update appended {appended} episodes to {args.bank_path}")
    if appended:
        print(f"Memory bank now has {count} episodes")


if __name__ == "__main__":
    main()
