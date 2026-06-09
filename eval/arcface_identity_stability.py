import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from PIL import Image


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _largest_face(face_infos):
    if len(face_infos) == 0:
        return None
    return sorted(
        face_infos,
        key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]),
    )[-1]


def _load_reference_embedding(
    face_model: FaceAnalysis,
    identity_image_path: Optional[str],
    identity_memory_path: Optional[str],
) -> np.ndarray:
    def _tensor_to_numpy_float32(tensor: torch.Tensor) -> np.ndarray:
        # NumPy conversion from bf16 tensors is unsupported in some torch builds.
        return tensor.detach().to(torch.float32).cpu().numpy()

    # Prefer the saved ArcFace/Antelope embedding when evaluating memory-driven generations.
    if identity_memory_path is not None:
        try:
            payload = torch.load(identity_memory_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(identity_memory_path, map_location="cpu")
        if "id_ante_embedding" in payload:
            return _tensor_to_numpy_float32(payload["id_ante_embedding"])[0]
        if "id_cond" in payload:
            return _tensor_to_numpy_float32(payload["id_cond"])[0, :512]
        raise ValueError(f"No ArcFace identity embedding found in: {identity_memory_path}")

    if identity_image_path is None:
        raise ValueError("Provide either --identity_image_path or --identity_memory_path.")

    image = np.array(Image.open(identity_image_path).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_info = _largest_face(face_model.get(image_bgr))
    if face_info is None:
        raise RuntimeError(f"No face detected in identity image: {identity_image_path}")
    return face_info["embedding"]


def _frame_embedding(face_model: FaceAnalysis, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    face_info = _largest_face(face_model.get(frame_bgr))
    if face_info is None:
        return None, None
    return face_info["embedding"], face_info["bbox"]


def _chunk_rows(frame_rows: List[Dict], chunk_size: int) -> List[Dict]:
    chunks: Dict[int, List[Dict]] = {}
    for row in frame_rows:
        chunks.setdefault(row["chunk_index"], []).append(row)

    chunk_rows = []
    for chunk_index in sorted(chunks):
        rows = chunks[chunk_index]
        similarities = [row["arcface_similarity"] for row in rows if row["face_detected"]]
        detected = len(similarities)
        total = len(rows)
        chunk_rows.append(
            {
                "chunk_index": chunk_index,
                "start_frame": rows[0]["frame_index"],
                "end_frame": rows[-1]["frame_index"],
                "sampled_frames": total,
                "detected_frames": detected,
                "missing_frames": total - detected,
                "detection_rate": detected / total if total else 0.0,
                "mean_similarity": float(np.mean(similarities)) if similarities else None,
                "min_similarity": float(np.min(similarities)) if similarities else None,
                "max_similarity": float(np.max(similarities)) if similarities else None,
                "chunk_size": chunk_size,
            }
        )
    return chunk_rows


def _similarity_decay_summary(chunk_rows: List[Dict]) -> Dict:
    valid_chunks = [row for row in chunk_rows if row["mean_similarity"] is not None]
    if not valid_chunks:
        return {
            "first_chunk_mean_similarity": None,
            "last_chunk_mean_similarity": None,
            "chunk_similarity_decay": None,
            "chunk_mean_similarity_slope": None,
        }

    first = valid_chunks[0]["mean_similarity"]
    last = valid_chunks[-1]["mean_similarity"]
    if len(valid_chunks) >= 2:
        x = np.asarray([row["chunk_index"] for row in valid_chunks], dtype=np.float32)
        y = np.asarray([row["mean_similarity"] for row in valid_chunks], dtype=np.float32)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = None

    return {
        "first_chunk_mean_similarity": first,
        "last_chunk_mean_similarity": last,
        "chunk_similarity_decay": float(last - first) if first is not None and last is not None else None,
        "chunk_mean_similarity_slope": slope,
    }


def _write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_arcface_identity_stability(
    model_path: str,
    video_path: str,
    identity_image_path: Optional[str] = None,
    identity_memory_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    sample_stride: int = 1,
    chunk_size: int = 49,
    det_size: int = 640,
):
    if sample_stride < 1:
        raise ValueError("--sample_stride must be >= 1")
    if chunk_size < 1:
        raise ValueError("--chunk_size must be >= 1")

    if not os.path.exists(model_path):
        print("Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=model_path)
    else:
        print(f"Model already exists in {model_path}, skipping download.")

    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    face_model = FaceAnalysis(name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=providers)
    face_model.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(det_size, det_size))

    reference_embedding = _load_reference_embedding(face_model, identity_image_path, identity_memory_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_rows = []
    frame_index = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_index % sample_stride != 0:
            frame_index += 1
            continue

        frame_embedding, bbox = _frame_embedding(face_model, frame_bgr)
        similarity = _cosine_similarity(reference_embedding, frame_embedding) if frame_embedding is not None else None
        x1, y1, x2, y2 = bbox.tolist() if bbox is not None else (None, None, None, None)

        # Chunk index is based on real frame position, so skipped sampling still maps to generated segments.
        frame_rows.append(
            {
                "frame_index": frame_index,
                "time_seconds": frame_index / fps if fps > 0 else None,
                "chunk_index": frame_index // chunk_size,
                "face_detected": frame_embedding is not None,
                "arcface_similarity": similarity,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
            }
        )
        frame_index += 1

    cap.release()

    chunk_rows = _chunk_rows(frame_rows, chunk_size)
    decay_summary = _similarity_decay_summary(chunk_rows)
    detected_scores = [row["arcface_similarity"] for row in frame_rows if row["face_detected"]]
    summary = {
        "video_path": video_path,
        "identity_image_path": identity_image_path,
        "identity_memory_path": identity_memory_path,
        "total_video_frames": total_frames,
        "sample_stride": sample_stride,
        "sampled_frames": len(frame_rows),
        "detected_frames": len(detected_scores),
        "missing_frames": len(frame_rows) - len(detected_scores),
        "detection_rate": len(detected_scores) / len(frame_rows) if frame_rows else 0.0,
        "mean_similarity": float(np.mean(detected_scores)) if detected_scores else None,
        "min_similarity": float(np.min(detected_scores)) if detected_scores else None,
        "max_similarity": float(np.max(detected_scores)) if detected_scores else None,
        "chunk_size": chunk_size,
        "num_chunks": len(chunk_rows),
        "lowest_similarity_chunk": min(
            (row for row in chunk_rows if row["mean_similarity"] is not None),
            key=lambda row: row["mean_similarity"],
            default=None,
        ),
        **decay_summary,
    }

    output_dir = output_dir or os.path.splitext(video_path)[0] + "_arcface_identity"
    os.makedirs(output_dir, exist_ok=True)
    frame_csv = os.path.join(output_dir, "frame_scores.csv")
    chunk_csv = os.path.join(output_dir, "chunk_scores.csv")
    summary_json = os.path.join(output_dir, "summary.json")

    _write_csv(frame_csv, frame_rows)
    _write_csv(chunk_csv, chunk_rows)
    with open(summary_json, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    print(f"Frame scores: {frame_csv}")
    print(f"Chunk scores: {chunk_csv}")
    print(f"Summary: {summary_json}")
    print(
        "ArcFace identity stability: "
        f"mean={summary['mean_similarity']}, min={summary['min_similarity']}, "
        f"detection_rate={summary['detection_rate']:.3f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Measure per-frame and per-chunk ArcFace identity stability.")
    parser.add_argument("--model_path", type=str, default="./ckpts", help="Path to ConsisID model directory.")
    parser.add_argument("--video_path", type=str, required=True, help="Generated video to evaluate.")
    parser.add_argument("--identity_image_path", type=str, default=None, help="Reference identity image.")
    parser.add_argument("--identity_memory_path", type=str, default=None, help="Saved identity memory .pt file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for frame/chunk CSV and summary JSON.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for ArcFace inference.")
    parser.add_argument("--sample_stride", type=int, default=1, help="Evaluate every Nth frame.")
    parser.add_argument("--chunk_size", type=int, default=49, help="Frames per generated chunk/segment.")
    parser.add_argument("--det_size", type=int, default=640, help="InsightFace detection size.")
    args = parser.parse_args()

    evaluate_arcface_identity_stability(
        model_path=args.model_path,
        video_path=args.video_path,
        identity_image_path=args.identity_image_path,
        identity_memory_path=args.identity_memory_path,
        output_dir=args.output_dir,
        device=args.device,
        sample_stride=args.sample_stride,
        chunk_size=args.chunk_size,
        det_size=args.det_size,
    )


if __name__ == "__main__":
    main()
