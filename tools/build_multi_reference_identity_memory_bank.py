import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.consisid_utils import prepare_face_models, process_face_embeddings_infer
from util.identity_memory import identity_conditioning_similarity, make_identity_episode, save_identity_episodes


def _iter_images(paths, directory):
    candidates = []
    if directory is not None:
        root = Path(directory)
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            candidates.extend(root.glob(suffix))
    candidates.extend(Path(path) for path in paths)
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path


def _save_rgb_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(src).convert("RGB"))
    image.save(dst)


def _extract_episode(
    image_path: Path,
    model_path: str,
    face_models,
    device: str,
    dtype: torch.dtype,
    metadata,
):
    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_mean, eva_std = face_models
    id_cond, id_vit_hidden, aligned_image, face_kps = process_face_embeddings_infer(
        face_helper_1,
        face_clip_model,
        face_helper_2,
        eva_mean,
        eva_std,
        face_main_model,
        device,
        dtype,
        str(image_path),
        is_align_face=True,
    )
    return make_identity_episode(
        id_cond,
        id_vit_hidden,
        aligned_image,
        face_kps,
        source=str(image_path),
        metadata=metadata,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a ConsisID identity memory bank from multiple real reference photos. "
            "Each candidate is verified against the base reference before it is added."
        )
    )
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--base_identity_image", type=str, required=True)
    parser.add_argument("--reference_image_paths", type=str, nargs="*", default=[])
    parser.add_argument("--reference_image_dir", type=str, default=None)
    parser.add_argument("--bank_path", type=str, required=True)
    parser.add_argument("--accepted_image_dir", type=str, required=True)
    parser.add_argument("--report_csv", type=str, required=True)
    parser.add_argument("--min_arcface_similarity", type=float, default=0.35)
    parser.add_argument("--min_combined_similarity", type=float, default=0.45)
    parser.add_argument("--include_base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_episodes", type=int, default=64)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"
    face_models = prepare_face_models(args.model_path, device, dtype)

    accepted_dir = Path(args.accepted_image_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_identity_image)
    if not base_path.exists():
        raise FileNotFoundError(f"Base identity image not found: {base_path}")

    base_episode = _extract_episode(
        base_path,
        args.model_path,
        face_models,
        device,
        dtype,
        {"role": "base_reference", "source_image": str(base_path)},
    )

    rows = []
    episodes = []
    if args.include_base:
        base_copy = accepted_dir / f"accepted_000_{base_path.name}"
        _save_rgb_copy(base_path, base_copy)
        base_episode["source"] = str(base_copy)
        episodes.append(base_episode)
        rows.append(
            {
                "image_path": str(base_path),
                "accepted": True,
                "reason": "base_reference",
                "arcface_similarity": 1.0,
                "vit_similarity": 1.0,
                "combined_similarity": 1.0,
                "accepted_copy": str(base_copy),
            }
        )

    for image_path in _iter_images(args.reference_image_paths, args.reference_image_dir):
        if not image_path.exists():
            rows.append(
                {
                    "image_path": str(image_path),
                    "accepted": False,
                    "reason": "missing_file",
                    "arcface_similarity": "",
                    "vit_similarity": "",
                    "combined_similarity": "",
                    "accepted_copy": "",
                }
            )
            continue
        if image_path.resolve() == base_path.resolve():
            continue

        try:
            episode = _extract_episode(
                image_path,
                args.model_path,
                face_models,
                device,
                dtype,
                {"role": "verified_reference", "source_image": str(image_path)},
            )
            similarity = identity_conditioning_similarity(base_episode["id_cond"], episode["id_cond"])
            accepted = (
                similarity["arcface_similarity"] >= args.min_arcface_similarity
                and similarity["score"] >= args.min_combined_similarity
            )
            reason = "accepted" if accepted else "identity_similarity_below_threshold"
            accepted_copy = ""
            if accepted:
                accepted_copy_path = accepted_dir / f"accepted_{len(episodes):03d}_{image_path.stem}.png"
                _save_rgb_copy(image_path, accepted_copy_path)
                episode["source"] = str(accepted_copy_path)
                episode["metadata"].update(
                    {
                        "arcface_similarity_to_base": similarity["arcface_similarity"],
                        "vit_similarity_to_base": similarity["vit_similarity"],
                        "combined_similarity_to_base": similarity["score"],
                    }
                )
                episodes.append(episode)
                accepted_copy = str(accepted_copy_path)
        except Exception as exc:
            accepted = False
            reason = f"extract_failed:{type(exc).__name__}:{exc}"
            similarity = {"arcface_similarity": "", "vit_similarity": "", "score": ""}
            accepted_copy = ""

        rows.append(
            {
                "image_path": str(image_path),
                "accepted": accepted,
                "reason": reason,
                "arcface_similarity": similarity["arcface_similarity"],
                "vit_similarity": similarity["vit_similarity"],
                "combined_similarity": similarity["score"],
                "accepted_copy": accepted_copy,
            }
        )

    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    if len(episodes) < 2:
        raise RuntimeError(
            "Fewer than two verified same-subject references were accepted. "
            f"See {report_path} for candidate scores."
        )

    save_identity_episodes(
        args.bank_path,
        episodes,
        metadata={
            "note": "Verified multi-reference identity memory bank",
            "base_identity_image": str(base_path),
            "min_arcface_similarity": args.min_arcface_similarity,
            "min_combined_similarity": args.min_combined_similarity,
        },
    )
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved multi-reference bank to {args.bank_path} ({len(episodes)} episodes)")
    print(f"Candidate report: {report_path}")


if __name__ == "__main__":
    main()
