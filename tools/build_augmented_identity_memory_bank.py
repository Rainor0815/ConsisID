import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.consisid_utils import prepare_face_models, process_face_embeddings_infer
from util.identity_memory import make_identity_episode, save_identity_episodes


def _center_crop_scale(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    crop_w = int(width * scale)
    crop_h = int(height * scale)
    left = max((width - crop_w) // 2, 0)
    top = max((height - crop_h) // 2, 0)
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def _variants(image: Image.Image):
    image = ImageOps.exif_transpose(image.convert("RGB"))
    return {
        "identity_original": image,
        "identity_crop_092": _center_crop_scale(image, 0.92),
        "identity_crop_085": _center_crop_scale(image, 0.85),
        "identity_bright_108": ImageEnhance.Brightness(image).enhance(1.08),
        "identity_contrast_112": ImageEnhance.Contrast(image).enhance(1.12),
        "identity_sharp_125": ImageEnhance.Sharpness(image).enhance(1.25),
        "identity_slight_blur": image.filter(ImageFilter.GaussianBlur(radius=0.35)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build an episodic identity memory bank from conservative same-reference image augmentations."
    )
    parser.add_argument("--model_path", type=str, default="ckpts")
    parser.add_argument("--identity_image_path", type=str, required=True)
    parser.add_argument("--bank_path", type=str, required=True)
    parser.add_argument("--augmented_image_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_episodes", type=int, default=64)
    args = parser.parse_args()

    image_path = Path(args.identity_image_path)
    output_dir = Path(args.augmented_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"

    face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_mean, eva_std = prepare_face_models(
        args.model_path, device, dtype
    )

    episodes = []
    for name, variant in _variants(Image.open(image_path)).items():
        variant_path = output_dir / f"{name}.png"
        variant.save(variant_path)
        id_cond, id_vit_hidden, aligned_image, face_kps = process_face_embeddings_infer(
            face_helper_1,
            face_clip_model,
            face_helper_2,
            eva_mean,
            eva_std,
            face_main_model,
            device,
            dtype,
            str(variant_path),
            is_align_face=True,
        )
        episodes.append(
            make_identity_episode(
                id_cond,
                id_vit_hidden,
                aligned_image,
                face_kps,
                source=str(variant_path),
                metadata={"augmentation": name, "base_identity_image": str(image_path)},
            )
        )

    if args.max_episodes > 0:
        episodes = episodes[-args.max_episodes :]
    save_identity_episodes(
        args.bank_path,
        episodes,
        metadata={
            "note": "Augmented same-reference episodic identity memory bank",
            "base_identity_image": str(image_path),
        },
    )
    print(f"Saved augmented identity memory bank to {args.bank_path} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
