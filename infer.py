import argparse
import os
import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video
from PIL import Image, ImageOps

from models.consisid_utils import prepare_face_models, process_face_embeddings_infer
from models.pipeline_consisid import ConsisIDPipeline
from models.transformer_consisid import ConsisIDTransformer3DModel
from util.identity_memory import (
    append_identity_episode,
    blend_identity_conditioning,
    make_identity_episode,
    retrieve_identity_episodes,
)
from util.rife_model import load_rife_model, rife_inference_with_latents
from util.utils import load_sd_upscale, upscale_batch_and_concatenate

def get_random_seed():
    return random.randint(0, 2**32 - 1)


def _validate_identity_png(identity_image_path: str):
    if not identity_image_path:
        raise ValueError("Identity image is required. Please provide --identity_image_path (PNG).")

    # Keep URL support for HF examples while enforcing PNG for local files.
    if identity_image_path.startswith(("http://", "https://")):
        return

    if not os.path.exists(identity_image_path):
        raise FileNotFoundError(f"Identity image not found: {identity_image_path}")

    if not identity_image_path.lower().endswith(".png"):
        raise ValueError(
            f"Identity image must be a PNG file for stable identity replacement: {identity_image_path}"
        )

    # Validate the image can be decoded before entering the pipeline.
    with Image.open(identity_image_path) as img:
        img.convert("RGB")


# Identity-memory helpers: keep this payload aligned with the tensors passed into ConsisIDPipeline.
def _tensor_summary(name, value):
    if torch.is_tensor(value):
        return f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    if isinstance(value, (list, tuple)):
        lines = [f"{name}: list[{len(value)}]"]
        for idx, item in enumerate(value):
            lines.append(_tensor_summary(f"{name}[{idx}]", item))
        return "\n".join(lines)
    if isinstance(value, np.ndarray):
        return f"{name}: shape={value.shape}, dtype={value.dtype}, device=n/a"
    return f"{name}: type={type(value).__name__}"


def _debug_print_identity_tensors(identity_tensors):
    print("Identity conditioning tensors:")
    for name, value in identity_tensors.items():
        print(f"  {_tensor_summary(name, value).replace(chr(10), chr(10) + '  ')}")


def _save_identity_memory(path, id_cond, id_vit_hidden, image, face_kps):
    # Store CPU tensors so the memory can be reloaded on any device/dtype later.
    id_cond_cpu = id_cond.detach().cpu()
    payload = {
        "format": "consisid_identity_memory_v1",
        "id_cond": id_cond_cpu,
        "id_ante_embedding": id_cond_cpu[:, :512],
        "id_cond_vit": id_cond_cpu[:, 512:],
        "id_vit_hidden": [tensor.detach().cpu() for tensor in id_vit_hidden],
        "face_kps": torch.as_tensor(face_kps).detach().cpu(),
        "conditioning_image": torch.from_numpy(np.array(image.convert("RGB"))).cpu(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    print(f"Saved identity memory to: {path}")


def _load_identity_memory(path, device, dtype):
    # Rehydrate the exact pipeline inputs and skip all reference-image face extraction.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if payload.get("format") != "consisid_identity_memory_v1":
        raise ValueError(f"Unsupported identity memory format in: {path}")

    id_cond = payload["id_cond"].to(device=device, dtype=dtype)
    id_vit_hidden = [tensor.to(device=device, dtype=dtype) for tensor in payload["id_vit_hidden"]]
    face_kps = payload["face_kps"].cpu()
    image_array = payload["conditioning_image"].cpu().numpy().astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(image_array))
    print(f"Loaded identity memory from: {path}")
    return id_cond, id_vit_hidden, image, face_kps


def generate_video(
    prompt: str,
    model_path: str,
    negative_prompt: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    img_file_path: str = None,
    identity_image_path: str = None,
    is_upscale: bool = False,
    is_frame_interpolation: bool = False,
    num_frames: int = 49,
    enable_model_cpu_offload: bool = False,
    enable_sequential_cpu_offload: bool = False,
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = True,
    save_identity_memory: str = None,
    load_identity_memory: str = None,
    episodic_identity_memory_path: str = None,
    episodic_top_k: int = 3,
    episodic_min_similarity: float = 0.45,
    episodic_base_weight: float = 1.0,
    episodic_exclude_exact_match: bool = True,
    episodic_exact_match_epsilon: float = 0.0,
    episodic_update_memory: bool = False,
    episodic_memory_max_episodes: int = 64,
    local_face_scale: float = None,
    prompt_enhancement_suffix: str = None,
    negative_prompt_append: str = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - negative_prompt (str): The description of the negative prompt.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - img_file_path (str): The path of the face image.
    - is_upscale (bool): Whether to apply super-resolution (video upscaling) to the generated video. Default is False.
    - is_frame_interpolation (bool): Whether to perform frame interpolation to increase the frame rate. Default is False.
    """
    # 0. Pre config
    device = "cuda"
    # Backward compatibility for old CLI flag.
    identity_image_path = identity_image_path or img_file_path
    if load_identity_memory is None:
        _validate_identity_png(identity_image_path)

    # Keep generation stable across runs when seed is set.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if os.path.exists(os.path.join(model_path, "transformer_ema")):
        subfolder = "transformer_ema"
    else:
        subfolder = "transformer"


    # 1. Prepare face models only when extracting identity from a reference image.
    if load_identity_memory is None:
        face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models(model_path, device, dtype)


    # 2. Load Pipeline.
    transformer = ConsisIDTransformer3DModel.from_pretrained(model_path, subfolder=subfolder, torch_dtype=dtype)
    if local_face_scale is not None:
        previous_scale = getattr(transformer, "local_face_scale", None)
        transformer.local_face_scale = float(local_face_scale)
        print(f"Set transformer.local_face_scale: {previous_scale} -> {transformer.local_face_scale}")
    pipe = ConsisIDPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)


    # 3. Move to device and apply optional memory-saving features.
    transformer.to(device, dtype=dtype)
    pipe.to(device)
    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    if enable_vae_slicing:
        pipe.vae.enable_slicing()
    if enable_vae_tiling:
        pipe.vae.enable_tiling()


    # 4. Prepare model input, either by extraction or by reusing a saved identity memory.
    if load_identity_memory is not None:
        id_cond, id_vit_hidden, image, face_kps = _load_identity_memory(load_identity_memory, device, dtype)
    else:
        id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2,
                                                                                eva_transform_mean, eva_transform_std,
                                                                                face_main_model, device, dtype,
                                                                                identity_image_path, is_align_face=True)
        if save_identity_memory is not None:
            _save_identity_memory(save_identity_memory, id_cond, id_vit_hidden, image, face_kps)

    raw_id_cond = id_cond
    raw_id_vit_hidden = id_vit_hidden
    if episodic_identity_memory_path is not None:
        retrieved_identity = retrieve_identity_episodes(
            episodic_identity_memory_path,
            raw_id_cond,
            top_k=episodic_top_k,
            min_similarity=episodic_min_similarity,
            exclude_exact_match=episodic_exclude_exact_match,
            exact_match_epsilon=episodic_exact_match_epsilon,
        )
        id_cond, id_vit_hidden, memory_debug = blend_identity_conditioning(
            raw_id_cond,
            raw_id_vit_hidden,
            retrieved_identity,
            base_weight=episodic_base_weight,
        )
        print(f"Episodic identity memory retrieval: {memory_debug}")
        if not retrieved_identity:
            print(
                "Warning: episodic identity memory did not retrieve any eligible episodes. "
                "Generation will use only the base identity conditioning."
            )
        if episodic_update_memory:
            episode = make_identity_episode(
                raw_id_cond,
                raw_id_vit_hidden,
                image,
                face_kps,
                source=identity_image_path or load_identity_memory,
                metadata={
                    "prompt": prompt,
                    "seed": seed,
                    "memory_debug": memory_debug,
                },
            )
            num_episodes = append_identity_episode(
                episodic_identity_memory_path,
                episode,
                max_episodes=episodic_memory_max_episodes,
                metadata={"note": "ConsisID episodic identity memory bank"},
            )
            print(f"Updated episodic identity memory bank: {episodic_identity_memory_path} ({num_episodes} episodes)")

    _debug_print_identity_tensors(
        {
            "id_cond": id_cond,
            "id_ante_embedding": id_cond[:, :512],
            "id_cond_vit": id_cond[:, 512:],
            "id_vit_hidden": id_vit_hidden,
            "face_kps": face_kps,
            "conditioning_image": torch.from_numpy(np.array(image.convert("RGB"))),
        }
    )

    if prompt_enhancement_suffix:
        prompt = f"{prompt.rstrip()} {prompt_enhancement_suffix.strip()}"
    if negative_prompt_append:
        negative_prompt = f"{(negative_prompt or '').rstrip()} {negative_prompt_append.strip()}".strip()

    prompt = prompt.strip('"')
    if negative_prompt:
        negative_prompt = negative_prompt.strip('"')


    # 5. Generate Identity-Preserving Video
    generator = torch.Generator(device).manual_seed(seed) if seed else None
    video_pt = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=generator,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        output_type="pt",
    ).frames

    del pipe
    del transformer
    free_memory()

    if is_upscale:
        print("Upscaling...")
        upscale_model = load_sd_upscale(f"{model_path}/model_real_esran/RealESRGAN_x4.pth", device)
        video_pt = upscale_batch_and_concatenate(upscale_model, video_pt, device)
    if is_frame_interpolation:
        print("Frame Interpolating...")
        frame_interpolation_model = load_rife_model(f"{model_path}/model_rife")
        video_pt = rife_inference_with_latents(frame_interpolation_model, video_pt)

    batch_size = video_pt.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = video_pt[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    # 6. Export the generated frames to a video file. fps must be 8 for original video.
    file_count = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
    video_path = f"{output_path}/{seed}_{file_count:04d}.mp4"
    export_to_video(batch_video_frames[0], video_path, fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using ConsisID")

    # ckpt arguments
    parser.add_argument("--model_path", type=str, default="ckpts", help="The path of the pre-trained model to be used")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    # input arguments
    parser.add_argument("--identity_image_path", type=str, default="asserts/example_images/2.png", help="Path to PNG identity reference image used for identity replacement.")
    parser.add_argument("--img_file_path", type=str, default=None, help="Deprecated alias for --identity_image_path.")
    parser.add_argument("--save_identity_memory", type=str, default=None, help="Save extracted identity conditioning tensors to this .pt file.")
    parser.add_argument("--load_identity_memory", type=str, default=None, help="Load identity conditioning tensors from this .pt file and skip reference-image extraction.")
    parser.add_argument("--episodic_identity_memory_path", type=str, default=None, help="Optional identity memory bank used to retrieve and blend historical identity episodes.")
    parser.add_argument("--episodic_top_k", type=int, default=3, help="Number of retrieved identity episodes to blend into conditioning.")
    parser.add_argument("--episodic_min_similarity", type=float, default=0.45, help="Minimum combined ArcFace/EVA similarity required for identity memory retrieval.")
    parser.add_argument("--episodic_base_weight", type=float, default=1.0, help="Weight assigned to the current identity features before blending retrieved memories.")
    parser.add_argument("--episodic_exclude_exact_match", action=argparse.BooleanOptionalAction, default=True, help="Skip memory episodes whose identity conditioning is numerically identical to the current query.")
    parser.add_argument("--episodic_exact_match_epsilon", type=float, default=0.0, help="Maximum identity-conditioning absolute difference treated as an exact self-match.")
    parser.add_argument("--episodic_update_memory", action="store_true", help="Append the current identity features to the episodic memory bank before generation.")
    parser.add_argument("--episodic_memory_max_episodes", type=int, default=64, help="Maximum number of episodes retained in the identity memory bank.")
    parser.add_argument("--prompt", type=str, default="The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel.")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Specify a negative prompt to guide the generation model away from certain undesired features or content.")
    parser.add_argument("--prompt_enhancement_suffix", type=str, default=None, help="Append a face-preservation suffix to the prompt for recovery attempts.")
    parser.add_argument("--negative_prompt_append", type=str, default=None, help="Append face-loss terms to the negative prompt for recovery attempts.")
    # output arguments
    parser.add_argument("--output_path", type=str, default="./output", help="The path where the generated video will be saved")
    # generation arguments
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    # auxiliary model arguments
    parser.add_argument("--is_upscale", action='store_true', help="Enable video upscaling (super-resolution) if this flag is set.")
    parser.add_argument("--is_frame_interpolation", action='store_true', help="Enable frame interpolation to increase frame rate if this flag is set.")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate (lower values reduce VRAM usage).")
    parser.add_argument("--enable_model_cpu_offload", action='store_true', help="Enable model CPU offload to reduce VRAM usage.")
    parser.add_argument("--enable_sequential_cpu_offload", action='store_true', help="Enable sequential CPU offload for maximum VRAM savings.")
    parser.add_argument("--enable_vae_slicing", action=argparse.BooleanOptionalAction, default=True, help="Enable VAE slicing to reduce VAE memory usage.")
    parser.add_argument("--enable_vae_tiling", action=argparse.BooleanOptionalAction, default=True, help="Enable VAE tiling to reduce VAE memory usage.")
    parser.add_argument("--local_face_scale", type=float, default=None, help="Override transformer.local_face_scale at inference time.")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print("Base Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=args.model_path)
    else:
        print(f"Base Model already exists in {args.model_path}, skipping download.")

    if args.is_upscale and not os.path.exists(f"{args.model_path}/model_rife"):
        print("Upscale Model not found, downloading from Hugging Face...")
        snapshot_download(repo_id="AlexWortega/RIFE", local_dir=f"{args.model_path}/model_rife")
    else:
        print(f"Upscale Model already exists in {args.model_path}, skipping download.")

    if args.is_frame_interpolation and not os.path.exists(f"{args.model_path}/model_real_esran"):
        print("Frame Interpolation Model not found, downloading from Hugging Face...")
        hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir=f"{args.model_path}/model_real_esran")
    else:
        print(f"Frame Interpolation Model already exists in {args.model_path}, skipping download.")

    generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
        seed=args.seed,
        img_file_path=args.img_file_path,
        identity_image_path=args.identity_image_path,
        is_upscale=args.is_upscale,
        is_frame_interpolation=args.is_frame_interpolation,
        num_frames=args.num_frames,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        enable_sequential_cpu_offload=args.enable_sequential_cpu_offload,
        enable_vae_slicing=args.enable_vae_slicing,
        enable_vae_tiling=args.enable_vae_tiling,
        save_identity_memory=args.save_identity_memory,
        load_identity_memory=args.load_identity_memory,
        episodic_identity_memory_path=args.episodic_identity_memory_path,
        episodic_top_k=args.episodic_top_k,
        episodic_min_similarity=args.episodic_min_similarity,
        episodic_base_weight=args.episodic_base_weight,
        episodic_exclude_exact_match=args.episodic_exclude_exact_match,
        episodic_exact_match_epsilon=args.episodic_exact_match_epsilon,
        episodic_update_memory=args.episodic_update_memory,
        episodic_memory_max_episodes=args.episodic_memory_max_episodes,
        local_face_scale=args.local_face_scale,
        prompt_enhancement_suffix=args.prompt_enhancement_suffix,
        negative_prompt_append=args.negative_prompt_append,
    )
