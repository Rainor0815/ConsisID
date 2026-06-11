import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


SINGLE_MEMORY_FORMAT = "consisid_identity_memory_v1"
BANK_MEMORY_FORMAT = "consisid_identity_memory_bank_v1"


def _safe_torch_load(path: str) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_cpu_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _as_float_tensor(value) -> torch.Tensor:
    return _to_cpu_tensor(value).to(torch.float32)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _as_float_tensor(a).flatten()
    b = _as_float_tensor(b).flatten()
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if denom.item() == 0:
        return 0.0
    return float(torch.dot(a, b).item() / denom.item())


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    a = _as_float_tensor(a)
    b = _as_float_tensor(b)
    if a.shape != b.shape:
        return float("inf")
    return float((a - b).abs().max().item())


def _identity_parts(id_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    id_cond = _as_float_tensor(id_cond)
    return id_cond[:, :512], id_cond[:, 512:]


def identity_conditioning_similarity(left_id_cond: torch.Tensor, right_id_cond: torch.Tensor) -> Dict[str, float]:
    left_ante, left_vit = _identity_parts(left_id_cond)
    right_ante, right_vit = _identity_parts(right_id_cond)
    arcface_similarity = _cosine_similarity(left_ante, right_ante)
    vit_similarity = _cosine_similarity(left_vit, right_vit)
    return {
        "arcface_similarity": arcface_similarity,
        "vit_similarity": vit_similarity,
        "score": 0.8 * arcface_similarity + 0.2 * vit_similarity,
    }


def make_identity_episode(
    id_cond: torch.Tensor,
    id_vit_hidden: List[torch.Tensor],
    image,
    face_kps,
    source: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    id_cond_cpu = _to_cpu_tensor(id_cond)
    id_ante_embedding, id_cond_vit = _identity_parts(id_cond_cpu)
    episode = {
        "id_cond": id_cond_cpu,
        "id_ante_embedding": id_ante_embedding,
        "id_cond_vit": id_cond_vit,
        "id_vit_hidden": [_to_cpu_tensor(tensor) for tensor in id_vit_hidden],
        "face_kps": _to_cpu_tensor(face_kps),
        "source": source,
        "metadata": metadata or {},
    }
    if image is not None:
        episode["conditioning_image"] = torch.from_numpy(np.array(image.convert("RGB"))).cpu()
    return episode


def _single_payload_to_episode(payload: Dict, source: str) -> Dict:
    if "id_ante_embedding" not in payload and "id_cond" in payload:
        payload["id_ante_embedding"] = payload["id_cond"][:, :512]
    if "id_cond_vit" not in payload and "id_cond" in payload:
        payload["id_cond_vit"] = payload["id_cond"][:, 512:]
    return {
        "id_cond": _to_cpu_tensor(payload["id_cond"]),
        "id_ante_embedding": _to_cpu_tensor(payload["id_ante_embedding"]),
        "id_cond_vit": _to_cpu_tensor(payload["id_cond_vit"]),
        "id_vit_hidden": [_to_cpu_tensor(tensor) for tensor in payload["id_vit_hidden"]],
        "face_kps": _to_cpu_tensor(payload["face_kps"]),
        "conditioning_image": _to_cpu_tensor(payload["conditioning_image"])
        if "conditioning_image" in payload
        else None,
        "source": source,
        "metadata": {"loaded_from_single_memory": True},
    }


def load_identity_episodes(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    payload = _safe_torch_load(path)
    payload_format = payload.get("format")
    if payload_format == BANK_MEMORY_FORMAT:
        return payload.get("episodes", [])
    if payload_format == SINGLE_MEMORY_FORMAT or "id_cond" in payload:
        return [_single_payload_to_episode(payload, path)]
    raise ValueError(f"Unsupported identity memory format in: {path}")


def save_identity_episodes(path: str, episodes: List[Dict], metadata: Optional[Dict] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "format": BANK_MEMORY_FORMAT,
            "episodes": episodes,
            "metadata": metadata or {},
        },
        path,
    )


def append_identity_episode(
    path: str,
    episode: Dict,
    max_episodes: Optional[int] = None,
    metadata: Optional[Dict] = None,
) -> int:
    episodes = load_identity_episodes(path)
    episodes.append(episode)
    if max_episodes is not None and max_episodes > 0 and len(episodes) > max_episodes:
        episodes = episodes[-max_episodes:]
    save_identity_episodes(path, episodes, metadata=metadata)
    return len(episodes)


def retrieve_identity_episodes(
    path: str,
    query_id_cond: torch.Tensor,
    top_k: int = 3,
    min_similarity: float = 0.45,
    exclude_exact_match: bool = True,
    exact_match_epsilon: float = 0.0,
) -> List[Dict]:
    if not path or top_k <= 0:
        return []

    query_ante, query_vit = _identity_parts(query_id_cond)
    scored = []
    for episode in load_identity_episodes(path):
        if exclude_exact_match and _max_abs_diff(query_id_cond, episode["id_cond"]) <= exact_match_epsilon:
            continue
        arcface_similarity = _cosine_similarity(query_ante, episode["id_ante_embedding"])
        vit_similarity = _cosine_similarity(query_vit, episode["id_cond_vit"])
        score = 0.8 * arcface_similarity + 0.2 * vit_similarity
        if score >= min_similarity:
            scored.append(
                {
                    "score": score,
                    "arcface_similarity": arcface_similarity,
                    "vit_similarity": vit_similarity,
                    "episode": episode,
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def blend_identity_conditioning(
    base_id_cond: torch.Tensor,
    base_id_vit_hidden: List[torch.Tensor],
    retrieved: List[Dict],
    base_weight: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
    if not retrieved:
        return base_id_cond, base_id_vit_hidden, {"num_retrieved": 0, "scores": []}

    device = base_id_cond.device
    dtype = base_id_cond.dtype
    weights = [max(float(base_weight), 0.0)] + [max(float(item["score"]), 0.0) for item in retrieved]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return base_id_cond, base_id_vit_hidden, {"num_retrieved": 0, "scores": []}

    normalized_weights = [weight / weight_sum for weight in weights]
    blended_id_cond = base_id_cond * normalized_weights[0]
    for weight, item in zip(normalized_weights[1:], retrieved):
        blended_id_cond = blended_id_cond + item["episode"]["id_cond"].to(device=device, dtype=dtype) * weight

    blended_hidden = []
    for hidden_index, base_hidden in enumerate(base_id_vit_hidden):
        blended = base_hidden * normalized_weights[0]
        for weight, item in zip(normalized_weights[1:], retrieved):
            memory_hidden = item["episode"]["id_vit_hidden"][hidden_index].to(device=base_hidden.device, dtype=base_hidden.dtype)
            blended = blended + memory_hidden * weight
        blended_hidden.append(blended)

    return (
        blended_id_cond,
        blended_hidden,
        {
            "num_retrieved": len(retrieved),
            "scores": [
                {
                    "score": item["score"],
                    "arcface_similarity": item["arcface_similarity"],
                    "vit_similarity": item["vit_similarity"],
                    "source": item["episode"].get("source"),
                }
                for item in retrieved
            ],
            "weights": normalized_weights,
        },
    )
