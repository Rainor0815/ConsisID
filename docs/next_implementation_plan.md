# Next Implementation Plan: Face-Preservation Recovery for ConsisID

Date: 2026-06-11

Recommended next coding task: implement a detector-guided face-preservation / generation-time recovery loop. Do not start with latent-history retrieval or model attention surgery. The last experiment showed identity similarity improved when faces existed, but chunks 1 and 2 had zero detected faces. The next experiment should prevent or recover from face disappearance.

## Objective

Generate sequential ConsisID chunks while enforcing face visibility and identity consistency at generation time. A chunk should not be accepted into the final long video unless it passes face detection and identity thresholds.

## Files Likely To Modify

- `infer.py`
  - Add CLI flags for runtime face/identity conditioning knobs:
    - `--local_face_scale`
    - `--prompt_enhancement_suffix`
    - `--negative_prompt_append`
    - optional `--conditioning_image_override` or clean-crop identity override if safe.
  - Pass `local_face_scale` into `transformer.local_face_scale` after loading the transformer.

- `eval/run_identity_persistence_experiment.py`
  - Extend rerun policy beyond current fixed rerun:
    - multiple recovery profiles;
    - seed offsets;
    - prompt suffixes;
    - negative prompt variants;
    - local face scale schedule;
    - accept/reject final segment by detection first.
  - Write `segment_recovery_attempts.csv/json`.

- `tools/guarded_memory_update_from_video.py`
  - Keep as online memory guard.
  - Add bbox area and detection confidence to accepted-frame report if available.

- `eval/arcface_identity_stability.py`
  - Add optional face bbox metrics:
    - mean face area ratio;
    - min face area ratio;
    - face center variance;
    - max consecutive missing frames.

- `util/identity_memory.py`
  - Add optional metadata helpers for face pose/quality once recovery loop works.
  - Not required for the first patch unless needed for clean-crop selection.

- New file: `util/face_recovery.py`
  - Proposed helper functions:
    - `build_recovery_profiles(base_prompt, negative_prompt, segment_index, args)`
    - `segment_passes_face_guard(summary, thresholds)`
    - `rank_segment_attempt(summary)`
    - `format_recovery_reason(summary)`

- New file: `eval/identity_prompt_suite_face_recovery.json`
  - Small prompt suite focused on known failure modes:
    - steady portrait;
    - viewpoint turn;
    - occlusion/reappearance;
    - background shift.

## Proposed Modules And Functions

### `util/face_recovery.py`

```python
def build_recovery_profiles(prompt, negative_prompt, segment_index, args):
    """Return ordered recovery attempts with prompt, negative prompt, seed offset, top_k, base_weight, local_face_scale."""
```

```python
def segment_passes_face_guard(summary, min_detection_rate, min_mean_similarity, min_frame_similarity=None):
    """Decide whether a generated segment is acceptable for the final long video."""
```

```python
def rank_segment_attempt(summary):
    """Sort attempts by detection rate, then mean similarity, then min similarity."""
```

### `infer.py`

```python
parser.add_argument("--local_face_scale", type=float, default=None)
parser.add_argument("--prompt_enhancement_suffix", type=str, default=None)
parser.add_argument("--negative_prompt_append", type=str, default=None)
```

Inside `generate_video()` after transformer load:

```python
if local_face_scale is not None:
    transformer.local_face_scale = local_face_scale
```

### `eval/run_identity_persistence_experiment.py`

Add CLI flags:

```text
--face_recovery
--face_recovery_max_attempts
--face_recovery_min_detection_rate
--face_recovery_min_mean_similarity
--face_recovery_seed_stride
--face_recovery_prompt_suffix
--face_recovery_negative_prompt
--face_recovery_local_face_scales
--face_recovery_top_ks
--face_recovery_base_weights
```

Rerun loop policy:

1. Generate segment attempt.
2. Evaluate segment immediately.
3. If pass, accept.
4. If fail, generate next recovery profile.
5. Select best attempt by detection-first ranking.
6. Only append memory from accepted attempts that pass stricter online memory guards.

## Experiment Configs

### Quick smoke

- Output: `output/identity_persistence_face_recovery_smoke_steady_3x49_seed42`
- Prompt suite: `eval/identity_prompt_suite_face_recovery.json`
- Category: `realistic`
- Prompt: `steady_portrait`
- Seed: `42`
- Segments: `3`
- Frames per segment: `49`
- Steps: `50`
- Base mode: episodic with `output/identity_memory_bank_multi_reference.pt`

Suggested recovery profiles:

| Attempt | Seed offset | top_k | base_weight | local_face_scale | Prompt / negative change |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 2 to 5 schedule | 1.0 to 0.35 schedule | default | original prompt |
| 1 | 101 | 6 | 0.25 | 1.25x current | add face-visible suffix and negative face-loss prompt |
| 2 | 211 | 6 | 0.20 | 1.50x current | closer portrait fallback, stricter negative prompt |

### Broader follow-up

- Output: `output/identity_persistence_face_recovery_realistic_3x49_seed42`
- Prompt suite: `eval/identity_prompt_suite.json`
- Categories: `realistic`
- Prompts: steady portrait, viewpoint turn, occlusion/reappearance, background shift.
- Compare modes:
  - baseline;
  - current episodic;
  - episodic plus face recovery.

## Metrics To Track

Primary:

- Overall detection rate.
- Per-chunk detection rate.
- Number of zero-face chunks.
- Maximum consecutive missing detected frames.
- Mean ArcFace similarity on detected frames.
- Min ArcFace similarity on detected frames.

Secondary:

- Face bbox area ratio mean/min.
- Face center stability.
- Number of rerun attempts per segment.
- Acceptance rate by attempt index.
- Guarded memory accepted/rejected frame count.
- Runtime overhead versus current episodic run.
- Video MD5 uniqueness to verify attempts are not identical.

Optional:

- CLIP/prompt score.
- Face pose/yaw bucket coverage if pose estimates are available from existing face models.
- Manual inspection thumbnails/contact sheet for accepted attempts.

## Expected Success Criteria

For the first steady-portrait 3x49 smoke test:

- Zero-face chunks: reduce from 2/3 to 0/3.
- Overall detection rate: at least 0.75.
- Per-chunk detection rate: at least 0.50 for every chunk.
- Mean ArcFace similarity: at least 0.55.
- Min ArcFace similarity: at least 0.42.
- Guarded online memory: no low-detection segment admitted.
- Runtime: no more than 3 attempts per segment in smoke testing.

For the broader realistic suite:

- Episodic plus recovery should beat current episodic on detection rate and no-face chunk count.
- Identity similarity should not drop below current episodic by more than 0.03 mean ArcFace.
- Memory guard rejection should remain conservative.

## Estimated Risk Level

- Prompt/negative prompt rerun profiles: low risk.
- Detection-first acceptance and attempt ranking: low risk.
- Runtime `local_face_scale` schedule: medium risk. It may over-constrain the video or create face copy-paste artifacts.
- Conditioning on generated clean crops: medium risk. Guardrails must remain strict to avoid poisoning memory.
- Gradient-based ID-aware guidance: high risk. It likely increases VRAM use and may require differentiable face-feature extraction or proxy losses.
- Latent/query injection: high risk. It touches model internals and can destabilize generation.

## Recommended Stop Point For The Next Patch

The next patch should stop after implementing and testing the low/medium-risk recovery loop:

1. Add recovery profiles and detection-first segment acceptance.
2. Expose `local_face_scale` as a runtime option.
3. Add face bbox/missing-streak metrics if straightforward.
4. Run the steady-portrait 3x49 smoke test.
5. Compare against the last result:
   - current stronger schedule detection rate: 0.2313;
   - current zero-face chunks: 2/3;
   - current mean similarity on detected frames: 0.5678.

Do not implement latent-history retrieval or query injection until the face-preservation recovery loop proves it can remove zero-face chunks.

