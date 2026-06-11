# Research: Next Methods for ConsisID Long-Horizon Identity Memory

Date: 2026-06-11

This report surveys recent long-video consistency, world-memory, retrieval, and identity-preservation methods that could inform the next ConsisID identity-memory experiment. The current ConsisID result to explain is: stronger late identity conditioning improved ArcFace similarity on detected frames, but later chunks still collapsed to zero face detections. That points to a missing face-preservation / generation-time recovery loop, not only a missing identity embedding bank.

## Current ConsisID Integration Context

Relevant local entry points:

- `infer.py`: extracts or loads identity tensors, retrieves identity episodes, blends `id_cond` / `id_vit_hidden`, then calls `ConsisIDPipeline`.
- `util/identity_memory.py`: stores identity episodes and retrieves by ArcFace plus VIT/EVA similarity.
- `tools/guarded_memory_update_from_video.py`: admits generated face crops only after detection and similarity guards pass.
- `eval/run_identity_persistence_experiment.py`: can run sequential chunks, evaluate each segment, rerun failed segments, and gate online memory updates.
- `models/pipeline_consisid.py`: denoising loop where latent callbacks, latent state persistence, or guidance schedules can be inserted.
- `models/transformer_consisid.py`: Local Facial Extractor cross-attention uses `id_cond` and `id_vit_hidden`; runtime `local_face_scale` or face-expert weighting could be exposed.

## Method Reviews

### 1. VideoMemory: Toward Consistent Video Generation via Memory Integration

- Link: https://arxiv.org/abs/2601.03655 and project page https://hit-perfect.github.io/VideoMemory/
- Main problem it solves: multi-shot narrative videos lose consistent characters, props, and backgrounds across distant shots.
- Key mechanism: entity-centric Dynamic Memory Banks for characters, props, and backgrounds. A storyboard/memory agent retrieves or creates entity references per shot, generates keyframes, then synthesizes shots. The memory bank is updated after shots while preserving entity identity.
- Training or inference-time: mostly an orchestration framework around external agents and generators; does not require full retraining of the base video generator in principle, though its full stack depends on keyframe and video models.
- Useful part for ConsisID: make identity memory explicitly entity-state based rather than a flat list of identity tensors. Store not only `id_cond`, but also selected reference images, generated clean crops, face bbox/pose/quality, prompt/shot state, and acceptance reason.
- Implementation difficulty: medium. The memory schema and retrieval policy are feasible now; keyframe-agent synthesis is larger.
- Risks with ConsisID: if the video generator ignores retrieved visual references or if face visibility collapses, a memory bank alone will not recover the subject. The previous run already showed that clean retrieval can still produce zero-face chunks.
- Concrete adaptation idea: add a `SubjectStateMemory` JSON/PT sidecar next to the identity bank with slots for base references, accepted generated crops, face pose, bbox area, detection confidence, and prompt state. Use it to select identity episodes plus a face-visible prompt/recovery policy per segment.

### 2. LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation

- Link: https://arxiv.org/abs/2606.02553 and code https://github.com/qixinhu11/LongLive-RAG
- Main problem it solves: autoregressive long video generation accumulates errors because sliding-window context only conditions on recent, potentially degraded latents.
- Key mechanism: self-generated latent history becomes a searchable memory. Each completed latent block is indexed, the next block queries the history, and top-K relevant historical latents are inserted into the context. The released implementation keeps the base generator frozen and trains only a small retrieval autoencoder/encoder.
- Training or inference-time: base generator frozen, but retrieval encoder training is used. A simpler heuristic latent/frame retrieval variant could be inference-only.
- Useful part for ConsisID: our current episodic identity retrieval operates only before generation. LongLive-RAG suggests retrieving non-local clean history during generation or per segment, instead of relying on the last chunk.
- Implementation difficulty: medium to high. A heuristic version that stores decoded frames/crops is medium; true latent-context insertion requires pipeline and attention changes.
- Risks with ConsisID: ConsisID is not an autoregressive latent-history model in the same way. The current pipeline generates one 49-frame clip at a time and concatenates outputs, so there is no native long-context latent attention to extend.
- Concrete adaptation idea: first build "LongLive-lite" for ConsisID: save generated latents or decoded clean face crops from successful chunks, retrieve the best previous clean face evidence for each new chunk, and condition via existing identity tensors plus prompt/reference augmentation. Defer true latent insertion.

### 3. WorldMem: Long-term Consistent World Simulation with Memory

- Link: https://arxiv.org/abs/2504.12369
- Main problem it solves: limited temporal context causes world simulation failures, especially 3D spatial consistency over long temporal/viewpoint gaps.
- Key mechanism: memory units store memory frames and states, such as pose and timestamp. A memory attention mechanism retrieves relevant memory frames based on state.
- Training or inference-time: architecture-level memory attention, so it generally requires model changes/training.
- Useful part for ConsisID: the state-conditioned memory idea. For faces, state can be approximate yaw/pitch, bbox size, frame index, prompt scene, and detection confidence.
- Implementation difficulty: medium for metadata-aware retrieval; high for memory attention inside the model.
- Risks with ConsisID: no camera pose/action state is available by default. Face pose estimates from InsightFace may be noisy and cannot fully describe scene state.
- Concrete adaptation idea: add pose-aware identity episode retrieval. Query with desired/current prompt segment plus target face pose bucket; retrieve references with similar pose when available, otherwise front-facing high-confidence references.

### 4. WorldPack: Compressed Memory Improves Spatial Consistency in Video World Modeling

- Link: https://arxiv.org/abs/2512.02473
- Main problem it solves: long-context world models are expensive and lose spatial consistency over long rollouts.
- Key mechanism: compressed memory with trajectory packing and memory retrieval, improving spatial consistency while using much shorter context.
- Training or inference-time: architecture/training oriented.
- Useful part for ConsisID: compress memory instead of appending every episode. Keep prototypes: frontal face, side face, smiling face, high-quality generated crop, and reject redundant/low-quality entries.
- Implementation difficulty: low to medium for memory compression policy; high for model-native compressed memory.
- Risks with ConsisID: over-compressing identity memory can discard rare pose/expression evidence needed later.
- Concrete adaptation idea: implement identity-bank compaction: cluster accepted episodes by ArcFace plus pose; retain one or two highest-quality representatives per cluster and retrieval diversity constraints.

### 5. MemoryPack / Pack and Force Your Memory: Long-form and Consistent Video Generation

- Link: https://arxiv.org/abs/2510.01784
- Main problem it solves: long-form video generation needs long-range dependencies and must reduce autoregressive error accumulation.
- Key mechanism: MemoryPack uses learnable context retrieval from textual and image information as global guidance; Direct Forcing reduces training-inference mismatch to curb error propagation.
- Training or inference-time: requires learned context retrieval and training changes; not a direct plug-in.
- Useful part for ConsisID: retrieve both text state and visual identity state. Current retrieval ignores prompt/shot context except metadata.
- Implementation difficulty: medium for a lightweight text+image retrieval score; high for Direct Forcing.
- Risks with ConsisID: Direct Forcing is not practical without training. Text-image retrieval could retrieve visually similar but wrong-pose references if poorly weighted.
- Concrete adaptation idea: add retrieval scoring terms: identity similarity, face quality, face pose, prompt compatibility, recency, and diversity. Use this to pick identity evidence for the next chunk.

### 6. VRAG: Learning World Models for Interactive Video Generation

- Link: https://arxiv.org/abs/2505.21996
- Main problem it solves: interactive video world models suffer compounding errors and insufficient memory mechanisms.
- Key mechanism: Video Retrieval-Augmented Generation with explicit global state conditioning. The paper argues naive longer context and naive retrieval can be insufficient without global state.
- Training or inference-time: architecture/training oriented, but the global-state idea can be adapted at inference time.
- Useful part for ConsisID: make global identity/face-presence state explicit: subject present, face visible, bbox area, frontal/profile, identity score, and memory freshness.
- Implementation difficulty: low for explicit state tracking and recovery decisions; high for true model conditioning on global state vectors.
- Risks with ConsisID: state tracking helps decide reruns, but the generator may not obey state unless prompts, reference image, or guidance are changed.
- Concrete adaptation idea: add a per-segment `identity_state.json` and recovery policy. If the predicted/observed state is "face disappeared", rerun with face-visible prompt constraints, stronger face guidance, different seed, and stricter output acceptance.

### 7. Video Storyboarding / Motion by Queries

- Link: https://arxiv.org/abs/2412.07750 and project page https://research.nvidia.com/labs/par/motion-by-queries/
- Main problem it solves: multi-shot character consistency without training while preserving natural motion.
- Key mechanism: self-attention query features encode motion, structure, and identity. Controlled Q-feature injection can share identity across shots while reducing the identity-motion tradeoff.
- Training or inference-time: training-free, but requires access to internal attention/query tensors.
- Useful part for ConsisID: a face-preserving "clean query anchor" from a successful chunk may be injected or blended in later chunks.
- Implementation difficulty: high for exact Q injection in ConsisID; medium if approximated as latent or hidden-state anchoring via callbacks.
- Risks with ConsisID: wrong layer/timestep injection can freeze motion, copy the face, or destabilize DiT hidden states. Face ROI token mapping is non-trivial.
- Concrete adaptation idea: start with a diagnostic hook that stores selected hidden states or attention inputs from a successful face-visible chunk, without using them. Then test limited injection only in late denoising steps and only on face-region tokens.

### 8. Collaborative Face Experts Fusion / Mixture of Facial Experts for Large Face Poses

- Link: https://arxiv.org/abs/2508.09476
- Main problem it solves: identity-preserving video generation fails under large face poses.
- Key mechanism: complementary face experts inside the DiT: identity expert for cross-pose invariant identity, semantic expert for high-level context, and detail expert for texture/color. A curated large-face-pose dataset improves training coverage.
- Training or inference-time: primarily requires model architecture and training.
- Useful part for ConsisID: ConsisID already has identity conditioning and Local Facial Extractor. We can emulate a small part by separating identity, semantic/VIT, and detail/reference-image weights during retrieval.
- Implementation difficulty: medium for retrieval-side expert weighting; high for actual expert modules.
- Risks with ConsisID: without training, manually reweighting embeddings may not produce the intended expert behavior.
- Concrete adaptation idea: expose separate blend weights for ArcFace identity (`id_ante_embedding`) and VIT/detail tokens (`id_cond_vit`, `id_vit_hidden`) instead of a single blended `id_cond` weight. Test whether stronger ArcFace but moderate VIT detail reduces copy-paste and drift.

### 9. Mv2ID: Identity-Consistent Video Generation under Large Facial-Angle Variations

- Link: https://arxiv.org/abs/2603.21299
- Main problem it solves: single-view reference-to-video methods fail under large face-angle changes; naive multi-view references can create view-dependent copy-paste artifacts.
- Key mechanism: multi-view conditioning, region-masking training to prevent shortcut learning, and reference-decoupled RoPE for separate positional treatment of video and reference tokens.
- Training or inference-time: training required.
- Useful part for ConsisID: multi-reference banks are useful, but must avoid naive averaging/copy-paste. Retrieval should be pose-aware and diversity-aware.
- Implementation difficulty: low to medium for pose-aware retrieval; high for RoPE/region masking training.
- Risks with ConsisID: our multi-reference blending can blur identity or overfit to the wrong view if the prompt calls for a different pose.
- Concrete adaptation idea: add face-pose buckets to `make_identity_episode()` metadata and retrieve references by pose diversity. Blend frontal identity with one nearest pose reference instead of all top-K.

### 10. TPIGE: Identity-Preserving Text-to-Video Generation via Training-Free Prompt, Image, and Guidance Enhancement

- Link: https://arxiv.org/abs/2509.01362 and code https://github.com/Andyplus1/IPT2V
- Main problem it solves: preserving identity in text-to-video without expensive fine-tuning.
- Key mechanism: Face-Aware Prompt Enhancement, Prompt-Aware Reference Image Enhancement, and ID-Aware Spatiotemporal Guidance Enhancement using unified gradients.
- Training or inference-time: training-free.
- Useful part for ConsisID: this is closest to the current likely next experiment. Our failure mode is not just low identity score; it is face disappearance. Prompt/image/guidance enhancement can explicitly preserve face visibility and identity during sampling.
- Implementation difficulty: low for prompt and rerun policy; medium to high for gradient-based ID-aware guidance inside the denoising loop.
- Risks with ConsisID: gradient guidance requires differentiable face/identity losses or proxy features at generation time, which increases memory and may conflict with bfloat16 inference.
- Concrete adaptation idea: implement inference-time face-recovery reruns first: prompt enhancement with face visibility, negative prompts for back-of-head/occlusion, seed sweep, local face scale schedule, and output acceptance by detection/ArcFace. Later test gradient guidance.

### 11. FreeLong / FreeLong++ / FreeSpec / FreeNoise

- Links: FreeLong++ https://arxiv.org/abs/2507.00162, FreeLong https://arxiv.org/abs/2407.19918, FreeNoise https://arxiv.org/abs/2310.15169, FreeSpec https://arxiv.org/abs/2605.06509
- Main problem it solves: extending short-video diffusion models to longer sequences without retraining often degrades temporal consistency and visual fidelity.
- Key mechanism: training-free temporal attention/noise/spectral strategies. FreeLong blends global low-frequency and local high-frequency features; FreeNoise reschedules noise for long-range correlation; FreeSpec reconstructs global/local spectral components.
- Training or inference-time: training-free.
- Useful part for ConsisID: if face disappearance is partly a temporal extension artifact, training-free long-context feature/noise handling may help stabilize layout and subject presence.
- Implementation difficulty: high inside ConsisID because the current experiment creates separate 49-frame chunks rather than one extended denoising window. Lower difficulty if used only as a seed/noise scheduling idea.
- Risks with ConsisID: these methods preserve global temporal consistency but do not specifically enforce face identity or face visibility.
- Concrete adaptation idea: implement cross-chunk noise/seed correlation and first-frame/layout carryover before attempting full attention-branch modifications.

### 12. Mixture of Contexts for Long Video Generation

- Link: https://arxiv.org/abs/2508.21058
- Main problem it solves: long video generation is a long-context memory problem, but full self-attention is too expensive.
- Key mechanism: learnable sparse attention routing. Each query selects informative history chunks plus mandatory anchors such as caption and local windows.
- Training or inference-time: requires model training or at least attention-routing changes.
- Useful part for ConsisID: retrieval should include mandatory anchors. For identity, mandatory anchors are base identity and cleanest known face crop, while optional anchors are similar pose/recent successful frames.
- Implementation difficulty: low for anchor policy in memory retrieval; high for sparse attention routing.
- Risks with ConsisID: without native long-context attention, routing cannot be fully applied.
- Concrete adaptation idea: make base identity and best clean face episode non-droppable anchors during retrieval, then add top-K context episodes around them.

### 13. StoryBooth: Training-free Multi-Subject Consistency for Improved Visual Storytelling

- Link: https://arxiv.org/abs/2504.05800
- Main problem it solves: multi-subject consistency suffers attention leakage between subjects.
- Key mechanism: region-based subject localization, bounded cross-frame self-attention, and token merging for fine-grained detail consistency.
- Training or inference-time: training-free, but requires modifying diffusion attention layers.
- Useful part for ConsisID: region-bounded identity sharing could prevent identity guidance from affecting non-face/background tokens and help preserve face details.
- Implementation difficulty: high for bounded attention; medium for region metadata and face ROI masks.
- Risks with ConsisID: we only have one subject in the current experiment, so attention leakage is less central than face disappearance.
- Concrete adaptation idea: add face ROI tracking and use it for diagnostics first. If stable, use ROI masks to limit future identity/attention injection to face tokens.

### 14. STAGE: Storyboard-Anchored Generation for Cinematic Multi-shot Narrative

- Link: https://arxiv.org/abs/2512.12372
- Main problem it solves: keyframe-based multi-shot generation lacks cross-shot consistency and cinematic structure.
- Key mechanism: structural storyboard with start-end frame pairs, multi-shot memory pack, dual encoding, and training for cinematic transitions.
- Training or inference-time: larger trained workflow.
- Useful part for ConsisID: explicit start/end keyframe structure could be useful for face-presence recovery: each chunk should have a valid face-visible anchor frame before video synthesis is accepted.
- Implementation difficulty: medium if simplified to keyframe validation; high for full STAGE.
- Risks with ConsisID: ConsisID does not currently generate explicit storyboard keyframes before video.
- Concrete adaptation idea: add a preflight keyframe/reference check: before generating a chunk, choose a face-visible reference crop and prompt variant; after generation, accept only if the start/end sampled frames contain the subject face.

### 15. Corgi / Cached Latent Memory Guided Video Generation

- Link: no primary source found under this exact name and description during this pass.
- Main problem it likely targets: reusing cached latent/feature history to stabilize or accelerate video generation.
- Key mechanism: not verified. The closest verified methods are LongLive-RAG for searchable latent history and recent diffusion feature caching/spectral methods for efficient/stable latent reuse.
- Training or inference-time: unknown.
- Useful part for ConsisID: cached clean latent/feature anchors are still a promising direction, but should be grounded in LongLive-RAG, Video Storyboarding, or FreeLong/FreeSpec rather than an unverified citation.
- Implementation difficulty: unknown for Corgi specifically; medium to high for cached latent anchoring.
- Risks with ConsisID: implementing from an unverified method risks building the wrong mechanism.
- Concrete adaptation idea: name the next local prototype `latent_history_retrieval` instead of `corgi`, and base it on saved ConsisID latents plus LongLive-RAG-style retrieval.

## What The Literature Suggests For Our Immediate Problem

The previous ConsisID experiment failed mainly because later chunks had no detectable face. Methods that only retrieve identity embeddings cannot fix a chunk where the subject face is absent. The most relevant sources for the immediate next step are:

1. TPIGE: because it directly proposes training-free prompt/image/guidance enhancement for identity-preserving video generation.
2. VideoMemory and VRAG: because they argue for explicit entity/global state tracking, not just passive retrieval.
3. LongLive-RAG and WorldMem: because they support non-local memory retrieval, but their full latent/memory-attention form is a larger change.

Therefore the likely best next experiment is still a face-preservation / generation-time recovery mechanism, with explicit state and guarded acceptance. Other papers do not suggest a better immediate first step; they suggest better medium and large follow-ups after the recovery loop exists.

## Top 3 Implementation Directions

### Direction 1: Detector-Guided Face-Preservation Recovery Loop

- Category: quick inference-time fix.
- Core idea: treat face visibility as a hard acceptance condition, not only an evaluation metric. If a segment has low face detection, rerun with prompt enhancement, negative prompt constraints, seed variation, stronger face module scale, and selected clean face evidence.
- Why this is top priority: our measured failure is zero-face chunks. This direction directly targets the bottleneck.
- Expected implementation effort: 1-2 focused coding sessions.
- Primary inspirations: TPIGE, VideoMemory global/entity state, VRAG explicit state.

### Direction 2: State-Aware Subject Memory Bank

- Category: medium-size engineering change.
- Core idea: evolve the identity bank into a stateful subject memory that stores identity tensors plus face crop, bbox, pose bucket, quality, source segment, prompt state, and acceptance reason. Retrieval becomes identity plus state plus quality, not only cosine similarity.
- Why this matters: multi-reference identity memory should retrieve the right evidence for the next shot/chunk, and should avoid ambiguous generated frames.
- Expected implementation effort: several sessions.
- Primary inspirations: VideoMemory, WorldMem, WorldPack, MemoryPack, Mv2ID.

### Direction 3: Latent / Hidden-State History Retrieval

- Category: larger research architecture change.
- Core idea: save clean latent or hidden-state anchors from successful chunks and retrieve them for future chunks. Start with diagnostics and non-invasive latent storage; later test attention/query/latent injection.
- Why this matters: identity embeddings may be too weak to preserve layout and face presence. Latent history can carry non-local visual context.
- Expected implementation effort: high; should follow the recovery loop and state memory.
- Primary inspirations: LongLive-RAG, Video Storyboarding, Mixture of Contexts, WorldMem.

## Recommendation by Scope

### A. Quick Inference-Time Fixes We Can Implement Now

- Add face-presence acceptance criteria before a segment is allowed into the final joined video.
- Add rerun modes:
  - stronger face-visible prompt suffix;
  - negative prompt for back-of-head, cropped face, occlusion, face out of frame, duplicate faces;
  - seed offset sweep;
  - lower motion / closer portrait prompt fallback;
  - stronger `local_face_scale` if exposed safely;
  - choose the best attempt by detection rate first, then mean/min similarity.
- Use selected previous clean face crops as retrieval references only if they pass guard thresholds.
- Track face bbox area and face center, not only detection rate.

### B. Medium-Size Engineering Changes

- Build `SubjectStateMemory` with pose/quality/retrieval metadata.
- Add pose-aware and diversity-aware retrieval:
  - base identity as mandatory anchor;
  - best clean generated crop as optional anchor;
  - nearest pose reference if available;
  - cap redundant same-pose episodes.
- Add a keyframe-like preflight/recovery workflow:
  - generate or choose a face-visible reference crop for each segment;
  - validate start/end sampled frames;
  - rerun if subject disappears.
- Add richer evaluation:
  - detection streak failures;
  - no-face chunk count;
  - face bbox area stability;
  - identity-memory poisoning rejection rate.

### C. Larger Research Architecture Changes

- LongLive-RAG-style latent history retrieval for ConsisID:
  - save latents from successful chunks;
  - train or approximate a retrieval encoder;
  - condition later chunks on retrieved latent history.
- Video Storyboarding-style query/hidden-state injection:
  - cache clean identity-preserving query/hidden features;
  - inject only in late denoising and face-region tokens.
- MoFE/CoFE-style face expert decomposition:
  - separate ArcFace identity, semantic/VIT, and detail/reference evidence;
  - eventually train a small adapter or gating module.

## Final Recommendation

The next coding task should be Direction 1: detector-guided face-preservation recovery. It is the fastest way to test the central hypothesis from the last experiment: identity drift becomes unrecoverable once the generated segment stops showing a detectable face. State-aware memory should be added alongside it, but the first success criterion should be simple: prevent zero-face chunks from entering the long-horizon sequence.

