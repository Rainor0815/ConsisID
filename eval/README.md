# <u>Evaluation Pipeline</u> by *ConsisID*
This repo describes how to evaluate customized model in the [ConsisID](https://arxiv.org/abs/2411.17440) paper.

## ⚙️ Download Data and Weight

The Evaluate Data is available at [HuggingFace](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data), which will be used to sample videos by your own models. The weights will be automatically downloaded, or you can download it with the following commands.

```bash
cd util
python download_weights_eval.py
```

Once ready, the weights will be organized in this format:

```
📦 ConsisiID/
├── 📂 ckpts/
│   ├── 📂 data_process/
│       ├── 📂 clip-vit-base-patch32
│   ├── 📂 face_encoder/
```

## 🗝️ Usage

### Way 1 - Step by step

```
# Get FaceSim-Score and FID-Score
python get_facesim_fid.py
# Get CLIPScore
python get_clipscore.py
```

### Way 2 - All in once

```
pip install natsort
python eval_all_in_once.py
```

We would like to thank [@z-jiaming](https://github.com/z-jiaming) for his work on this [eval_all-in-once.py](https://github.com/PKU-YuanGroup/ConsisID/blob/main/eval/eval_all_in_once.py).

### Realistic identity persistence gate

Before evaluating stylized/anime/manga prompts, run the realistic-only persistence gate from `eval_command.txt`.

The gate intentionally checks for:

- at least 3 chunks (`--num_frames 49 --segments 3 --chunk_size 49`)
- high realistic identity similarity (`--min_mean_similarity`, `--min_frame_similarity`)
- low face-detection loss
- low chunk-to-chunk identity decay
- non-identical baseline and episodic videos
- measurable episodic improvement over baseline

It also rejects an episodic bank that only contains the exact same tensors as the active `identity_memory.pt`. A self-only bank can smoke-test loading, but it cannot prove memory-based identity preservation.

## 🔒 Limitation

- The currently released [video_caption_eval_old.csv](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data/blob/main/video_caption_eval_old.csv) is of low quality, and we further performed prompt refine on it in the article.  And we will release the latest csv in the future.
- The current code has not yet standardized the output format and currently only supports measuring a single video or prompt. We will continue to update the code in the future.
