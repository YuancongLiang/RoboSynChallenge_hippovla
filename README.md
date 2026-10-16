# HIPPOVLA for RoboSynChallenge

This repository packages inference and evaluation for the checkpoint selected in the deployment guide: **RynnBrain-8B + DiT-B, 14-dimensional delta actions, 16-step chunks, HIPPO Plus memory**. One shared checkpoint covers all ten submission tasks. Code is prepared from the working deployment at `/data/milu/RoboSynChallenge_ws/robosyn_starvla_policy_server`.

中文上传步骤与网站表单说明：[提交指南](docs/SUBMISSION_GUIDE.zh-CN.md)。表单草稿：[SUBMISSION_TEMPLATE.md](docs/SUBMISSION_TEMPLATE.md)。打包验证结果见 [VALIDATION.md](docs/VALIDATION.md)。

The GitHub repository contains code, matching training configuration, and normalization statistics. Policy weights and the base VLM are downloaded separately. No simulator assets, training datasets, Python environments, or benchmark results are bundled.

## 1. Download models

After uploading the checkpoint to your Hugging Face model repository, pin its immutable revision:

```bash
export POLICY_REPO=YOUR_ACCOUNT/YOUR_MODEL
export POLICY_REVISION=YOUR_IMMUTABLE_HF_REVISION
hf download "$POLICY_REPO" --revision "$POLICY_REVISION" --local-dir models/hippovla

export BASE_REVISION=YOUR_VERIFIED_RYNNBRAIN_REVISION
hf download Alibaba-DAMO-Academy/RynnBrain-8B --revision "$BASE_REVISION" \
  --local-dir models/RynnBrain8B
```

`hf` is supplied by `huggingface_hub`. The base snapshot used on the source server has no recorded upstream revision; select and record the matching snapshot rather than assuming the latest one is identical. The selected policy's exact identity is in [checkpoint_identity.json](docs/checkpoint_identity.json).

```text
models/
├── hippovla/
│   ├── config.yaml
│   ├── dataset_statistics.json
│   └── final_model/pytorch_model.pt
└── RynnBrain8B/
    ├── config.json
    ├── model-*.safetensors
    ├── tokenizer_config.json
    ├── preprocessor_config.json
    └── ... complete Hugging Face snapshot
```

The `.pt` loader reads metadata from `Path(checkpoint).parents[1]`. Keep the checkpoint in `final_model/` or `checkpoints/`, one level below its metadata directory. Keep the original training config and statistics together. `STARVLA_BASE_VLM` overrides the machine-specific base model path at runtime.

## 2. Install inference environment

Reference environment: Linux, Python 3.10, NVIDIA A100 80 GB, torch 2.6.0+cu124, torchvision 0.21.0+cu124, flash-attn 2.8.3, numpy 1.26.4. GPU memory requirements vary; use BF16. Install a torch/CUDA-compatible flash-attn build.

```bash
conda create -n hippovla python=3.10 -y
conda activate hippovla
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
bash install_policy_server.sh
python -m pip install flash-attn==2.8.3 --no-build-isolation
PYTHONPATH="$PWD/transformer/src:$PWD" python verify_install.py
```

The bundled custom Transformers implements the checkpoint's memory architecture. Do not replace it with a stock Transformers installation. No training framework or training data is needed for inference.

## 3. Start policy server

From the repository root, in the inference environment:

```bash
# Positional arguments: checkpoint, complete base VLM directory, physical GPU, port.
bash run_policy_server.sh \
  "$PWD/models/hippovla/final_model/pytorch_model.pt" \
  "$PWD/models/RynnBrain8B" 0 5711
```

Wait for the WebSocket listening message and check for loading errors. Model initialization uses strict checkpoint loading. The default checkpoint and base paths are the ones above; `bash run_policy_server.sh` uses GPU 0 / port 5711. `STARVLA_PYTHON` chooses the interpreter; `ROBOSYN_USE_BF16=1` and `ROBOSYN_SERVER_IDLE_TIMEOUT=-1` are defaults. The service binds `0.0.0.0` for the evaluator connection.

## 4. Install official evaluation environment and adapter

Install [RoboSynChallenge](https://github.com/EDEM-AI/RoboSynChallenge) and EmbodiChain using the [official installation guide](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html). Use their dedicated environment for simulation. Our source checkout references are recorded in [provenance.json](docs/provenance.json); these do not claim unmodified upstream checkouts.

```bash
export ROBOSYNCHALLENGE_PATH=/absolute/path/to/RoboSynChallenge
export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain
export ROBOSYN_PYTHON=/absolute/path/to/EmbodiChain/.venv/bin/python

"$ROBOSYN_PYTHON" -m pip install -r requirements-eval.txt
"$ROBOSYN_PYTHON" scripts/install_eval_adapter.py "$ROBOSYNCHALLENGE_PATH"
ffmpeg -version
```

`ffmpeg` must support `libx264`. The installer adds `policy/hippovla/` and refuses to overwrite an existing directory. It exports `get_model`, `eval`, `reset_model`, `encode_obs`, and `encode_action` as required by the [official policy interface](https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html).

Server and simulation can use separate GPUs or hosts. For a remote server, set `ROBOSYN_POLICY_HOST` to its address. Both machines need the matching metadata/checkpoint directory layout; the client reads configuration and statistics locally. Do not export the server's CUDA mask into the evaluation environment; the launcher removes it and selects the simulator GPU with `--gpu_id`.

## 5. Run all ten tasks

Keep the server running. From the repository root, using the simulation interpreter:

```bash
"$ROBOSYN_PYTHON" scripts/eval.py all --gpu 1 --episodes 50 \
  --output eval_runs/hippovla_all_001
```

All tasks use `models/hippovla/final_model/pytorch_model.pt` from the same pinned HF revision. Individual task commands:

| Task | Command |
|---|---|
| click_bell | `"$ROBOSYN_PYTHON" scripts/eval.py click_bell --gpu 1 --episodes 50 --output eval_runs/click_bell` |
| drawer_open_place | `"$ROBOSYN_PYTHON" scripts/eval.py drawer_open_place --gpu 1 --episodes 50 --output eval_runs/drawer_open_place` |
| handle_basket | `"$ROBOSYN_PYTHON" scripts/eval.py handle_basket --gpu 1 --episodes 50 --output eval_runs/handle_basket` |
| item_assembly | `"$ROBOSYN_PYTHON" scripts/eval.py item_assembly --gpu 1 --episodes 50 --output eval_runs/item_assembly` |
| items_handover | `"$ROBOSYN_PYTHON" scripts/eval.py items_handover --gpu 1 --episodes 50 --output eval_runs/items_handover` |
| manipulate_pipette | `"$ROBOSYN_PYTHON" scripts/eval.py manipulate_pipette --gpu 1 --episodes 50 --output eval_runs/manipulate_pipette` |
| mixer_operating | `"$ROBOSYN_PYTHON" scripts/eval.py mixer_operating --gpu 1 --episodes 50 --output eval_runs/mixer_operating` |
| sample_loading | `"$ROBOSYN_PYTHON" scripts/eval.py sample_loading --gpu 1 --episodes 50 --output eval_runs/sample_loading` |
| table_rearrangement | `"$ROBOSYN_PYTHON" scripts/eval.py table_rearrangement --gpu 1 --episodes 50 --output eval_runs/table_rearrangement` |
| water_pouring | `"$ROBOSYN_PYTHON" scripts/eval.py water_pouring --gpu 1 --episodes 50 --output eval_runs/water_pouring` |

Use `--checkpoint`, `--host`, `--port`, `--seed`, and `--robosyn-root` to override paths and connection settings. `--dry-run` prints resolved commands without starting simulation. Existing output directories are rejected. Each task runs sequentially in one environment. Maximum steps come from that task's current `configs/TASK/random/gym_config.json`; lighting/material randomization is preserved.

Each task gets `eval.log`, `command.json`, `gym_config_source.json` (the source task config, not a full runtime state), and `videos/`. The adapter saves all three camera frame sequences and encodes a 25 FPS MP4 for each at episode end. Failed recordings surface as exceptions. The launcher `summary.json` records process exit codes only; read the official logs for completed episodes, exceptions, and success counts before reporting results. A zero exit code alone does not establish 50 completed episodes.

## Action and memory semantics

The policy returns normalized actions. The adapter performs: continuous min-max denormalization over **all 14 dimensions**, then `absolute[t] = measured_state + cumsum(delta)[t]` using a fresh measured state for every new chunk. It preserves `[left_arm(6), left_gripper, right_arm(6), right_gripper]`, executes 16 steps per chunk, and does not binarize grippers. Gripper ratios are clipped to [0,1] then mapped to the robot's actual joint limits; `table_rearrangement` uses physical units directly. The policy network receives three RGB views resized to 224×224; measured state is used for local delta reconstruction and is not sent to the model.

Memory matches training: 5 sparse historical frames, interval 10, chronological order, gated fusion, slot attention enabled, injection levels [2], query `latest_history`, temporal encoding `none`. The adapter integrates the measured-state decoding and three-view recording previously supplied by local evaluation wrappers, so no historical run directory is required.

`items_handover` and `sample_loading` were absent from the documented training task list; their ratio mappings need further validation. `table_rearrangement` retains an unresolved training action/state unit consistency risk. This package does not claim a new full benchmark success rate.

## Upload and provenance

Upload extracted source files to GitHub. Stage weights with `scripts/prepare_hf_upload.py` and upload them to a separate HF model repository. Fill the URL/revision/data-source fields in `submission.json`, the HF model card and [submission template](docs/SUBMISSION_TEMPLATE.md). The public submission frontend reported new requests closed on 2026-10-16; use the applicable organizer-approved request/correction process. Private GitHub access and gated HF access rules are detailed in the Chinese submission guide.

The source checkpoint SHA-256 is `c47a13143bc04d9a5d9dcd2ce148e7592b440905986c07953fad538417b0077b` (17,915,145,078 bytes). Metadata is copied unchanged from the same training run. Code attribution and separate model terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
