# HIPPOVLA: complete composed policy

This directory is the complete inference package for the RoboSynChallenge submission. The official composer can copy **only `policy/hippovla/`** into its benchmark checkout. No other files from our repository are required. The model service remains a separate process from the simulator, using the same WebSocket client/server architecture as frozen submission `80326a6a80f6f0ded34d2f7847915f31cea33f2a`.

The packaging correction preserves model code, client code, evaluation configuration, weights, seeds and timing semantics. `docs/inference_code_identity.json` records SHA-256 identities of 2,267 unchanged files from that frozen commit. The prediction horizon remains 16; `eval_action_steps: 4`, `replan_action_steps: 4`, seed 0, DDIM 10 steps, BF16 and idle timeout -1 remain unchanged. Runtime source is shipped here; Python dependencies and the official benchmark/simulator are installed separately. Model weights may live outside this directory.

## Contents

```text
hippovla/
  deploy_policy.py, deploy_policy.yml, history_flow_utils.py, eval.sh
  deployment/model_server/     # model service, protocol and runtime guard
  starVLA/                    # RynnBrainOFT, action head and inference helpers
  transformer/                # complete custom Transformers source + setup/license
  install_policy_server.sh, run_policy_server.sh, verify_install.py
  requirements-policy-server.txt, requirements-eval.txt
  scripts/                    # model download, installation, evaluation, verification
  models/hippovla/             # unchanged metadata; weights downloaded separately
  docs/                       # identity and upstream verification records
  submission.json             # all ten task mappings and immutable model URLs
```

## Immutable models

| Dependency | Immutable snapshot | Required contents |
|---|---|---|
| Policy, shared by all ten tasks | [ovoovo/hippovla @ 352a6cf7b419734fd0b13ce75a4205c2073c03ee](https://huggingface.co/ovoovo/hippovla/tree/352a6cf7b419734fd0b13ce75a4205c2073c03ee) | `final_model/pytorch_model.pt`, `config.yaml`, `dataset_statistics.json` |
| Base VLM | [Alibaba-DAMO-Academy/RynnBrain-8B @ 04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B/tree/04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe) | Complete snapshot: all four safetensors shards, index, config, tokenizer, processor and chat template |

Policy [weight file](https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt), [config](https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/config.yaml) and [statistics](https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/dataset_statistics.json) are required together. Weight SHA-256: `c47a13143bc04d9a5d9dcd2ce148e7592b440905986c07953fad538417b0077b`; size: 17,915,145,078 bytes. Upstream verification records for both models are in `docs/`.

The loader reads metadata at `Path(checkpoint).parents[1]`: keep `final_model/pytorch_model.pt` one level below `config.yaml` and `dataset_statistics.json`. The launcher supplies `STARVLA_BASE_VLM` to override the original machine-specific training path without editing the configuration.

## Install and start from the composed directory

Use separate inference and official simulation environments. Reference inference environment: Linux, Python 3.10, NVIDIA A100 80 GB, torch 2.6.0+cu124, torchvision 0.21.0+cu124, flash-attn 2.8.3 and numpy 1.26.4. Run these commands from this `hippovla` directory, including when it has been copied into the official checkout:

```bash
conda create -n hippovla python=3.10 -y
conda activate hippovla
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
bash install_policy_server.sh
python -m pip install flash-attn==2.8.3 --no-build-isolation
python verify_install.py
python scripts/verify_policy_bundle.py --runtime

# Downloads both complete snapshots, using submission.json's immutable revisions.
python scripts/download_models.py --models-dir /absolute/path/to/models

bash run_policy_server.sh \
  /absolute/path/to/models/hippovla/final_model/pytorch_model.pt \
  /absolute/path/to/models/RynnBrain8B 0 5711
```

The launcher's positional arguments are `CHECKPOINT BASE_VLM GPU PORT [CONFIG]`. With no arguments it uses `models/hippovla/final_model/pytorch_model.pt`, `models/RynnBrain8B`, GPU 0, port 5711. `STARVLA_PYTHON` selects the inference interpreter. The script resolves its own directory, exports only bundled project import paths, verifies the custom `direct_mean_pool_v1` memory implementation, loads the checkpoint strictly and binds `0.0.0.0`. It can be invoked from any working directory, with absolute or caller-relative model paths. The service must be listening before evaluation starts. No server autostart is added to the timed client path.

Do not install stock Transformers over the bundled custom source. Flash-attn must match the torch/CUDA installation. Run `python verify_install.py` in the inference environment to check dependencies and GPU access.

## Official simulation environment

The official RoboSynChallenge checkout and EmbodiChain supply `scripts/eval_policy.py`, task configurations and simulator assets. They are external benchmark dependencies; the model service needs none of their modules. The adapter targets the official four-return interface checked at RoboSynChallenge commit `64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6`.

```bash
# Use the benchmark's interpreter for these commands.
export ROBOSYN_PYTHON=/absolute/path/to/EmbodiChain/.venv/bin/python
export ROBOSYNCHALLENGE_PATH=/absolute/path/to/RoboSynChallenge
export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain
"$ROBOSYN_PYTHON" -m pip install -r requirements-eval.txt
```

If the official composer already installed `policy/hippovla`, use it directly. For a separate local benchmark checkout, copy the complete policy once:

```bash
"$ROBOSYN_PYTHON" scripts/install_eval_adapter.py "$ROBOSYNCHALLENGE_PATH"
```

The installer refuses to overwrite an existing policy and excludes model weights/caches. Evaluation requires ffmpeg with libx264 for the unchanged three-camera 25 FPS recording. Choose the simulator GPU separately from the server GPU:

```bash
bash "$ROBOSYNCHALLENGE_PATH/policy/hippovla/eval.sh" \
  click_bell /absolute/path/to/models/hippovla/final_model/pytorch_model.pt 1 50

# Or run all ten tasks with per-task output directories and official task limits:
"$ROBOSYN_PYTHON" scripts/eval.py all \
  --robosyn-root "$ROBOSYNCHALLENGE_PATH" \
  --checkpoint /absolute/path/to/models/hippovla/final_model/pytorch_model.pt \
  --gpu 1 --episodes 50 --output /absolute/path/to/new_eval_output
```

`eval.sh` arguments are `TASK CHECKPOINT [GPU=0] [EPISODES=50] [--KEY VALUE ...]`. Set `ROBOSYN_POLICY_HOST` for a remote service and `ROBOSYN_SERVER_PORT` for another port. Both processes need the matching checkpoint metadata layout. The client executes 4 actions from each 16-step prediction and returns `(observation, info, truncated, inference_times_s)`. Timing covers preprocessing, memory preparation, the synchronous WebSocket request, delta decoding and executable-action transfer; cached actions, simulation steps and video encoding add no inference samples. Only official `is_task_success` checks are used.

## Isolated verification

Copy this directory into an empty location outside the source repository; do not copy repository siblings. In the inference environment:

```bash
cd /isolated/location/policy/hippovla
unset PYTHONPATH
python scripts/verify_policy_bundle.py --runtime
python verify_install.py
# Start run_policy_server.sh with external model paths as above, then:
python scripts/smoke_inference.py \
  --checkpoint /absolute/path/to/models/hippovla/final_model/pytorch_model.pt \
  --port 5711 --output /tmp/hippovla_smoke.json
```

The verifier checks frozen source identities, entrypoints, all ten mappings and that loaded project modules reside inside the copied policy. The smoke test performs real model inference through the unchanged adapter using synthetic RGB/qpos transitions: finite `(1,16,14)` outputs at steps 0/4/8, three positive timing samples, 12 environment steps and the four-return interface. This is not a full simulation benchmark. It disables test video recording and uses the existing `table_rearrangement` physical gripper mapping because the mock environment has no robot joint limits.

## Unchanged task mappings

Every task uses `final_model/pytorch_model.pt` at policy revision `352a6cf7b419734fd0b13ce75a4205c2073c03ee` and the base revision above. All ten explicit immutable weight URLs are recorded in `submission.json`.

| Task | Checkpoint file |
|---|---|
| click_bell | final_model/pytorch_model.pt |
| drawer_open_place | final_model/pytorch_model.pt |
| handle_basket | final_model/pytorch_model.pt |
| item_assembly | final_model/pytorch_model.pt |
| items_handover | final_model/pytorch_model.pt |
| manipulate_pipette | final_model/pytorch_model.pt |
| mixer_operating | final_model/pytorch_model.pt |
| sample_loading | final_model/pytorch_model.pt |
| table_rearrangement | final_model/pytorch_model.pt |
| water_pouring | final_model/pytorch_model.pt |

`items_handover` and `sample_loading` retain the prior training-task coverage limitation; `table_rearrangement` retains the prior action/state unit consistency risk. No new success-rate claim is made. Policy training used only official RoboSynChallenge simulation data, as previously confirmed by the team; the exact dataset release/revision was not recorded. Model terms and code attribution are in `THIRD_PARTY_NOTICES.md` and the retained licenses.
