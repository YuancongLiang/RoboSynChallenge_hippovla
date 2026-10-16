# Submission form draft

Use the clean GitHub commit containing these files. This document does not submit an evaluation request.

**Artifact name:** HIPPOVLA-RynnBrain8B-DiTB-delta16

**Experiment name:** robosynchallenge_hippo_plus_seed42_final

**Short description:** A shared bimanual manipulation policy built on RynnBrain-8B with a DiT-B action head and HIPPO Plus visual memory. Predicts 16 steps of 14-dimensional normalized joint deltas. All ten tasks use the same final checkpoint.

**Code URL:** Run `git rev-parse HEAD` in the clean submission checkout, then append the resulting commit to `https://github.com/YuancongLiang/RoboSynChallenge_hippovla/tree/`. Submit that immutable URL.

**Hugging Face checkpoints:** Repeat `https://huggingface.co/ovoovo/hippovla/tree/352a6cf7b419734fd0b13ce75a4205c2073c03ee` in all ten task fields. Repository root contains `config.yaml`, `dataset_statistics.json`, and `final_model/pytorch_model.pt`.

**Data source:** Policy training used only official RoboSynChallenge simulation data (100% simulated, 0% real), with no additional policy-training datasets, as confirmed by the team on 2026-10-16. Local mixture: `robosynchallenge_all`. The exact upstream dataset release/revision was not recorded. Base-model pretraining is separate; base VLM: `Alibaba-DAMO-Academy/RynnBrain-8B`, revision `04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe`.

**Run and dependency notes:**

Use the pinned GitHub commit and HF revision above. Follow README.md for two separate Python environments: policy inference (Python 3.10, torch 2.6.0+cu124 / torchvision 0.21.0+cu124 reference, flash-attn 2.8.3, bundled custom Transformers) and official RoboSynChallenge/EmbodiChain evaluation. Install `requirements-policy-server.txt` and the bundled Transformers with `install_policy_server.sh`; install `requirements-eval.txt` in the benchmark environment. ffmpeg must provide libx264. Start `bash run_policy_server.sh CHECKPOINT BASE_VLM SERVER_GPU 5711`. Install the adapter using `python scripts/install_eval_adapter.py "$ROBOSYNCHALLENGE_PATH"`.

For every task below run `"$ROBOSYN_PYTHON" scripts/eval.py TASK --gpu EVAL_GPU --episodes 50 --output eval_runs/TASK`. The adapter predicts 16 actions and executes the first 4 before replanning. It returns the official four values with a timing sample for each actual prediction, excluding cached actions and environment execution. All stopping checks use the official success interface. The installed standalone entrypoint is `bash "$ROBOSYNCHALLENGE_PATH/policy/hippovla/eval.sh" TASK CHECKPOINT EVAL_GPU 50`. Task/checkpoint mapping:

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

Normalized outputs are decoded using continuous min-max across all 14 dimensions, then reconstructed as measured state + cumulative deltas per chunk. CobotMagic action order is preserved. Grippers are mapped from ratio to actual joint limits except table_rearrangement (physical units). Memory: 5 frames at interval 10, gated fusion, slot attention, injection level [2], latest_history query, no temporal encoding. No state is sent to the policy network; measured state is used locally to decode deltas. Environment randomization and task limits come from the official task config. Known task coverage/unit limitations are documented in README.md. No new full benchmark score is claimed by this packaging step.
