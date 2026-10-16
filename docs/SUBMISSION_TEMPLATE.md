# Submission form draft

Replace all `YOUR_*` values before submitting. This is a draft, not a completed request.

**Artifact name:** HIPPOVLA-RynnBrain8B-DiTB-delta16

**Experiment name:** robosynchallenge_hippo_plus_seed42_final

**Short description:** A shared bimanual manipulation policy built on RynnBrain-8B with a DiT-B action head and HIPPO Plus visual memory. Predicts 16 steps of 14-dimensional normalized joint deltas. All ten tasks use the same final checkpoint.

**Code URL:** https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY/tree/YOUR_COMMIT

**Hugging Face checkpoints:** Repeat `https://huggingface.co/YOUR_ACCOUNT/YOUR_MODEL/tree/YOUR_REVISION` in all ten task fields. Repository root contains `config.yaml`, `dataset_statistics.json`, and `final_model/pytorch_model.pt`.

**Data source:** TO COMPLETE FROM ACTUAL TRAINING RECORDS. State dataset repositories/versions, simulated and real data sources, task coverage, and any additional data. Config records `data_mix=robosynchallenge_all`; this alone is not sufficient provenance. Base VLM: Alibaba-DAMO-Academy/RynnBrain-8B; provide the actual snapshot revision.

**Run and dependency notes:**

Use the pinned GitHub commit and HF revision above. Follow README.md for two separate Python environments: policy inference (Python 3.10, torch 2.6.0+cu124 / torchvision 0.21.0+cu124 reference, flash-attn 2.8.3, bundled custom Transformers) and official RoboSynChallenge/EmbodiChain evaluation. Install `requirements-policy-server.txt` and the bundled Transformers with `install_policy_server.sh`; install `requirements-eval.txt` in the benchmark environment. ffmpeg must provide libx264. Start `bash run_policy_server.sh CHECKPOINT BASE_VLM SERVER_GPU 5711`. Install the adapter using `python scripts/install_eval_adapter.py "$ROBOSYNCHALLENGE_PATH"`.

For every task below run `"$ROBOSYN_PYTHON" scripts/eval.py TASK --gpu EVAL_GPU --episodes 50 --output eval_runs/TASK`. Task/checkpoint mapping:

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
