# HIPPOVLA for RoboSynChallenge

The **complete inference package is [`policy/hippovla/`](policy/hippovla/README.md)**. The official composer may copy only that directory; it includes the client adapter, model service, StarVLA modules, custom Transformers, installation and launch scripts, model metadata, immutable dependency URLs and all ten task mappings.

This corrects the incomplete packaging of frozen submission `80326a6a80f6f0ded34d2f7847915f31cea33f2a`. Model/client implementation and evaluation configuration are unchanged: RynnBrain-8B + DiT-B, 14-dimensional delta actions, 16-step prediction horizon, HIPPO Plus memory, four-step replanning, seed 0 and the same inference timing boundaries. A SHA-256 identity record covers 2,267 unchanged files. Weights are downloaded separately and need not be committed.

Follow the [self-contained runtime README](policy/hippovla/README.md) from the composed directory:

```bash
cd policy/hippovla
# In the inference environment, after installing compatible torch/torchvision:
bash install_policy_server.sh
python -m pip install flash-attn==2.8.3 --no-build-isolation
python verify_install.py
python scripts/verify_policy_bundle.py --runtime
python scripts/download_models.py --models-dir /absolute/path/to/models
bash run_policy_server.sh \
  /absolute/path/to/models/hippovla/final_model/pytorch_model.pt \
  /absolute/path/to/models/RynnBrain8B 0 5711
```

Reference environment: Linux, Python 3.10, NVIDIA A100 80 GB, torch 2.6.0+cu124, torchvision 0.21.0+cu124, flash-attn 2.8.3, numpy 1.26.4. The custom Transformers source ships inside the policy. Simulation uses the separate official RoboSynChallenge/EmbodiChain environment, with `policy/hippovla/requirements-eval.txt` and ffmpeg/libx264.

Immutable dependencies, unchanged for every task:

- [Policy checkpoint and metadata: ovoovo/hippovla @ 352a6cf7b419734fd0b13ce75a4205c2073c03ee](https://huggingface.co/ovoovo/hippovla/tree/352a6cf7b419734fd0b13ce75a4205c2073c03ee).
- [Complete RynnBrain-8B base model @ 04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B/tree/04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe).

Both [`submission.json`](submission.json) and the identical bundled [`policy/hippovla/submission.json`](policy/hippovla/submission.json) list all ten tasks with explicit immutable weight URLs. The checkpoint SHA-256 remains `c47a13143bc04d9a5d9dcd2ce148e7592b440905986c07953fad538417b0077b` (17,915,145,078 bytes).

Root `run_policy_server.sh`, `install_policy_server.sh`, `verify_install.py`, `scripts/eval.py` and `scripts/install_eval_adapter.py` forward to the bundled implementations. Their default model paths now resolve inside `policy/hippovla/models/`; supply explicit paths for weights previously downloaded into root `models/`. Root `models/` retains the original metadata and download notes for existing administrative upload tools.

Packaging checks and limits: [validation](docs/PACKAGING_VALIDATION.md). Backend response instructions and text: [correction response](docs/PACKAGING_CORRECTION.md). Submit the corrected published immutable code commit and **all ten mappings, including unchanged ones**, using the latest official response form appended by the backend, as requested in the evaluation team's email.

Historical preparation records: [validation history](docs/VALIDATION.md), [provenance](docs/provenance.json), [checkpoint identity](docs/checkpoint_identity.json), [base model verification](docs/base_model_verification.json). Code attribution and separate model terms: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Policy training used only official RoboSynChallenge simulation data (100% simulated, 0% real), as confirmed by the team on 2026-10-16; the exact dataset release/revision was not recorded. Base-model pretraining follows the upstream RynnBrain documentation. `items_handover` and `sample_loading` retain the prior task-coverage limitation; `table_rearrangement` retains the prior action/state unit consistency risk. No new full benchmark success rate is claimed.
