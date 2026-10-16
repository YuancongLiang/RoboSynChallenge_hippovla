---
language:
- en
- zh
tags:
- robotics
- robosynchallenge
- hippovla
base_model: Alibaba-DAMO-Academy/RynnBrain-8B
---

# HIPPOVLA RoboSynChallenge checkpoint

RynnBrainOFT / RynnBrain-8B + DiT-B, 14 continuous joint deltas, horizon 16, HIPPO Plus visual memory. One shared checkpoint for all ten RoboSynChallenge tasks.

Keep this structure when downloading:

```text
config.yaml
dataset_statistics.json
checkpoint_identity.json
final_model/pytorch_model.pt
```

Evaluation code: [YuancongLiang/RoboSynChallenge_hippovla](https://github.com/YuancongLiang/RoboSynChallenge_hippovla). Use the immutable GitHub commit URL supplied in the evaluation request. See that commit's README for task commands, dependencies and limitations. Use its bundled custom Transformers; stock Transformers is incompatible with the checkpoint memory implementation. The deployment predicts 16 actions and executes the first 4 before requesting a fresh prediction; this scheduling setting does not change the weights.

Policy training used only official RoboSynChallenge simulation data: 100% simulated, 0% real, with no additional policy-training datasets, as confirmed by the team on 2026-10-16. The local mixture is `robosynchallenge_all`. The exact upstream dataset release/revision was not recorded. This statement covers policy training; upstream base-model pretraining follows the RynnBrain-8B documentation. `items_handover` and `sample_loading` were not in the documented training task list. The table_rearrangement action/state unit risk remains unresolved.

Base VLM: [Alibaba-DAMO-Academy/RynnBrain-8B at 04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B/tree/04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe). The source snapshot's four weight shards and all runtime configuration/tokenizer/processor files were verified against upstream content hashes.

Checkpoint SHA-256: `c47a13143bc04d9a5d9dcd2ce148e7592b440905986c07953fad538417b0077b`; size: 17,915,145,078 bytes. All ten task fields may reference this same model revision and checkpoint file.

The weight license is not newly assigned by this packaging step. Confirm the applicable model and training data terms before choosing a repository license.
