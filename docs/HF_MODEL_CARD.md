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

Evaluation code: **TO COMPLETE with the GitHub repository and immutable commit**. See that repository's README for all task commands, dependencies and limitations. Use its bundled custom Transformers; stock Transformers is incompatible with the checkpoint memory implementation.

Training data provenance: **TO COMPLETE from actual training records**. The configuration's local mixture name is `robosynchallenge_all`; it does not establish a complete dataset citation. `items_handover` and `sample_loading` were not in the documented training task list. The table_rearrangement action/state unit risk remains unresolved.

The weight license is not newly assigned by this packaging step. Confirm the applicable model and training data terms before choosing a repository license.
