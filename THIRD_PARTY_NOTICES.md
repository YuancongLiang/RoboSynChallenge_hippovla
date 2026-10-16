# Third-party code and models

- StarVLA inference code retains its existing copyright notices and root LICENSE.
- `transformer/` contains a modified Hugging Face Transformers 4.57.6 source snapshot, including HIPPO Plus memory modules and Qwen3-VL changes. Apache-2.0 license text is retained at `transformer/LICENSE`; source headers remain intact. Install this snapshot rather than the stock package.
- RynnBrain-8B model files are distributed separately by Alibaba DAMO Academy: https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B. Follow the upstream model's terms.
- RoboSynChallenge and EmbodiChain are separately installed dependencies. This repository does not redistribute their simulation assets or environments.
- The root code license does not assign new licensing terms to the separately distributed trained checkpoint or its training data.
