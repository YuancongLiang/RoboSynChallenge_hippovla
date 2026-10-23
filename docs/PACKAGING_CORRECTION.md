# 本次打包补交说明

本次修复对应主办方邮件中的代码 commit `80326a6a80f6f0ded34d2f7847915f31cea33f2a` 和 evaluation bundle `e6acf9ed2151cd3d237e48fb42ba66ec6e40c955`。

## 已修复

- `policy/hippovla/` 现在包含完整客户端、模型服务、StarVLA 推理模块、自定义 Transformers、安装/启动/模型下载/校验脚本及必要元数据。官方仅复制此目录即可使用。
- 所有项目导入和启动路径均指向该目录内部；权重和完整基础模型可以保存在外部，通过绝对路径传入。
- `docs/inference_code_identity.json` 校验原冻结版本的 2,267 个文件。模型实现、适配器、动作解码、评测 YAML、种子、预测/重规划步数及计时边界保持不变。
- 所有任务仍使用同一原 checkpoint 和原 RynnBrain-8B 版本；未重新训练或修改权重。独立启动说明位于 `policy/hippovla/README.md`。

验证结果见 [PACKAGING_VALIDATION.md](PACKAGING_VALIDATION.md)，机器可读结果见 [packaging_smoke_result.json](packaging_smoke_result.json)。

## 发布和填写

本文件是补交材料，不会自动发布代码或提交表单。检查修改后，将完整改动提交并推送到原代码仓库，记录**实际包含本次修复且已发布**的 commit：

```bash
git status --short
# Review and commit the packaging correction, then push it to the code repository.
git rev-parse HEAD
```

将新 commit 拼成 `https://github.com/YuancongLiang/RoboSynChallenge_hippovla/tree/<NEW_CORRECTED_COMMIT>`。不要再次填写旧的 `80326a6...`，也不要填浮动的分支链接。

按此次邮件要求，通过**后台追加的最新官方 response form**提交新固定代码 URL 及下表**全部十个任务映射，包括未变更的映射**。当前对话没有该 response form 的实际地址；请从后台最新通知打开。此处不把原始提交入口或邮件回复当作该表单的替代入口。

固定基础模型完整快照：https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B/tree/04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe

策略权重相对路径：`final_model/pytorch_model.pt`。如果表单需要仓库目录 URL，十项均填 `https://huggingface.co/ovoovo/hippovla/tree/352a6cf7b419734fd0b13ce75a4205c2073c03ee`，并在 checkpoint 文件路径字段填上面的相对路径；如果需要文件 URL，使用下表。

| Task | Immutable checkpoint file URL |
|---|---|
| click_bell | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| drawer_open_place | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| handle_basket | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| item_assembly | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| items_handover | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| manipulate_pipette | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| mixer_operating | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| sample_loading | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| table_rearrangement | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |
| water_pouring | https://huggingface.co/ovoovo/hippovla/resolve/352a6cf7b419734fd0b13ce75a4205c2073c03ee/final_model/pytorch_model.pt |

`config.yaml` 与 `dataset_statistics.json` 必须从相同策略 revision 一并下载；完整固定链接记录于 `submission.json`。RynnBrain-8B 需要完整快照，不能只下载配置或其中一个分片。

## 可粘贴的英文说明

```text
This is a packaging-only correction of fixed submission 80326a6a80f6f0ded34d2f7847915f31cea33f2a (evaluation bundle e6acf9ed2151cd3d237e48fb42ba66ec6e40c955).

Corrected immutable code URL:
https://github.com/YuancongLiang/RoboSynChallenge_hippovla/tree/<NEW_CORRECTED_COMMIT>

The complete inference implementation now ships under policy/hippovla/, including the model service, StarVLA runtime, custom Transformers, launch/install/download/verification scripts, and matching metadata. See policy/hippovla/README.md for startup from the composed policy directory. Weights remain external.

The model and adapter implementation, checkpoint weights, seeds, action scheduling and timing semantics are unchanged. SHA-256 checks verify 2,267 unchanged source/configuration files against the frozen submission. All ten task checkpoint mappings are unchanged and are included in full in submission.json and in this response.

The complete RynnBrain-8B dependency is pinned at:
https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B/tree/04f588d50bf6f31b6b9f03f4d16d98dd0d22bcfe

The policy checkpoint and its config/statistics are pinned at:
https://huggingface.co/ovoovo/hippovla/tree/352a6cf7b419734fd0b13ce75a4205c2073c03ee
Checkpoint file: final_model/pytorch_model.pt
SHA-256: c47a13143bc04d9a5d9dcd2ce148e7592b440905986c07953fad538417b0077b

Validation: only policy/hippovla was copied outside the repository; offline startup strictly loaded the full checkpoint, and the real adapter obtained three finite (1,16,14) predictions at steps 0/4/8 with the official four-return interface. No full simulator benchmark was run. See docs/PACKAGING_VALIDATION.md.
```
